import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.morphology import skeletonize
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
PLATE_SIZE_MM = 150
PLATE_POSITION_ROBOT = [0.10775, 0.062, 0.17]  # Base position for robot
RECT_POSITIONS = [
    (0.09, 0.135, 0.18, 0.265),
    (0.27, 0.135, 0.36, 0.265),
    (0.45, 0.135, 0.54, 0.285),
    (0.61, 0.135, 0.75, 0.265),
    (0.80, 0.135, 0.95, 0.295),
]

# Model Losses and Metrics
@tf.keras.utils.register_keras_serializable()
def f1_metric(y_true, y_pred, threshold=0.3):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    TP = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    pred_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = TP / (pred_positives + tf.keras.backend.epsilon())
    recall = TP / (positives + tf.keras.backend.epsilon())
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    dice_loss = 1 - (2 * tf.reduce_sum(y_true * y_pred) + 1e-7) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7
    )
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * dice_loss + 0.5 * bce_loss

# Preprocess Image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image

# Extract Petri Dish
def extract_petri_dish(image):
    _, thresholded = cv2.threshold(image, 57, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logging.warning("No contours detected.")
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    max_side = max(w, h)
    x_center, y_center = x + w // 2, y + h // 2
    x_start = max(0, x_center - max_side // 2)
    y_start = max(0, y_center - max_side // 2)
    return image[y_start : y_start + max_side, x_start : x_start + max_side]

# Predict Root Mask
def predict_root_mask(image, model, patch_size=128, stride=64):
    h, w = image.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(np.stack([patch] * 3, axis=-1) / 255.0)
    patches = np.array(patches)
    predictions = model.predict(patches, verbose=0)
    mask = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for idx, pred in enumerate(predictions):
        y = (idx // ((w - patch_size) // stride + 1)) * stride
        x = (idx % ((w - patch_size) // stride + 1)) * stride
        mask[y : y + patch_size, x : x + patch_size] += pred[..., 0]
        count[y : y + patch_size, x : x + patch_size] += 1
    return (mask / np.maximum(count, 1)) > 0.5

# Connect Fragmented Roots
def connect_roots(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=15)

# Filter and Skeletonize Roots
def filter_and_skeletonize_roots(mask, rect_positions, min_area=200):
    """
    Filter roots based on highest point and area within rectangles, 
    limit to the largest object in each rectangle, and skeletonize them.
    """
    h, w = mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    root_data = []

    for rect_idx, (x_start, y_start, x_end, y_end) in enumerate(rect_positions):
        rect_x_start = int(x_start * w)
        rect_y_start = int(y_start * h)
        rect_x_end = int(x_end * w)
        rect_y_end = int(y_end * h)

        valid_objects = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            object_pixels = np.column_stack(np.where(labels == i))
            highest_point = object_pixels[np.argmin(object_pixels[:, 0])]
            if rect_x_start <= highest_point[1] <= rect_x_end and rect_y_start <= highest_point[0] <= rect_y_end:
                valid_objects.append((i, area))

        if not valid_objects:
            continue

        # Select the largest object in the rectangle
        largest_object = max(valid_objects, key=lambda x: x[1])
        root_data.append((rect_idx + 1, largest_object[0]))
        filtered_mask[labels == largest_object[0]] = 255

    # Skeletonize filtered roots
    skeletonized_mask = skeletonize(filtered_mask > 0).astype(np.uint8) * 255

    return filtered_mask, skeletonized_mask, root_data, labels

# Find Root Tips
def find_root_tips_and_convert_to_robot_coordinates(
    skeletonized_mask, root_data, labels, conversion_factor, petri_dish, plate_position_robot, output_path
):
    """
    Find root tips in pixel coordinates, convert them to mm, and then to robot coordinates.
    Annotate the image and save it with the converted coordinates.
    """
    annotated_image = cv2.cvtColor(petri_dish, cv2.COLOR_GRAY2BGR)
    root_tips_robot = []
    for rect_id, obj_id in root_data:
        # Get coordinates of the root tip
        coords = np.column_stack(np.where(labels == obj_id))
        if len(coords) == 0:
            continue
        # Find the lowest point of the root tip
        lowest_point = coords[np.argmax(coords[:, 0])]  # Max y-coordinate
        y_pixel, x_pixel = lowest_point

        # Convert to mm
        x_mm = x_pixel * conversion_factor
        y_mm = y_pixel * conversion_factor

        # Convert to robot coordinates
        x_robot = x_mm / 1000 + plate_position_robot[0]
        y_robot = y_mm / 1000 + plate_position_robot[1]
        z_robot = plate_position_robot[2]

        # Append robot coordinates
        root_tips_robot.append([x_robot, y_robot, z_robot])

        # Annotate the image
        cv2.circle(annotated_image, (x_pixel, y_pixel), radius=5, color=(0, 0, 255), thickness=-1)

        # Log the conversions
        logging.info(
            f"Pixel: ({x_pixel}, {y_pixel}), MM: ({x_mm:.2f}, {y_mm:.2f}), Robot: ({x_robot:.5f}, {y_robot:.5f}, {z_robot:.5f})"
        )

    # Save the annotated image
    cv2.imwrite(output_path, annotated_image)
    logging.info(f"Annotated image saved to: {output_path}")

    return root_tips_robot

# Run Pipeline with Conversion
def run_pipeline(image_path, model, plate_position_robot, output_dir=r"C:\Users\szala\Documents\GitHub\rl2\highlighted_roottips"):
    try:
        logging.info(f"Processing image: {image_path}")
        os.makedirs(output_dir, exist_ok=True)

        # Preprocess the image
        image = preprocess_image(image_path)
        petri_dish = extract_petri_dish(image)

        # Predict root mask
        root_mask = predict_root_mask(petri_dish, model)

        # Connect roots and filter
        connected_mask = connect_roots(root_mask)
        filtered_mask, skeletonized_mask, root_data, labels = filter_and_skeletonize_roots(
            connected_mask, RECT_POSITIONS
        )

        # Calculate conversion factor
        plate_size_pixels = petri_dish.shape[1]  # Assuming square plate
        conversion_factor = PLATE_SIZE_MM / plate_size_pixels

        # Find root tips and convert coordinates
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        root_tips_robot = find_root_tips_and_convert_to_robot_coordinates(
            skeletonized_mask, root_data, labels, conversion_factor, petri_dish, plate_position_robot, output_path
        )

        return root_tips_robot
    except Exception as e:
        logging.error(f"Failed to process {image_path}: {e}")
        return []

