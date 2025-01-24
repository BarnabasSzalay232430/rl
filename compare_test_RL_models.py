import json
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from sim_class_robotics_pipeline import Simulation
from stable_baselines3 import PPO
import os

# Set the current working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
MODEL_1_PATH = "C:/Users/szala/Documents/GitHub/rl/best_model.zip"  # Path to the first RL model
MODEL_2_PATH = "C:/Users/szala/Documents/GitHub/rl/best_personal_model.zip"  # Path to the second RL model
CV_MODEL_PATH = "C:/Users/szala/Documents/GitHub/renforcement_learning_232430/232430_unet_model_128px_v9md_checkpoint.keras"  # Path to the CV model
TIME_STEP = 1.0  # Time interval for updates
ACCURACY_THRESHOLD = 0.001  # Distance threshold to consider the goal position reached
HOLD_DURATION = 50  # Number of consecutive steps within the threshold to confirm success
MAX_ITERATIONS = 1000  # Maximum number of iterations for each simulation
START_POSITION = [0.10775, 0.062, 0.17]  # Initial pipette position

# Helper Functions
def load_rl_model(model_path):
    """
    Load a pre-trained reinforcement learning (RL) model.

    Args:
        model_path (str): Path to the RL model file.

    Returns:
        PPO: Loaded RL model.
    """
    logging.info(f"Loading RL model from: {model_path}")
    return PPO.load(model_path)

def initialize_simulation_and_coordinates(cv_model_path):
    """
    Initialize the simulation and extract the plate image and root coordinates.

    Args:
        cv_model_path (str): Path to the CV model for root tip detection.

    Returns:
        tuple: The initialized simulation, plate image path, and extracted root coordinates.

    Raises:
        ValueError: If no root coordinates are extracted.
    """
    simulation = Simulation(num_agents=1, render=False, rl_model=False, cv_model_path=cv_model_path)
    plate_image = simulation.get_plate_image()
    root_coordinates = simulation.get_root_coordinates()

    if not root_coordinates:
        raise ValueError("No root coordinates were extracted from the plate image.")

    logging.info(f"Initialized simulation with plate image: {plate_image}")
    logging.info(f"Extracted root coordinates: {root_coordinates}")

    return simulation, plate_image, root_coordinates

def reinitialize_simulation_with_image(plate_image, root_coordinates, cv_model_path, rl_model):
    """
    Reinitialize a simulation with a specific plate image and pre-extracted root coordinates.

    Args:
        plate_image (str): Path to the plate image to use.
        root_coordinates (list): Pre-extracted root coordinates.
        cv_model_path (str): Path to the CV model.
        rl_model: RL model to use.

    Returns:
        Simulation: Reinitialized simulation object with the specified plate image and coordinates.
    """
    simulation = Simulation(num_agents=1, render=False, rl_model=rl_model, cv_model_path=cv_model_path)
    simulation.plate_image_path = plate_image  # Use the pre-loaded plate image
    simulation.root_coordinates = root_coordinates  # Use the pre-extracted coordinates

    logging.info(f"Reinitialized simulation with pre-loaded plate image: {plate_image}")
    return simulation

def benchmark_models():
    """
    Benchmark the performance of two RL models.

    Returns:
        tuple: Results for the two RL models.
    """
    # Step 1: Initialize the first simulation and extract plate image and root coordinates
    simulation, plate_image, root_coordinates = initialize_simulation_and_coordinates(CV_MODEL_PATH)

    # Step 2: Load the two RL models
    model_1 = load_rl_model(MODEL_1_PATH)
    model_2 = load_rl_model(MODEL_2_PATH)

    # Step 3: Reinitialize simulations with the same plate image and coordinates
    simulation_model_1 = reinitialize_simulation_with_image(plate_image, root_coordinates, CV_MODEL_PATH, model_1)
    simulation_model_2 = reinitialize_simulation_with_image(plate_image, root_coordinates, CV_MODEL_PATH, model_2)

    # Benchmarking results
    model_1_results = []
    model_2_results = []

    for idx, goal_position in enumerate(root_coordinates):
        logging.info(f"Benchmarking for root tip {idx + 1}: {goal_position}")

        # Benchmark Model 1
        time_1, distance_1 = run_rl_simulation(simulation_model_1, model_1, goal_position)
        model_1_results.append({"time": time_1, "distance": distance_1})

        # Benchmark Model 2
        time_2, distance_2 = run_rl_simulation(simulation_model_2, model_2, goal_position)
        model_2_results.append({"time": time_2, "distance": distance_2})

    return model_1_results, model_2_results

def run_rl_simulation(simulation, rl_model, goal_position):
    """
    Run the RL simulation for a single root tip.

    Args:
        simulation (Simulation): The simulation environment.
        rl_model (PPO): Pre-trained RL model.
        goal_position (list): Target position to reach.

    Returns:
        tuple: Elapsed time and final distance to the target position.
    """
    simulation.reset(num_agents=1)
    simulation.set_start_position(*START_POSITION)

    in_threshold_counter = 0
    current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))
    start_time = time.time()

    for _ in range(MAX_ITERATIONS):
        # Create observation and predict action using RL model
        observation = np.concatenate((current_position, goal_position))
        action, _ = rl_model.predict(observation, deterministic=True)
        simulation.run([np.concatenate([action, [0.0]])])  # Run simulation step

        # Update current position and calculate distance to the goal
        current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))
        distance_to_goal = np.linalg.norm(current_position - goal_position)

        if distance_to_goal <= ACCURACY_THRESHOLD:
            in_threshold_counter += 1
            if in_threshold_counter >= HOLD_DURATION:
                elapsed_time = time.time() - start_time
                return elapsed_time, distance_to_goal
        else:
            in_threshold_counter = 0

    elapsed_time = time.time() - start_time
    return elapsed_time, distance_to_goal

def visualize_results(model_1_results, model_2_results):
    """
    Visualize the benchmarking results using bar plots.

    Args:
        model_1_results (list): Results for the first RL model.
        model_2_results (list): Results for the second RL model.
    """
    model_1_times = [r["time"] for r in model_1_results]
    model_2_times = [r["time"] for r in model_2_results]
    model_1_distances = [r["distance"] for r in model_1_results]
    model_2_distances = [r["distance"] for r in model_2_results]

    # Plot time comparison
    plt.figure()
    plt.bar(["Team Model", "Personal Model"], [np.mean(model_1_times), np.mean(model_2_times)], 
            yerr=[np.std(model_1_times), np.std(model_2_times)], capsize=5)
    plt.title("Average Time to Reach Goal")
    plt.ylabel("Time (s)")
    plt.savefig("average_time_to_goal.png")
    plt.show()

    # Plot distance comparison
    plt.figure()
    plt.bar(["Team Model", "Personal Moodel"], [np.mean(model_1_distances), np.mean(model_2_distances)], 
            yerr=[np.std(model_1_distances), np.std(model_2_distances)], capsize=5)
    plt.title("Average Final Distance to Goal")
    plt.ylabel("Distance (m)")
    plt.savefig("average_distance_to_goal.png")
    plt.show()

def main():
    """
    Main function to benchmark RL models and visualize results.
    """
    model_1_results, model_2_results = benchmark_models()

    # Save results to a JSON file
    with open("benchmark_results.json", "w") as f:
        json.dump({"Model 1": model_1_results, "Model 2": model_2_results}, f, indent=4)

    # Visualize the benchmarking results
    visualize_results(model_1_results, model_2_results)

if __name__ == "__main__":
    main()
