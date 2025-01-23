"""
This script simulates a robotic pipeline using a PID controller to move a robotic pipette to target positions
extracted from an image of a petri dish. The pipeline includes extracting root tip coordinates, running a PID-based
control simulation, and logging results.

"""

import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from pid_class import PIDController
from sim_class_robotics_pipeline import Simulation
import time

# Set the current working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
IMAGE_DIRECTORY = r"C:\Users\szala\Documents\GitHub\rl2\textures\_plates"
COORDINATES_FILE = r"C:\Users\szala\Documents\GitHub\renforcement_learning_232430\roottip_coordinates.json"
MODEL_PATH = r"C:\Users\szala\Documents\GitHub\renforcement_learning_232430\232430_unet_model_128px_v9md_checkpoint.keras"
START_POSITION = [0.10775, 0.062, 0.17]  # Initial pipette position in robot space
PID_GAINS = {"Kp": [15.0, 15.0, 15.0], "Ki": [0.0, 0.0, 0.0], "Kd": [0.8, 0.8, 0.8]}  # PID controller gains
TIME_STEP = 1.0  # Time interval for PID updates
ACCURACY_THRESHOLD = 0.001  # Distance threshold for goal achievement
HOLD_DURATION = 50  # Time steps to hold at the target to confirm success
MAX_ITERATIONS = 1000  # Maximum iterations for the simulation


def load_coordinates(file_path):
    """
    Load root tip coordinates from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing coordinates.

    Returns:
        dict: Loaded coordinates. Empty if file does not exist.
    """
    if not os.path.exists(file_path):
        logging.warning(f"Coordinates file not found: {file_path}")
        return {}
    with open(file_path, 'r') as f:
        return json.load(f)


def save_coordinates(file_path, data):
    """
    Save root tip coordinates to a JSON file.

    Args:
        file_path (str): Path to save the JSON file.
        data (dict): Data to save (root tip coordinates).
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Root coordinates saved to: {file_path}")


def run_pid_simulation(simulation, pid_gains, time_step, goal_position, max_iterations, accuracy_threshold, hold_duration):
    """
    Simulate a PID-controlled robot to reach a specified goal position.

    Args:
        simulation (Simulation): The simulation environment.
        pid_gains (dict): PID gains for the controller.
        time_step (float): Time interval for PID updates.
        goal_position (list): Target position for the robot [x, y, z].
        max_iterations (int): Maximum number of iterations.
        accuracy_threshold (float): Distance threshold for goal achievement.
        hold_duration (int): Required hold time at the goal.

    Returns:
        bool: True if goal is reached, False otherwise.
    """
    # Initialize PID controller
    controller = PIDController(pid_gains['Kp'], pid_gains['Ki'], pid_gains['Kd'], time_step)

    # Reset the simulation and set the start position
    simulation.reset(num_agents=1)
    simulation.set_start_position(*START_POSITION)
    controller.reset()

    # Initialize variables for tracking progress
    in_threshold_counter = 0
    current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))

    logging.info(f"Starting PID control to reach goal position: {goal_position}")

    for iteration in range(max_iterations):
        # Calculate control signals using PID
        control_signals = controller.compute(current_position, goal_position)
        action = np.clip(control_signals, -1, 1)  # Clip control signals to valid range

        # Apply the action in the simulation
        simulation.run([np.concatenate([action, [0.0]])])  # Append 0.0 for unused dimensions

        # Update the current position and calculate the distance to the goal
        current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))
        distance_to_goal = np.linalg.norm(current_position - goal_position)

        logging.info(f"Iteration {iteration + 1}: Current Position: {current_position}, Distance to Goal: {distance_to_goal:.6f}")

        # Check if the goal is reached within the accuracy threshold
        if distance_to_goal <= accuracy_threshold:
            in_threshold_counter += 1
            if in_threshold_counter >= hold_duration:
                logging.info(f"Goal position reached at iteration {iteration + 1}.")
                logging.info(f"Reached root tip at {current_position}. Attempting inoculation...")

                # Drop a droplet at the pipette position
                droplet_position = simulation.drop(robotId=simulation.robotIds[0])
                logging.info(f"Inoculation completed. Droplet placed at {droplet_position}.")

                # Render the droplet in the simulation
                for _ in range(100):  # Step simulation for visibility
                    p.stepSimulation()
                    time.sleep(1 / 240.0)

                return True
        else:
            in_threshold_counter = 0  # Reset counter if not within threshold

    logging.warning("Max iterations reached. Goal not achieved.")
    return False


def main_simulation(model_path):
    """
    Main function to run the PID-controlled simulation for all root tips.

    Args:
        model_path (str): Path to the CV model used for root tip extraction.
    """
    # Initialize the simulation with the given CV model
    simulation = Simulation(num_agents=1, render=True, cv_model_path=model_path, rl_model=None)

    try:
        # Extract root tip coordinates from the loaded plate image
        logging.info(f"Random image loaded: {simulation.get_plate_image()}")
        root_tips = simulation.get_root_coordinates()

        if not root_tips:
            logging.error("No root tips found in the image. Exiting.")
            return

        logging.info(f"Extracted root tip coordinates: {root_tips}")

        # Simulate robot movements for each root tip
        for idx, coordinate in enumerate(root_tips):
            logging.info(f"Starting simulation for root tip {idx + 1}.")
            success = run_pid_simulation(
                simulation=simulation,
                pid_gains=PID_GAINS,
                time_step=TIME_STEP,
                goal_position=coordinate,
                max_iterations=MAX_ITERATIONS,
                accuracy_threshold=ACCURACY_THRESHOLD,
                hold_duration=HOLD_DURATION,
            )
            if success:
                logging.info(f"Successfully inoculated root tip {idx + 1}.")
            else:
                logging.error(f"Failed to inoculate root tip {idx + 1}.")

    finally:
        # Clean up and close the simulation
        simulation.close()
        logging.info("Simulation completed.")


# Main Execution
if __name__ == "__main__":
    main_simulation(MODEL_PATH)
