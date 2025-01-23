from stable_baselines3 import PPO  # Adjust if using a different algorithm
from stable_baselines3.common.env_checker import check_env
import logging
import os
import json
import numpy as np
import pybullet as p
from sim_class_robotics_pipeline import Simulation
import time
from cv_pipeline import run_pipeline


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

# Constants
START_POSITION = [0.10775, 0.062, 0.17]  # Initial pipette position in robot space
ACCURACY_THRESHOLD = 0.001
HOLD_DURATION = 50
MAX_ITERATIONS = 1000
MODEL_ZIP_PATH = r"C:\Users\szala\Documents\GitHub\rl\best_model.zip"
CV_MODEL_PATH = r"C:\Users\szala\Documents\GitHub\renforcement_learning_232430\232430_unet_model_128px_v9md_checkpoint.keras"

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_rl_model(model_path):
    """Load the RL model from the specified path."""
    if not os.path.exists(model_path):
        logging.error(f"RL model file not found: {model_path}")
        raise FileNotFoundError(f"RL model file not found: {model_path}")
    logging.info(f"Loading RL model from: {model_path}")
    return PPO.load(model_path)


def run_rl_simulation(simulation, rl_model, goal_position, max_iterations, accuracy_threshold, hold_duration):
    """Run RL simulation for a single root tip coordinate."""
    simulation.reset(num_agents=1)
    simulation.set_start_position(*START_POSITION)

    in_threshold_counter = 0
    current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))

    logging.info(f"Starting RL control to reach goal position: {goal_position}")
    for iteration in range(max_iterations):
        # Define observation (e.g., current state + goal position)
        observation = np.concatenate((current_position, goal_position))

        # Predict action using RL model
        action, _ = rl_model.predict(observation, deterministic=True)

        # Run simulation for the action
        simulation.run([np.concatenate([action, [0.0]])])  # Append [0.0] for other control dimensions

        current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))
        distance_to_goal = np.linalg.norm(current_position - goal_position)
        logging.info(f"Iteration {iteration + 1}: Current Position: {current_position}, Distance to Goal: {distance_to_goal:.6f}")

        if distance_to_goal <= accuracy_threshold:
            in_threshold_counter += 1
            if in_threshold_counter >= hold_duration:
                logging.info(f"Goal position reached at iteration {iteration + 1}.")
                logging.info(f"Reached root tip at {current_position}. Attempting inoculation...")

                # Drop a droplet at the current pipette position
                droplet_position = simulation.drop(robotId=simulation.robotIds[0])
                logging.info(f"Inoculation completed. Droplet placed at {droplet_position}.")
                
                # Step the simulation to render the droplet
                for _ in range(100):  # Run steps to make the droplet visible
                    p.stepSimulation()
                    time.sleep(1 / 240.0)
                
                return True

        else:
            in_threshold_counter = 0

    logging.warning("Max iterations reached. Goal not achieved.")
    return False


def main_simulation():
    """Main pipeline to simulate robot movement based on root tip coordinates."""
    rl_model = load_rl_model(MODEL_ZIP_PATH)
    simulation = Simulation(num_agents=1, render=True, rl_model=rl_model, cv_model_path=CV_MODEL_PATH)

    try:
        # Load the image and extract root tips
        logging.info(f"Random image loaded: {simulation.get_plate_image()}")
        root_tips = simulation.get_root_coordinates()

        if not root_tips:
            logging.error("No root tips found in the image. Exiting.")
            return

        logging.info(f"Extracted root tip coordinates: {root_tips}")

        # Simulate robot movements for each root tip
        for idx, coordinate in enumerate(root_tips):
            logging.info(f"Starting simulation for root tip {idx + 1}.")
            success = run_rl_simulation(
                simulation=simulation,
                rl_model=rl_model,
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
        simulation.close()
        logging.info("Simulation completed.")


# Main Execution
if __name__ == "__main__":
    main_simulation()
