import logging
import os
import json
import numpy as np
import pybullet as p
from pid_class import PIDController
from sim_class_robotics_pipeline import Simulation
from stable_baselines3 import PPO
import time

# Set the current working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
START_POSITION = [0.10775, 0.062, 0.17]  # Initial pipette position in robot space
ACCURACY_THRESHOLD = 0.001  # Distance threshold to determine goal achievement
HOLD_DURATION = 50  # Number of consecutive steps within the threshold required for success
MAX_ITERATIONS = 1000  # Maximum iterations allowed for a simulation
TIME_STEP = 1.0  # Time interval for PID control
PID_GAINS = {"Kp": [15.0, 15.0, 15.0], "Ki": [0.0, 0.0, 0.0], "Kd": [0.8, 0.8, 0.8]}  # PID gains
MODEL_ZIP_PATH = r"C:\Users\szala\Documents\GitHub\rl\best_model.zip"  # Path to the RL model
CV_MODEL_PATH = r"C:\Users\szala\Documents\GitHub\renforcement_learning_232430\232430_unet_model_128px_v9md_checkpoint.keras"  # Path to the CV model


def load_rl_model(model_path):
    """
    Load a pre-trained RL model from the specified path.

    Args:
        model_path (str): Path to the RL model file.

    Returns:
        PPO: Loaded RL model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_path):
        logging.error(f"RL model file not found: {model_path}")
        raise FileNotFoundError(f"RL model file not found: {model_path}")

    logging.info(f"Loading RL model from: {model_path}")
    return PPO.load(model_path)


def run_pid_simulation(simulation, goal_position):
    """
    Simulate a robot using PID control to reach a specified goal position.

    Args:
        simulation (Simulation): The simulation environment.
        goal_position (list): Target position for the robot [x, y, z].

    Returns:
        bool: True if the goal is successfully reached and inoculated, False otherwise.
    """
    # Initialize the PID controller
    controller = PIDController(PID_GAINS['Kp'], PID_GAINS['Ki'], PID_GAINS['Kd'], TIME_STEP)
    simulation.reset(num_agents=1)
    simulation.set_start_position(*START_POSITION)
    controller.reset()

    in_threshold_counter = 0
    current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))

    logging.info(f"Starting PID control to reach goal position: {goal_position}")

    for iteration in range(MAX_ITERATIONS):
        # Compute control signals and apply actions
        control_signals = controller.compute(current_position, goal_position)
        action = np.clip(control_signals, -1, 1)
        simulation.run([np.concatenate([action, [0.0]])])  # Last value for unused dimensions

        # Update the robot's position and calculate distance to the goal
        current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))
        distance_to_goal = np.linalg.norm(current_position - goal_position)

        logging.info(f"Iteration {iteration + 1}: Current Position: {current_position}, Distance to Goal: {distance_to_goal:.6f}")

        # Check if the goal is within the accuracy threshold
        if distance_to_goal <= ACCURACY_THRESHOLD:
            in_threshold_counter += 1
            if in_threshold_counter >= HOLD_DURATION:
                logging.info(f"Goal position reached at iteration {iteration + 1}.")
                logging.info(f"Reached root tip at {current_position}. Attempting inoculation...")

                # Drop a droplet at the pipette position
                droplet_position = simulation.drop(robotId=simulation.robotIds[0])
                logging.info(f"Inoculation completed. Droplet placed at {droplet_position}.")

                # Render the droplet for visibility
                for _ in range(100):
                    p.stepSimulation()
                    time.sleep(1 / 240.0)

                return True
        else:
            in_threshold_counter = 0

    logging.warning("Max iterations reached. Goal not achieved.")
    return False


def run_rl_simulation(simulation, rl_model, goal_position):
    """
    Simulate a robot using an RL controller to reach a specified goal position.

    Args:
        simulation (Simulation): The simulation environment.
        rl_model (PPO): Pre-trained RL model.
        goal_position (list): Target position for the robot [x, y, z].

    Returns:
        bool: True if the goal is successfully reached and inoculated, False otherwise.
    """
    simulation.reset(num_agents=1)
    simulation.set_start_position(*START_POSITION)

    in_threshold_counter = 0
    current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))

    logging.info(f"Starting RL control to reach goal position: {goal_position}")

    for iteration in range(MAX_ITERATIONS):
        # Formulate observation and predict action using the RL model
        observation = np.concatenate((current_position, goal_position))
        action, _ = rl_model.predict(observation, deterministic=True)
        simulation.run([np.concatenate([action, [0.0]])])

        # Update the robot's position and calculate distance to the goal
        current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))
        distance_to_goal = np.linalg.norm(current_position - goal_position)

        logging.info(f"Iteration {iteration + 1}: Current Position: {current_position}, Distance to Goal: {distance_to_goal:.6f}")

        # Check if the goal is within the accuracy threshold
        if distance_to_goal <= ACCURACY_THRESHOLD:
            in_threshold_counter += 1
            if in_threshold_counter >= HOLD_DURATION:
                logging.info(f"Goal position reached at iteration {iteration + 1}.")
                logging.info(f"Reached root tip at {current_position}. Attempting inoculation...")

                # Drop a droplet at the pipette position
                droplet_position = simulation.drop(robotId=simulation.robotIds[0])
                logging.info(f"Inoculation completed. Droplet placed at {droplet_position}.")

                # Render the droplet for visibility
                for _ in range(100):
                    p.stepSimulation()
                    time.sleep(1 / 240.0)

                return True
        else:
            in_threshold_counter = 0

    logging.warning("Max iterations reached. Goal not achieved.")
    return False


def main_simulation(use_rl=True):
    """
    Main function to run the simulation using either PID control or an RL controller.

    Args:
        use_rl (bool): Whether to use the RL controller. Defaults to True.
    """
    # Load the RL model if specified
    rl_model = None
    if use_rl:
        logging.info("Using RL model for the simulation.")
        rl_model = load_rl_model(MODEL_ZIP_PATH)
    else:
        logging.info("Using PID control for the simulation.")

    # Initialize the simulation environment
    simulation = Simulation(num_agents=1, render=True, rl_model=rl_model, cv_model_path=CV_MODEL_PATH)

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

            if use_rl:
                success = run_rl_simulation(simulation, rl_model, coordinate)
            else:
                success = run_pid_simulation(simulation, coordinate)

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
    use_rl = True  # Set to False to use PID control
    main_simulation(use_rl=use_rl)
