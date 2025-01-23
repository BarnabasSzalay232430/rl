import json
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from sim_class_robotics_pipeline import Simulation
from pid_class import PIDController
from stable_baselines3 import PPO
import os

# Set the current working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
PID_GAINS = {"Kp": [15.0, 15.0, 15.0], "Ki": [0.0, 0.0, 0.0], "Kd": [0.8, 0.8, 0.8]}  # PID controller gains
TIME_STEP = 1.0  # Time interval for PID updates
ACCURACY_THRESHOLD = 0.001  # Distance threshold to consider the goal position reached
HOLD_DURATION = 50  # Number of consecutive steps within the threshold to confirm success
MAX_ITERATIONS = 1000  # Maximum number of iterations for each simulation
START_POSITION = [0.10775, 0.062, 0.17]  # Initial pipette position
MODEL_ZIP_PATH = "C:/Users/szala/Documents/GitHub/rl/best_model.zip"  # Path to the RL model
CV_MODEL_PATH = "C:/Users/szala/Documents/GitHub/renforcement_learning_232430/232430_unet_model_128px_v9md_checkpoint.keras"  # Path to the CV model

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

    This function initializes the simulation with a random plate image, extracts the
    plate image path and root coordinates using the CV model, and returns them.

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


def reinitialize_simulation_with_image(plate_image, root_coordinates, cv_model_path, rl_model=None):
    """
    Reinitialize a simulation with a specific plate image and pre-extracted root coordinates.

    This function ensures that the same plate image and root coordinates are used across
    different controllers (PID and RL) to maintain consistency during benchmarking.

    Args:
        plate_image (str): Path to the plate image to use.
        root_coordinates (list): Pre-extracted root coordinates.
        cv_model_path (str): Path to the CV model.
        rl_model: RL model to use, if any (default: None).

    Returns:
        Simulation: Reinitialized simulation object with the specified plate image and coordinates.
    """
    simulation = Simulation(num_agents=1, render=False, rl_model=rl_model, cv_model_path=cv_model_path)
    simulation.plate_image_path = plate_image  # Use the pre-loaded plate image
    simulation.root_coordinates = root_coordinates  # Use the pre-extracted coordinates

    logging.info(f"Reinitialized simulation with pre-loaded plate image: {plate_image}")
    return simulation


def benchmark_controllers():
    """
    Benchmark the performance of PID and RL controllers.

    This function benchmarks the performance of both the PID and RL controllers using
    the same plate image and root coordinates to ensure consistency. It logs and returns
    the results for both controllers.

    Returns:
        tuple: Results for the PID controller and RL controller.
    """
    # Step 1: Initialize the first simulation and extract plate image and root coordinates
    simulation_pid, plate_image, root_coordinates = initialize_simulation_and_coordinates(CV_MODEL_PATH)

    # Step 2: Reinitialize the RL simulation with the same plate image and coordinates
    rl_model = load_rl_model(MODEL_ZIP_PATH)
    simulation_rl = reinitialize_simulation_with_image(plate_image, root_coordinates, CV_MODEL_PATH, rl_model)

    # Benchmarking results
    pid_results = []
    rl_results = []

    for idx, goal_position in enumerate(root_coordinates):
        logging.info(f"Benchmarking for root tip {idx + 1}: {goal_position}")

        # Benchmark PID controller
        pid_time, pid_distance = run_pid_simulation(simulation_pid, goal_position)
        pid_results.append({"time": pid_time, "distance": pid_distance})

        # Benchmark RL controller
        rl_time, rl_distance = run_rl_simulation(simulation_rl, rl_model, goal_position)
        rl_results.append({"time": rl_time, "distance": rl_distance})

    return pid_results, rl_results


def run_pid_simulation(simulation, goal_position):
    """
    Run the PID simulation for a single root tip.

    Args:
        simulation (Simulation): The simulation environment.
        goal_position (list): Target position to reach.

    Returns:
        tuple: Elapsed time and final distance to the target position.
    """
    controller = PIDController(PID_GAINS['Kp'], PID_GAINS['Ki'], PID_GAINS['Kd'], TIME_STEP)
    simulation.reset(num_agents=1)
    simulation.set_start_position(*START_POSITION)
    controller.reset()

    in_threshold_counter = 0
    current_position = np.array(simulation.get_pipette_position(simulation.robotIds[0]))
    start_time = time.time()

    for _ in range(MAX_ITERATIONS):
        # Compute control signals using PID
        control_signals = controller.compute(current_position, goal_position)
        action = np.clip(control_signals, -1, 1)  # Clip control signals to valid range
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


def visualize_results(pid_results, rl_results):
    """
    Visualize the benchmarking results using bar plots.

    Args:
        pid_results (list): Results for the PID controller.
        rl_results (list): Results for the RL controller.
    """
    pid_times = [r["time"] for r in pid_results]
    rl_times = [r["time"] for r in rl_results]
    pid_distances = [r["distance"] for r in pid_results]
    rl_distances = [r["distance"] for r in rl_results]

    # Plot time comparison
    plt.figure()
    plt.bar(["PID", "RL"], [np.mean(pid_times), np.mean(rl_times)], yerr=[np.std(pid_times), np.std(rl_times)], capsize=5)
    plt.title("Average Time to Reach Goal")
    plt.ylabel("Time (s)")
    plt.show()

    # Plot distance comparison
    plt.figure()
    plt.bar(["PID", "RL"], [np.mean(pid_distances), np.mean(rl_distances)], yerr=[np.std(pid_distances), np.std(rl_distances)], capsize=5)
    plt.title("Average Final Distance to Goal")
    plt.ylabel("Distance (m)")
    plt.show()


def main():
    """
    Main function to benchmark controllers and visualize results.

    This function benchmarks both the PID and RL controllers, saves the results
    to a JSON file, and visualizes the results using bar plots.
    """
    pid_results, rl_results = benchmark_controllers()

    # Save results to a JSON file
    with open("benchmark_results.json", "w") as f:
        json.dump({"PID": pid_results, "RL": rl_results}, f, indent=4)

    # Visualize the benchmarking results
    visualize_results(pid_results, rl_results)


if __name__ == "__main__":
    main()
