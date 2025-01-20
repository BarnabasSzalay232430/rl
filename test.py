
from sim_class import Simulation
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

if __name__ == "__main__":
    # Number of agents for the simulation
    num_agents = 1

    # Initialize the simulation
    sim = Simulation(num_agents=num_agents, render=True)

    try:
        # Run the simulation for a few steps to verify the image on the plate
        print(f"Plate image path: {sim.get_plate_image()}")
        
        # Run the simulation for a few steps
        sim.run(actions=[[0, 0, 0, 0]], num_steps=1000)
        
        print("Simulation ran successfully. Check the plate for the loaded image.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the simulation
        sim.close()
