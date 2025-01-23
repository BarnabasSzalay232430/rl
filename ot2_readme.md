### Reinforcement Learning Environment for Pipette Control  
This repository contains a custom reinforcement learning (RL) environment for controlling a pipette in a 3D workspace. The environment is compatible with the Gymnasium API and has been tested using stable-baselines3. It is designed to facilitate reinforcement learning research in robotics simulation.

Project Structure
.
├── ot2_env.py                # Script 1: Defines the custom Gym environment (OT2Env)
├── test_env.py               # Script 2: Tests the environment compatibility and functionality
├── ot2_readme.md                 # Instructions for environment setup and usage

Environment Description
The OT2Env class implements a custom RL environment that follows the Gymnasium interface. The agent controls a pipette's position in 3D space to reach a predefined goal while optimizing for efficiency.

Features:
Action Space: Continuous control of the pipette in the x, y, and z directions (spaces.Box).
Observation Space: Includes the pipette's current position and the goal position (spaces.Box).
Reward Function: Encourages proximity to the goal and penalizes unnecessary steps.
Termination Criteria:
Success when the pipette reaches the goal within a small threshold.
Episode truncation when the maximum number of steps is reached.
Setup Instructions
Prerequisites
Ensure you have Python 3.8 or higher installed. Clone this repository and navigate to its directory.


git clone https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git
Install Required Libraries
Install the dependencies


Required Libraries:
gymnasium: For creating and managing the RL environment.
stable-baselines3: For RL algorithm support and environment compatibility checks.
numpy: For numerical operations.
matplotlib: For visualization (optional).
cv2 (OpenCV): For image processing (used by the simulation).
random: For generating random goal positions.
math: For mathematical operations.
skimage: For connected component and skeletonization analysis.
To ensure compatibility, use the versions specified in requirements.txt.

Usage
Environment Setup
The custom environment is defined in ot2_env.py. To create an instance of the environment:


# Create the environment
from ot2_env import OT2Env


Testing the Environment
The script test_wrapper.py checks the environment's compatibility with the Gymnasium API and tests its functionality. Run the script using:

bash
Copy
Edit
python test_env.py
Expected output:

The environment compatibility check (check_env) should pass without errors.
Random actions will be sampled, and details such as rewards, termination status, and observations will be printed for each step.
Key Scripts
1. ot2_env.py
Defines the OT2Env class, implementing the custom Gymnasium-compatible RL environment.

Initialization (__init__): Sets up the simulation and defines the action and observation spaces.
Step (step): Executes an action, calculates rewards, and returns the next state and episode status.
Reset (reset): Resets the environment for a new episode.
Close (close): Cleans up resources.
2. test_env.py
Tests the OT2Env implementation:

Validates the Gym API compliance using check_env.
Runs one or more episodes with random actions.
Prints step-wise details, including actions, rewards, and termination status.
Sample Output
During Testing (test_env.py):
yaml
Copy
Edit
--- Starting Episode 1 ---
Step: 1, Action: [ 0.224 -0.873  0.517], Reward: -0.546, Terminated: False, Truncated: False
Step: 2, Action: [-0.647  0.173  0.811], Reward: -0.623, Terminated: False, Truncated: False
...
Episode finished after 120 steps.
Extending the Project
Integrate RL Algorithms: Use frameworks like stable-baselines3 to train agents in the environment.
Visualizations: Add rendering for real-time tracking of the pipette's position and trajectory.
Multi-Agent Support: Expand the simulation to support multiple agents.