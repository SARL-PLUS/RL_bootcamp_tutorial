"""
Simon Hirlaender
This script sets up and verifies a Model Predictive Control (MPC) policy
on a predefined environment task using the AwakeSteering simulation.
The script performs the following steps:
1. Loads a predefined task for verification.
2. Initializes the environment with the specified task and Degrees of Freedom (DoF).
3. Configures the model parameters for MPC, including scaling the action matrix.
4. Defines the MPC policy using a lambda function.
5. Creates a folder to save verification results.
6. Verifies the external MPC policy on the specific environment and saves the results.

Dependencies:
- MPC module for the model_predictive_control function.
- Visualize_policy_validation module for the verify_external_policy_on_specific_env function.
- environment_awake_steering module for AwakeSteering, DoFWrapper, and load_prefdefined_task functions.

"""

# TODO: save data for accelerated verification
import os

# Importing required functions and classes
from helper_scripts.MPC_script import model_predictive_control
from helper_scripts.Visualize_policy_validation import verify_external_policy_on_specific_env
from environment.environment_awake_steering import AwakeSteering
from environment.helpers import load_predefined_task, DoFWrapper

# Load a predefined task for verification
verification_task = load_predefined_task(0)
action_matrix = verification_task['goal'][0]
DoF = 1  # Degrees of Freedom
nr_validation_episodes = 10  # Number of validation episodes
mpc_horizon = 5  # Number of steps for MPC horizon

# Initialize the environment with the specified task and DoF
env = DoFWrapper(AwakeSteering(task=verification_task), DoF)

# Model parameters for MPC
action_matrix = action_matrix[:DoF, :DoF]  # Adjust action matrix according to DoF
action_matrix_scaled = action_matrix * env.unwrapped.action_scale  # Scale the action matrix
threshold = -env.threshold  # Define the threshold for the MPC

# Define the policy for MPC
policy_mpc = lambda x: model_predictive_control(x, mpc_horizon, action_matrix_scaled, threshold, plot=False)

# Create folder to save verification results
save_folder = 'Figures/mpc_verification'
os.makedirs(save_folder, exist_ok=True)

# Verify the external policy on the specific environment
verify_external_policy_on_specific_env(
    env, [policy_mpc],
    tasks=verification_task,
    episodes=nr_validation_episodes,
    title='MPC',
    save_folder=save_folder,
    policy_labels=['MPC'],
    DoF=DoF,
    seed_set=[0, 2, 3, 4, 5, 7, 8, 9, 10, 11]
)
