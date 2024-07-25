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
from environment.helpers import load_predefined_task, DoFWrapper, read_yaml_file, get_model_parameters

environment_settings = read_yaml_file('config/environment_setting.yaml')

# Load a predefined task for verification
verification_task = load_predefined_task(environment_settings['task_setting']['task_nr'], environment_settings['task_setting']['task_location'])

DoF = environment_settings['degrees-of-freedom']  # Degrees of Freedom
validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes
mpc_horizon = environment_settings['mpc-settings']['horizon-length'] # Number of steps for MPC horizon

# Initialize the environment with the specified task and DoF
env = DoFWrapper(AwakeSteering(task=verification_task), DoF)

action_matrix_scaled, threshold = get_model_parameters(env)

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
    seed_set=validation_seeds
)
