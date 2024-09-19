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

# TODO: save data for accelerated verification in the verification functions
import os

from environment.helpers import read_yaml_file, get_model_parameters, load_env_config
# Importing required functions and classes
from helper_scripts.MPC_script import model_predictive_control
from helper_scripts.Visualize_policy_validation import verify_external_policy_on_specific_env

environment_settings = read_yaml_file('config/environment_setting.yaml')
predefined_task = environment_settings['task_setting']['task_nr']
task_location = environment_settings['task_setting']['task_location']

# # Load a predefined task for verification
# verification_task = load_predefined_task(predefined_task, task_location)

DoF = environment_settings['degrees-of-freedom']  # Degrees of Freedom
validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes


# Train on different size of the environment
env = load_env_config(env_config='config/environment_setting.yaml')


# MPC specific parameters
mpc_horizon = environment_settings['mpc-settings']['horizon-length'] # Number of steps for MPC horizon
action_matrix_scaled, threshold = get_model_parameters(env)
# Define the policy for MPC
policy_mpc = lambda x: model_predictive_control(x, mpc_horizon, action_matrix_scaled, threshold, plot=False)

# Create folder to save verification results
# Specific for RL training
optimization_type = 'MPC'
experiment_name = f'predefined_task_{predefined_task}'
save_folder_figures = os.path.join(optimization_type, experiment_name, 'Figures', f'Dof_{DoF}')
save_folder_results = os.path.join(optimization_type, experiment_name, 'MPC_results', f'Dof_{DoF}')
save_folder_name_results = os.path.join(save_folder_results, 'MPC_results.pkl')
# save_folder_weights = os.path.join(optimization_type, experiment_name, 'Weights', 'verification')
os.makedirs(save_folder_figures, exist_ok=True)
os.makedirs(save_folder_results, exist_ok=True)

# Verify the external policy on the specific environment
verify_external_policy_on_specific_env(
    env, [policy_mpc],
    episodes=nr_validation_episodes,
    title='MPC',
    save_folder=save_folder_figures,
    policy_labels=['MPC'],
    DoF=DoF,
    seed_set=validation_seeds,
    save_results=save_folder_name_results
)
