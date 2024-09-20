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

from environment.environment_helpers import read_experiment_config, get_model_parameters, load_env_config
# Importing required functions and classes
from helper_scripts.MPC import model_predictive_control
from helper_scripts.general_helpers import verify_external_policy_on_specific_env, make_experiment_folder

environment_settings = read_experiment_config('config/environment_setting.yaml')

DoF = environment_settings['degrees-of-freedom']  # Degrees of Freedom
validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes

# Train on different size of the environment
env = load_env_config(env_config='config/environment_setting.yaml')

# MPC specific parameters
mpc_horizon = environment_settings['mpc-settings']['horizon-length'] # Number of steps for MPC horizon
mpc_tol = environment_settings['mpc-settings']['tol'] # Tolerance of MPC
mpc_disp = environment_settings['mpc-settings']['disp'] # Display optimization details of MPC

action_matrix_scaled, threshold = get_model_parameters(env)
# Define the policy for MPC
policy_mpc = lambda x: model_predictive_control(x, mpc_horizon, action_matrix_scaled, threshold,
                                                plot=False, tol=mpc_tol, disp=mpc_disp)

# Create folder to save verification results
# Specific for MPC
optimization_type = 'MPC'
algorithm = ''

save_folder_figures = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Figures')
save_folder_results = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='MPC_results')
save_name_results = os.path.join(save_folder_results, 'MPC_results.pkl')

# Verify the external policy on the specific environment
verify_external_policy_on_specific_env(
    env, [policy_mpc],
    episodes=nr_validation_episodes,
    title='MPC',
    save_folder=save_folder_figures,
    policy_labels=['MPC'],
    DoF=DoF,
    seed_set=validation_seeds,
    save_results=save_name_results
)
