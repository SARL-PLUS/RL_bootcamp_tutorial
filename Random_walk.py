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

from environment.environment_helpers import read_experiment_config, load_env_config
# Importing required functions and classes
from helper_scripts.general_helpers import verify_external_policy_on_specific_env, make_experiment_folder

environment_settings = read_experiment_config('config/environment_setting.yaml')

DoF = environment_settings['degrees-of-freedom']  # Degrees of Freedom
validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes

# Train on different size of the environment
env = load_env_config(env_config='config/environment_setting.yaml')

optimization_type = 'random_walk'
algorithm = ''

save_folder_figures = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Figures')
save_folder_results = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Random walk')
save_name_results = os.path.join(save_folder_results, 'Random_walk_results.pkl')

print(save_name_results)
print(save_folder_figures)

# Verify the external policy on the specific environment
verify_external_policy_on_specific_env(
    env, [None],
    episodes=nr_validation_episodes,
    title='Random',
    save_folder=save_folder_figures,
    policy_labels=['Random'],
    DoF=DoF,
    seed_set=validation_seeds,
    save_results=save_name_results
)
