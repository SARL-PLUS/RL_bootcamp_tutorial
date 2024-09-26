import glob
import os
import numpy as np
from environment.environment_helpers import read_experiment_config, load_env_config, get_model_parameters
from helper_scripts.MPC import model_predictive_control
from helper_scripts.general_helpers import make_experiment_folder, verify_external_policy_on_specific_env_regret, \
    load_latest_policy

# ToDo: store the trajectories of the MPC policy

environment_settings = read_experiment_config('config/environment_setting.yaml')
predefined_task = environment_settings['task_setting']['task_nr']
task_location = environment_settings['task_setting']['task_location']

# # Load a predefined task for verification
# verification_task = load_predefined_task(predefined_task, task_location)

DoF = environment_settings['degrees-of-freedom']  # Degrees of Freedom
validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes


# Train on different size of the environment
env = load_env_config(env_config='config/environment_setting.yaml')
verification_task = env.get_task()


# # MPC specific parameters
# mpc_horizon = environment_settings['mpc-settings']['horizon-length'] # Number of steps for MPC horizon

# # Define the policy for MPC
# policy_mpc = lambda x: model_predictive_control(x, mpc_horizon, action_matrix_scaled, threshold, plot=False)
action_matrix_scaled, threshold = get_model_parameters(env)
b_inv = np.linalg.inv(action_matrix_scaled)
def policy_response_matrix(state):
    action = -b_inv @ state
    action_abs_max = max(abs(action))
    if action_abs_max > 1:
        action = action / action_abs_max
    return action

experiment_name = f'predefined_task_{predefined_task}'

# Create the save folder for figures
save_folder = os.path.join('results', 'comparison', experiment_name, 'Figures', f'Dof_{DoF}')
os.makedirs(save_folder, exist_ok=True)

# Define the save folder for results
save_folder_results = make_experiment_folder('MPC', '', environment_settings,
                                             purpose='MPC_results', generate=False)
save_folder_name_results = os.path.join(save_folder_results, 'MPC_results.pkl')

# Check if save_folder_name_results exists
if not os.path.exists(save_folder_name_results):
    raise FileNotFoundError(f"The results file does not exist: {save_folder_name_results}. "
                            f"Did you run the MPC_approach.py script wiht this configuration?")

# If it exists, you can proceed with your operations on save_folder_name_results
print(f"Results file found: {save_folder_name_results}")

policy_rl_agent, algorithm = load_latest_policy(environment_settings)

policy_benchmark = 'policy_mpc_stored'

verify_external_policy_on_specific_env_regret(env, [policy_rl_agent, 'policy_mpc_stored'],
                                              policy_benchmark='policy_mpc_stored',
                                              tasks=verification_task, episodes=nr_validation_episodes,
                                              title=f'Regret to {policy_benchmark} of {algorithm}',
                                              save_folder=save_folder+'_MPC',
                                              policy_labels=[algorithm, 'MPC'],
                                              DoF=DoF, read_results=save_folder_name_results)

policy_benchmark = policy_response_matrix

verify_external_policy_on_specific_env_regret(env, [policy_rl_agent, policy_response_matrix],
                                              policy_benchmark=policy_response_matrix,
                                              tasks=verification_task, episodes=nr_validation_episodes,
                                              title=f'Regret to {"policy_response_matrix"} of {algorithm}',
                                              save_folder=save_folder+'_Response_matrix',
                                              policy_labels=[algorithm, 'policy_response_matrix'],
                                              DoF=DoF, read_results=save_folder_name_results)
