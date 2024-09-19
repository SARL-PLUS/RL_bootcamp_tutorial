import glob
import os

import numpy as np
from sb3_contrib import TRPO
from stable_baselines3 import PPO

from environment.helpers import read_yaml_file, load_env_config, get_model_parameters
from helper_scripts.MPC_script import model_predictive_control
from helper_scripts.Visualize_policy_validation import verify_external_policy_on_specific_env_regret

# ToDo: store the trajectories of the MPC policy

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
verification_task = env.get_task()


# MPC specific parameters
mpc_horizon = environment_settings['mpc-settings']['horizon-length'] # Number of steps for MPC horizon
action_matrix_scaled, threshold = get_model_parameters(env)
# Define the policy for MPC
policy_mpc = lambda x: model_predictive_control(x, mpc_horizon, action_matrix_scaled, threshold, plot=False)

b_inv = np.linalg.inv(action_matrix_scaled)
def policy_response_matrix(state):
    action = -b_inv @ state
    action_abs_max = max(abs(action))
    if action_abs_max > 1:
        action = action / action_abs_max
    return action

experiment_name = f'predefined_task_{predefined_task}'

# Create the save folder for figures
save_folder = os.path.join('Comparing', experiment_name, 'Figures', f'Dof_{DoF}', 'Comparison')
os.makedirs(save_folder, exist_ok=True)

# Define the save folder for results
save_folder_results = os.path.join('MPC', experiment_name, 'MPC_results', f'Dof_{DoF}')
save_folder_name_results = os.path.join(save_folder_results, 'MPC_results.pkl')

# Check if save_folder_name_results exists
if not os.path.exists(save_folder_name_results):
    raise FileNotFoundError(f"The results file does not exist: {save_folder_name_results}")

# If it exists, you can proceed with your operations on save_folder_name_results
print(f"Results file found: {save_folder_name_results}")


optimization_type = 'RL'


# Select on algorithm

algorithm = environment_settings['rl-settings']['algorithm']

# Construct the save folder path for weights
save_folder_weights = os.path.join(optimization_type, algorithm, experiment_name, 'Weights', f'Dof_{DoF}')

# Get the list of files in the weights folder
files = glob.glob(os.path.join(save_folder_weights, '*'))
files.sort()

# Check if there are any files found
if not files:
    raise FileNotFoundError(f"No model files found in {save_folder_weights} to load. Run training (Train_policy_gradients_off_the_shelf.py)!")

# Get the latest model file
latest_model_file = files[-1]
print(f"Loading the latest model from: {latest_model_file}")

# Load the model
if algorithm == 'TRPO':
    model = TRPO.load(latest_model_file)  # , verbose=1, clip_range=.1, learning_rate=5e-4, gamma=1)
else:
    model = PPO.load(latest_model_file)

def policy_rl_agent(state):
 return model.predict(state, deterministic=True)[0]

verify_external_policy_on_specific_env_regret(env, ['policy_mpc_stored', 'policy_mpc_stored', policy_rl_agent], policy_benchmark='policy_mpc_stored', tasks=verification_task,
                                              episodes=nr_validation_episodes, title=f'Regret to MPC of {algorithm} and analytical approach', save_folder=save_folder+'_2',
                                              policy_labels=[algorithm, 'MPC', 'Response_matrix'], DoF=DoF, save_results=save_folder_name_results)
