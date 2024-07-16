import glob
import os
import numpy as np
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from helper_scripts.MPC_script import model_predictive_control
from helper_scripts.Visualize_policy_validation import verify_external_policy_on_specific_env
from environment.environment_awake_steering import AwakeSteering, DoFWrapper, load_predefined_task


# ToDo: find bug in initialisation

predefined_task = 1
verification_task = load_predefined_task(predefined_task)
action_matrix = verification_task['goal'][0]
DoF = 1
nr_validation_episodes = 10
env = DoFWrapper(AwakeSteering(task=verification_task), DoF)


# Model parameters for MPC
action_matrix = action_matrix[:DoF, :DoF]
action_matrix_scaled = action_matrix * env.action_scale * env.state_scale
threshold = -env.threshold

mpc_horizon = 5  # Number of steps
def policy_mpc(state):
    return model_predictive_control(state, mpc_horizon, action_matrix_scaled, threshold, plot=False)

model = PPO("MlpPolicy", env, verbose=1)


algorithm = 'TRPO'

experiment_name = f'predefined_task_{predefined_task}'
save_folder_figures = os.path.join(algorithm, experiment_name, 'Figures_compare')
save_folder_weights = os.path.join(algorithm, experiment_name, 'Weights')

files = glob.glob(os.path.join(save_folder_weights, '*'))
files_DoF = [file for file in files if f'verification_{DoF}' in file][-1]

print(files)
if algorithm == 'TRPO':
    model = TRPO.load(files_DoF)  # , verbose=1, clip_range=.1, learning_rate=5e-4, gamma=1)
else:
    model = PPO.load(files_DoF)

vec_env = model.get_env()

def policy_ppo(state):
 return model.predict(state)[0]

b_inv = np.linalg.inv(action_matrix_scaled)
def policy_response_matrix(state):
    action = -b_inv @ state
    action_abs_max = max(abs(action))
    if action_abs_max > 1:
        action = action / action_abs_max
    return action


save_folder = save_folder_figures
os.makedirs(save_folder, exist_ok=True)

verify_external_policy_on_specific_env(env, [policy_mpc, policy_ppo, policy_response_matrix], tasks=verification_task,
                                       episodes=nr_validation_episodes, title=f'MPC vs. {algorithm} vs analytical approach', save_folder=save_folder+'_1',
                                       policy_labels=['MPC', algorithm, 'Response_matrix'], DoF=DoF)

verify_external_policy_on_specific_env(env, [policy_ppo, policy_mpc, policy_response_matrix], tasks=verification_task,
                                       episodes=nr_validation_episodes, title=f'MPC vs. {algorithm} vs analytical approach', save_folder=save_folder+'_1',
                                       policy_labels=[algorithm, 'MPC', 'Response_matrix'], DoF=DoF)

verify_external_policy_on_specific_env(env, [policy_response_matrix, policy_ppo, policy_mpc], tasks=verification_task,
                                       episodes=nr_validation_episodes, title=f'MPC vs. {algorithm} vs analytical approach', save_folder=save_folder+'_1',
                                       policy_labels=['Response_matrix', algorithm, 'MPC'], DoF=DoF)
