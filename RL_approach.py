import os
import time

from matplotlib import pyplot as plt
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from tqdm import tqdm

from environment.environment_helpers import load_env_config, read_experiment_config
from helper_scripts.general_helpers import make_experiment_folder, plot_progress, \
    verify_external_policy_on_specific_env

# Todo: Make plots interactive!!!

# Here we select one possible MDP out of a set of MDPs - not important at this stage
environment_settings = read_experiment_config('config/environment_setting.yaml')
predefined_task = environment_settings['task_setting']['task_nr']
task_location = environment_settings['task_setting']['task_location']
DoF = environment_settings['degrees-of-freedom']

# Train on different size of the environment
env = load_env_config(env_config='config/environment_setting.yaml')


validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes


# Specific for RL training savings
# Select on algorithm
algorithm = environment_settings['rl-settings']['algorithm']
optimization_type = 'RL'
save_folder_figures = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Figures')
save_folder_weights = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Weights')


if algorithm == 'TRPO':
    model = TRPO("MlpPolicy", env)
elif algorithm == 'PPO':
    model = PPO("MlpPolicy", env)
else:
    print('Select valid algorithm')

success_rates, mean_rewards, x_plot = [], [], []

total_steps = environment_settings['rl-settings']['total_steps']
evaluation_steps =  environment_settings['rl-settings']['evaluation_steps']
increments = total_steps // evaluation_steps

for i in tqdm(range(0, evaluation_steps)):
    num_samples = increments * i
    save_folder_figures_individual = os.path.join(save_folder_figures, f'{num_samples:07}')
    save_folder_weights_individual = os.path.join(save_folder_weights, f'{num_samples:07}')
    if i > 0:
        model.learn(total_timesteps=increments)
    model.save(save_folder_weights_individual)
    # vec_env = model.get_env()
    policy = lambda x: model.predict(x, deterministic=True)[0]
    title = f'{algorithm}_{DoF}_{num_samples} samples, threshold={env.threshold}'
    success_rate, mean_reward = verify_external_policy_on_specific_env(env, [policy],
                                                                       num_samples=num_samples,
                                                                       episodes=nr_validation_episodes,
                                                                       title=title,
                                                                       save_folder=save_folder_figures,
                                                                       policy_labels=[algorithm], DoF=DoF,
                                                                       nr_validation_episodes=nr_validation_episodes,
                                                                       seed_set=validation_seeds)

    print(success_rate)
    time.sleep(2)
    success_rates.append(success_rate)
    mean_rewards.append(mean_reward)

    x_plot.append(num_samples)
    plot_progress(x_plot, mean_rewards, success_rates, DoF, num_samples=num_samples,
                  nr_validation_episodes=nr_validation_episodes, algorithm=algorithm, save_figure=save_folder_figures)

x = [i * increments for i in (range(0, evaluation_steps))]
plot_progress(x_plot, mean_rewards, success_rates, DoF, num_samples=num_samples,
              nr_validation_episodes=nr_validation_episodes, algorithm=algorithm, save_figure=save_folder_figures)

model = TRPO.load(save_folder_weights_individual)

# vec_env = model.get_env()
policy = lambda x: model.predict(x, deterministic=True)[0]

verify_external_policy_on_specific_env(env, [policy],
                                       num_samples=num_samples,
                                       episodes=10, title=algorithm,
                                       save_folder=save_folder_figures, policy_labels=[algorithm],
                                       DoF=DoF, nr_validation_episodes=nr_validation_episodes,
                                       seed_set=validation_seeds)
