import os
import time

from matplotlib import pyplot as plt
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from tqdm import tqdm

from environment.helpers import load_env_config, read_yaml_file
from helper_scripts.Visualize_policy_validation import verify_external_policy_on_specific_env

# Todo: Make plots interactive and add variance

# Here we select one possible MDP out of a set of MDPs - not important at this stage
environment_settings = read_yaml_file('config/environment_setting.yaml')
predefined_task = environment_settings['task_setting']['task_nr']
task_location = environment_settings['task_setting']['task_location']

# For Olga-change here
# Train on different size of the environment
env = load_env_config(env_config='config/environment_setting.yaml')
DoF = env.DoF
# verification_task = env.get_task()

validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes

# Specific for RL training savings
# Select on algorithm
algorithm = environment_settings['rl-settings']['algorithm']  #
# algorithm = 'PPO'  #

optimization_type = 'RL'
experiment_name = f'predefined_task_{predefined_task}'
save_folder_figures = os.path.join(optimization_type, algorithm, experiment_name, 'Figures', f'Dof_{DoF}')
save_folder_weights = os.path.join(optimization_type, algorithm, experiment_name, 'Weights', f'Dof_{DoF}')

os.makedirs(save_folder_figures, exist_ok=True)
os.makedirs(save_folder_weights, exist_ok=True)


def plot_progress(x, mean_rewards, success_rate, DoF, num_samples, nr_validation_episodes, save_figure=False):
    """
    Plot the progress of training over episodes or time.

    Parameters:
    - x: Iterable with x-axis values (e.g., episode numbers).
    - mean_rewards: Iterable with mean rewards corresponding to x.
    - success_rate: Iterable with success rates corresponding to x.
    - DoF: Degree of Freedom for the AWAKE environment.
    - num_samples: Number of samples used for training.
    """
    fig, ax = plt.subplots()
    ax.plot(x, mean_rewards, color='blue', label='Mean Rewards')
    ax.set_xlabel('Interactions with the system')
    ax.set_ylabel('Cumulative Reward', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_title(
        f"{algorithm} on AWAKE with DoF: {DoF} and trained on {num_samples} samples\n averaged over {nr_validation_episodes} validation episodes")

    ax1 = ax.twinx()
    ax1.plot(x, success_rate, color='green', label='Success Rate')
    ax1.set_ylabel('Success Rate (%)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    # Creating a legend that includes all labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    if save_figure:
        save_name = os.path.join(save_folder_figures, f'Learning_{num_samples}')
        plt.savefig(save_name + '.pdf', format='pdf')  # Specify the format as needed
        plt.savefig(save_name + '.png', format='png')  # Specify the format as needed
    plt.show()


if algorithm == 'TRPO':
    model = TRPO("MlpPolicy", env)
elif algorithm == 'PPO':
    model = PPO("MlpPolicy", env)
else:
    print('Select valid algorithm')

success_rates, mean_rewards, x_plot = [], [], []

# For Olga-change here
total_steps = int(2e5)
evaluation_steps = 10
increments = total_steps // evaluation_steps

for i in tqdm(range(0, evaluation_steps)):
    num_samples = increments * i
    save_folder_figures_individual = os.path.join(save_folder_figures, f'{num_samples:07}')
    save_folder_weights_individual = os.path.join(save_folder_weights, f'{num_samples:07}')
    if i > 0:
        model.learn(total_timesteps=increments)
    model.save(save_folder_weights_individual)
    vec_env = model.get_env()
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
                  nr_validation_episodes=nr_validation_episodes, save_figure=True)

x = [i * increments for i in (range(0, evaluation_steps))]
plot_progress(x_plot, mean_rewards, success_rates, DoF, num_samples=num_samples,
              nr_validation_episodes=nr_validation_episodes, save_figure=True)

model = TRPO.load(save_folder_weights_individual)

vec_env = model.get_env()
policy = lambda x: model.predict(x, deterministic=True)[0]

verify_external_policy_on_specific_env(env, [policy],
                                       num_samples=num_samples,
                                       episodes=10, title=algorithm,
                                       save_folder=save_folder_figures, policy_labels=[algorithm],
                                       DoF=DoF, nr_validation_episodes=nr_validation_episodes,
                                       seed_set=validation_seeds)
