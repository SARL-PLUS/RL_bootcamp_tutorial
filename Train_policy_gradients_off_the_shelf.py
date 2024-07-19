import os
import time

from matplotlib import pyplot as plt
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from tqdm import tqdm

from environment.helpers import load_predefined_task
from helper_scripts.Visualize_policy_validation import verify_external_policy_on_specific_env
from environment.environment_awake_steering import DoFWrapper, AwakeSteering

# Todo: Make plots interactive and add variance

# Select on algorithm
algorithm = 'TRPO'  #
algorithm = 'PPO'  #

# Here we select one possible MDP out of a set of MDPs - not important at this stage
predefined_task = 0
verification_task = load_predefined_task(predefined_task)
optimization_type = 'RL'
experiment_name = f'predefined_task_{predefined_task}'
save_folder_figures = os.path.join(optimization_type, algorithm, experiment_name, 'Figures', 'verification')
save_folder_weights = os.path.join(optimization_type, algorithm, experiment_name, 'Weights', 'verification')

os.makedirs(save_folder_figures, exist_ok=True)
os.makedirs(save_folder_weights, exist_ok=True)


def plot_progress(x, mean_rewards, success_rate, DoF, num_samples, nr_validation_episodes):
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
    plt.show()


# For Olga-change here
# Train on different size of the environment
for DoF in [1]:
    env = DoFWrapper(AwakeSteering(task=verification_task), DoF)
    if algorithm == 'TRPO':
        model = TRPO("MlpPolicy", env)
    elif algorithm == 'PPO':
        model = PPO("MlpPolicy", env)
    else: print('Select valid algorithm')

    success_rates, mean_rewards, x_plot = [], [], []

    # For Olga-change here
    total_steps = int(1e4)
    nr_steps = 50
    increments = total_steps // nr_steps

    nr_validation_episodes = 10
    seed_set = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11]

    for i in tqdm(range(0, nr_steps)):
        num_samples = increments * i
        save_folder_figures_individual = f'{save_folder_figures}_{DoF}_{num_samples}'
        save_folder_weights_individual = f'{save_folder_weights}_{DoF}_{num_samples}'
        if i > 0:
            model.learn(total_timesteps=increments)
        model.save(save_folder_weights_individual)
        vec_env = model.get_env()
        policy = lambda x: model.predict(x)[0]
        title = f'{algorithm}_{DoF}_{num_samples} samples, threshold={env.threshold}'
        success_rate, mean_reward = verify_external_policy_on_specific_env(env, [policy], tasks=verification_task,
                                                                           episodes=nr_validation_episodes,
                                                                           title=title,
                                                                           save_folder=save_folder_figures_individual,
                                                                           policy_labels=[algorithm], DoF=DoF,
                                                                           nr_validation_episodes=nr_validation_episodes,
                                                                           seed_set=seed_set)

        print(success_rate)
        time.sleep(5)
        success_rates.append(success_rate)
        mean_rewards.append(mean_reward)

        x_plot.append(num_samples)
        plot_progress(x_plot, mean_rewards, success_rates, DoF, num_samples=num_samples,
                      nr_validation_episodes=nr_validation_episodes)

    x = [i * increments for i in (range(0, nr_steps))]
    plot_progress(x_plot, mean_rewards, success_rates, DoF, num_samples=num_samples,
                  nr_validation_episodes=nr_validation_episodes)

    model = TRPO.load(save_folder_weights_individual)

    vec_env = model.get_env()
    policy = lambda x: model.predict(x)[0]

    verify_external_policy_on_specific_env(env, [policy], tasks=verification_task, episodes=10, title=algorithm,
                                           save_folder=save_folder_figures_individual, policy_labels=[algorithm],
                                           DoF=DoF, nr_validation_episodes=nr_validation_episodes,
                                           seed_set=seed_set)
