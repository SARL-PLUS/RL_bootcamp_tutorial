import glob
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from tqdm import tqdm

colors = plt.cm.rainbow(np.linspace(0, 1, 10))
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[color for color in colors])
colors_for_different_appoaches = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

import os
import shutil
import logging
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_experiment_folder(
    optimization_type: str,
    algorithm: str,
    environment_settings: Dict[str, Any],
    purpose: str,
    generate: bool = True,
    delete: bool = False
) -> str:
    """
    Creates a directory path for storing experiment results based on the provided parameters.

    Args:
        optimization_type (str): The type of optimization being used (e.g., 'RL', 'MPC').
        algorithm (str): The specific algorithm name (e.g., 'PPO', 'TRPO').
        environment_settings (Dict[str, Any]): Configuration settings for the environment.
            Must contain:
                - "task_setting": Dict with "task_nr" key.
                - "degrees-of-freedom": int.
        purpose (str): The purpose or phase of the experiment (e.g., 'weights', 'results').
        generate (bool, optional): Whether to create the directory if it does not exist. Defaults to True.
        delete (bool, optional): Whether to delete the existing directory before creating a new one. Defaults to False.

    Returns:
        str: The path to the created or existing experiment folder.

    Raises:
        KeyError: If required keys are missing in environment_settings.
        OSError: If the directory cannot be created or deleted.
    """
    try:
        experiment_name = f'predefined_task_{environment_settings["task_setting"]["task_nr"]}'
        degrees_of_freedom = environment_settings["degrees-of-freedom"]
    except KeyError as e:
        logger.error(f"Missing key in environment_settings: {e}")
        raise KeyError(f"Missing key in environment_settings: {e}")

    save_folder = os.path.join(
        'results',
        optimization_type,
        algorithm,
        experiment_name,
        purpose,
        f'Dof_{degrees_of_freedom}'
    )

    if delete:
        if os.path.exists(save_folder):
            try:
                shutil.rmtree(save_folder)
                logger.info(f"Deleted existing folder: {save_folder}")
            except OSError as e:
                logger.error(f"Error deleting folder {save_folder}: {e}")
                raise

    if generate:
        try:
            os.makedirs(save_folder, exist_ok=True)
            logger.info(f"Created folder: {save_folder}")
        except OSError as e:
            logger.error(f"Error creating folder {save_folder}: {e}")
            raise

    return save_folder



#Todo: legend for states and actions

def ep_mean_return(rewards_per_task):
    mean_returns = np.mean([np.sum(rews) for rews in rewards_per_task])
    return mean_returns


def ep_success_rate(rewards_per_task, threshold):
    success_rate = np.mean([rews[-1] >= threshold for rews in rewards_per_task])
    return success_rate


def ep_success_rates(rewards_per_task, threshold):
    success_rate = np.array([rew[-1] >= threshold for rew in rewards_per_task])
    return success_rate


def plot_episode_lengths(ax, ax_twin, ep_len_per_task, color, label=None):
    for i, ep_len in enumerate(ep_len_per_task):
        x = np.arange(len(ep_len))
        ax.plot(x, ep_len, label=f"Len Task {i} - {label}", drawstyle='steps', color=color)
        ax_twin.plot(x, np.cumsum(ep_len), label=f"Len Task {i} - {label}", ls=':', drawstyle='steps',
                     color=color)
    ax_twin.set_ylabel('Cumulative episode length')


def plot_actions_states(ax, data_per_task, label, ep_len_per_task, ls='-'):
    for i, data in enumerate(data_per_task):
        lens = [len(entry) for entry in data]
        episode_lengths = ep_len_per_task[i]
        episode_lengths = [length + 1 for length in episode_lengths]
        cumulative_lengths = np.cumsum([0] + episode_lengths)
        for i, (start, length) in enumerate(zip(cumulative_lengths, lens)):
            ax.axvline(x=start, color='grey', linestyle='--', alpha=0.5)
            end = start + length
            ax.plot(range(start, end), data[i], label=f"{label} Task {i}", marker='o', ls=ls)
        ax.set_xlim(-1, cumulative_lengths[-1])


def plot_rms_states(ax, ax_twin, data_per_task, label, ep_len_per_task, rewards_per_task, threshold, ls='-'):
    for i, data in enumerate(data_per_task):
        lens = [len(entry) for entry in data]
        episode_lengths = ep_len_per_task[i]
        episode_lengths = [length + 1 for length in episode_lengths]
        cumulative_lengths = np.cumsum([0] + episode_lengths)

        for i, (start, length) in enumerate(zip(cumulative_lengths, lens)):
            ax.axvline(x=start, color='grey', linestyle='--', alpha=0.5)
            end = start + length
            data_rms = [-np.sqrt(np.mean(np.square(entry))) for entry in data[i]]
            ax.plot(range(start, end), data_rms, label=f"{label} Task {i}", marker='o', ls=ls)
            ax.plot(range(start + 1, end), rewards_per_task[0][i], label=f"{label} Task {i}", marker='x', ls=':', c='r')
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)
        ax.set_xlim(-1, cumulative_lengths[-1])


def plot_rewards(ax, ax_twin, rewards_per_task, success_rates, color, label=None):
    for i, rews in enumerate(rewards_per_task):
        ep_returns = [np.sum(r) for r in rews]
        # x = np.linspace(0, len(ep_returns), len(ep_returns))
        x = np.arange(len(ep_returns))
        ax.plot(x, ep_returns, label=f"Reward Task {i} - {label}", drawstyle='steps', color=color)
    color = 'tab:red'
    ax_twin.plot(x, success_rates, c=color, drawstyle='steps')
    ax_twin.set_ylabel('Success', color=color)
    ax_twin.tick_params(axis='y', labelcolor=color)
    ax_twin.set_ylim(-0.1, 1.1)


def plot_regrets(ax, ax_twin, rewards_per_task,  rewards_per_task_benchmark, success_rates, color, label=None):
    for i in range(len(rewards_per_task)):
        ep_returns = np.array([np.sum(r) for r in rewards_per_task[i]])
        ep_returns_benchmark = np.array([np.sum(r) for r in rewards_per_task_benchmark[i]])
        print(label, ep_returns, ep_returns_benchmark)
        # x = np.linspace(0, len(ep_returns), len(ep_returns))
        x = np.arange(len(ep_returns))
        ax.plot(x, ep_returns_benchmark-ep_returns, label=f"Regret Task {i} - {label}", drawstyle='steps',
                color=color)
    color = 'tab:red'
    ax_twin.plot(x, success_rates, c=color, drawstyle='steps')
    ax_twin.set_ylabel('Success', color=color)
    ax_twin.tick_params(axis='y', labelcolor=color)
    ax_twin.set_ylim(-0.1, 1.1)


def test_policy(env, policy=None, episodes=50, seed_set=None):
    rewards_per_task, ep_len_per_task, actions_per_task, states_per_task = [], [], [], []

    if policy is None:
        policy = lambda x: env.action_space.sample()

    trajectories = \
        create_trajectories(env, policy, episodes, seed_set=seed_set)

    all_observations = [trajectory['observations'] for trajectory in trajectories]
    all_actions = [trajectory['actions'] for trajectory in trajectories]
    all_rewards = [trajectory['rewards'] for trajectory in trajectories]
    all_lengths = [trajectory['length'] for trajectory in trajectories]

    ep_len_per_task.append(all_lengths)
    rewards_per_task.append(all_rewards)
    actions_per_task.append(all_actions)
    states_per_task.append(all_observations)

    return rewards_per_task, ep_len_per_task, actions_per_task, states_per_task


def create_trajectories(env, policy, episodes, seed_set=None):
    trajectories = []
    if seed_set is None:
        seed_set = range(episodes)
    for seed in tqdm(seed_set, total=episodes):
        trajectory = {'observations': [], 'actions': [], 'rewards': [], 'length': 0}
        observation, info = env.reset(seed=seed)
        trajectory['observations'].append(observation.copy())

        done = False
        while not done:
            action = policy(observation)
            observation, reward, terminated, truncated, infos = env.step(action)
            trajectory['observations'].append(observation.copy())
            trajectory['actions'].append(action.copy())
            trajectory['rewards'].append(reward.copy())
            done = terminated or truncated
            trajectory['length'] += 1
            # if trajectory['length'] >= 10:
            #     done = True
        trajectories.append(trajectory)

    return trajectories


def verify_external_policy_on_specific_env(env, policies, episodes=50, **kwargs):
    labels = kwargs['policy_labels']
    seed_set = kwargs['seed_set'] if 'seed_set' in kwargs else None

    fig = plt.figure(figsize=(10, 10))
    ax = []
    ax.append(fig.add_subplot(511))
    ax.append(fig.add_subplot(512, sharex=ax[0]))
    ax.append(fig.add_subplot(513))
    ax.append(fig.add_subplot(514, sharex=ax[2]))
    ax.append(fig.add_subplot(515, sharex=ax[2]))

    ax[0].tick_params('x', labelbottom=False)
    ax[2].tick_params('x', labelbottom=False)
    ax[3].tick_params('x', labelbottom=False)

    ax0_twin = ax[0].twinx()
    ax1_twin = ax[1].twinx()
    ax2_twin = ax[2].twinx()
    # policies = [policies] if not isinstance(policies, (list, tuple)) else policies



    success_rates, mean_rewards = [], []
    for i, policy in enumerate(policies):
        label = labels[i]
        if policy == 'policy_mpc_stored' and 'read_results' in kwargs:
            save_folder_name_results = kwargs['read_results']
            # Load the dictionary from the file using pickle
            with open(save_folder_name_results, "rb") as f:
                save_dict = pickle.load(f)
            # Extract data from the dictionary
            rewards_per_task = save_dict.get('rewards_per_task')
            ep_len_per_task = save_dict.get('ep_len_per_task')
            actions_per_task = save_dict.get('actions_per_task')
            states_per_task = save_dict.get('states_per_task')

        else:
            rewards_per_task, ep_len_per_task, actions_per_task, states_per_task = test_policy(env, policy,
                                                                                               episodes,
                                                                                               seed_set=seed_set)

        if 'save_results' in kwargs:
            save_folder_name_results = kwargs['save_results']
            save_dict= {'rewards_per_task': rewards_per_task, 'ep_len_per_task': ep_len_per_task,
                        'actions_per_task': actions_per_task, 'states_per_task': states_per_task}

            # Save the dictionary to the file using pickle
            with open(save_folder_name_results, "wb") as f:
                pickle.dump(save_dict, f)

        for task_nr, ep_ret in enumerate(map(ep_mean_return, rewards_per_task)):
            print(f"Mean return for Task nr.{task_nr} - {label}: {ep_ret}")
            mean_rewards.append(ep_ret)

        for task_nr, ep_suc in enumerate(
                map(lambda rews: ep_success_rates(rews, threshold=env.threshold), rewards_per_task)):
            print(f"Success rate Task nr.{task_nr} - {label}: {np.mean(ep_suc)}")
            success_rate_per_tasks = ep_suc
            success_rates.append(success_rate_per_tasks)

        # Plot actions and states and reward per step
        if i == 0:
            plot_actions_states(ax[4], actions_per_task, "Actions", ep_len_per_task)
            ax[4].set_ylabel("Actions")
            ax[4].set_xlabel("Step")
            plot_actions_states(ax[3], states_per_task, "States", ep_len_per_task)
            ax[3].set_ylabel("States")
            plot_rms_states(ax[2], ax2_twin, states_per_task, "States", ep_len_per_task, rewards_per_task,
                            threshold=env.threshold)
            ax[2].set_ylabel("negative RMS")
            ax[2].set_ylim(-1, 0)

        # Plot episode lengths
        plot_episode_lengths(ax[0], ax0_twin, ep_len_per_task, label=label, color=colors_for_different_appoaches[i])
        ax[0].set_ylabel("Ep. lengths")
        ax[0].legend()
        # Plot rewards
        plot_rewards(ax[1], ax1_twin, rewards_per_task, success_rate_per_tasks, label=label,
                     color=colors_for_different_appoaches[i])

        ax[1].set_ylabel("Cum. Reward")
        ax[1].set_xlabel("Episode")
        ax[1].legend()

    # Handle additional kwargs
    title = kwargs.get('title', '') + f' - policy shown from {labels[0]}'
    if title:
        plt.suptitle(title)
    fig.align_ylabels(ax)
    plt.tight_layout()
    if 'save_folder' in kwargs:
        save_folder = kwargs.get('save_folder')
        save_name = f'Verification_{kwargs.get("num_samples")}'
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder,save_name + '.pdf'), format='pdf')  # Specify the format as needed
        plt.savefig(os.path.join(save_folder,save_name + '.png'), format='png')  # Specify the format as needed
    plt.show()

    return np.mean(success_rates), mean_rewards

def verify_external_policy_on_specific_env_regret(env, policies, policy_benchmark, episodes=50, **kwargs):
    labels = kwargs['policy_labels']
    seed_set = kwargs['seed_set'] if 'seed_set' in kwargs else None

    fig = plt.figure(figsize=(10, 10))
    ax = []
    ax.append(fig.add_subplot(511))
    ax.append(fig.add_subplot(512, sharex=ax[0]))
    ax.append(fig.add_subplot(513))
    ax.append(fig.add_subplot(514, sharex=ax[2]))
    ax.append(fig.add_subplot(515, sharex=ax[2]))

    ax[0].tick_params('x', labelbottom=False)
    ax[2].tick_params('x', labelbottom=False)
    ax[3].tick_params('x', labelbottom=False)

    ax0_twin = ax[0].twinx()
    ax1_twin = ax[1].twinx()
    ax2_twin = ax[2].twinx()

    if policy_benchmark == 'policy_mpc_stored' and 'read_results' in kwargs:
        save_location_name_results = kwargs['read_results']
        # Load the dictionary from the file using pickle
        with open(save_location_name_results, "rb") as f:
            save_dict = pickle.load(f)
        # Extract data from the dictionary
        rewards_per_task_benchmark = save_dict.get('rewards_per_task')
        ep_len_per_task_benchmark = save_dict.get('ep_len_per_task')
        actions_per_task_benchmark = save_dict.get('actions_per_task')
        states_per_task_benchmark = save_dict.get('states_per_task')

    else:
        rewards_per_task_benchmark, ep_len_per_task_benchmark, actions_per_task_benchmark, states_per_task_benchmark = test_policy(env, policy_benchmark,
                                                                                           episodes, seed_set=seed_set)



    success_rates, mean_rewards = [], []
    for i, policy in enumerate(policies):
        label = labels[i]
        if policy == 'policy_mpc_stored' and 'read_results' in kwargs:
            save_location_name_results = kwargs['read_results']
            # Load the dictionary from the file using pickle
            with open(save_location_name_results, "rb") as f:
                save_dict = pickle.load(f)
            # Extract data from the dictionary
            rewards_per_task = save_dict.get('rewards_per_task')
            ep_len_per_task = save_dict.get('ep_len_per_task')
            actions_per_task = save_dict.get('actions_per_task')
            states_per_task = save_dict.get('states_per_task')

        else:
            rewards_per_task, ep_len_per_task, actions_per_task, states_per_task = test_policy(env, policy,
                                                                                               episodes,
                                                                                               seed_set=seed_set)

        if 'save_results' in kwargs:
            save_location_name_results = kwargs['save_results']
            save_dict= {'rewards_per_task': rewards_per_task, 'ep_len_per_task': ep_len_per_task,
                        'actions_per_task': actions_per_task, 'states_per_task': states_per_task}

            # Save the dictionary to the file using pickle
            with open(save_location_name_results, "wb") as f:
                pickle.dump(save_dict, f)

        for task_nr, ep_ret in enumerate(map(ep_mean_return, rewards_per_task)):
            print(f"Mean return for Task nr.{task_nr} - {label}: {ep_ret}")
            mean_rewards.append(ep_ret)

        for task_nr, ep_suc in enumerate(
                map(lambda rews: ep_success_rates(rews, threshold=env.threshold), rewards_per_task)):
            print(f"Success rate Task nr.{task_nr} - {label}: {np.mean(ep_suc)}")
            success_rate_per_tasks = ep_suc
            success_rates.append(success_rate_per_tasks)

        # Plot actions and states
        if i == 0:
            plot_actions_states(ax[4], actions_per_task, "Actions", ep_len_per_task)
            ax[4].set_ylabel("Actions")
            ax[4].set_xlabel("Step")
            plot_actions_states(ax[3], states_per_task, "States", ep_len_per_task)
            ax[3].set_ylabel("States")
            plot_rms_states(ax[2], ax2_twin, states_per_task, "States", ep_len_per_task, rewards_per_task,
                            threshold=env.threshold)
            ax[2].set_ylabel("negative RMS")
            ax[2].set_ylim(-1, 0)

        # Plot episode lengths
        print('Plot episode lengths', label, ep_len_per_task)
        plot_episode_lengths(ax[0], ax0_twin, ep_len_per_task, label=label, color=colors_for_different_appoaches[i])
        ax[0].set_ylabel("Ep. lengths")
        ax[0].legend()

        # Plot regrets
        plot_regrets(ax[1], ax1_twin, rewards_per_task, rewards_per_task_benchmark, success_rate_per_tasks, label=label,
                     color=colors_for_different_appoaches[i])

        ax[1].set_ylabel("Regret")
        ax[1].set_xlabel("Episode")
        ax[1].legend()

    # Handle additional kwargs
    title = kwargs.get('title', '') + f' - policy shown from {labels[0]}'
    if title:
        plt.suptitle(title)
    fig.align_ylabels(ax)
    plt.tight_layout()
    if 'save_folder' in kwargs:
        save_folder = kwargs.get('save_folder')
        save_name = f'Verification_{kwargs.get("num_samples")}'
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder,save_name + '.pdf'), format='pdf')  # Specify the format as needed
        plt.savefig(os.path.join(save_folder,save_name + '.png'), format='png')  # Specify the format as needed
    plt.show()


    return np.mean(success_rates), mean_rewards


def plot_progress(x, mean_rewards, success_rate, DoF, num_samples, nr_validation_episodes, algorithm, save_figure=False):
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
        save_name = os.path.join(save_figure, f'Learning_{num_samples}')
        plt.savefig(save_name + '.pdf', format='pdf')  # Specify the format as needed
        plt.savefig(save_name + '.png', format='png')  # Specify the format as needed
    plt.show()


def load_latest_policy(environment_settings):
    optimization_type = 'RL'

    # Select on algorithm

    algorithm = environment_settings['rl-settings']['algorithm']

    # Construct the save folder path for weights
    save_folder_weights = make_experiment_folder(optimization_type, algorithm, environment_settings,
                                                 purpose='Weights', generate=False)
    # Get the list of files in the weights folder
    files = glob.glob(os.path.join(save_folder_weights, '*'))
    files.sort()

    # Check if there are any files found
    if not files:
        raise FileNotFoundError(
            f"No model files found in {save_folder_weights} to load. Run training (Train_policy_gradients_off_the_shelf.py)!")

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

    return policy_rl_agent, algorithm

if __name__ == "__main__":
    pass
