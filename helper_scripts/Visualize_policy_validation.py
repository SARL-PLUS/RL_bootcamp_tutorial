import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def ep_mean_return(rewards_per_task):
    mean_returns = np.mean([np.sum(rews) for rews in rewards_per_task])
    return mean_returns


def ep_success_rate(rewards_per_task, threshold):
    success_rate = np.mean([rews[-1] >= threshold for rews in rewards_per_task])
    return success_rate


def ep_success_rates(rewards_per_task, threshold):
    success_rate = np.array([rew[-1] >= threshold for rew in rewards_per_task])
    return success_rate


def plot_episode_lengths(ax, ax_twin, ep_len_per_task, label=None):
    for i, ep_len in enumerate(ep_len_per_task):
        x = np.arange(len(ep_len))
        ax.plot(x, ep_len, label=f"Len Task {i} - {label}", drawstyle='steps')
        ax_twin.plot(x, np.cumsum(ep_len), label=f"Len Task {i} - {label}", ls=':', drawstyle='steps')
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


def plot_rewards(ax, ax_twin, rewards_per_task, success_rates, label=None):
    for i, rews in enumerate(rewards_per_task):
        ep_returns = [np.sum(r) for r in rews]
        x = np.linspace(0, len(ep_returns), len(ep_returns))
        ax.plot(x, ep_returns, label=f"Reward Task {i} - {label}", drawstyle='steps')
    color = 'tab:red'
    ax_twin.plot(x, success_rates, c=color, drawstyle='steps')
    ax_twin.set_ylabel('Success Rate', color=color)
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
    for seed in seed_set:
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
    DoF = kwargs['DoF'] if 'DoF' in kwargs else 10
    seed_set = kwargs['seed_set'] if 'seed_set' in kwargs else None
    colors = plt.cm.rainbow(np.linspace(0, 1, DoF))
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[color for color in colors])

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
        rewards_per_task, ep_len_per_task, actions_per_task, states_per_task = test_policy(env, policy,
                                                                                           episodes, seed_set=seed_set)

        for task_nr, ep_ret in enumerate(map(ep_mean_return, rewards_per_task)):
            print(f"Mean return for Task nr.{task_nr} - {label}: {ep_ret}")
            mean_rewards.append(ep_ret)

        for task_nr, ep_suc in enumerate(
                map(lambda rews: ep_success_rates(rews, threshold=env.threshold), rewards_per_task)):
            print(f"Success rate Task nr.{task_nr} - {label}: {np.mean(ep_suc)}")
            success_rate_per_taks = ep_suc
            success_rates.append(success_rate_per_taks)

        # Plot actions and states
        if i == 0:
            print('Plotting actions and states', i)
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
        plot_episode_lengths(ax[0], ax0_twin, ep_len_per_task, label=label)
        ax[0].set_ylabel("Ep. lengths")
        ax[0].legend()
        # Plot rewards
        plot_rewards(ax[1], ax1_twin, rewards_per_task, success_rate_per_taks, label=label)
        ax[1].set_ylabel("Cum. reward")
        ax[1].set_xlabel("Episode")
        ax[1].legend()

    # Handle additional kwargs
    title = kwargs.get('title', '') + f'- policy shown from {labels[0]}'
    if title:
        plt.suptitle(title)
    fig.align_ylabels(ax)
    plt.tight_layout()
    if 'save_folder' in kwargs:
        plt.savefig(f"{kwargs.get('save_folder')}.pdf", format='pdf')  # Specify the format as needed
        plt.savefig(f"{kwargs.get('save_folder')}.png", format='png')  # Specify the format as needed
    plt.show()

    return np.mean(success_rates), mean_rewards


if __name__ == "__main__":
    pass
