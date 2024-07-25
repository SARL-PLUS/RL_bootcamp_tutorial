import gymnasium as gym
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from environment.environment_awake_steering import AwakeSteering


class EpisodeData:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_done = False

    def add_step(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def end_episode(self):
        self.is_done = True


class SmartEpisodeTrackerWithPlottingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episodes = []
        self.current_episode = None
        self._setup_plotting()

    def _setup_plotting(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(6, 8), tight_layout=True
        )
        self.cumulative_step = 0
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.colors_states = cm.rainbow(np.linspace(0, 1, self.n_states))
        self.colors_actions = cm.rainbow(np.linspace(0, 1, self.n_actions))
        plt.show(block=False)


    def _update_plots(self):
        self.ax1.clear()
        self.ax2.clear()

        cumulative_step = 0

        # Function to plot data for an episode
        def plot_episode(episode, start_step):
            if not episode:
                return start_step

            trajectory = (
                np.array(episode.states)
                if episode.states
                else np.zeros((0, self.n_states))
            )
            steps = range(start_step, start_step + len(trajectory))

            for i in range(self.n_states):
                self.ax1.plot(steps, trajectory[:, i], color=self.colors_states[i])

            for i in range(self.n_actions):
                action_values = [
                    action[i] if action is not None and i < len(action) else np.nan
                    for action in episode.actions
                ]
                self.ax2.plot(
                    steps,
                    action_values,
                    color=self.colors_actions[i],
                    ls="--",
                    marker=".",
                )

            return start_step + len(trajectory)

        # Plot data for each completed episode
        for episode in self.episodes:
            cumulative_step = plot_episode(episode, cumulative_step)

        # Plot data for the current (incomplete) episode
        cumulative_step = plot_episode(self.current_episode, cumulative_step)

        self.ax1.set_title("Trajectories for Each Episode")
        self.ax1.set_xlabel("Cumulative Step")
        self.ax1.set_ylabel("State Value")
        self.ax1.grid()

        self.ax2.set_title("Actions for Each Episode")
        self.ax2.set_xlabel("Cumulative Step")
        self.ax2.set_ylabel("Action Value")
        self.ax2.grid()

        # Update legends
        legend_handles_states = [
            mlines.Line2D([], [], color=self.colors_states[i], label=f"State {i + 1}")
            for i in range(self.n_states)
        ]
        legend_handles_actions = [
            mlines.Line2D([], [], color=self.colors_actions[i], label=f"Action {i + 1}")
            for i in range(self.n_actions)
        ]

        # self.ax1.legend(handles=legend_handles_states)
        # self.ax2.legend(handles=legend_handles_actions)
        self.ax1.legend(
            handles=legend_handles_states, loc="upper left", bbox_to_anchor=(1, 1)
        )
        self.ax2.legend(
            handles=legend_handles_actions, loc="upper left", bbox_to_anchor=(1, 1)
        )

        self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

    def step(self, action):
        observation, reward, done, _, info = self.env.step(action)

        if self.current_episode is None:
            self.current_episode = EpisodeData()

        self.current_episode.add_step(observation, action, reward)

        if done:
            self.current_episode.end_episode()
            self.episodes.append(self.current_episode)
            self.current_episode = None

        self._update_plots()
        return observation, reward, done, False, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        if self.current_episode is not None and not self.current_episode.is_done:
            self.current_episode.end_episode()
            self.episodes.append(self.current_episode)

        self.current_episode = EpisodeData()
        self.current_episode.add_step(
            observation, None, None
        )  # Initial state with no action or reward

        self._update_plots()
        return observation, info


def plot_trajectories_and_actions(env, n_episodes):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    cumulative_step = 0  # To track the cumulative number of steps across episodes

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    colors_states = cm.rainbow(np.linspace(0, 1, n_states))
    colors_actions = cm.rainbow(np.linspace(0, 1, n_actions))

    # Create legend handles
    legend_handles_states = [
        mlines.Line2D([], [], color=colors_states[i], label=f"State {i + 1}")
        for i in range(n_states)
    ]
    legend_handles_actions = [
        mlines.Line2D([], [], color=colors_actions[i], label=f"Action {i + 1}")
        for i in range(n_actions)
    ]

    for _ in range(n_episodes):
        _, _ = env.reset()
        done = False
        trajectory = []
        actions = []

        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            trajectory.append(next_state)
            actions.append(action)

        trajectory = np.array(trajectory)
        actions = np.array(actions)

        steps = range(cumulative_step, cumulative_step + len(trajectory))

        for i in range(n_states):
            ax1.plot(steps, trajectory[:, i], color=colors_states[i])
        for i in range(n_actions):
            ax2.plot(steps, actions[:, i], color=colors_actions[i])

        cumulative_step += len(trajectory)

    ax1.set_title("Trajectories for Each Episode")
    ax1.set_xlabel("Cumulative Step")
    ax1.set_ylabel("State Value")
    ax1.legend(handles=legend_handles_states)
    ax1.grid()

    ax2.set_title("Actions for Each Episode")
    ax2.set_xlabel("Cumulative Step")
    ax2.set_ylabel("Action Value")
    ax2.legend(handles=legend_handles_actions)
    ax2.grid()

    plt.show()


if __name__ == "__main__":
    # Initialize the environment
    env = AwakeSteering(drift_amplitude=0.25)
    # plot_trajectories_and_actions(env, 3)  # Plot for 3 episodes

    wrapped_env = SmartEpisodeTrackerWithPlottingWrapper(env)

    for _ in range(10):  # Number of episodes
        obs, _ = wrapped_env.reset()
        done = False
        while not done:
            action = wrapped_env.action_space.sample()
            obs, reward, done, _, _ = wrapped_env.step(action)

    plt.ioff()  # Turn off interactive mode
    plt.show()
