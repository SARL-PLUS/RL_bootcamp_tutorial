import math
import random
from enum import Enum
from typing import Optional, Dict, Any

import gymnasium as gym
import numpy as np
from cpymad.madx import Madx
from gymnasium import spaces
from matplotlib import pyplot as plt


class Plane(Enum):
    horizontal = 0
    vertical = 1

class DynamicsHelper:
    # init
    def __init__(self):
        self.twiss = self._generate_optics()
        self.response_scale = 0.5
        self.twiss_bpms = self.twiss[self.twiss["keyword"] == "monitor"]
        self.twiss_correctors = self.twiss[self.twiss["keyword"] == "kicker"]

    def _calculate_response(self, bpmsTwiss, correctorsTwiss, plane):
        bpms = bpmsTwiss.index.values.tolist()
        correctors = correctorsTwiss.index.values.tolist()
        bpms.pop(0)
        correctors.pop(-1)
        rmatrix = np.zeros((len(bpms), len(correctors)))
        for i, bpm in enumerate(bpms):
            for j, corrector in enumerate(correctors):
                if plane == Plane.horizontal:
                    bpm_beta = bpmsTwiss.betx[bpm]
                    corrector_beta = correctorsTwiss.betx[corrector]
                    bpm_mu = bpmsTwiss.mux[bpm]
                    corrector_mu = correctorsTwiss.mux[corrector]
                else:
                    bpm_beta = bpmsTwiss.bety[bpm]
                    corrector_beta = correctorsTwiss.bety[corrector]
                    bpm_mu = bpmsTwiss.muy[bpm]
                    corrector_mu = correctorsTwiss.muy[corrector]

                if bpm_mu > corrector_mu:
                    rmatrix[i][j] = (
                            math.sqrt(bpm_beta * corrector_beta)
                            * math.sin((bpm_mu - corrector_mu) * 2.0 * math.pi)
                            * self.response_scale
                    )
                else:
                    rmatrix[i][j] = 0.0
        return rmatrix

    def generate_optics(self, randomize=True):
        twiss = self._generate_optics(randomize)
        twiss_bpms = twiss[twiss["keyword"] == "monitor"]
        twiss_correctors = twiss[twiss["keyword"] == "kicker"]
        responseH = self._calculate_response(
            twiss_bpms, twiss_correctors, Plane.horizontal
        )
        responseV = self._calculate_response(
            twiss_bpms, twiss_correctors, Plane.vertical
        )
        return responseH, responseV

    def recalculate_response(self):
        responseH = self._calculate_response(self.twiss_bpms, self.twiss_correctors, Plane.horizontal)
        responseV = self._calculate_response(self.twiss_bpms, self.twiss_correctors, Plane.vertical)
        return responseH, responseV

    def _generate_optics(self, randomize=False):
        OPTIONS = ["WARN"]  # ['ECHO', 'WARN', 'INFO', 'DEBUG', 'TWISS_PRINT']
        MADX_OUT = [f"option, -{ele};" for ele in OPTIONS]
        madx = Madx(stdout=False)
        madx.input("\n".join(MADX_OUT))
        tt43_ini = "environment/electron_design.mad"
        madx.call(file=tt43_ini, chdir=True)
        madx.use(sequence="tt43", range="#s/plasma_merge")
        quads = {}
        variation_range = (0.75, 1.25)
        if randomize:
            for ele, value in dict(madx.globals).items():
                if "kq" in ele:
                    # quads[ele] = value * 0.8
                    quads[ele] = value * np.random.uniform(variation_range[0], variation_range[1], size=None)
                    # pass
        madx.globals.update(quads)
        madx.input(
            "initbeta0:beta0,BETX=5,ALFX=0,DX=0,DPX=0,BETY=5,ALFY=0,DY=0.0,DPY=0.0,x=0,px=0,y=0,py=0;"
        )
        twiss_cpymad = madx.twiss(beta0="initbeta0").dframe()

        return twiss_cpymad

    def sample_tasks(self, num_tasks):
        # Generate goals using list comprehension for more concise code
        goals = [self.generate_optics() for _ in range(num_tasks)]

        # Create tasks with goals and corresponding IDs using list comprehension
        tasks = [{"goal": goal, "id": idx} for idx, goal in enumerate(goals)]
        return tasks

    def get_origin_task(self, idx=0):
        # Generate goals using list comprehension for more concise code
        goal = self.generate_optics(randomize=False)
        # Create tasks with goals and corresponding IDs using list comprehension
        task = {"goal": goal, "id": idx}
        return task


class AwakeSteering(gym.Env):
    """
    Gym environment for beam steering using reinforcement learning.
    """

    metadata = {"render_modes": []}

    def __init__(self, twiss=None, task=None, train=False, **kwargs):
        """
        Initialize the AwakeSteering environment.

        Args:
            twiss: Optional Twiss parameters.
            task: Task dictionary containing 'goal' and 'id'.
            train: Training mode flag.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.__version__ = "1.0"
        self.MAX_TIME = kwargs.get("MAX_TIME", 100)
        self.boundary_conditions = kwargs.get("boundary_conditions", False)
        self.state_scale = 1.0

        self.threshold = -0.1  # Corresponds to 1 mm scaled.

        self.current_episode = -1
        self.current_steps = 0

        seed = kwargs.get('seed', None)
        self.seed(seed)
        self.maml_helper = DynamicsHelper()
        self.plane = kwargs.get("plane", Plane.horizontal)

        if task is None:
            task = self.maml_helper.get_origin_task()
        self.reset_task(task)

        self.setup_dimensions()

        self.verification_tasks_loc = kwargs.get("verification_tasks_loc", None)

    def setup_dimensions(self):
        """
        Set up the dimensions of the action and observation spaces based on the response matrices.
        """
        num_bpms = len(self.maml_helper.twiss_bpms) - 1
        num_correctors = len(self.maml_helper.twiss_correctors) - 1

        # Define action and observation space limits
        self.high_action = np.ones(num_correctors, dtype=np.float32)
        self.low_action = -self.high_action

        self.high_observation = np.ones(num_bpms, dtype=np.float32) * self.state_scale
        self.low_observation = -self.high_observation

        self.action_space = spaces.Box(
            low=self.low_action, high=self.high_action, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observation, high=self.high_observation, dtype=np.float32
        )

        # Set the response matrix based on the plane
        if self.plane == Plane.horizontal:
            self.rmatrix = self.responseH
        else:
            self.rmatrix = self.responseV

        # Use pseudo-inverse for numerical stability
        self.rmatrix_inverse = np.linalg.pinv(self.rmatrix)

    def step(self, action: np.ndarray):
        """
        Execute one time step within the environment.

        Args:
            action: The action to be taken.

        Returns:
            observation: The agent's observation of the current environment.
            reward: Amount of reward returned after previous action.
            done: Whether the episode has ended.
            truncated: Whether the episode was truncated.
            info: Additional information about the environment.
        """
        delta_kicks = np.clip(action, self.low_action, self.high_action)
        self.state += self.rmatrix.dot(delta_kicks)
        self.state = np.clip(self.state, self.low_observation, self.high_observation)
        return_state = self.state.copy()

        reward = self._get_reward(return_state)

        self.current_steps += 1
        done = reward > self.threshold
        truncated = self.current_steps >= self.MAX_TIME
        if truncated:
            done = True

        # Find all indices where the absolute value of return_state exceeds or equals 1
        violations = np.argwhere(np.abs(return_state) >= 1).flatten()
        if violations.size > 0:
            # Violations detected
            # Take the index of the first violation
            violation_index = violations[0]
            # Set return_state from the point of violation onwards
            # Assign the sign (+1 or -1) of the violating element to the remaining elements
            return_state[violation_index:] = np.sign(return_state[violation_index])
            # Compute a penalty reward based on the modified return_state
            reward = -np.sqrt(np.mean(np.square(return_state)))
        info = {"task": self._id, 'time': self.current_steps}
        return return_state, reward, done, truncated, info

    def _get_reward(self, observation: np.ndarray) -> float:
        """
        Compute the reward based on the current observation.

        Args:
            observation: The current state observation.

        Returns:
            The computed reward.
        """
        # Negative Euclidean norm (L2 norm) as a penalty
        return -np.sqrt(np.mean(np.square(observation)))

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """
        Reset the environment to an initial state.

        Args:
            seed: Seed for randomness.
            options: Additional options for reset.

        Returns:
            observation: The initial observation.
            info: Additional information.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
            self.observation_space.seed(seed)
        self.is_finalized = False
        self.current_steps = 0
        self.current_episode += 1
        self.state = np.clip(self.observation_space.sample(), -1, 1)*self.init_scaling
        return_state = self.state.copy()
        return return_state, {}

    def seed(self, seed: Optional[int] = None):
        """
        Set the seed for the environment's random number generator(s).

        Args:
            seed: The seed value.

        Returns:
            A list containing the seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def sample_tasks(self, num_tasks: int):
        """
        Sample a list of tasks for meta-learning.

        Args:
            num_tasks: The number of tasks to sample.

        Returns:
            A list of task dictionaries.
        """
        tasks = self.maml_helper.sample_tasks(num_tasks)
        return tasks

    def get_origin_task(self, idx: int = 0):
        """
        Get the original task with default optics.

        Args:
            idx: Task ID.

        Returns:
            The original task dictionary.
        """
        task = self.maml_helper.get_origin_task(idx=idx)
        return task

    def reset_task(self, task: Dict[str, Any]):
        """
        Reset the environment with a new task.

        Args:
            task: The task dictionary containing 'goal' and 'id'.
        """
        self._task = task
        self._goal = task["goal"]
        self._id = task["id"]

        self.responseH = self._goal[0]
        self.responseV = self._goal[1]

    def get_task(self):
        """
        Get the current task.

        Returns:
            The current task dictionary.
        """
        return self._task

if __name__ == "__main__":
    # Create an instance of the environment
    env = AwakeSteering()

    num_episodes = 5        # Number of episodes to run
    max_steps_per_episode = env.MAX_TIME  # Maximum steps per episode

    all_states = []
    all_actions = []
    all_rewards = []
    episode_ends = []

    total_steps = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []

        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            # Sample a random action
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)

            # Record the state, action, and reward
            episode_states.append(state.copy())
            episode_actions.append(action.copy())
            episode_rewards.append(reward)

            state = next_state
            step += 1
            total_steps += 1

        # Append episode data to the overall lists
        all_states.extend(episode_states)
        all_actions.extend(episode_actions)
        all_rewards.extend(episode_rewards)

        # Record the total steps so far to know where the episode ended
        episode_ends.append(total_steps)

    # Convert lists to NumPy arrays
    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    all_rewards = np.array(all_rewards)

    # Plotting the results
    num_subplots = 3
    fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 10))

    time_steps = np.arange(len(all_states))

    # Plot states
    for i in range(all_states.shape[1]):
        axs[0].plot(time_steps, all_states[:, i], label=f'State {i+1}')
    axs[0].set_title('States over Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('State Value')
    axs[0].legend(loc='upper right')

    # Plot actions
    for i in range(all_actions.shape[1]):
        axs[1].plot(time_steps, all_actions[:, i], label=f'Action {i+1}')
    axs[1].set_title('Actions over Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Action Value')
    axs[1].legend(loc='upper right')

    # Plot rewards
    axs[2].plot(time_steps, all_rewards, label='Reward')
    axs[2].set_title('Rewards over Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Reward')
    axs[2].legend(loc='upper right')

    # Mark episode boundaries with vertical dashed lines
    for ax in axs:
        for ep_end in episode_ends:
            ax.axvline(x=ep_end, color='k', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()

