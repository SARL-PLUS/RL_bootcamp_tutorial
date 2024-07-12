import os
import os
import pickle
import random
from enum import Enum
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


from Visualize_policy_validation import verify_external_policy_on_specific_env
from helpers import MamlHelpers


class DoFWrapper(gym.Wrapper):
    def __init__(self, env, DoF):
        super(DoFWrapper, self).__init__(env)
        self.DoF = DoF
        # self.threshold = -0.005 * DoF if (DoF <= 6) else -0.1
        self.threshold = -0.1
        self.env = env

        self.observation_space = spaces.Box(low=env.observation_space.low[:self.DoF],
                                            high=env.observation_space.high[:self.DoF],
                                            # shape=env.observation_space.shape[:self.DoF],
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=env.action_space.low[:self.DoF],
                                       high=env.action_space.high[:self.DoF],
                                       # shape=env.action_space.shape[:self.DoF],
                                       dtype=np.float32)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed(seed=seed)
        observation, info = self.env.reset(seed=seed)
        observation = observation[:self.DoF]
        return observation, info

    def step(self, action):
        # Initialize a zero-filled array for the action space
        full_action_space = np.zeros(self.env.action_space.shape)
        full_action_space[:self.DoF] = action  # Set the first 'DoF' elements with the provided action

        # Execute the action in the environment
        observation, reward, terminated, truncated, info = self.env.step(full_action_space)

        # Reset termination status and check for step limit
        truncated = self.env.current_steps >= self.env.MAX_TIME

        # Focus only on the degrees of freedom for observations
        observation = observation[:self.DoF]

        # Update the reward based on the current observation
        reward = self.env._get_reward(observation)

        # Terminate if the reward exceeds the threshold
        if reward >= self.threshold:
            terminated = True

        # observation = observation * self.pot_function(
        #     observation)  # Ensure observation is a NumPy array
        reward = self.env._get_reward(observation)

        # Check for any violations where the absolute values in observations exceed 1
        violations = np.where(abs(observation) >= 1)[0]
        if violations.size > 0:
            # Modify observation from the first violation onward
            observation[violations[0]:] = np.sign(observation[violations[0]])

            # Recalculate reward after modification
            # observation = observation * self.pot_function(
            #     observation)  # Ensure observation is a NumPy array
            reward = self.env._get_reward(observation)
            # truncated = True  # Terminate due to the violation
        #
        return observation, reward, terminated, truncated, info

    def seed(self, seed):
        self.env.seed(seed)

    def pot_function(self, x, k=1000, x0=1):
        """
        Compute a potential function using a modified sigmoid to handle deviations.
        The output scales transformations symmetrically for positive and negative values of x.
        """
        # Precompute the exponential terms to use them efficiently.
        exp_pos = np.exp(k * (x - x0))
        exp_neg = np.exp(k * (-x - x0))

        # Calculate the transformation symmetrically for both deviations
        result = (1 - 1 / (1 + exp_pos)) + (1 - 1 / (1 + exp_neg))

        # Scale and shift the output between 1 and 10
        return 1 + 10 * result


class AwakeSteering(gym.Env):

    def __init__(self, twiss=[], task={}, train=False, **kwargs):
        self.__version__ = "0.2"
        self.MAX_TIME = 100
        self.state_scale = 1
        self.action_scale = 1
        self.threshold = -0.1  # Corresponds to 1 mm scaled.

        self.current_episode = -1
        self.current_steps = 0

        self.seed(kwargs.get('seed'))
        self.maml_helper = MamlHelpers()
        self.plane = kwargs.get("plane", Plane.horizontal)

        if not task:
            task = self.maml_helper.get_origin_task()
        self.reset_task(task)

        self.setup_dimensions()

        self.verification_tasks_loc = kwargs.get("verification_tasks_loc")

    def setup_dimensions(self):
        num_bpms = len(self.maml_helper.twiss_bpms) - 1
        num_correctors = len(self.maml_helper.twiss_correctors) - 1

        self.positions = np.zeros(num_bpms)
        self.settings = np.zeros(num_correctors)
        self.high = np.ones(num_correctors)
        self.low = -self.high

        self.action_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)

        if self.plane == Plane.horizontal:
            self.rmatrix = self.responseH
        else:
            self.rmatrix = self.responseV

        self.rmatrix_inverse = np.linalg.inv(self.rmatrix)

    def step(self, action):
        delta_kicks = np.clip(action, -1., 1) * self.action_scale
        self.state += self.rmatrix.dot(delta_kicks)
        return_state = np.clip(self.state, -1, 1)

        reward = -np.sqrt(np.mean(np.square(return_state)))

        self.current_steps += 1
        done = (reward > self.threshold)
        truncated = (self.current_steps >= self.MAX_TIME)

        violations = np.abs(return_state) >= 1
        if violations.any():
            return_state[violations] = np.sign(return_state[violations])
            reward = -np.sqrt(np.mean(np.square(return_state)))
            truncated = True

        return return_state, reward, done, truncated, {"task": self._id, 'time': self.current_steps}

    import numpy as np



    def _get_reward(self, observation):
        """
        Compute the reward using a sigmoid transformation on each observation,
        then calculate the negative of the Euclidean norm (L2 norm) as a penalty.
        """

        return -np.sqrt(np.mean(np.square(observation)))

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed(seed)
        self.is_finalized = False
        self.current_steps = 0
        self.current_episode += 1
        self.state = self.observation_space.sample()
        return self.state, {}

    def seed(self, seed):
        print(f'set seed {seed}')
        random.seed(seed)
        np.random.seed(seed)

    # MAML specific function, while training samples fresh new tasks and for testing it uses previously saved tasks
    def sample_tasks(self, num_tasks):
        tasks = self.maml_helper.sample_tasks(num_tasks)
        return tasks

    def get_origin_task(self, idx):
        task = self.maml_helper.get_origin_task(idx=idx)
        return task

    def reset_task(self, task):
        self._task = task
        self._goal = task["goal"]
        self._id = task["id"]

        self.responseH = self._goal[0]
        self.responseV = self._goal[1]


class Plane(Enum):
    horizontal = 0
    vertical = 1


def load_prefdefined_task(task_nr):
    # Define file location and name
    verification_tasks_loc = 'configs'
    filename = 'verification_tasks.pkl'  # Adding .pkl extension for clarity
    # Construct the full file path
    full_path = os.path.join(verification_tasks_loc, filename)

    with open(full_path, "rb") as input_file:  # Load in tasks
        tasks = pickle.load(input_file)
    return tasks[task_nr]


if __name__ == "__main__":
    # find_thresholds_for_DoFs(episodes=10000)
    DoF = 2
    env = DoFWrapper(AwakeSteering(), DoF)
    policy = lambda s: env.action_space.sample()
    verify_external_policy_on_specific_env(env, [policy], episodes=15, policy_labels=['rand'], DoF=DoF)
