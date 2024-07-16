import random
from enum import Enum
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.helpers import MamlHelpers, DoFWrapper
from helper_scripts.Visualize_policy_validation import verify_external_policy_on_specific_env


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

    def _get_reward(self, observation):
        """
        Compute the reward using a sigmoid transformation on each observation,
        then calculate the negative of the Euclidean norm (L2 norm) as a penalty.
        """

        return -np.sqrt(np.mean(np.square(observation)))

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed(seed)
            self.observation_space.seed(seed)
        self.is_finalized = False
        self.current_steps = 0
        self.current_episode += 1
        self.state = self.observation_space.sample()
        return self.state, {}

    def seed(self, seed):
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





if __name__ == "__main__":
    # find_thresholds_for_DoFs(episodes=10000)
    DoF = 2
    env = DoFWrapper(AwakeSteering(), DoF)
    policy = lambda s: env.action_space.sample()
    verify_external_policy_on_specific_env(env, [policy], episodes=15, policy_labels=['rand'], DoF=DoF)
