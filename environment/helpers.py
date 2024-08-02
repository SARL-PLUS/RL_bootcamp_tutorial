import math
import os
import pickle
from enum import Enum
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from cpymad.madx import Madx
from gymnasium import Wrapper
from gymnasium import spaces

from environment.environment_awake_steering import AwakeSteering


class Plane(Enum):
    horizontal = 0
    vertical = 1





class DoFWrapper(gym.Wrapper):
    def __init__(self, env, DoF, **kwargs):
        super(DoFWrapper, self).__init__(env)
        self.DoF = DoF
        # self.threshold = -0.005 * DoF if (DoF <= 6) else -0.1
        self.threshold = -0.1
        self.env = env
        (self.boundary_conditions) = kwargs.get("boundary_conditions", False)

        self.observation_space = spaces.Box(low=env.observation_space.low[:self.DoF],
                                            high=env.observation_space.high[:self.DoF],
                                            # shape=env.observation_space.shape[:self.DoF],
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=env.action_space.low[:self.DoF],
                                       high=env.action_space.high[:self.DoF],
                                       # shape=env.action_space.shape[:self.DoF],
                                       dtype=np.float32)

    def reset(self, seed: Optional[int] = None):
        observation, info = self.env.reset(seed=seed)
        observation = observation[:self.DoF]
        return observation, info

    def step(self, action):
        # Initialize a zero-filled array for the action space
        full_action_space = np.zeros(self.env.action_space.shape)
        full_action_space[:self.DoF] = action  # Set the first 'DoF' elements with the provided action

        # Execute the action in the environment
        observation, reward, terminated, truncated, info = self.env.step(full_action_space)

        # # Reset termination status and check for step limit
        # truncated = self.env.current_steps >= self.env.MAX_TIME

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
            reward = self.env._get_reward(observation)*100
            # print('reward', reward)
            terminated = self.boundary_conditions

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


class Awake_Benchmarking_Wrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.invV = None
        self.invH = None
        self.optimal_rewards = None
        self.optimal_actions = None
        self.optimal_states = None

    def reset(self, **kwargs):
        #     print('reset', self.current_steps, self._get_reward(return_value))
        return_initial_state, _ = self.env.reset(**kwargs)

        self.invH, self.invV = np.linalg.inv(self.env.responseH / 100) / 100, np.linalg.inv(
            self.env.responseV / 100) / 100
        self.optimal_states, self.optimal_actions, self.optimal_rewards = self._get_optimal_trajectory(
            return_initial_state)
        return return_initial_state, {}

    def policy_optimal(self, state):
        # invrmatrix = self.invH if self.plane == 'horizontal' else self.invV
        invrmatrix = self.invH
        action = -invrmatrix.dot(state * self.env.state_scale)
        # action = np.clip(action, -1, 1)
        action_abs_max = max(abs(action))
        if action_abs_max > 1:
            action /= action_abs_max
        return action

    def get_k_for_state(self, state):
        # invrmatrix = self.invH if self.plane == 'horizontal' else self.invV
        invrmatrix = self.invH
        k = invrmatrix.dot(state * self.env.unwrapped.state_scale) * self.env.unwrapped.action_scale
        return k

    def get_optimal_trajectory(self):
        return self.optimal_states, self.optimal_actions, self.optimal_rewards

    def _get_optimal_trajectory(self, init_state):
        max_iterations = 25
        states = [init_state]
        actions = []
        # Todo: reward scaling
        rewards = [self.env._get_reward(init_state) * self.env.reward_scale]

        self.env.kicks_0_opt = self.env.kicks_0.copy()
        self.env.kicks_0 = self.get_k_for_state(init_state)
        self.env.is_finalized = False

        for i in range(max_iterations):
            action = self.policy_optimal(states[i])
            actions.append(action)
            state, reward, is_finalized, _, _ = self.env.step(action)

            states.append(state)
            rewards.append(reward)

            if is_finalized:
                break

        if i < max_iterations - 1:
            # nan_state = [np.nan] * self.env.observation_space.shape[-1]
            # nan_action = [np.nan] * self.env.action_space.shape[-1]
            # states[i + 2:] = [nan_state] * (max_iterations - i - 1)
            # actions[i + 1:] = [nan_action] * (max_iterations - i - 1)
            states.append([np.nan] * self.env.observation_space.shape[-1])
            actions.append([np.nan] * self.env.action_space.shape[-1])
            rewards.append(np.nan)

        self.env.kicks_0 = self.env.kicks_0_opt.copy()
        self.env.is_finalized = False
        return states, actions, rewards

    def draw_optimal_trajectories(self, init_state, nr_trajectories=5):
        states_frames, actions_frames, rewards_frames = [], [], []
        len_mean = []

        for i in range(nr_trajectories):
            states, actions, rewards = self._get_optimal_trajectory(init_state)
            states_frames.append(pd.DataFrame(states))
            actions_frames.append(pd.DataFrame(actions))
            rewards_frames.append(pd.DataFrame(rewards))
            # actions end with np.nan to find episode ends
            len_mean.append(len(actions) - 1)

        mean_length = np.mean(len_mean)
        # print(mean_length)

        states_df = pd.concat(states_frames, ignore_index=True)
        actions_df = pd.concat(actions_frames, ignore_index=True)
        rewards_df = pd.concat(rewards_frames, ignore_index=True)

        fig, axs = plt.subplots(3, figsize=(10, 10))
        for df, ax in zip([states_df, rewards_df, actions_df], axs):
            df.plot(ax=ax)

        plt.suptitle(f'Mean Length of Episodes: {mean_length}')
        plt.tight_layout()
        plt.show()
        plt.pause(1)


def load_predefined_task(task_nr, full_path):
    with open(full_path, "rb") as input_file:  # Load in tasks
        tasks = pickle.load(input_file)
    return tasks[task_nr]


# Helper functions for plotting
def plot_results(states, actions, rewards, env, title):
    fig, axs = plt.subplots(3)
    axs[0].plot(states)
    axs[0].set_title('States')
    axs[1].plot(actions)
    axs[1].set_title('Actions')
    axs[2].plot(rewards)
    axs[2].set_title('Rewards')
    axs[2].axhline(env.unwrapped.threshold, c='r')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_optimal_policy(states_opt_list, actions_opt_list, returns_opt_list, env):
    states_df = pd.concat([pd.DataFrame(states) for states in states_opt_list], ignore_index=True)
    actions_df = pd.concat([pd.DataFrame(actions) for actions in actions_opt_list], ignore_index=True)
    returns_df = pd.concat([pd.DataFrame(rewards) for rewards in returns_opt_list], ignore_index=True)
    episode_lengths = [len(rewards) - 1 for rewards in returns_opt_list]

    fig, axs = plt.subplots(4, figsize=(10, 10))
    states_df.plot(ax=axs[0])
    axs[0].set_title('States')
    actions_df.plot(ax=axs[1])
    axs[1].set_title('Actions')
    returns_df.plot(ax=axs[2])
    axs[2].axhline(env.unwrapped.threshold, c='r')
    axs[2].set_title('Return')
    axs[3].plot(episode_lengths)
    axs[3].set_title('Length')
    plt.suptitle('Optimal Policy')
    plt.tight_layout()
    plt.show()


def read_yaml_file(filepath):
    """
    Reads a YAML file and returns the data.

    Args:
    - filepath (str): The path to the YAML file.

    Returns:
    - dict: The data from the YAML file.
    """
    try:
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("The file could not be found.")
    except yaml.YAMLError as exc:
        print("Error parsing YAML:", exc)


# Model parameters for MPC from env
def get_model_parameters(environment):
    """
    Extract and process model parameters from a given environment.

    This function retrieves the action matrix and threshold for model predictive control (MPC)
    from the environment's underlying attributes. It adjusts the action matrix based on the
    degrees of freedom (DoF) and scales it appropriately.

    Args:
        environment: An environment object that should have attributes `unwrapped`, `DoF`,
                     `action_scale`, and `threshold`.

    Returns:
        tuple: A tuple containing the scaled action matrix and the MPC threshold.

    Raises:
        AttributeError: If the necessary attributes are not found in the environment.
    """
    try:
        # Extract the action matrix from the environment.
        action_matrix = environment.unwrapped.rmatrix

        # Adjust the action matrix size based on the Degrees of Freedom (DoF).
        action_matrix_reduced = action_matrix[:environment.DoF, :environment.DoF]

        # Scale the action matrix by the environment's action scale factor.
        action_matrix_scaled = action_matrix_reduced * environment.unwrapped.action_scale

        # Define the threshold for MPC, converting it to a positive value if necessary.
        threshold = -environment.threshold if environment.threshold < 0 else environment.threshold

        return action_matrix_scaled, threshold

    except AttributeError as error:
        raise AttributeError("Missing one or more required attributes from the environment: " + str(error))


# Example usage
# Assuming `env` is your environment object loaded with all necessary attributes.
# action_matrix, mpc_threshold = get_model_parameters(env)

class RewardScalingWrapper(gym.Wrapper):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale
        self.state_scale = 1.0

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale

        observation = self.state_scale * observation + np.random.uniform(low=-0.001, high=0.001, size=observation.shape)
        return observation, scaled_reward, done, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = self.state_scale * observation + np.random.uniform(low=-0.001, high=0.001, size=observation.shape)
        return observation, info


def load_env_config(env_config='config/environment_setting.yaml'):
    environment_settings = read_yaml_file(env_config)

    # Load a predefined task for verification
    verification_task = load_predefined_task(environment_settings['task_setting']['task_nr'],
                                             environment_settings['task_setting']['task_location'])

    DoF = environment_settings['degrees-of-freedom']  # Degrees of Freedom

    MAX_TIME = environment_settings['terminal-conditions']['MAX-TIME']
    boundary_conditions = environment_settings['terminal-conditions']['boundary-conditions']

    env = DoFWrapper(AwakeSteering(task=verification_task, MAX_TIME=MAX_TIME), DoF,
                     boundary_conditions=boundary_conditions)

    return env
