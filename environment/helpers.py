import logging
import os
import pickle
from enum import Enum
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from gymnasium import Wrapper
from gymnasium import spaces

from environment.environment_awake_steering import AwakeSteering


class Plane(Enum):
    horizontal = 0
    vertical = 1

from typing import Optional, Any, Dict


class DoFWrapper(gym.Wrapper):
    """
    Gym Wrapper to limit the environment to a subset of degrees of freedom (DoF).
    This wrapper modifies the action and observation spaces to include only the first 'DoF' elements.
    It also modifies the reward and termination conditions based on the subset of observations.
    """

    def __init__(self, env: gym.Env, DoF: int, **kwargs):
        """
        Initialize the DoFWrapper.

        Args:
            env: The original Gym environment.
            DoF: The number of degrees of freedom to limit the action and observation spaces to.
            **kwargs: Additional keyword arguments, such as 'threshold' and 'boundary_conditions'.
        """
        super().__init__(env)
        self.DoF = DoF
        self.threshold = kwargs.get("threshold", -0.1)
        self.boundary_conditions = kwargs.get("boundary_conditions", False)
        self.env.init_scaling = kwargs.get("init_scaling", 1.0)
        self.action_scale = kwargs.get("action_scale", 1.0)
        print('self.action_scale: ', self.action_scale)


        # Modify the action and observation spaces
        self.action_space = spaces.Box(
            low=env.action_space.low[:DoF],
            high=env.action_space.high[:DoF],
            dtype=env.action_space.dtype,
        )
        self.observation_space = spaces.Box(
            low=env.observation_space.low[:DoF],
            high=env.observation_space.high[:DoF],
            dtype=env.observation_space.dtype,
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """
        Reset the environment and return the initial observation limited to the specified DoF.

        Args:
            seed: Optional seed for the environment.
            options: Optional dictionary of options.

        Returns:
            observation: The initial observation limited to the specified DoF.
            info: Additional information from the environment.
        """
        observation, info = self.env.reset(seed=seed, options=options)
        observation = observation[: self.DoF]
        return observation, info

    def step(self, action: np.ndarray):
        """
        Step the environment with the given action, limited to the specified DoF.

        Args:
            action: The action to take, limited to the DoF.

        Returns:
            observation: The observation after the action, limited to the DoF.
            reward: The reward after the action.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode was truncated.
            info: Additional information from the environment.
        """

        action = action*self.action_scale
        # Initialize a zero-filled array for the full action space
        full_action = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        full_action[: self.DoF] = action  # Set the first 'DoF' elements with the provided action

        # Execute the action in the environment
        observation, reward, terminated, truncated, info = self.env.step(full_action)

        # Focus only on the degrees of freedom for observations
        observation = observation[: self.DoF]

        # Update the reward based on the current observation
        reward = self.env._get_reward(observation)

        # Check for termination based on the reward threshold
        if reward >= self.threshold:
            terminated = True

        # Check for any violations where the absolute values in observations exceed 1
        violations = np.where(np.abs(observation) >= 1)[0]
        if violations.size > 0:
            # Modify observation from the first violation onward
            first_violation = violations[0]
            observation[first_violation:] = np.sign(observation[first_violation])

            # Recalculate reward after modification
            reward = self.env._get_reward(observation) * 100

            # Terminate if boundary conditions are set
            terminated = self.boundary_conditions

        return observation, reward, terminated, truncated, info

    def seed(self, seed: Optional[int] = None):
        """
        Set the seed for the environment's random number generator(s).

        Args:
            seed: The seed value.

        Returns:
            A list containing the seed.
        """
        return self.env.seed(seed)

    # Optional function, not currently used
    def pot_function(self, x: np.ndarray, k: float = 1000, x0: float = 1) -> np.ndarray:
        """
        Compute a potential function using a modified sigmoid to handle deviations.
        The output scales transformations symmetrically for positive and negative values of x.

        Args:
            x: Input array.
            k: Steepness of the sigmoid.
            x0: Center point of the sigmoid.

        Returns:
            Transformed array with values scaled between 1 and 11.
        """
        # Precompute the exponential terms to use them efficiently.
        exp_pos = np.exp(k * (x - x0))
        exp_neg = np.exp(k * (-x - x0))

        # Calculate the transformation symmetrically for both deviations
        result = (1 - 1 / (1 + exp_pos)) + (1 - 1 / (1 + exp_neg))

        # Scale and shift the output between 1 and 11
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


def load_predefined_task(task_nr, task_location):
    # Check if the file exists
    if not os.path.exists(task_location):
        task_location = os.path.join(os.getcwd(), '..', task_location)

    with open(task_location, "rb") as input_file:  # Load in tasks
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
        action_matrix_scaled = action_matrix_reduced * environment.action_scale

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


def load_env_config(env_config: str = 'config/environment_setting.yaml') -> Any:
    """
    Load the environment configuration from a YAML file and initialize the environment.

    Args:
        env_config (str): Path to the environment configuration YAML file.

    Returns:
        env: An initialized environment object based on the configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If required configuration keys are missing.
    """

    # Read environment settings from the YAML configuration file
    logging.info(f"Loading environment configuration from {env_config}")
    environment_settings = read_yaml_file(env_config)

    # Extract task settings with error handling
    try:
        task_settings = environment_settings['task_setting']
        task_nr = task_settings['task_nr']
        task_location = task_settings['task_location']
    except KeyError as e:
        raise KeyError(f"Missing key in 'task_setting' configuration: {e}")

    # Load a predefined task for verification
    verification_task = load_predefined_task(
        task_nr=task_nr,
        task_location=task_location
    )
    logging.info(f"Loaded task number {task_nr} from {task_location}")

    # Get the Degrees of Freedom (DoF) setting from the configuration
    try:
        DoF = environment_settings['degrees-of-freedom']  # Degrees of Freedom
    except KeyError as e:
        raise KeyError(f"Missing key 'degrees-of-freedom' in configuration: {e}")

    # Get terminal conditions with error handling
    try:
        terminal_conditions = environment_settings['terminal-conditions']
        MAX_TIME = terminal_conditions['MAX-TIME']
        boundary_conditions = terminal_conditions['boundary-conditions']
    except KeyError as e:
        raise KeyError(f"Missing key in 'terminal-conditions' configuration: {e}")

    # Get initial scaling parameters, providing a default if necessary
    init_scaling = environment_settings['init_scaling']  # Default scaling factor is 1.0
    action_scale = environment_settings['action_scale']

    # Initialize the base environment with the loaded task and maximum time
    base_env = AwakeSteering(
        task=verification_task,
        MAX_TIME=MAX_TIME
    )

    # Initialize the environment wrapper with DoF, boundary conditions, and initial scaling
    env = DoFWrapper(
        env=base_env,
        DoF=DoF,
        boundary_conditions=boundary_conditions,
        init_scaling=init_scaling,
        action_scale=action_scale
    )

    logging.info("Environment initialized successfully.")
    return env

