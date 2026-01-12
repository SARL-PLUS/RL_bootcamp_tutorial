"""
Simon Hirlaender
This script sets up and verifies a Model Predictive Control (MPC) policy
on a predefined environment task using the AwakeSteering simulation.
The script performs the following steps:
1. Loads a predefined task for verification.
2. Initializes the environment with the specified task and Degrees of Freedom (DoF).
3. Configures the model parameters for MPC, including scaling the action matrix.
4. Defines the MPC policy using a lambda function.
5. Creates a folder to save verification results.
6. Verifies the external MPC policy on the specific environment and saves the results.

Dependencies:
- MPC module for the model_predictive_control function.
- Visualize_policy_validation module for the verify_external_policy_on_specific_env function.
- environment_awake_steering module for AwakeSteering, DoFWrapper, and load_prefdefined_task functions.

"""

# TODO: save data for accelerated verification in the verification functions
import os

from environment.environment_helpers import read_experiment_config, load_env_config
# Importing required functions and classes
from helper_scripts.general_helpers import verify_external_policy_on_specific_env, make_experiment_folder

environment_settings = read_experiment_config('config/environment_setting.yaml')

DoF = environment_settings['degrees-of-freedom']  # Degrees of Freedom
validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes

# Train on different size of the environment
env = load_env_config(env_config='config/environment_setting.yaml')

#seed = 42
#env.action_space.seed(seed)


optimization_type = 'random_walk'
algorithm = ''

save_folder_figures = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Figures')
save_folder_results = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Random walk')
save_name_results = os.path.join(save_folder_results, 'Random_walk_results.pkl')

print(save_name_results)
print(save_folder_figures)

'''
import numpy as np
import matplotlib.pyplot as plt

num_experiments = 1
all_success_rates = []
all_mean_rewards = []


fixed_seed_set = [ 46]

for exp in range(num_experiments):
    print(f"Running experiment {exp + 1}/{num_experiments}")
    success_rate, mean_rewards = verify_external_policy_on_specific_env(
        env, [None],
        #seed=84,  # Global seed for reproducibility if needed
        episodes=len(fixed_seed_set),  # Number of episodes equals the number of fixed seeds
        title=f'Random Experiment {exp + 1}',
        save_folder=save_folder_figures,
        policy_labels=['Random'],
        DoF=DoF,
        #seed_set=fixed_seed_set,  # Use your fixed list of seeds here
        save_results=save_name_results  # Adjust save path if necessary to avoid overwriting
    )

    # Store the outputs for later analysis
    all_success_rates.append(success_rate)
    all_mean_rewards.append(mean_rewards)


all_success_rates = np.array(all_success_rates)
all_mean_rewards = np.array(all_mean_rewards)
mean_success_rate = np.mean(all_success_rates)
std_success_rate = np.std(all_success_rates)

print(f"Overall Mean Success Rate: {mean_success_rate:.3f} ± {std_success_rate:.3f}")
mean_rewards_avg = np.mean(all_mean_rewards, axis=0)
std_rewards_avg = np.std(all_mean_rewards, axis=0)
episodes = np.arange(len(mean_rewards_avg))
plt.errorbar(episodes, mean_rewards_avg, yerr=std_rewards_avg, fmt='o', capsize=5)
plt.xlabel('Episode (or Task Index)')
plt.ylabel('Mean Reward')
plt.title('Mean Rewards across 5 Random Experiments')
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt


#seed
fixed_seed_set = [43, 44, 45, 46, 47]


all_success_rates = []
all_mean_rewards = []
state_trajectories = {}





for seed in fixed_seed_set:
    print(f"Running experiment with seed {seed}")
    env.action_space.seed(seed)# für gleiche ergebnisse
    success_rate, mean_rewards = verify_external_policy_on_specific_env(
        env, [None],
        episodes=1,
        title=f'Random Experiment with seed {seed}',
        save_folder=save_folder_figures,
        policy_labels=['Random'],
        DoF=DoF,
        seed_set=[seed],
        save_results=save_name_results
    )
    all_success_rates.append(success_rate )
    all_mean_rewards.append(mean_rewards[0])
    state_trajectories[seed] = mean_rewards
all_success_rates = np.array(all_success_rates)
all_mean_rewards = np.array(all_mean_rewards)
mean_success_rate = np.mean( all_success_rates)
std_success_rate = np.std(all_success_rates)
mean_reward = np.mean( all_mean_rewards)
std_reward = np.std(all_mean_rewards)

print("Statistical Analysis:")
print(f"Overall Mean Success Rate: {mean_success_rate:.3f} ± {std_success_rate:.3f}")
print(f"Overall Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.errorbar(fixed_seed_set, all_mean_rewards, yerr=std_reward, fmt='o', capsize=5, label='Mean Reward per Seed')
plt.xlabel("Seed")
plt.ylabel("Mean Reward")
plt.title("Mean Reward per Fixed Seed")
plt.legend()
plt.grid(True)
fig, axes = plt.subplots(len(fixed_seed_set), 1, figsize=(8, len(fixed_seed_set) * 3), sharex=True)

for idx, seed in enumerate(fixed_seed_set):
    ax = axes[idx]
    states = state_trajectories[seed]
    episodes = np.arange(1, len(states) + 1)
    ax.plot(episodes, states, marker='o', label=f"Seed {seed}")
    ax.set_ylabel("State Value")
    ax.set_title(f"State Trajectory for Seed {seed}")
    ax.legend()
    ax.grid(True)

plt.xlabel("Episode")
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
for seed in fixed_seed_set:
    states = state_trajectories[seed]
    episodes = np.arange(1, len(states) + 1)
    plt.plot(episodes, states, marker='o', linestyle='-', label=f"Seed {seed}")



plt.xlabel("Episode")
plt.ylabel("State Value")
plt.title("State Trajectories for All Seeds")
plt.legend()
plt.grid(True)
plt.show()



