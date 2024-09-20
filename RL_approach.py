import os
import time
import yaml

from matplotlib import pyplot as plt
from sb3_contrib import TRPO
from stable_baselines3 import PPO
from tqdm import tqdm

from environment.environment_helpers import load_env_config, read_experiment_config
from helper_scripts.general_helpers import make_experiment_folder, plot_progress, \
    verify_external_policy_on_specific_env

# Todo: Make plots interactive!!!

# Load the configuration
with open('config/environment_setting.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract RL settings
rl_settings = config['rl-settings']
algorithm = rl_settings['algorithm']

# Extract algorithm-specific parameters
if algorithm == 'PPO':
    algo_params = rl_settings.get('ppo', {})
elif algorithm == 'TRPO':
    algo_params = rl_settings.get('trpo', {})
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")

# Here we select one possible MDP out of a set of MDPs - not important at this stage
environment_settings = read_experiment_config('config/environment_setting.yaml')
predefined_task = environment_settings['task_setting']['task_nr']
task_location = environment_settings['task_setting']['task_location']
DoF = environment_settings['degrees-of-freedom']

# Train on different size of the environment
env = load_env_config(env_config='config/environment_setting.yaml')


validation_seeds = environment_settings['validation-settings']['validation-seeds']
nr_validation_episodes = len(validation_seeds)  # Number of validation episodes


# Specific for RL training savings
# Select on algorithm
algorithm = environment_settings['rl-settings']['algorithm']
optimization_type = 'RL'
save_folder_figures = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Figures')
save_folder_weights = make_experiment_folder(optimization_type, algorithm, environment_settings, purpose='Weights')


if algorithm == 'TRPO':
    model = TRPO("MlpPolicy", env,
                 learning_rate=algo_params.get('learning_rate', 1e-3),
                 n_steps=algo_params.get('n_steps', 2048),
                 batch_size=algo_params.get('batch_size', 128),
                 gamma=algo_params.get('gamma', 0.99),
                 cg_max_steps=algo_params.get('cg_max_steps', 15),
                 cg_damping=algo_params.get('cg_damping', 0.1),
                 line_search_shrinking_factor=algo_params.get('line_search_shrinking_factor', 0.8),
                 line_search_max_iter=algo_params.get('line_search_max_iter', 10),
                 n_critic_updates=algo_params.get('n_critic_updates', 10),
                 gae_lambda=algo_params.get('gae_lambda', 0.95),
                 use_sde=algo_params.get('use_sde', False),
                 sde_sample_freq=algo_params.get('sde_sample_freq', -1),
                 normalize_advantage=algo_params.get('normalize_advantage', True),
                 target_kl=algo_params.get('target_kl', 0.01),
                 sub_sampling_factor=algo_params.get('sub_sampling_factor', 1))
elif algorithm == 'PPO':
    model = PPO("MlpPolicy", env,
                learning_rate=algo_params.get('learning_rate', 3e-4),
                n_steps=algo_params.get('n_steps', 2048),
                batch_size=algo_params.get('batch_size', 64),
                n_epochs=algo_params.get('n_epochs', 10),
                gamma=algo_params.get('gamma', 0.99),
                gae_lambda=algo_params.get('gae_lambda', 0.95),
                clip_range=algo_params.get('clip_range', 0.2),
                ent_coef=algo_params.get('ent_coef', 0.0),
                vf_coef=algo_params.get('vf_coef', 0.5),
                max_grad_norm=algo_params.get('max_grad_norm', 0.5),
                use_sde=algo_params.get('use_sde', False),
                sde_sample_freq=algo_params.get('sde_sample_freq', -1))
else:
    print('Select valid algorithm')

success_rates, mean_rewards, x_plot = [], [], []

total_steps = environment_settings['rl-settings']['total_steps']
evaluation_steps =  environment_settings['rl-settings']['evaluation_steps']
increments = total_steps // evaluation_steps

for i in tqdm(range(0, evaluation_steps)):
    num_samples = increments * i
    save_folder_figures_individual = os.path.join(save_folder_figures, f'{num_samples:07}')
    save_folder_weights_individual = os.path.join(save_folder_weights, f'{num_samples:07}')
    if i > 0:
        model.learn(total_timesteps=increments)
    model.save(save_folder_weights_individual)
    # vec_env = model.get_env()
    policy = lambda x: model.predict(x, deterministic=True)[0]
    title = f'{algorithm}_{DoF}_{num_samples} samples, threshold={env.threshold}'
    success_rate, mean_reward = verify_external_policy_on_specific_env(env, [policy],
                                                                       num_samples=num_samples,
                                                                       episodes=nr_validation_episodes,
                                                                       title=title,
                                                                       save_folder=save_folder_figures,
                                                                       policy_labels=[algorithm], DoF=DoF,
                                                                       nr_validation_episodes=nr_validation_episodes,
                                                                       seed_set=validation_seeds)

    print(success_rate)
    time.sleep(2)
    success_rates.append(success_rate)
    mean_rewards.append(mean_reward)

    x_plot.append(num_samples)
    plot_progress(x_plot, mean_rewards, success_rates, DoF, num_samples=num_samples,
                  nr_validation_episodes=nr_validation_episodes, algorithm=algorithm, save_figure=save_folder_figures)

x = [i * increments for i in (range(0, evaluation_steps))]
plot_progress(x_plot, mean_rewards, success_rates, DoF, num_samples=num_samples,
              nr_validation_episodes=nr_validation_episodes, algorithm=algorithm, save_figure=save_folder_figures)

if algorithm == 'TRPO':
    model = TRPO.load(save_folder_weights_individual)
elif algorithm == 'PPO':
    model = PPO.load(save_folder_weights_individual)

# vec_env = model.get_env()
policy = lambda x: model.predict(x, deterministic=True)[0]

verify_external_policy_on_specific_env(env, [policy],
                                       num_samples=num_samples,
                                       episodes=10, title=algorithm,
                                       save_folder=save_folder_figures, policy_labels=[algorithm],
                                       DoF=DoF, nr_validation_episodes=nr_validation_episodes,
                                       seed_set=validation_seeds)
