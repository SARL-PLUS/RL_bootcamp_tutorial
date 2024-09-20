import time
import logging
from typing import Any, Tuple

import matplotlib

matplotlib.use('TkAgg')  # Force the use of the TkAgg backend for external windows

from helper_scripts.gp_mpc_controller import GpMpcController
from environment.environment_helpers import (
    read_experiment_config,
    load_env_config,
    RewardScalingWrapper, SmartEpisodeTrackerWithPlottingWrapper,
)
from helper_scripts.utils import init_visu_and_folders, init_control, close_run
# from wrapper import SmartEpisodeTrackerWithPlottingWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def init_graphics_and_controller(
    env: Any, num_steps: int, params_controller_dict: dict
) -> Tuple[Any, GpMpcController]:
    """
    Initialize the visualization object and the controller.

    Args:
        env: The environment instance.
        num_steps: Total number of steps for the environment.
        params_controller_dict: Dictionary containing controller parameters.

    Returns:
        A tuple containing the live plot object and the controller object.
    """
    live_plot_obj = init_visu_and_folders(
        env=env, num_steps=num_steps, params_controller_dict=params_controller_dict
    )

    ctrl_obj = GpMpcController(
        observation_space=env.observation_space,
        action_space=env.action_space,
        params_dict=params_controller_dict,
    )

    return live_plot_obj, ctrl_obj


def adjust_params_for_DoF(params_controller_dict: dict, DoF: int) -> None:
    """
    Adjust the parameters in params_controller_dict based on the degrees of freedom (DoF).

    Args:
        params_controller_dict: Dictionary containing controller parameters.
        DoF: Degrees of freedom from the environment.
    """
    # change_list = ["memory", "gp_init"]
    change_list = ["gp_init"]
    for var in change_list:
        for key in params_controller_dict[var]:
            params_controller_dict[var][key] = params_controller_dict[var][key][:DoF]
            logger.debug(f"{var} {key}: {params_controller_dict[var][key]}")

    controller_keys = [
        "target_state_norm",
        "weight_state",
        "weight_state_terminal",
        "target_action_norm",
        "weight_action",
        "obs_var_norm",
    ]
    for key in controller_keys:
        params_controller_dict["controller"][key] = params_controller_dict["controller"][key][:DoF]
        logger.debug(f"controller {key}: {params_controller_dict['controller'][key]}")


def main():
    """
    Main function to run the GP-MPC controller with the environment.
    """
    params_controller_dict = read_experiment_config("config/data_driven_mpc_config.yaml")

    num_steps = params_controller_dict.get("num_steps_env", 1000)
    num_repeat_actions = params_controller_dict["controller"].get("num_repeat_actions", 1)
    random_actions_init = params_controller_dict.get("random_actions_init", 0)


    env = load_env_config(env_config="config/environment_setting.yaml")
    DoF = env.DoF

    adjust_params_for_DoF(params_controller_dict, DoF)

    # env = SmartEpisodeTrackerWithPlottingWrapper(
    #     RewardScalingWrapper(env, scale=1.0)
    # )
    env = SmartEpisodeTrackerWithPlottingWrapper(
        env
    )

    live_plot_obj, ctrl_obj = init_graphics_and_controller(
        env, num_steps, params_controller_dict
    )

    (
        ctrl_obj,
        env,
        live_plot_obj,
        obs,
        action,
        cost,
        obs_prev_ctrl,
        obs_lst,
        actions_lst,
        rewards_lst,
    ) = init_control(
        ctrl_obj=ctrl_obj,
        env=env,
        live_plot_obj=live_plot_obj,
        random_actions_init=random_actions_init,
        num_repeat_actions=num_repeat_actions,
    )

    info_dict = None
    done = False
    for step in range(random_actions_init, num_steps):
        time_start = time.time()

        if step % num_repeat_actions == 0:
            if info_dict is not None:
                predicted_state = info_dict.get("predicted states", [None])[0]
                predicted_state_std = info_dict.get("predicted states std", [None])[0]
                # check_storage = True
            else:
                predicted_state = None
                predicted_state_std = None
            #     check_storage = False

            # Add memory before computing action
            ctrl_obj.add_memory(
                obs=obs_prev_ctrl,
                action=action,
                obs_new=obs,
                reward=-cost,
                # check_storage=check_storage,
                predicted_state=predicted_state,
                predicted_state_std=predicted_state_std,
            )

            if done:
                obs, _ = env.reset()

            # Compute the action
            action, info_dict = ctrl_obj.compute_action(obs_mu=obs)

            if params_controller_dict.get("verbose", False):
                for key, value in info_dict.items():
                    logger.info(f"{key}: {value}")

        # Perform action on the system
        obs_new, reward, done, _, _ = env.step(action)
        cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs, action)

        # Update visualization if enabled
        try:
            if live_plot_obj is not None:
                live_plot_obj.update(
                    obs=obs, cost=cost, action=action, info_dict=info_dict
                )
        except Exception as e:
            logger.error(f"Error updating live plot: {e}")

        # Update observations
        obs_prev_ctrl = obs
        obs = obs_new

        logger.debug(f"Time loop: {time.time() - time_start:.4f} s")

    # Close resources after the loop
    close_run(ctrl_obj=ctrl_obj, env=env)


if __name__ == "__main__":
    main()
