import hydra
from omegaconf import DictConfig
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import random
from pathlib import Path
from src.utils.utils import seed_erverything




@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> int:

    if "seed" in cfg:
        seed_erverything(seed=cfg.seed)

    # instantiate envirionments
    envs = AsyncVectorEnv([
        hydra.utils.instantiate(cfg.environment) for _ in range(cfg.num_envs)
    ])

    # instantiate policy
    policy = hydra.utils.instantiate(
        cfg.policy,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
    )
    # seed for reproducibility reasons
    policy.action_space.seed(random.randint(0, 2**16))

    # instantiate agent
    agent = hydra.utils.instantiate(
        cfg.agent,
        qnet_local=policy.Qnet,
    ).to(cfg.device)

  
    logger =  hydra.utils.instantiate(cfg.logger) if "logger" in cfg else None

    episode_returns = []

    for episode in range(cfg.max_episodes):

        obs, _ = envs.reset(seed=random.randint(0, 2**16))
        episode_returns.append(0)

        while True:
            
            actions = policy(obs)
            next_obs, rewards, terminated, truncated, _ = envs.step(actions=actions)
            agent.step(obs, actions, rewards, next_obs, terminated)

            episode_returns[-1] += np.mean(rewards).item()


            if (any(terminated) or any(truncated)):
                break

            obs = next_obs


        if logger: 
            logger.add_scalar("Return", episode_returns[-1], episode + 1)

        # update policy (e.g. decay epsilon in case of epsilon greedy policy)
        policy.update()

        avg_return = -np.inf if len(episode_returns) < cfg.avg_window else \
            np.mean(episode_returns[-cfg.avg_window:])

        if logger:
            logger.add_scalar("Avg Return", avg_return, episode + 1)

            if hasattr(policy, "epsilon"):
                logger.add_scalar("epsilon", policy.epsilon, episode + 1)
        

        
        
        print("Episode %d: Return: %g, Avg Return: %g"\
            %(episode, episode_returns[-1], avg_return))

        if len(episode_returns) >= cfg.avg_window and avg_return >= cfg.solved_at:
            print("Environment solved in %d episodes."%(episode + 1))
            break
            

    
    print("Saving checkpoint...")
    policy.save_checkpoint(Path(cfg.checkpoint_path, "checkpoint_%d.pt"%episode))
    

    return episode







if __name__ == "__main__":
    main()


