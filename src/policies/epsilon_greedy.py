
from typing import Callable, Union
from gymnasium.spaces import Space, Discrete
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class RandomPolicy():
    def __init__(
            self,
            action_space: Discrete,
            observation_space: Space,
            ) -> None:
        
        self.action_space = action_space
        self.observation_space = observation_space

    def __call__(self, obs: np.array) -> np.array:
        Q = np.random.rand(obs.shape[0], self.action_space.n)
        return np.argmax(Q, axis=1)

    def update(self) -> None:
        pass
    


class GreedyPolicy(RandomPolicy):

    def __init__(
            self,
            action_space: Discrete,
            observation_space: Space,
            Qnet: Callable[[], nn.Module],
            ) -> None:
        
        super().__init__(action_space, observation_space)
        self.Qnet = Qnet(
            observation_space=observation_space,
            action_space=action_space,
        )


    def __call__(self, obs: np.array) -> None:

        # expand dimensionality w.r.t batch size in the singel env case
        if obs.shape == self.observation_space.shape:
            obs = np.expand_dims(obs, axis=0)

        # if beeing trained the Qnet might run on an accelerator
        device = next(self.Qnet.parameters()).device
        
        self.Qnet.eval()
        with torch.no_grad():
            Q = self.Qnet(torch.FloatTensor(obs).to(device)).cpu().numpy()

        return np.argmax(Q, axis=1)
        

    def save_checkpoint(self, file_path: Union[str, Path]):
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.Qnet.state_dict(), file_path)

    def load_checkpoint(self, file_path):
        self.Qnet.load_state_dict(torch.load(file_path))



class EpsilonGreedy(GreedyPolicy):
    def __init__(
            self, 
            action_space: Discrete,
            observation_space: Space,
            Qnet: nn.Module,
            eps_max: float,
            eps_min: float,
            eps_decay: float,
            ) -> None:
        super().__init__(action_space, observation_space, Qnet)

        assert eps_decay > 0 and eps_decay < 1
        assert eps_max > 0 and eps_max <= 1
        assert eps_min >= 0 and eps_min < eps_max
        
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.epsilon = eps_max
        self.eps_decay = eps_decay


    def reset(self) -> None:
        self.epsilon = self.eps_max


    def update(self) -> None:
        self.epsilon = np.clip(
            self.epsilon*self.eps_decay, self.eps_min, self.eps_max
        )


    def __call__(self, obs: np.array, greedy=False) ->np.array:

        # expand dimensionality w.r.t batch size in the singel env case
        if obs.shape == self.observation_space.shape:
            obs = np.expand_dims(obs, axis=0)

        # TODO: sample greedy actions
        actions = ...

        # TODO: sample random actions
        random_actions = ...

        # TODO: pick random action over greedy action with probability self.epsilon
        return actions

