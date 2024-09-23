
from typing import Optional, Union
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from copy import deepcopy
import random
import numpy as np
from collections import deque




class ReplayBuffer():
    def __init__(self, buffer_size: int):
        self.memory = deque(maxlen=buffer_size)  

    def __len__(self):
        return len(self.memory)

    @property
    def maxlen(self):
        return self.memory.maxlen

    def queue(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size: int):
        """
        Sample a batch of experiences randomly from the experience buffer
        """

        assert len(self.memory) >= batch_size, "Cannot sample from buffer, insufficient data!"

        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, terminated = zip(*transitions)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminated)




class DeepQAgent():
    def __init__(
            self,
            buffer: ReplayBuffer,
            qnet_local: nn.Module,
            criterion: nn.Module,
            optimizer: Optimizer,
            batch_size: int,
            gamma: float,
            tau: float = 1e-3,
            num_updates: int = 1,
            update_every: int = 1,
            burn_in: Optional[int]=None
        ) -> None:
        
        assert batch_size > 0 and batch_size < buffer.maxlen
        assert not burn_in or (burn_in > 0 and burn_in >= batch_size)
        assert gamma > 0 and gamma <= 1
        assert tau > 0 and tau < 1


        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.burn_in = burn_in if burn_in else batch_size

        self.buffer = buffer
        self.qnet_local = qnet_local
        self.qnet_target = deepcopy(qnet_local)


        self.criterion = criterion
        self.optimizer = optimizer(params=self.qnet_local.parameters())


        self.num_updates = num_updates
        self.update_every = update_every
        self.t_step = 0

        # take care that both models are on the same device
        self.to(self.device)



    @property
    def device(self) -> torch.device:
        return next(self.qnet_local.parameters()).device


    def to(self, device: Union[torch.device, str]):
        self.qnet_local.to(device)
        self.qnet_target.to(device)


    def step(
            self,   
            obs: np.array,
            actions: np.array,
            rewards: np.array,
            next_obs: np.array,
            terminated:np.array,
            ) -> None:
        """
        Peform agent transitions like
            - queue experiences
            - update local Q-network via gradient decent
            - softupdate target Q-network via parameter mixing 
        """
        

        # queue data in single env case
        if np.isscalar(rewards):
            transition = (obs, actions, rewards, next_obs, terminated)
            self.buffer.queue(transition=transition)

        # queue data in multienv case
        else:
            for transition in zip(obs, actions, rewards, next_obs, terminated):
                self.buffer.queue(transition=transition)


        # let's start to update our networks
        if len(self.buffer) >= self.burn_in: 
            self.t_step += 1
    
            if (self.t_step % self.update_every) == 0:
                for _ in range(self.num_updates):
                    self.update()

                self.t_step = 0

            # slowly mix local network parameter weights into target network
            self.update_target_network()
                

    def compute_targets(
            self,
            next_obs: torch.FloatTensor,
            rewards: torch.FloatTensor,
            terminated: torch.FloatTensor,
            ) -> torch.FloatTensor:
        """
        Compute targets for L2 minimization

            target = R + gamma*max_a[Q(obs_{k+1}, a)]

        """
        # TODO: compute targets
        target_q_values = ...
        return target_q_values
            



    def update(self) -> None:
        "Update the network parameters of the local network"


        # sample batch of experiences
        obs, actions, rewards, next_obs, terminated = \
            self.buffer.sample(batch_size=self.batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        terminated = torch.FloatTensor(terminated).to(self.device)
        
        # TODO: compute q_values
        q_values = ...

        target_q_values = self.compute_targets(next_obs, rewards, terminated)

        # TODO: compute loss
        loss = self.criterion(...)

        # TODO: update network weights via backpropagation


    def update_target_network(self):
        # TODO: copy local network parameters to target network from time to time or mix them slowly
        pass

