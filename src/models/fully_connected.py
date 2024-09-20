from typing import Sequence, Optional
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Space, Discrete
from torch.distributions import Normal




class QNet(nn.Sequential):
    def __init__(
            self,
            observation_space: Space,
            action_space: Discrete,
            hidden_layers: Sequence[int],
            activation: nn.Module = nn.ReLU,
            seed: Optional[int] = None,
        ):

        if seed:
            torch.manual_seed(seed=seed)

        in_features = [int(np.prod(observation_space.shape)), *hidden_layers]
        out_features = [*hidden_layers, action_space.n]
        
        block = []

        for layer, (in_feat, out_feat) in enumerate(zip(in_features, out_features)):
            block.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            if layer < len(in_features)-1:
                block.append(activation())


        super().__init__(*block)




class NoisyQNet(QNet):
    def __init__(self, log_std_init: float = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_outputs = self[-1].out_features
        self.log_std = nn.Parameter(
            data=log_std_init*torch.ones(num_outputs),
            requires_grad=True,
        )

    def forward(self, x):
        x = super().forward(x)
        epsilon = torch.randn_like(x)
        return x + epsilon*self.log_std.exp().unsqueeze(dim=0)

     
