from typing import Sequence, Optional
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Space, Discrete





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




