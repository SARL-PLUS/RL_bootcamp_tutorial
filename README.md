# Tutorial in Reinforcement Learning of the [RL-Bootcamp Salzburg 24](https://sarl-plus.github.io/RL-Bootcamp/)


## Tutorial Day 1: Getting hands on Deep Q-Learning
Before actually tackling the beam steering environment with Proximal Policy Optimisation (Tutorial Day 2), we'll first tackle a much simpler environment with Deep Q-Learning, the algorithm that essentially kickstarted deep reinforcement learning in 2015.


## Setup
We provide a requirements.txt file for pip install, but as it is really challenging to resolve all package conflicts for Linux, Windows and OSx users on the one hand, and also take care of cuda tookit versions, we suggest to create a conda or python virtual environment, install python version 3.11, activate the environment and run

```
(your_env_id)$ bash install_dependencies.sh cpu|cu118|cu121
```
where the user needs to specify either `cpu` for cpu only installation or cu118 or cu121 for cuda tookkit 11.8 or 12.1 respectively.

## Goal
The goal is to get a basic understanding of how the algorithm works. As Deep Q-Learning is one of the easier algorithms to implement, we have a fair chance of getting it done within the short timeframe of the tutorial. Therefore, the goal is to write your own implementation without relying on third-party deep reinforcement learning libraries. 


## Framework
Writing your own deep reinforcement learning algorithms can be tedious because, frankly, deep reinforcement learning code is a nightmare to debug. Hence, we provide you already a code skeleton where most of the supporting functionality is implemented. 

The provided framework uses Hydra for configuration management. At first glance this looks like overkill for such a simple tutorial on DQN, but in fact it simplifies code development and, more importantly, provides complete reproducibility for free. The latter is quite handy as there are a lot of hyperparameters to take care of. 

### File Structure
The file structure of the framework is shown below and can be roughly divided into 3 parts, the training script `train.py`, the `configs` folder and the `src` folder. Not surprisingly, all the configuration goes into `configs` and all the code should go into `src` in the appropriate sub-folders. The idea of the configuration is to have it hirarchical, which keeps the parameters in the right place. There is a main configuration file `configs/train.yaml` which determines the parameters directly related to the training code and loads the individual configurations of the instances required to run the code, as seen in the defaults list at the top of `configs/train.yaml`´. 


```
RL_bootcamp_tutorial
│
├── configs
│   ├── train.yaml               # Global training configuration goes here
│   ├── agent
│   │   └── dqn.yaml             # Agent specific configuration goes here
│   ├── enviroment
│   │   └── default.yaml         # Environment specific configuration goes here
│   ├── hparams_search
│   │   └── optuna.yaml          # Config for hyperparameter search (e.g. advanced samplers)
│   ├── logger
│   │   └── tensorboard.yaml     # Logger config fpr progress visualization
│   └── policy
│      └── epsilon_greedy.yaml   # Policy specific configuration goes here
│
├── src
│   ├── agents
│   │   └── dqn.py               # Code for the Deep Q algorithm goes here
│   ├── models
│   │   └── fully_connected.py   # Code specifing the architecture of the torch model
│   ├── policies
│   │   └── epsilon_greedy.py    # Code for the policy we aim to sample from
│   └── utils
│       └── utils.py             # Helper code goes in here
│
├── README.md
├── train.py                     # Run this to train the agent with config train.yaml
└── inference.ipynb              # Once you trained your agent you can visualize it's performance here

```

### Output
You may wonder why these effort with the configuration is beneficial. First, having the configuration beeing hirarchichal significaltly improves readability. Second, and most importantly, our configuration is automatically tracked and linked to the output it produces. 
Hydra automatically creates a output directory in for each individual run for us. Depending if we run `train.py` in single or multirun mode hydra generates an output directory either be named `logs/runs/${task_name}/%Y-%m-%d_%H-%M-%S` or `logs/multiruns/${task_name}/%Y-%m-%d_%H-%M-%S/${job.num}`, where `${task_name}` is an identifer we can use to name our experiemnts and `${job.num}` is the corresponding job identifier.

Inside this output directory you will find a `.hydra` folder, a `tensorboard` folder if you configured the tensorboard logger correctly (which is default here) and a `checkpoints` folder where  the model's state dictionary is stored once the training has finished.
In `.hydra/config.yaml` the configuration which was actually used for training is stored. Moreover, hydra keeps track of changes with respect to the main configuration in `.hydra/overrides.yaml`.


### Usage
To run the training code with the default config just run `python train.py`. We can easily override any 
defined key in the configuration. For example, let's assume we want to use a different configuration file for the agent in `configs/agent/agent57` plus we want it change the activation function for the torch model to ELU. For better reproducability we als spend a `task_name` accordingly. In command line this looks like:

```
$ python train.py task_name=the_other_task agent=agent57 policy.Qnet.activation._target_=torch.nn.ELU
```

If we want to perform a grid search over some parameters we can use hydra's multirun support seamlessly.

```
$ python train.py -m task_name=grid_search_batchsize agent.batch_size=64,128,256
```

By default, hydra starts the jobs sequentially. However, if we provide a more sophisticated configuration for the hyperparameter search, we can speed things up considerably. In `configs\hparam_search\optuna.yaml` we have provided a basic configuration to use `optuna` to sample hyperparameters, giving us a tool to optimise for hyperparameters. In addition, by modifying the launcher configuration, we can use more sophisticated plugins to start jobs, giving us a significant speedup through parallelization. 

```
$ python train.py -m task_name=hparams_search hparams_search=optuna
```

## Logging
By default we incorporated logger support via tensorboard. Hence, you can wath your agent learn in realtime. Just load the python environment and run

```
$ tensorboard --logdir logs
```


## Enviornment
Due it's simplicity we use the `LunderLanderV2` environment for the tutorial. The environment is considered to be solved if the average reward of the last 100 episodes is at least 200.

### Action Space
Type: `Discrete(4)`
  - 0: all engines off
  - 1: fire left orientation engine
  - 2: fire main engine
  - 3: fire right orientation engine

### Observation Space
Type: Box2D(8)
  - dim 1, 2: x- and y-coordinate
  - dim 3, 4: linear velocities in x and y direction
  - dim 5, 6: landers orientation and angular velocity
  - dim 7, 8: Indicator for contact left and right leg
  
### Reward
The reward is composition of 
  - negative absolute position error w.r.t to the landing pad
  - negative absolute velocity error w.r.t (we do not want to crash land)
  - negative absolute orientation error (we want to land on the legs)
  - negative absolute angular velocity error (same motivation as for the linear velocity)
  - contact with legs is reward by +10 for each leg
  - side engine usage is bit costly: -0.03
  - main engine usage is definite costly: -0.3
  - crash landing is punished: -100

  
  
## Tasks
In order to make to code work you need to complete the code for two classes.

First you need to complete the code for the `EpsilonGreedy` policy in `src.policies.epsilon_greedy.py`.
```
def __call__(self, obs: np.array, greedy=False) ->np.array:
  # TODO: sample greedy actions
  actions = ...

  # TODO: sample random actions
  random_actions = ...

  # TODO: pick random action over greedy action with probability self.epsilon
  return actions
```

Second you need to implement the agent's `update`, `compute_targets` and `update_target_network` methods to make the algorithm work. 

```

def compute_targets(
  self,
  next_obs: torch.FloatTensor,
  rewards: torch.FloatTensor,
  terminated: torch.FloatTensor,
  ) -> torch.FloatTensor:

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

```


## Results
If your agents manage to solve the LunarLanderV2 environment, where we consider an environment to be solved if the average return of the last 100 episodes is at least 200, the agent should perform similarly to the one below.  

<video width="640" height="480" controls>
  <source src="./LunarLander-v2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

You can run our own inference by modifiying the `run` path in `inference.ipynb`.



## Bonus
In case you feel underchallenged with this simple task we suggest some improvements on the agent like:

- Double Deep Q-Learning 
- A noisy network layer for exploration
- A reward shaping filter
- ...
