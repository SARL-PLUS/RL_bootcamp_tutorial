import math
import random
from enum import Enum
from typing import Optional

import gymnasium as gym
import numpy as np
from cpymad.madx import Madx
from gymnasium import spaces

from helper_scripts.Visualize_policy_validation import verify_external_policy_on_specific_env

class MamlHelpers:
    # init
    def __init__(self):
        self.twiss = self._generate_optics()
        self.response_scale = 0.5
        self.twiss_bpms = self.twiss[self.twiss["keyword"] == "monitor"]
        self.twiss_correctors = self.twiss[self.twiss["keyword"] == "kicker"]

    def _calculate_response(self, bpmsTwiss, correctorsTwiss, plane):
        bpms = bpmsTwiss.index.values.tolist()
        correctors = correctorsTwiss.index.values.tolist()
        bpms.pop(0)
        correctors.pop(-1)
        rmatrix = np.zeros((len(bpms), len(correctors)))
        for i, bpm in enumerate(bpms):
            for j, corrector in enumerate(correctors):
                if plane == Plane.horizontal:
                    bpm_beta = bpmsTwiss.betx[bpm]
                    corrector_beta = correctorsTwiss.betx[corrector]
                    bpm_mu = bpmsTwiss.mux[bpm]
                    corrector_mu = correctorsTwiss.mux[corrector]
                else:
                    bpm_beta = bpmsTwiss.bety[bpm]
                    corrector_beta = correctorsTwiss.bety[corrector]
                    bpm_mu = bpmsTwiss.muy[bpm]
                    corrector_mu = correctorsTwiss.muy[corrector]

                if bpm_mu > corrector_mu:
                    rmatrix[i][j] = (
                            math.sqrt(bpm_beta * corrector_beta)
                            * math.sin((bpm_mu - corrector_mu) * 2.0 * math.pi)
                            * self.response_scale
                    )
                else:
                    rmatrix[i][j] = 0.0
        return rmatrix

    def generate_optics(self, randomize=True):
        twiss = self._generate_optics(randomize)
        twiss_bpms = twiss[twiss["keyword"] == "monitor"]
        twiss_correctors = twiss[twiss["keyword"] == "kicker"]
        responseH = self._calculate_response(
            twiss_bpms, twiss_correctors, Plane.horizontal
        )
        responseV = self._calculate_response(
            twiss_bpms, twiss_correctors, Plane.vertical
        )
        return responseH, responseV

    def recalculate_response(self):
        responseH = self._calculate_response(self.twiss_bpms, self.twiss_correctors, Plane.horizontal)
        responseV = self._calculate_response(self.twiss_bpms, self.twiss_correctors, Plane.vertical)
        return responseH, responseV

    def _generate_optics(self, randomize=False):
        OPTIONS = ["WARN"]  # ['ECHO', 'WARN', 'INFO', 'DEBUG', 'TWISS_PRINT']
        MADX_OUT = [f"option, -{ele};" for ele in OPTIONS]
        madx = Madx(stdout=False)
        madx.input("\n".join(MADX_OUT))
        tt43_ini = "environment/electron_design.mad"
        madx.call(file=tt43_ini, chdir=True)
        madx.use(sequence="tt43", range="#s/plasma_merge")
        quads = {}
        variation_range = (0.75, 1.25)
        if randomize:
            for ele, value in dict(madx.globals).items():
                if "kq" in ele:
                    # quads[ele] = value * 0.8
                    quads[ele] = value * np.random.uniform(variation_range[0], variation_range[1], size=None)
                    # pass
        madx.globals.update(quads)
        madx.input(
            "initbeta0:beta0,BETX=5,ALFX=0,DX=0,DPX=0,BETY=5,ALFY=0,DY=0.0,DPY=0.0,x=0,px=0,y=0,py=0;"
        )
        twiss_cpymad = madx.twiss(beta0="initbeta0").dframe()

        return twiss_cpymad

    def sample_tasks(self, num_tasks):
        # Generate goals using list comprehension for more concise code
        goals = [self.generate_optics() for _ in range(num_tasks)]

        # Create tasks with goals and corresponding IDs using list comprehension
        tasks = [{"goal": goal, "id": idx} for idx, goal in enumerate(goals)]
        return tasks

    def get_origin_task(self, idx=0):
        # Generate goals using list comprehension for more concise code
        goal = self.generate_optics(randomize=False)
        # Create tasks with goals and corresponding IDs using list comprehension
        task = {"goal": goal, "id": idx}
        return task

class AwakeSteering(gym.Env):

    def __init__(self, twiss=[], task={}, train=False, **kwargs):
        self.rmatrix = None
        self.rmatrix_inverse = None
        self.__version__ = "1.0"
        self.MAX_TIME = kwargs.get("MAX_TIME", 100)
        (self.boundary_conditions) = kwargs.get("boundary_conditions", False)
        self.state_scale = 1
        self.action_scale = 1
        self.threshold = -0.1  # Corresponds to 1 mm scaled.

        self.current_episode = -1
        self.current_steps = 0

        self.seed(kwargs.get('seed'))
        self.maml_helper = MamlHelpers()
        self.plane = kwargs.get("plane", Plane.horizontal)

        if not task:
            task = self.maml_helper.get_origin_task()
        self.reset_task(task)

        self.setup_dimensions()

        self.verification_tasks_loc = kwargs.get("verification_tasks_loc")

    def setup_dimensions(self):
        num_bpms = len(self.maml_helper.twiss_bpms) - 1
        num_correctors = len(self.maml_helper.twiss_correctors) - 1

        self.positions = np.zeros(num_bpms)
        self.settings = np.zeros(num_correctors)
        self.high = np.ones(num_correctors)
        self.low = -self.high

        self.action_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)

        if self.plane == Plane.horizontal:
            self.rmatrix = self.responseH
        else:
            self.rmatrix = self.responseV

        self.rmatrix_inverse = np.linalg.inv(self.rmatrix)

    def step(self, action):
        delta_kicks = np.clip(action, -1., 1) * self.action_scale
        self.state += self.rmatrix.dot(delta_kicks)
        self.state = np.clip(self.state, -1, 1)
        return_state = self.state

        reward = -np.sqrt(np.mean(np.square(return_state)))

        self.current_steps += 1
        done = (reward > self.threshold)
        truncated = (self.current_steps >= self.MAX_TIME)
        if truncated:
            done = True

        violations = np.abs(return_state) >= 1
        if violations.any():
            return_state[violations] = np.sign(return_state[violations])
            reward = -np.sqrt(np.mean(np.square(return_state)))
            done = self.boundary_conditions

        return return_state, reward, done, truncated, {"task": self._id, 'time': self.current_steps}

    def _get_reward(self, observation):
        """
        Compute the reward using a sigmoid transformation on each observation,
        then calculate the negative of the Euclidean norm (L2 norm) as a penalty.
        """

        return -np.sqrt(np.mean(np.square(observation)))

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed(seed)
            self.observation_space.seed(seed)
        self.is_finalized = False
        self.current_steps = 0
        self.current_episode += 1
        self.state = self.observation_space.sample()
        self.state = np.clip(self.state, -1, 1)
        return_state = self.state
        return return_state, {}

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    # MAML specific function, while training samples fresh new tasks and for testing it uses previously saved tasks
    def sample_tasks(self, num_tasks):
        tasks = self.maml_helper.sample_tasks(num_tasks)
        return tasks

    def get_origin_task(self, idx):
        task = self.maml_helper.get_origin_task(idx=idx)
        return task

    def reset_task(self, task):
        self._task = task
        self._goal = task["goal"]
        self._id = task["id"]

        self.responseH = self._goal[0]
        self.responseV = self._goal[1]

    def get_task(self):
        return self._task


class Plane(Enum):
    horizontal = 0
    vertical = 1




#
# if __name__ == "__main__":
#     # find_thresholds_for_DoFs(episodes=10000)
#     DoF = 2
#     env = DoFWrapper(AwakeSteering(), DoF)
#     policy = lambda s: env.action_space.sample()
#     verify_external_policy_on_specific_env(env, [policy], episodes=15, policy_labels=['rand'], DoF=DoF)
