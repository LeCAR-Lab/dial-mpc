from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import jax
from brax.base import System
from brax.envs.base import PipelineEnv


@dataclass
class BaseEnvConfig:
    task_name: str = 'default'
    randomize_tasks: bool = False  # Whether to randomize the task.
    # P gain, or a list of P gains for each joint.
    kp: Union[float, jax.Array] = 30.0
    # D gain, or a list of D gains for each joint.
    kd: Union[float, jax.Array] = 1.0
    debug: bool = False
    # dt of the environment step, not the underlying simulator step.
    dt: float = 0.02
    # timestep of the underlying simulator step. user is responsible for making sure it matches their model.
    timestep: float = 0.02
    backend: str = 'mjx'  # backend of the environment.
    # control method for the joints, either "torque" or "position"
    leg_control: str = "torque"
    action_scale: float = 1.0  # scale of the action space.


class BaseEnv(PipelineEnv):
    def __init__(self, config: BaseEnvConfig):
        assert config.dt % config.timestep == 0, 'timestep must be divisible by dt'
        self._config = config
        n_frames = int(config.dt / config.timestep)
        sys = self.make_system(config)
        super().__init__(sys, config.backend, n_frames, config.debug)

    def make_system(self, config: BaseEnvConfig) -> System:
        """
        Make the system for the environment. Called in BaseEnv.__init__.
        """
        raise NotImplementedError
