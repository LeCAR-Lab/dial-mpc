from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import jax
import jax.numpy as jnp
from functools import partial

from brax.base import System
from brax.envs.base import PipelineEnv

from dial_mpc.config.base_env_config import BaseEnvConfig


class BaseEnv(PipelineEnv):
    def __init__(self, config: BaseEnvConfig):
        assert config.dt % config.timestep == 0, "timestep must be divisible by dt"
        self._config = config
        n_frames = int(config.dt / config.timestep)
        sys = self.make_system(config)
        super().__init__(sys, config.backend, n_frames, config.debug)

        # joint limit definitions
        self.physical_joint_range = self.sys.jnt_range[1:]
        self.joint_range = self.physical_joint_range
        self.joint_torque_range = self.sys.actuator_ctrlrange

        # number of everything
        self._nv = self.sys.nv
        self._nq = self.sys.nq

    def make_system(self, config: BaseEnvConfig) -> System:
        """
        Make the system for the environment. Called in BaseEnv.__init__.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        act_normalized = (
            act * self._config.action_scale + 1.0
        ) / 2.0  # normalize to [0, 1]
        joint_targets = self.joint_range[:, 0] + act_normalized * (
            self.joint_range[:, 1] - self.joint_range[:, 0]
        )  # scale to joint range
        joint_targets = jnp.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )
        return joint_targets

    @partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act: jax.Array, pipline_state) -> jax.Array:
        joint_target = self.act2joint(act)

        q = pipline_state.qpos[7:]
        q = q[: len(joint_target)]
        qd = pipline_state.qvel[6:]
        qd = qd[: len(joint_target)]
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd

        tau = jnp.clip(
            tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1]
        )
        return tau
