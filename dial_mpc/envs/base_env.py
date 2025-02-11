from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import jax
import jax.numpy as jnp
from functools import partial

from brax.base import System
from brax.io import mjcf

import mujoco
from mujoco import mjx
from mujoco.mjx import Data, Model

from dial_mpc.config.base_env_config import BaseEnvConfig


@jax.tree_util.register_dataclass
@dataclass
class DialState:
    data: Data
    info: Dict[str, Any]
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class BaseEnv:
    def __init__(self, config: BaseEnvConfig):
        assert config.dt % config.timestep == 0, "timestep must be divisible by dt"
        self._config = config
        self.decimation = int(config.dt / config.timestep)
        self.mj_model: Model = self._load_mj_model(config)
        self.mjx_model: Model = self._make_mjx_model(config)
        self.system: System = self._make_system(config)
        # joint limit definitions
        self.physical_joint_range = self.mjx_model.jnt_range[1:]
        self.joint_range = self.physical_joint_range
        self.joint_torque_range = self.mjx_model.actuator_ctrlrange

        # number of everything
        self.nv = self.mjx_model.nv
        self.nq = self.mjx_model.nq
        self.nu = self.mjx_model.nu

    def _load_mj_model(self, config: BaseEnvConfig) -> Model:
        """
        Load the mujoco model for the environment. Called in BaseEnv.__init__.
        """
        raise NotImplementedError

    def _make_mjx_model(self, config: BaseEnvConfig) -> Model:
        """
        Make the model for the environment. Called in BaseEnv.__init__.
        """
        mj_model = self._load_mj_model(config)
        mjx_model = mjx.put_model(mj_model)
        return mjx_model

    def _make_system(self, config: BaseEnvConfig) -> System:
        """
        Make the system for the environment. Called in BaseEnv.__init__.
        """
        mj_model = self._load_mj_model(config)
        mj_model.opt.timestep = config.dt
        return mjcf.load_model(mj_model)

    @partial(jax.jit, static_argnums=(0,))
    def physics_step(self, mjx_data: Data) -> Data:
        """
        Take a physics step using the mujoco mjx backend.
        """
        def f(data, _):
            return mjx.step(self.mjx_model, data), None
        return jax.lax.scan(f, mjx_data, None, self.decimation)[0]

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
    def act2tau(self, act: jax.Array, data: Data) -> jax.Array:
        joint_target = self.act2joint(act)

        q = data.qpos[7:]
        q = q[: len(joint_target)]
        qd = data.qvel[6:]
        qd = qd[: len(joint_target)]
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd

        tau = jnp.clip(
            tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1]
        )
        return tau

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DialState, action: jax.Array) -> DialState:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def multiple_step(self, state: DialState, actions: jax.Array) -> DialState:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def multiple_step_physics_step(self, mjx_data: Data, actions: jax.Array) -> Tuple[Data, Data]:
        """
        Take a physics step using the mujoco mjx backend.
        """
        def f(data, action):
            data = data.replace(ctrl=action)
            data_next = self.physics_step(data)
            return data_next, data_next
        return jax.lax.scan(f, mjx_data, actions)
