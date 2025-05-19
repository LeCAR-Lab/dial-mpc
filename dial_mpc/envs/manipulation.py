from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import jax
import jax.numpy as jnp
from functools import partial

from brax import math
import brax.base as base
from brax.base import System
from brax import envs as brax_envs
from brax.envs.base import PipelineEnv, State
from brax.io import html, mjcf, model

import mujoco
from mujoco import mjx

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.function_utils import global_to_body_velocity, get_foot_step
from dial_mpc.utils.io_utils import get_model_path


@dataclass
class AllegroReorientEnvConfig(BaseEnvConfig):
    kp: Union[float, jax.Array] = 1.0
    kd: Union[float, jax.Array] = 0.1


class AllegroReorientEnv(BaseEnv):
    def __init__(self, config: AllegroReorientEnvConfig):
        super().__init__(config)

        self._object_body_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "object"
        )
        self._init_q = jnp.array(self.sys.mj_model.keyframe("in_hand_reorient").qpos)

    def make_system(self, config: AllegroReorientEnvConfig) -> System:
        model_path = get_model_path("wonik_allegro", "scene_left.xml")
        mj_model = mujoco.MjModel.from_xml_path(model_path.as_posix())
        sys = mjcf.load_model(mj_model)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.5]),
            "pos_tar": jnp.array([0.0, 0.0, 0.13]),
            "step": 0,
        }

        obs = jnp.zeros(1)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        joint_targets = self.act2joint(action)
        if self._config.leg_control == "position":
            ctrl = joint_targets
        elif self._config.leg_control == "torque":
            raise NotImplementedError
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # reward
        ball_ang_vel = xd.ang[self._object_body_idx - 1] * jnp.pi / 180.0
        ball_pos = x.pos[self._object_body_idx - 1]
        reward_ang_vel = -jnp.sum(jnp.square(ball_ang_vel - state.info["ang_vel_tar"]))
        reward_pos = -jnp.sum(jnp.square(ball_pos - state.info["pos_tar"]))
        reward_joint_angle_deviation = -jnp.sum(jnp.square(pipeline_state.q[7:] - self._init_q[7:]))

        reward = (reward_ang_vel * 1.0
                  + reward_pos * 5.0
                  + reward_joint_angle_deviation * 0.1)
        # done
        done = jnp.zeros(1)
        done = jnp.where(state.info["step"] >= 100, 1, done)

        # update state
        state_info = {
            "rng": rng,
            "ang_vel_tar": state.info["ang_vel_tar"],
            "pos_tar": state.info["pos_tar"],
            "step": state.info["step"] + 1,
        }

        obs = jnp.zeros(1)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        act_normalized = (
            act * self._config.action_scale + 1.0
        ) / 2.0  # normalize to [0, 1]
        joint_targets = self.joint_range[:, 0] + self._init_q[7:] + act_normalized * (
            self.joint_range[:, 1] - self.joint_range[:, 0]
        )  # scale to joint range
        joint_targets = jnp.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )
        return joint_targets

brax_envs.register_environment("allegro_reorient", AllegroReorientEnv)