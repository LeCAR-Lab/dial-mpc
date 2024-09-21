from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np

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
import dial_mpc.envs as dial_envs


@dataclass
class UnitreeH1WalkEnvConfig(BaseEnvConfig):
    kp: Union[float, jax.Array] = jnp.array([
        200.0, 200.0, 200.0,  # left hips
        200.0, 60.0,  # left knee, ankle
        200.0, 200.0, 200.0,  # right hips
        200.0, 60.0,  # right knee, ankle
            200.0,  # torso
        60.0, 60.0, 60.0, 60.0,  # left shoulder, elbow
        60.0, 60.0, 60.0, 60.0,  # right shoulder, elbow
    ])
    kd: Union[float, jax.Array] = jnp.array([
        5.0, 5.0, 5.0,  # left hips
        5.0, 1.5,  # left knee, ankle
        5.0, 5.0, 5.0,  # right hips
        5.0, 1.5,  # right knee, ankle
            5.0,  # torso
        1.5, 1.5, 1.5, 1.5,  # left shoulder, elbow
        1.5, 1.5, 1.5, 1.5,  # right shoulder, elbow
    ])
    default_vx: float = 1.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 2.0
    gait: str = "jog"


class UnitreeH1WalkEnv(BaseEnv):
    def __init__(self, config: UnitreeH1WalkEnvConfig):
        super().__init__(config)

        # some body indices
        self._pelvis_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "pelvis"
        )
        self._torso_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "torso_link"
        )

        self._left_foot_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "left_foot"
        )
        self._right_foot_idx = mujoco.mj_name2id(
            self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "right_foot"
        )
        self._feet_site_id = jnp.array(
            [self._left_foot_idx, self._right_foot_idx], dtype=jnp.int32
        )
        # gait phase
        self._gait = config.gait
        self._gait_phase = {
            "stand": jnp.zeros(2),
            "walk": jnp.array([0.0, 0.5]),
            "jog": jnp.array([0.0, 0.5]),
        }
        self._gait_params = {
            # ratio, cadence, amplitude
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "walk": jnp.array([0.5, 1.2, 0.2]),
            "jog": jnp.array([0.3, 2, 0.2]),
        }

        # joint limits and initial pose
        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self._default_pose = self.sys.mj_model.keyframe("home").qpos[7:]
        # joint sampling range
        self._joint_range = jnp.array([
                [-0.3, 0.3],
                [-0.3, 0.3],
                [-1.0, 1.0],
                [0.0, 1.74],
                [-0.6, 0.4],

                [-0.3, 0.3],
                [-0.3, 0.3],
                [-1.0, 1.0],
                [0.0, 1.74],
                [-0.6, 0.4],

                [-0.5, 0.5],

                [-0.78, 0.78],
                [-0.3, 0.3],
                [-0.3, 0.3],
                [-0.3, 0.3],

                [-0.78, 0.78],
                [-0.3, 0.3],
                [-0.3, 0.3],
                [-0.3, 0.3],
        ])
        # self._joint_range = self._physical_joint_range

    def make_system(self, config: UnitreeH1WalkEnvConfig) -> System:
        model_path = get_model_path("unitree_h1", "mjx_scene_h1_walk.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "pos_tar": jnp.array([0.0, 0.0, 1.3]),
            "vel_tar": jnp.zeros(3),
            "ang_vel_tar": jnp.zeros(3),
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(2),
            "z_feet_tar": jnp.zeros(2),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": jnp.zeros(2, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(2),
        }

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        joint_targets = self.act2joint(action)
        if self._config.leg_control == "position":
            ctrl = joint_targets
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action, state.pipeline_state)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # switch to new target if randomize_target is True
        def dont_randomize():
            return (
                jnp.array([self._config.default_vx, self._config.default_vy, 0.0]),
                jnp.array([0.0, 0.0, self._config.default_vyaw]),
            )

        def randomize():
            return self.sample_command(cmd_rng)

        vel_tar, ang_vel_tar = jax.lax.cond(
            (state.info["randomize_target"]) & (state.info["step"] % 500 == 0),
            randomize,
            dont_randomize,
        )
        state.info["vel_tar"] = jnp.minimum(
            vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time, vel_tar
        )
        state.info["ang_vel_tar"] = jnp.minimum(
            ang_vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time,
            ang_vel_tar,
        )

        # reward
        # gaits reward
        # z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        amplitude, cadence, duty_ratio = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            amplitude, cadence, duty_ratio, phases, state.info["step"] * self.dt
        )
        # reward_gaits = -jnp.sum(((z_feet_tar - z_feet)) ** 2)
        z_feet = jnp.array(
            [
                jnp.min(pipeline_state.contact.dist[:2]),
                jnp.min(pipeline_state.contact.dist[2:]),
            ]
        )
        reward_gaits = -jnp.sum((z_feet_tar - z_feet) ** 2)
        # foot contact data based on z-position
        # pytype: disable=attribute-error
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2]
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt
        reward_air_time = jnp.sum((state.info["feet_air_time"] - 0.1) * first_contact)
        # position reward
        pos_tar = (
            state.info["pos_tar"] + state.info["vel_tar"] * self.dt * state.info["step"]
        )
        pos = x.pos[self._torso_idx - 1]
        reward_pos = -jnp.sum((pos - pos_tar) ** 2)
        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # yaw orientation reward
        yaw_tar = (
            state.info["yaw_tar"]
            + state.info["ang_vel_tar"][2] * self.dt * state.info["step"]
        )
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        d_yaw = yaw - yaw_tar
        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))
        # stay to norminal pose reward
        # reward_pose = -jnp.sum(jnp.square(joint_targets - self._default_pose))
        # velocity reward
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        reward_vel = -jnp.sum((vb[:2] - state.info["vel_tar"][:2]) ** 2)
        reward_ang_vel = -jnp.sum((ab[2] - state.info["ang_vel_tar"][2]) ** 2)
        # height reward
        reward_height = -jnp.sum(
            (x.pos[self._torso_idx - 1, 2] - state.info["pos_tar"][2]) ** 2
        )
        # energy reward
        # reward_energy = -jnp.sum((ctrl * pipeline_state.qvel[6:] / 160.0) ** 2)
        reward_energy = -jnp.sum((ctrl / self._joint_torque_range[:, 1]) ** 2)
        # stay alive reward
        reward_alive = 1.0 - state.done
        # reward
        reward = (
            reward_gaits * 5.0
            + reward_air_time * 0.0
            + reward_pos * 0.0
            + reward_upright * 0.5
            + reward_yaw * 0.1
            # + reward_pose * 0.0
            + reward_vel * 1.0
            + reward_ang_vel * 1.0
            + reward_height * 0.5
            + reward_energy * 0.01
            + reward_alive * 0.0
        )

        # done
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:]
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self._joint_range[:, 0])
        done |= jnp.any(joint_angles > self._joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18
        done = done.astype(jnp.float32)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        obs = jnp.concatenate(
            [
                state_info["vel_tar"],
                state_info["ang_vel_tar"],
                pipeline_state.ctrl,
                pipeline_state.qpos,
                vb,
                ab,
                pipeline_state.qvel[6:],
            ]
        )
        return obs

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)

    def sample_command(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        lin_vel_x = [-1.5, 1.5]  # min max [m/s]
        lin_vel_y = [-0.5, 0.5]  # min max [m/s]
        ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_lin_vel_cmd = jnp.array([lin_vel_x[0], lin_vel_y[0], 0.0])
        new_ang_vel_cmd = jnp.array([0.0, 0.0, ang_vel_yaw[0]])
        return new_lin_vel_cmd, new_ang_vel_cmd


brax_envs.register_environment("unitree_h1_walk", UnitreeH1WalkEnv)
