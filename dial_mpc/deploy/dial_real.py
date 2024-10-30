import os
import time
import csv
import sys
import importlib
from multiprocessing import shared_memory
from threading import Thread
from typing import List, Union
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import art
import yaml

from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelFactoryInitialize,
    ChannelPublisher,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowState_,
    unitree_go_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread

from dial_mpc.config.base_env_config import BaseEnvConfig
from dial_mpc.core.dial_config import DialConfig
import dial_mpc.utils.unitree_legged_const as unitree
from dial_mpc.utils.io_utils import (
    load_dataclass_from_dict,
    get_model_path,
    get_example_path,
)
from dial_mpc.examples import deploy_examples
from dial_mpc.deploy.localization import load_plugin, get_available_plugins


@dataclass
class DialRealConfig:
    robot_name: str
    scene_name: str
    real_leg_control: str
    record: bool
    network_interface: str
    real_kp: Union[float, List[float]]
    real_kd: Union[float, List[float]]
    initial_position_ctrl: List[float]
    low_cmd_pub_dt: float
    localization_plugin: str
    localization_timeout_sec: float


class DialReal:
    def __init__(
        self,
        real_config: DialRealConfig,
        env_config: BaseEnvConfig,
        dial_config: DialConfig,
        plugin_config: dict,
    ):
        self.leg_control = real_config.real_leg_control
        if self.leg_control != "position" and self.leg_control != "torque":
            raise ValueError("Invalid leg control mode")
        self.record = real_config.record
        self.data = []
        # control related
        self.kp = real_config.real_kp
        self.kd = real_config.real_kd
        self.current_kp = 0.0
        self.mocap_odom = None
        self.ctrl_dt = env_config.dt
        self.n_acts = dial_config.Hsample + 1
        self.t = 0.0
        self.stand_ctrl = np.array(real_config.initial_position_ctrl, dtype=np.float32)
        self.low_cmd_pub_dt = real_config.low_cmd_pub_dt

        # load localization plugin
        self.localization_plugin = load_plugin(real_config.localization_plugin)
        if self.localization_plugin is None:
            raise ValueError(
                f'Failed to load localization plugin "{real_config.localization_plugin}". Please see error messages above. Valid plugins are: {get_available_plugins()}'
            )
        self.localization_plugin = self.localization_plugin(plugin_config)
        self.localization_timeout_sec = real_config.localization_timeout_sec

        # mujoco setup
        self.mj_model = mujoco.MjModel.from_xml_path(
            get_model_path(real_config.robot_name, real_config.scene_name).as_posix()
        )
        self.mj_model.opt.timestep = real_config.low_cmd_pub_dt
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.viewer = mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=True
        )

        # parameters
        self.Nx = self.mj_model.nq + self.mj_model.nv
        self.Nu = self.mj_model.nu

        # get home keyframe
        self.default_q = self.mj_model.keyframe("home").qpos
        self.default_u = self.mj_model.keyframe("home").ctrl

        # communication setup
        # publisher
        self.time_shm = shared_memory.SharedMemory(
            name="time_shm", create=True, size=32
        )
        self.time_shared = np.ndarray(1, dtype=np.float32, buffer=self.time_shm.buf)
        self.time_shared[0] = 0.0
        self.state_shm = shared_memory.SharedMemory(
            name="state_shm", create=True, size=self.Nx * 32
        )
        self.state_shared = np.ndarray(
            (self.Nx,), dtype=np.float32, buffer=self.state_shm.buf
        )
        # listener
        self.acts_shm = shared_memory.SharedMemory(
            name="acts_shm", create=True, size=self.n_acts * self.Nu * 32
        )
        self.acts_shared = np.ndarray(
            (self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.acts_shm.buf
        )
        self.acts_shared[:] = self.default_u
        self.refs_shm = shared_memory.SharedMemory(
            name="refs_shm", create=True, size=self.n_acts * self.Nu * 3 * 32
        )
        self.refs_shared = np.ndarray(
            (self.n_acts, self.Nu, 3), dtype=np.float32, buffer=self.refs_shm.buf
        )
        self.refs_shared[:] = 1.0
        self.plan_time_shm = shared_memory.SharedMemory(
            name="plan_time_shm", create=True, size=32
        )
        self.plan_time_shared = np.ndarray(
            1, dtype=np.float32, buffer=self.plan_time_shm.buf
        )
        self.plan_time_shared[0] = -self.ctrl_dt

        self.tau_shm = shared_memory.SharedMemory(
            name="tau_shm", create=True, size=self.n_acts * self.Nu * 32
        )
        self.tau_shared = np.ndarray(
            (self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.tau_shm.buf
        )

        # unitree pubs and subs
        self.crc = CRC()
        ChannelFactoryInitialize(0, real_config.network_interface)
        self.low_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_pub.Init()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q = unitree.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = unitree.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
        self.low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_sub.Init(self.on_low_state, 1)

        # visualization thread
        self.vis_thread = Thread(target=self.visualize)
        self.vis_thread.Start()

    def visualize(self):
        while True:
            mujoco.mj_step(self.mj_model, self.mj_data)
            self.viewer.sync()
            time.sleep(0.05)

    def on_low_state(self, msg: LowState_):
        localization_output = self.localization_plugin.get_state()
        if localization_output is None:
            return
        now = time.time()
        localization_time = self.localization_plugin.get_last_update_time()
        if now - localization_time > self.localization_timeout_sec:
            print(f"[WARN] Localization plugin timeout: {now - localization_time} s")
            return

        q = np.zeros(self.mj_model.nq)
        dq = np.zeros(self.mj_model.nv)

        # copy body pose and velocity from localization plugin
        q[:7] = localization_output[:7]
        dq[0:3] = localization_output[7:10]

        # rotate angular velocity into the world frame
        rot = R.from_quat([q[4], q[5], q[6], q[3]]).as_matrix()
        # ang_vel_body = np.array([self.mocap_odom.twist.twist.angular.x, self.mocap_odom.twist.twist.angular.y, self.mocap_odom.twist.twist.angular.z])
        ang_vel_body = np.array([msg.imu_state.gyroscope]).flatten()
        ang_vel_world = rot @ ang_vel_body
        dq[3:6] = ang_vel_world

        # update joint positions and velocities
        for i in range(12):
            q[7 + i] = msg.motor_state[i].q
            dq[6 + i] = msg.motor_state[i].dq

        state = np.concatenate([q, dq])
        self.state_shared[:] = state
        self.mj_data.qpos = q
        self.mj_data.qvel = dq

    def main_loop(self):
        while True:
            t0 = time.time()
            if self.plan_time_shared[0] < 0.0:
                self.mj_data.ctrl = self.stand_ctrl
            else:
                delta_time = self.t - self.plan_time_shared[0]
                delta_step = int(delta_time / self.ctrl_dt)
                if delta_step >= self.n_acts or delta_step < 0:
                    delta_step = self.n_acts - 1
                self.mj_data.ctrl = self.acts_shared[delta_step]
                taus = self.tau_shared[delta_step].copy()

                # mujoco.mj_step(self.mj_model, self.mj_data)
                self.t += self.low_cmd_pub_dt
                self.time_shared[:] = self.t

                # self.data.append(np.concatenate([[time.time()], self.mj_data.qpos, self.mj_data.qvel, self.mj_data.ctrl]))

            # publish control
            for i in range(12):
                if self.plan_time_shared[0] < 0.0 or self.leg_control == "position":
                    self.low_cmd.motor_cmd[i].q = self.mj_data.ctrl[i]
                    self.low_cmd.motor_cmd[i].kp = (
                        min(self.current_kp, self.kp)
                        if type(self.kp) is float
                        else min(self.current_kp, self.kp[i])
                    )
                    self.low_cmd.motor_cmd[i].dq = 0
                    self.low_cmd.motor_cmd[i].kd = (
                        self.kd if type(self.kd) is float else self.kd[i]
                    )
                    self.low_cmd.motor_cmd[i].tau = 0
                    self.current_kp += 0.005  # ramp up kp to start the robot smoothly
                else:
                    self.low_cmd.motor_cmd[i].q = 0.0
                    self.low_cmd.motor_cmd[i].kp = 0.0
                    self.low_cmd.motor_cmd[i].dq = 0.0
                    self.low_cmd.motor_cmd[i].kd = (
                        self.kd if type(self.kd) is float else self.kd[i]
                    )
                    self.low_cmd.motor_cmd[i].tau = taus[i] * 1.0
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.low_pub.Write(self.low_cmd)

            if self.plan_time_shared[0] >= 0.0 and self.record:
                self.data.append(
                    np.concatenate(
                        [
                            [time.time()],
                            self.mj_data.qpos,
                            self.mj_data.qvel,
                            self.mj_data.ctrl,
                        ]
                    )
                )

            t1 = time.time()
            duration = t1 - t0
            if duration < self.low_cmd_pub_dt:
                time.sleep(self.low_cmd_pub_dt - duration)
            else:
                print(f"[WARN] Real loop overruns: {duration * 1000} ms")

    def close(self):
        self.time_shm.close()
        self.time_shm.unlink()
        self.state_shm.close()
        self.state_shm.unlink()
        self.acts_shm.close()
        self.acts_shm.unlink()
        self.plan_time_shm.close()
        self.plan_time_shm.unlink()


def main(args=None):
    art.tprint("LeCAR @ CMU\nDIAL-MPC\nREAL", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    group.add_argument(
        "--example",
        type=str,
        default=None,
        help="Example to run",
    )
    group.add_argument(
        "--list-examples",
        action="store_true",
        help="List available examples",
    )
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    parser.add_argument(
        "--network-interface",
        type=str,
        default=None,
        help="Network interface override",
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default=None,
        help="Custom localization plugin to import dynamically",
    )
    args = parser.parse_args()

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)
    if args.plugin is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.plugin)

    if args.list_examples:
        print("Available examples:")
        for example in deploy_examples:
            print(f"  - {example}")
        return
    if args.example is not None:
        if args.example not in deploy_examples:
            print(f"Example {args.example} not found.")
            return
        config_dict = yaml.safe_load(
            open(get_example_path(args.example + ".yaml"), "r")
        )
    else:
        config_dict = yaml.safe_load(open(args.config, "r"))

    if args.network_interface is not None:
        config_dict["network_interface"] = args.network_interface

    real_config = load_dataclass_from_dict(DialRealConfig, config_dict)
    env_config = load_dataclass_from_dict(BaseEnvConfig, config_dict)
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    real_env = DialReal(real_config, env_config, dial_config, config_dict)

    try:
        real_env.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        if real_env.record:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            data = np.array(real_env.data)
            output_dir = os.path.join(
                dial_config.output_dir,
                f"sim_{dial_config.env_name}_{env_config.task_name}_{timestamp}",
            )
            os.makedirs(output_dir)
            np.save(os.path.join(output_dir, "states"), data)

        real_env.close()


if __name__ == "__main__":
    main()
