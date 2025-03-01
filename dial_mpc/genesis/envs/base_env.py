from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import torch

import genesis as gs

from dial_mpc.config.base_env_config import BaseEnvConfig


class BaseEnv:
    def __init__(self, config: BaseEnvConfig, num_envs: int, render: bool = False):
        assert torch.allclose(config.dt % config.timestep, 0.0), "timestep must be divisible by dt"
        self._config = config
        self.scene = self._make_scene(config, num_envs, render)

        # joint limit definitions
        self.physical_joint_range, self.joint_range, self.joint_torque_range = self._get_joint_ranges()

    def _make_scene(self, config: BaseEnvConfig, num_envs: int, render: bool) -> gs.Scene:
        """
        Make the scene for the environment. Called in BaseEnv.__init__.
        """
        raise NotImplementedError
    
    def _get_joint_ranges(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the joint ranges for the robot.
        physical_joint_range: (num_dofs, 2)
        joint_range: (num_dofs, 2)
        joint_torque_range: (num_dofs, 2)
        """
        raise NotImplementedError

    def act2joint(self, act: torch.Tensor) -> torch.Tensor:
        act_normalized = (
            act * self._config.action_scale + 1.0
        ) / 2.0  # normalize to [0, 1]
        joint_targets = self.joint_range[:, 0] + act_normalized * (
            self.joint_range[:, 1] - self.joint_range[:, 0]
        )  # scale to joint range
        joint_targets = torch.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )
        return joint_targets
