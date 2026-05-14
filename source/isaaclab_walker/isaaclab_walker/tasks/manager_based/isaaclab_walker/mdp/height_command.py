# Copyright (c) 2025, Yonghyun Jo.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator for base height (pos_z) control."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformHeightCommand(CommandTerm):
    """Command generator that samples a target base height from a uniform distribution.

    The command is a scalar target height (z-position) for the robot base,
    expressed in world frame. Shape: (num_envs, 1).
    """

    cfg: UniformHeightCommandCfg

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # command buffer: (num_envs, 1)
        self.height_command_w = torch.zeros(self.num_envs, 1, device=self.device)
        self.is_default_height_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # metrics
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeight range: {self.cfg.ranges.height}\n"
        msg += f"\tDefault height probability: {self.cfg.rel_default_height_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base height command. Shape is (num_envs, 1)."""
        return self.height_command_w

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        current_height = self.robot.data.root_pos_w[:, 2]
        self.metrics["error_height"] += (
            torch.abs(self.height_command_w[:, 0] - current_height) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        # sample height
        self.height_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
        # determine default-height environments
        self.is_default_height_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_default_height_envs

    def _update_command(self):
        # enforce default height for designated envs
        default_env_ids = self.is_default_height_env.nonzero(as_tuple=False).flatten()
        self.height_command_w[default_env_ids, 0] = self.cfg.default_height


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for the uniform height command generator."""

    class_type: type = UniformHeightCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    default_height: float = 0.89
    """Default standing height [m]. Used for default-height environments."""

    rel_default_height_envs: float = 0.5
    """Probability of environments that use the default standing height. Defaults to 0.5."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the height command."""

        height: tuple[float, float] = MISSING
        """Range for the target base height command [m]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the height command."""
