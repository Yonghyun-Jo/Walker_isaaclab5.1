# Copyright (c) 2025, Yonghyun Jo.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator for base pose (height + roll + pitch + yaw) control."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformBasePoseCommand(CommandTerm):
    """Command generator that samples target base pose [height, roll, pitch, yaw].

    The command is a 4D vector per environment:
      [0] height (m)   — target base z-position in world frame
      [1] roll (rad)    — target roll angle (gravity-aligned frame)
      [2] pitch (rad)   — target pitch angle (gravity-aligned frame)
      [3] yaw (rad)     — target yaw offset (relative to current heading)

    Shape: (num_envs, 4).

    Each axis can be independently frozen to its default value via
    rel_default_*_envs probability. This enables phased activation:
    Phase 1: only height active (roll/pitch/yaw all default)
    Phase 2: height + pitch active
    Phase 3: all active
    """

    cfg: UniformBasePoseCommandCfg

    def __init__(self, cfg: UniformBasePoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # command buffer: (num_envs, 4) = [height, roll, pitch, yaw]
        self.pose_command = torch.zeros(self.num_envs, 4, device=self.device)
        # Initialize to default pose
        self.pose_command[:, 0] = cfg.default_height
        self.pose_command[:, 1] = cfg.default_roll
        self.pose_command[:, 2] = cfg.default_pitch
        self.pose_command[:, 3] = cfg.default_yaw

        # Per-axis default masks
        self.is_default_height = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_default_roll = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_default_pitch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_default_yaw = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # metrics
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformBasePoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeight range: {self.cfg.ranges.height}\n"
        msg += f"\tRoll range: {self.cfg.ranges.roll}\n"
        msg += f"\tPitch range: {self.cfg.ranges.pitch}\n"
        msg += f"\tYaw range: {self.cfg.ranges.yaw}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base pose command. Shape is (num_envs, 4)."""
        return self.pose_command

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # Height error
        current_height = self.robot.data.root_pos_w[:, 2]
        self.metrics["error_height"] += (
            torch.abs(self.pose_command[:, 0] - current_height) / max_command_step
        )
        # Pitch error (using projected gravity x ≈ sin(pitch))
        pitch_cmd = self.pose_command[:, 2]
        pitch_actual = self.robot.data.projected_gravity_b[:, 0]
        self.metrics["error_pitch"] += (
            torch.abs(pitch_cmd - pitch_actual) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        # Sample each axis
        self.pose_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
        self.pose_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.roll)
        self.pose_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.pitch)
        self.pose_command[env_ids, 3] = r.uniform_(*self.cfg.ranges.yaw)
        # Determine default masks per axis
        self.is_default_height[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_default_height_envs
        self.is_default_roll[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_default_roll_envs
        self.is_default_pitch[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_default_pitch_envs
        self.is_default_yaw[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_default_yaw_envs

    def _update_command(self):
        # Enforce defaults for designated envs (per axis)
        h_ids = self.is_default_height.nonzero(as_tuple=False).flatten()
        r_ids = self.is_default_roll.nonzero(as_tuple=False).flatten()
        p_ids = self.is_default_pitch.nonzero(as_tuple=False).flatten()
        y_ids = self.is_default_yaw.nonzero(as_tuple=False).flatten()
        self.pose_command[h_ids, 0] = self.cfg.default_height
        self.pose_command[r_ids, 1] = self.cfg.default_roll
        self.pose_command[p_ids, 2] = self.cfg.default_pitch
        self.pose_command[y_ids, 3] = self.cfg.default_yaw


@configclass
class UniformBasePoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform base pose command generator."""

    class_type: type = UniformBasePoseCommand

    asset_name: str = MISSING

    # Default values (standing upright)
    default_height: float = 0.89
    default_roll: float = 0.0
    default_pitch: float = 0.0
    default_yaw: float = 0.0

    # Per-axis probability of using default value (1.0 = always default = frozen)
    rel_default_height_envs: float = 0.5
    rel_default_roll_envs: float = 1.0    # Phase 1-2: frozen
    rel_default_pitch_envs: float = 1.0   # Phase 1: frozen, Phase 2: 0.5
    rel_default_yaw_envs: float = 1.0     # Phase 1-2: frozen

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING
        roll: tuple[float, float] = (-0.2, 0.2)
        pitch: tuple[float, float] = (-0.3, 0.1)
        yaw: tuple[float, float] = (-0.3, 0.3)

    ranges: Ranges = MISSING
