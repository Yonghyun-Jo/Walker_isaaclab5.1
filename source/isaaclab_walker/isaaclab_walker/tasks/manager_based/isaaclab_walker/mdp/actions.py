from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
import omni.log
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from .torque_lut import AnkleTorqueLUT, KneeTorqueLUT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class LowerBodyActions(ActionTerm):
    cfg: LowerBodyActionsCfg
    _asset: Articulation
    _scale: torch.Tensor | float
    _clip: torch.Tensor

    def __init__(self, cfg: LowerBodyActionsCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.lower_joint_names)
        self._joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.lower_joint_names]
        self._joint_names = list(self.cfg.lower_joint_names)
        self._upper_joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.upper_joint_names]
        self._default_upper_joint_pos = self._asset.data.default_joint_pos[:, self._upper_joint_ids]
        self._p_gains = torch.tensor(self.cfg.p_gains, device=self.device)
        self._d_gains = torch.tensor(self.cfg.d_gains, device=self.device)
        self._torque_limits = torch.tensor(self.cfg.torque_limits, device=self.device)
        self._lower_joint_pos_limits = torch.tensor(self.cfg.joint_pos_limits, device=self.device)
        self._rand_motor_scale_range = torch.tensor(self.cfg.rand_motor_scale_range, device=self.device)

        # -- Angle-dependent torque LUTs (4-bar linkage joints) --
        self._use_torque_lut = self.cfg.torque_lut_dir is not None
        if self._use_torque_lut:
            from pathlib import Path
            lut_dir = Path(self.cfg.torque_lut_dir)
            self._knee_lut_L = KneeTorqueLUT(str(lut_dir / "L_knee_lut_new.csv"), device=self.device)
            self._knee_lut_R = KneeTorqueLUT(str(lut_dir / "R_knee_lut_new.csv"), device=self.device)
            self._ankle_lut_L = AnkleTorqueLUT(str(lut_dir / "L_ankle_lut_new.csv"), device=self.device)
            self._ankle_lut_R = AnkleTorqueLUT(str(lut_dir / "R_ankle_lut_new.csv"), device=self.device)
            # lower joint name -> local index within _joint_ids
            name_to_idx = {name: i for i, name in enumerate(self.cfg.lower_joint_names)}
            self._lut_knee_L = name_to_idx["L_Knee_Joint"]
            self._lut_knee_R = name_to_idx["R_Knee_Joint"]
            self._lut_ankle_pitch_L = name_to_idx["L_AnklePitch_Joint"]
            self._lut_ankle_roll_L = name_to_idx["L_AnkleRoll_Joint"]
            self._lut_ankle_pitch_R = name_to_idx["R_AnklePitch_Joint"]
            self._lut_ankle_roll_R = name_to_idx["R_AnkleRoll_Joint"]
            omni.log.info(f"Torque LUT loaded from {lut_dir}")

        self._num_lower = len(self._joint_ids)
        self._num_upper = len(self._upper_joint_ids)
        self._num_joints = self._num_lower
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        processed = actions * self._scale
        if self.cfg.clip is not None:
            processed = torch.clamp(processed, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        processed = processed.clamp(-1.0, 1.0)
        self._raw_actions[:] = processed
        self._processed_actions[:] = processed

    def _get_lower_torque_limits(self) -> torch.Tensor:
        """Return torque limits for lower joints: (num_envs, num_lower).

        If LUTs are loaded, knee/ankle limits are angle-dependent.
        Otherwise, falls back to the static config values.
        """
        limits = self._torque_limits[:self._num_lower].unsqueeze(0).expand(self.num_envs, -1).clone()
        if not self._use_torque_lut:
            return limits
        q = self._asset.data.joint_pos[:, self._joint_ids]
        # Knee: 1D lookup
        limits[:, self._lut_knee_L] = self._knee_lut_L.query(q[:, self._lut_knee_L])
        limits[:, self._lut_knee_R] = self._knee_lut_R.query(q[:, self._lut_knee_R])
        # Ankle: 2D lookup
        lp_L, lr_L = self._ankle_lut_L.query(q[:, self._lut_ankle_pitch_L], q[:, self._lut_ankle_roll_L])
        limits[:, self._lut_ankle_pitch_L] = lp_L
        limits[:, self._lut_ankle_roll_L] = lr_L
        lp_R, lr_R = self._ankle_lut_R.query(q[:, self._lut_ankle_pitch_R], q[:, self._lut_ankle_roll_R])
        limits[:, self._lut_ankle_pitch_R] = lp_R
        limits[:, self._lut_ankle_roll_R] = lr_R
        return limits

    def apply_actions(self):
        joint_ids_ordered = self._joint_ids + self._upper_joint_ids

        # per-step motor strength randomization
        rand_motor_scale_full = math_utils.sample_uniform(
            float(self._rand_motor_scale_range[0]),
            float(self._rand_motor_scale_range[1]),
            (self.num_envs, len(joint_ids_ordered)),
            device=self.device,
        )

        lower_limits = self._get_lower_torque_limits()
        upper_limits = self._torque_limits[self._num_lower:].view(1, -1)

        if self.cfg.pd_control:
            # PD mode (delta_default): q_des_lower = q_default_lower + delta_q
            lower_lim = self._lower_joint_pos_limits[:, 0].view(1, -1).repeat(self.num_envs, 1)
            upper_lim = self._lower_joint_pos_limits[:, 1].view(1, -1).repeat(self.num_envs, 1)

            delta_q = self.processed_actions
            q_default_lower = self._asset.data.default_joint_pos[:, self._joint_ids]
            q_des_lower = q_default_lower + delta_q
            q_des_lower = torch.clamp(q_des_lower, min=lower_lim, max=upper_lim)

            q_lower = self._asset.data.joint_pos[:, self._joint_ids]
            qd_lower = self._asset.data.joint_vel[:, self._joint_ids]
            tau_lower = self._p_gains[:self._num_lower] * (q_des_lower - q_lower) + self._d_gains[:self._num_lower] * (-qd_lower)

            # upper body: PD hold default pose
            q_upper = self._asset.data.joint_pos[:, self._upper_joint_ids]
            qd_upper = self._asset.data.joint_vel[:, self._upper_joint_ids]
            q_des_upper = self._default_upper_joint_pos
            tau_upper = self._p_gains[self._num_lower:] * (q_des_upper - q_upper) + self._d_gains[self._num_lower:] * (-qd_upper)

            # safety clamp to torque limits
            tau_lower = torch.clamp(tau_lower, min=-lower_limits, max=lower_limits)
            tau_upper = torch.clamp(tau_upper, min=-upper_limits, max=upper_limits)

            # apply motor strength randomization
            tau_lower = tau_lower * rand_motor_scale_full[:, :self._num_lower]
            tau_upper = tau_upper * rand_motor_scale_full[:, self._num_lower:]
            target_effort = torch.cat([tau_lower, tau_upper], dim=1)

        else:
            # Torque-direct mode
            tau_lower = self.processed_actions * lower_limits * rand_motor_scale_full[:, :self._num_lower]
            tau_upper = self._p_gains[self._num_lower:] * (self._default_upper_joint_pos - self._asset.data.joint_pos[:, self._upper_joint_ids]) \
                      + self._d_gains[self._num_lower:] * (-self._asset.data.joint_vel[:, self._upper_joint_ids])
            tau_upper = torch.clamp(tau_upper, min=-upper_limits, max=upper_limits)
            tau_upper = tau_upper * rand_motor_scale_full[:, self._num_lower:]
            target_effort = torch.cat([tau_lower, tau_upper], dim=1)

        self._asset.set_joint_effort_target(target_effort, joint_ids=joint_ids_ordered)

from dataclasses import MISSING
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

@configclass
class LowerBodyActionsCfg(ActionTermCfg):
    class_type: type[ActionTerm] = LowerBodyActions
    lower_joint_names: list[str] = MISSING
    upper_joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    p_gains: list[float] = MISSING
    d_gains: list[float] = MISSING
    torque_limits: list[float] = MISSING
    joint_pos_limits: list[tuple[float, float]] = MISSING
    pd_control: bool = True
    rand_motor_scale_range: tuple[float, float] = (1.0, 1.0)
    torque_lut_dir: str | None = None
    """Path to directory containing torque LUT CSVs. None disables angle-dependent limits."""
