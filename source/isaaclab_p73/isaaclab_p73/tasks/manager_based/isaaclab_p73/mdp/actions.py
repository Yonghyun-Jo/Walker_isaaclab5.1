from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
import omni.log
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class LowerBodyActions(ActionTerm):
    cfg: LowerBodyActionsCfg
    _asset: Articulation
    _scale: torch.Tensor | float
    _clip: torch.Tensor

    # NOTE:
    # ActionManager instantiates terms as: term_cfg.class_type(term_cfg, env)
    # Therefore, ActionTerm implementations must follow (cfg, env) argument order.
    def __init__(self, cfg: LowerBodyActionsCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.lower_joint_names)
        self._joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.lower_joint_names]
        # Keep action-channel names consistent with actual action application order (`_joint_ids`).
        self._joint_names = list(self.cfg.lower_joint_names)
        self._upper_joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.upper_joint_names]
        self._default_upper_joint_pos = self._asset.data.default_joint_pos[:, self._upper_joint_ids]
        self._p_gains = torch.tensor(self.cfg.p_gains, device=self.device)
        self._d_gains = torch.tensor(self.cfg.d_gains, device=self.device)
        self._torque_limits = torch.tensor(self.cfg.torque_limits, device=self.device)
        self._lower_joint_pos_limits = torch.tensor(self.cfg.joint_pos_limits, device=self.device)
        self._rand_motor_scale_range = torch.tensor(self.cfg.rand_motor_scale_range, device=self.device)

        self._num_joints = len(self._joint_ids)
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
            # resolve the dictionary config
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

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Process actions from the policy.

        Match TOCABI "delta_default" semantics:
        - Apply configured scale/clip first.
        - Enforce RL action space semantics by clamping to [-1, 1].
        - Store the final tensor as both `raw_actions` and `processed_actions`.

        Note:
        - When `pd_control=True`, `processed_actions` is interpreted as Δq in radians in `apply_actions()`,
          and we compute q_des = q_default + Δq there.
        - We intentionally do NOT unscale to joint limits here (that would convert actions to absolute angles).
        """
        processed = actions * self._scale
        if self.cfg.clip is not None:
            processed = torch.clamp(processed, min=self._clip[:, :, 0], max=self._clip[:, :, 1])

        # enforce [-1, 1] semantics after scale/clip
        processed = processed.clamp(-1.0, 1.0)

        self._raw_actions[:] = processed
        self._processed_actions[:] = processed

    def apply_actions(self):
        # set position targets
        # self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        # self._asset.set_joint_position_target(self._default_upper_joint_pos, joint_ids=self._upper_joint_ids)
        joint_ids_ordered = self._joint_ids + self._upper_joint_ids

        # per-step motor strength randomization (applied consistently in both modes)
        rand_motor_scale_full = math_utils.sample_uniform(
            float(self._rand_motor_scale_range[0]),
            float(self._rand_motor_scale_range[1]),
            (self.num_envs, len(joint_ids_ordered)),
            device=self.device,
        )

        if self.cfg.pd_control:
            # ------------------------------------------------------------
            # PD mode (delta_default): q_des_lower = q_default_lower + Δq
            # ------------------------------------------------------------
            # lower joint position limits from config (shape: 12x2)
            lower_lim = self._lower_joint_pos_limits[:, 0].view(1, -1).repeat(self.num_envs, 1)
            upper_lim = self._lower_joint_pos_limits[:, 1].view(1, -1).repeat(self.num_envs, 1)

            # Δq in radians (after scale/clip + [-1,1] clamp in process_actions)
            delta_q = self.processed_actions
            q_default_lower = self._asset.data.default_joint_pos[:, self._joint_ids]
            q_des_lower = q_default_lower + delta_q
            q_des_lower = torch.clamp(q_des_lower, min=lower_lim, max=upper_lim)

            q_lower = self._asset.data.joint_pos[:, self._joint_ids]
            qd_lower = self._asset.data.joint_vel[:, self._joint_ids]
            tau_lower = (self._p_gains[:12] / 9.0) * (q_des_lower - q_lower) + (self._d_gains[:12] / 3.0) * (-qd_lower)

            # upper body: PD hold default pose (unchanged behavior)
            q_upper = self._asset.data.joint_pos[:, self._upper_joint_ids]
            qd_upper = self._asset.data.joint_vel[:, self._upper_joint_ids]
            q_des_upper = self._default_upper_joint_pos
            tau_upper = (self._p_gains[12:] / 9.0) * (q_des_upper - q_upper) + (self._d_gains[12:] / 3.0) * (-qd_upper)

            # safety clamp to torque limits (match TOCABI)
            tau_lower = torch.clamp(
                tau_lower,
                min=-self._torque_limits[:12].view(1, -1),
                max=self._torque_limits[:12].view(1, -1),
            )
            tau_upper = torch.clamp(
                tau_upper,
                min=-self._torque_limits[12:].view(1, -1),
                max=self._torque_limits[12:].view(1, -1),
            )

            # apply motor strength randomization
            tau_lower = tau_lower * rand_motor_scale_full[:, :12]
            tau_upper = tau_upper * rand_motor_scale_full[:, 12:]
            target_effort = torch.cat([tau_lower, tau_upper], dim=1)

        else:
            # ------------------------------------------------------------
            # Torque-direct mode: normalized torque in [-1,1] * torque_limits
            # ------------------------------------------------------------
            tau_lower = self.processed_actions * self._torque_limits[:12] * rand_motor_scale_full[:, :12]
            tau_upper = (self._p_gains[12:] / 9.0) * (self._default_upper_joint_pos - self._asset.data.joint_pos[:, self._upper_joint_ids]) \
                      + (self._d_gains[12:] / 3.0) * (-self._asset.data.joint_vel[:, self._upper_joint_ids])
            tau_upper = torch.clamp(
                tau_upper,
                min=-self._torque_limits[12:].view(1, -1),
                max=self._torque_limits[12:].view(1, -1),
            )
            tau_upper = tau_upper * rand_motor_scale_full[:, 12:]
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
    # Lower-body joint position limits (12 joints): list of (lower, upper) in radians.
    joint_pos_limits: list[tuple[float, float]] = MISSING
    pd_control: bool = True
    rand_motor_scale_range: tuple[float, float] = (1.0, 1.0)
