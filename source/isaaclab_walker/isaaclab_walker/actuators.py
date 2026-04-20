"""Custom actuator: DelayedPDActuator with angle-dependent torque LUT and motor strength randomization."""

from __future__ import annotations

import torch
from isaaclab.actuators.actuator_pd import DelayedPDActuator
from isaaclab.actuators.actuator_pd_cfg import DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from isaaclab_walker.tasks.manager_based.isaaclab_walker.mdp.torque_lut import (
    AnkleTorqueLUT,
    KneeTorqueLUT,
)


class DelayedPDActuatorLUT(DelayedPDActuator):
    """DelayedPDActuator extended with angle-dependent torque LUT and per-step motor strength randomization.

    For knee and ankle joints, effort limits are overridden each step by querying CSV-based lookup tables
    that map joint angles to maximum torques (accounting for 4-bar linkage geometry).
    All other joints use the static ``effort_limit`` from the config.

    Motor strength randomization multiplies the clipped effort by a uniform random scale sampled
    independently for each (env, joint) pair every physics step.
    """

    cfg: DelayedPDActuatorLUTCfg

    def __init__(self, cfg: DelayedPDActuatorLUTCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # -- Torque LUT setup --
        self._use_lut = cfg.torque_lut_dir is not None
        if self._use_lut:
            from pathlib import Path

            lut_dir = Path(cfg.torque_lut_dir)
            self._knee_lut_L = KneeTorqueLUT(str(lut_dir / "L_knee_lut_new.csv"), device=self._device)
            self._knee_lut_R = KneeTorqueLUT(str(lut_dir / "R_knee_lut_new.csv"), device=self._device)
            self._ankle_lut_L = AnkleTorqueLUT(str(lut_dir / "L_ankle_lut_new.csv"), device=self._device)
            self._ankle_lut_R = AnkleTorqueLUT(str(lut_dir / "R_ankle_lut_new.csv"), device=self._device)

            # Resolve LUT joint indices within this actuator group
            name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
            self._lut_knee_L = name_to_idx["L_Knee_Joint"]
            self._lut_knee_R = name_to_idx["R_Knee_Joint"]
            self._lut_ankle_pitch_L = name_to_idx["L_AnklePitch_Joint"]
            self._lut_ankle_roll_L = name_to_idx["L_AnkleRoll_Joint"]
            self._lut_ankle_pitch_R = name_to_idx["R_AnklePitch_Joint"]
            self._lut_ankle_roll_R = name_to_idx["R_AnkleRoll_Joint"]

        # -- Motor strength randomization --
        self._motor_scale_range = cfg.rand_motor_scale_range
        # Persistent buffer so observations can read the last-used scale
        self.motor_strength_scale = torch.ones(self._num_envs, self.num_joints, device=self._device)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # 1) Apply command delay
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        # 2) PD torque computation: tau = Kp*(q_des - q) + Kd*(qd_des - qd) + tau_ff
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts

        # 3) Effort clipping (LUT-based for knee/ankle, static for others)
        if self._use_lut:
            limits = self.effort_limit.clone()
            # Knee: 1D lookup
            limits[:, self._lut_knee_L] = self._knee_lut_L.query(joint_pos[:, self._lut_knee_L])
            limits[:, self._lut_knee_R] = self._knee_lut_R.query(joint_pos[:, self._lut_knee_R])
            # Ankle: 2D lookup (pitch, roll) -> (tau_pitch, tau_roll)
            lp_L, lr_L = self._ankle_lut_L.query(
                joint_pos[:, self._lut_ankle_pitch_L], joint_pos[:, self._lut_ankle_roll_L]
            )
            limits[:, self._lut_ankle_pitch_L] = lp_L
            limits[:, self._lut_ankle_roll_L] = lr_L
            lp_R, lr_R = self._ankle_lut_R.query(
                joint_pos[:, self._lut_ankle_pitch_R], joint_pos[:, self._lut_ankle_roll_R]
            )
            limits[:, self._lut_ankle_pitch_R] = lp_R
            limits[:, self._lut_ankle_roll_R] = lr_R
            self.applied_effort = torch.clamp(self.computed_effort, -limits, limits)
        else:
            self.applied_effort = self._clip_effort(self.computed_effort)

        # 4) Per-step motor strength randomization
        if self._motor_scale_range is not None:
            self.motor_strength_scale.uniform_(self._motor_scale_range[0], self._motor_scale_range[1])
            self.applied_effort = self.applied_effort * self.motor_strength_scale

        # Write back as effort command (explicit actuator)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class DelayedPDActuatorLUTCfg(DelayedPDActuatorCfg):
    """Configuration for DelayedPDActuator with torque LUT and motor strength randomization."""

    class_type: type = DelayedPDActuatorLUT

    torque_lut_dir: str | None = None
    """Path to directory containing torque LUT CSVs (knee/ankle). None disables angle-dependent limits."""

    rand_motor_scale_range: tuple[float, float] | None = None
    """Uniform range (low, high) for per-step motor strength randomization. None disables."""
