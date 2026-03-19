from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_P73_LOWER_JOINT_NAMES_ORDERED: list[str] = [
    "L_HipRoll_motor",
    "L_HipPitch_motor",
    "L_HipYaw_motor",
    "L_KneeUpper_motor",
    "L_AnkleM1_motor",
    "L_AnkleM2_motor",
    "R_HipRoll_motor",
    "R_HipPitch_motor",
    "R_HipYaw_motor",
    "R_KneeUpper_motor",
    "R_AnkleM1_motor",
    "R_AnkleM2_motor",
]


def _assert_p73_joint_order(env: "ManagerBasedRLEnv") -> None:
    """Verify that symmetry-critical action/observation orders match expected lower-body order."""
    env_u = getattr(env, "unwrapped", env)
    if env_u.__dict__.get("_p73_symmetry_joint_order_ok", False):
        return

    try:
        action_term = env_u.action_manager.get_term("joint_pos")
        # P73 LowerBodyActions applies actions using `_joint_ids` order.
        # `_joint_names` from `find_joints()` can be articulation-resolved order
        # (e.g., interleaved L/R), so derive runtime action order from `_joint_ids`.
        action_joint_ids = list(getattr(action_term, "_joint_ids"))
        action_asset = getattr(action_term, "_asset")
        action_joint_names = [action_asset.data.joint_names[i] for i in action_joint_ids]
    except Exception as e:
        raise RuntimeError(
            "Failed to access runtime action order from action term 'joint_pos' via '_joint_ids'."
        ) from e

    if action_joint_names != _P73_LOWER_JOINT_NAMES_ORDERED:
        mismatch = next(
            (
                (i, exp, got)
                for i, (exp, got) in enumerate(zip(_P73_LOWER_JOINT_NAMES_ORDERED, action_joint_names))
                if exp != got
            ),
            None,
        )
        raise RuntimeError(
            "P73 symmetry assumes a fixed 12-DOF lower-body order, but runtime action order differs.\n"
            f"First mismatch: idx={mismatch[0] if mismatch else 'N/A'} "
            f"expected={mismatch[1] if mismatch else 'N/A'} got={mismatch[2] if mismatch else 'N/A'}\n"
            "Fix: align ActionsCfg.joint_pos.lower_joint_names / _joint_ids with symmetry order."
        )

    cfg = cast(Any, getattr(env_u, "cfg", env.cfg))
    for term_name in ("motor_joint_pos", "motor_joint_vel"):
        term_cfg = cfg.observations.policy.__dict__[term_name]
        asset_cfg = term_cfg.params.get("asset_cfg", None)
        if asset_cfg is None:
            continue
        robot = env_u.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        func_name = getattr(getattr(term_cfg, "func", None), "__name__", "")
        if func_name in ("joint_pos_ordered_rel", "joint_vel_ordered"):
            obs_joint_names = list(getattr(asset_cfg, "joint_names", []) or [])
        else:
            if joint_ids == slice(None):
                obs_joint_names = list(getattr(robot, "joint_names"))
            else:
                obs_joint_names = [robot.joint_names[i] for i in joint_ids]
        if obs_joint_names != _P73_LOWER_JOINT_NAMES_ORDERED:
            mismatch = next(
                (
                    (i, exp, got)
                    for i, (exp, got) in enumerate(zip(_P73_LOWER_JOINT_NAMES_ORDERED, obs_joint_names))
                    if exp != got
                ),
                None,
            )
            raise RuntimeError(
                "P73 symmetry assumes a fixed 12-DOF lower-body order, but runtime observation order differs.\n"
                f"Term: policy.{term_name}\n"
                f"First mismatch: idx={mismatch[0] if mismatch else 'N/A'} "
                f"expected={mismatch[1] if mismatch else 'N/A'} got={mismatch[2] if mismatch else 'N/A'}\n"
                "Fix: keep policy motor_joint_pos/motor_joint_vel joint_names aligned with lower_joint_names."
            )

    env_u.__dict__["_p73_symmetry_joint_order_ok"] = True


def _flip_lowerbody_12_axis_aware(joint_tensor: torch.Tensor) -> torch.Tensor:
    """Mirror P73 lower-body 12D vector with axis-aware sign rules.

    Order:
      [0] L_HipRoll_motor
      [1] L_HipPitch_motor
      [2] L_HipYaw_motor
      [3] L_KneeUpper_motor
      [4] L_AnkleM1_motor
      [5] L_AnkleM2_motor
      [6] R_HipRoll_motor
      [7] R_HipPitch_motor
      [8] R_HipYaw_motor
      [9] R_KneeUpper_motor
      [10] R_AnkleM1_motor
      [11] R_AnkleM2_motor

    Rule updated from URDF-axis review:
      - HipRoll: swap only (no sign flip)
      - Others: swap + sign flip
    """
    if joint_tensor.shape[1] != 12:
        raise ValueError(f"Expected 12D lower-body tensor. Got: {joint_tensor.shape}.")
    out = torch.zeros_like(joint_tensor)

    # Left -> Right
    out[:, 6] = joint_tensor[:, 0]     # HipRoll: swap only
    out[:, 7] = -joint_tensor[:, 1]    # HipPitch: swap + negate
    out[:, 8] = -joint_tensor[:, 2]    # HipYaw: swap + negate
    out[:, 9] = -joint_tensor[:, 3]    # KneeUpper: swap + negate
    out[:, 10] = -joint_tensor[:, 4]   # AnkleM1: swap + negate
    out[:, 11] = -joint_tensor[:, 5]   # AnkleM2: swap + negate

    # Right -> Left
    out[:, 0] = joint_tensor[:, 6]     # HipRoll: swap only
    out[:, 1] = -joint_tensor[:, 7]    # HipPitch: swap + negate
    out[:, 2] = -joint_tensor[:, 8]    # HipYaw: swap + negate
    out[:, 3] = -joint_tensor[:, 9]    # KneeUpper: swap + negate
    out[:, 4] = -joint_tensor[:, 10]   # AnkleM1: swap + negate
    out[:, 5] = -joint_tensor[:, 11]   # AnkleM2: swap + negate
    return out


def _flip_measured_6_swap_negate(joint_tensor: torch.Tensor) -> torch.Tensor:
    """Mirror measured 6D joints [L3, R3] by left-right swap and sign inversion."""
    if joint_tensor.shape[1] != 6:
        raise ValueError(f"Expected 6D measured-joint tensor. Got: {joint_tensor.shape}.")
    out = torch.zeros_like(joint_tensor)
    out[:, 0:3] = -joint_tensor[:, 3:6]
    out[:, 3:6] = -joint_tensor[:, 0:3]
    return out


def p73_data_augmentation_lowerbody_mirror(
    obs: torch.Tensor | None,
    actions: torch.Tensor | None,
    env: "ManagerBasedRLEnv",
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """P73 lower-body symmetry augmentation helper for mirror loss path.

    Policy single-frame layout is fixed to 59D:
      [0:3]   base_ang_vel
      [3:6]   projected_gravity
      [6:9]   velocity_commands
      [9:11]  gait_phase_sin/cos (2D, no-flip)
      [11:23] motor_joint_pos (12D)
      [23:35] motor_joint_vel (12D)
      [35:41] measured_joint_pos (6D)
      [41:47] measured_joint_vel (6D)
      [47:59] actions(last_processed_action, 12D)
    """
    if (obs is not None and obs_type == "policy") or actions is not None:
        _assert_p73_joint_order(env)

    if obs is not None and obs.ndim == 1:
        # Keep function robust for single-sample minibatches passed as 1D vectors.
        obs = obs.unsqueeze(0)

    if actions is not None and actions.ndim == 1:
        actions = actions.unsqueeze(0)

    if obs is not None:
        if obs_type != "policy":
            raise ValueError(f"P73 lower-body symmetry currently supports obs_type='policy' only. Got: {obs_type}")
        single_frame_dim = 59
        if obs.shape[1] % single_frame_dim != 0:
            raise ValueError(f"Policy obs dim must be a multiple of 59. Got obs_dim={obs.shape[1]}.")
        history_length = obs.shape[1] // single_frame_dim
        obs_flipped = obs.clone()
        for i in range(history_length):
            start = i * single_frame_dim
            base_ang_vel = obs[:, start + 0 : start + 3]
            projected_gravity = obs[:, start + 3 : start + 6]
            vel_cmd = obs[:, start + 6 : start + 9]
            gait_phase = obs[:, start + 9 : start + 11]  # sin, cos (no flip)
            motor_joint_pos = obs[:, start + 11 : start + 23]
            motor_joint_vel = obs[:, start + 23 : start + 35]
            measured_joint_pos = obs[:, start + 35 : start + 41]
            measured_joint_vel = obs[:, start + 41 : start + 47]
            last_actions = obs[:, start + 47 : start + 59]

            base_ang_vel_flipped = base_ang_vel * torch.tensor([-1.0, 1.0, -1.0], device=obs.device)
            projected_gravity_flipped = projected_gravity * torch.tensor([-1.0, 1.0, 1.0], device=obs.device)
            vel_cmd_flipped = vel_cmd * torch.tensor([1.0, -1.0, -1.0], device=obs.device)
            motor_joint_pos_flipped = _flip_lowerbody_12_axis_aware(motor_joint_pos)
            motor_joint_vel_flipped = _flip_lowerbody_12_axis_aware(motor_joint_vel)
            measured_joint_pos_flipped = _flip_measured_6_swap_negate(measured_joint_pos)
            measured_joint_vel_flipped = _flip_measured_6_swap_negate(measured_joint_vel)
            last_actions_flipped = _flip_lowerbody_12_axis_aware(last_actions)

            obs_flipped[:, start : start + single_frame_dim] = torch.cat(
                [
                    base_ang_vel_flipped,
                    projected_gravity_flipped,
                    vel_cmd_flipped,
                    gait_phase,
                    motor_joint_pos_flipped,
                    motor_joint_vel_flipped,
                    measured_joint_pos_flipped,
                    measured_joint_vel_flipped,
                    last_actions_flipped,
                ],
                dim=1,
            )
        obs_augmented = torch.cat([obs, obs_flipped], dim=0)
    else:
        obs_augmented = None

    if actions is not None:
        actions_flipped = _flip_lowerbody_12_axis_aware(actions)
        actions_augmented = torch.cat([actions, actions_flipped], dim=0)
    else:
        actions_augmented = None

    return obs_augmented, actions_augmented
