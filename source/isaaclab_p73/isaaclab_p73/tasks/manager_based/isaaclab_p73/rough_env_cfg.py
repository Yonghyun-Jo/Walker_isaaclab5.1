# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from . import mdp

from isaaclab_p73 import P73_CFG  # isort: skip


@configclass
class KangarooRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    body_height_tracking = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,  # stronger anti-crouch penalty
        params={"target_height": 0.93},
    )

    # === 2. Stability & Safety Rewards ==============================================

    # 2-1. Base Orientation Error
    flat_base_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-10.0,  # -10.0 -> -15.0
    )

    # 2-2. Linear Velocity Z
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.2,  # -0.2 -> 0.0
    )

    # 2-3. Angular Velocity XY (override parent default: -0.05)
    # NOTE: Tune weight here (set 0.0 to disable).
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.2,
    )

    # 2-4. Undesired Contacts (Collision penalty)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,  # -4.5 -> -5.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                "base_link",
                ".*_Calf_Link",
                ".*_KneeUpper_Link",
                ".*_KneeLower_Link",
            ]),
            "threshold": 1.0,
        },
    )

    # === 3. Joint Control & Limits ==============================================

    # 3-1-1. Joint position limits (Penalize ankle joints)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0, # -1.5 -> -2.0
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_AnklePitch_Joint", ".*_AnkleRoll_Joint"])},
    )

    # 3-1-2. Joint position limits (Penalize all joints)
    all_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


    # 3-2-1. Joint Velocity Penalty (All joints - for slower, smoother whole-body movement)
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,  # -0.005 -> -0.01
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=".*",  # All joints for whole-body smoothness
            )
        },
    )

    # === 4. Joint Deviation & Limits ==============================================

    # --- Hip axis-wise constraints (L/R pairs) ---
    joint_deviation_hipyaw = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-0.80, # -0.50 -> -0.80
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_motor", "R_HipYaw_motor"]),
            "target_offset": 0.0,
            # Asymmetric example (direction-dependent):
            # - positive side (flexion): wider deadband, softer stiffness
            # - negative side (extension): narrow deadband, harder stiffness
            "deadband_pos": 0.0,   # ~57 deg (flexion allowance; tune to your gait range)
            "deadband_neg": 0.0,   # ~11.5 deg (extension constraint)
            "stiffness_pos": 1.0,
            "stiffness_neg": 1.0,
        },
    )


    joint_deviation_hiproll = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-1.0, # -0.40 -> -1.0
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipRoll_motor", "R_HipRoll_motor"]),
            "target_offset": 0.0,
            # Asymmetric example (direction-dependent):
            # - positive side (flexion): wider deadband, softer stiffness
            # - negative side (extension): narrow deadband, harder stiffness
            "deadband_pos": 0.3,   # ~17 deg 벌림
            "deadband_neg": 0.1,   # ~5.7 deg 오므림
            "stiffness_pos": 1.5,
            "stiffness_neg": 2.5,
        },
    )

    # --- Knee pitch constraint (L/R pair) ---
    # NOTE: pos/neg is defined by e = (q - q_target). If your knee flexion sign is opposite,
    # swap *_pos and *_neg values.
    joint_deviation_left_kneepitch = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-0.60, # -0.20 -> -0.60
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_KneePitch_Joint"]),
            "target_offset": 0.15, # ~8.6 deg
            # Asymmetric example (direction-dependent):
            # - positive side (knee flexion): allow large range with soft stiffness
            # - negative side (hyper-extension): keep tight with hard stiffness
            "deadband_pos": 2.0,   # ~114 deg
            "deadband_neg": 0.01,  # ~0.57 deg
            "stiffness_pos": 0.2,
            "stiffness_neg": 5.0,
        },
    )

    joint_deviation_right_kneepitch = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-0.60, # -0.20 -> -0.60
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["R_KneePitch_Joint"]),
            "target_offset": -0.15, # ~8.6 deg
            "deadband_pos": 0.01,   # ~0.57 deg
            "deadband_neg": 2.0,   # ~114 deg
            "stiffness_pos": 5.0,
            "stiffness_neg": 0.2,
        },
    )

    # === 5. Action & Energy Efficiency ==============================================

    # 5-1. Action Rate Smoothness
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.03,  # -0.03 -> -0.03 -> -0.035
    )

    # 5-1b. Action Acceleration Smoothness (2nd-order difference)
    # Penalizes oscillatory/chattering action profiles by discouraging rapid changes in action-rate.
    action_accel_l2 = RewTerm(
        func=mdp.action_accel_l2,
        weight=-0.002, # -0.002
    )

    # 5-2. Joint Acceleration Limits
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7, # -0.0 -> -2.5e-7
    )

    # 5-3. Joint Torque Limits
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-4.0e-6, # -4.0e-6
    )

    # === 6. Feet Contact & Stability ==============================================

    # 6-1. Feet Air Time (Biped version - prevents in-place stepping)
    # Walking: reward air time (발 들기 장려)
    # Standing: reward contact time (제자리 stepping 억제!)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            # IMPORTANT: explicit [L, R] order
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.45,
            "velocity_threshold": 0.05,
            # Stop alignment switch: keep walking-like until feet are aligned when cmd~0
            "stop_cmd_vel_max": 0.05,
            "yaw_threshold_deg": 5.0,
            "pos_threshold_m": 0.02,
            "stance_width_m": 0.24,
            "contact_threshold": 5.0,
        },
    )


    # 6-1b. Low-Speed Feet Alignment Penalty (NEW - 정지(저속)에서 발 정렬 강제)
    # Gate: cmd==0 && base(vel)<=0.05 일 때만 활성화  (vel_body_name="base_link")
    # Penalty: yaw/stance 오차의 threshold 초과분(연속값)을 강하게 페널티
    low_speed_feet_alignment_penalty = RewTerm(
        func=mdp.low_speed_feet_alignment_penalty,
        weight=-10.0, # -5.0 -> -10.0
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "cmd_zero_max": 1.0e-3,
            "body_vel_max": 0.07,
            "vel_body_name": "base_link",
            "yaw_threshold_deg": 5.0, # 10.0 -> 5.0
            "pos_threshold_m": 0.02,
            "stance_width_m": 0.24,
            "contact_threshold": 5.0,
            "yaw_scale": 1.0,
            "pos_scale": 1.0,
            "require_contact": False,
        },
    )

    # 6-1c. Low-Speed Double Support Penalty (NEW - 정지(저속)에서 양발 접촉 강제)
    # Gate: cmd==0 && base(vel)<=0.05 일 때만 활성화 (vel_body_name="base_link")
    # Penalty: 양발 모두 contact_threshold 이상 접촉이 아니면(0발/1발 포함) 1.0 페널티
    low_speed_double_support_penalty = RewTerm(
        func=mdp.low_speed_double_support_penalty,
        weight=-10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "cmd_zero_max": 1.0e-3,
            "body_vel_max": 0.07,
            "vel_body_name": "base_link",
            "contact_threshold": 1.0,
        },
    )


    # 6-2. Feet Slide (Increased penalty for precise trajectory tracking)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
        },
    )


    # 6-3. Feet Ground Parallel (발이 지면과 평행하지 않을 때 페널티)
    feet_ground_parallel = RewTerm(
        func=mdp.feet_ground_parallel,
        weight=-5.0,  # -10.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "threshold": 1.0,  # 접촉력 임계값 (N)
        },
    )

    # 6-4. Feet Parallel 양발의 yaw 방향이 다를 때 페널티
    feet_parallel = RewTerm(
        func=mdp.feet_parallel,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "threshold": 1.0,  # 접촉력 임계값 (N)
        },
    )

    # 6-4a. Head↔Feet Yaw Mismatch (각 발이 head heading과 다를 때 penalty; per-foot contact gated)
    # - 기준(reference)은 reference_body_name으로 확장 가능: "base_link" (P73 기본)
    head_feet_yaw_mismatch_l2 = RewTerm(
        func=mdp.ref_feet_yaw_mismatch_l2,
        weight=-10.0, # -10.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "reference_body_name": "base_link",
            "threshold": 1.0,  # 접촉력 임계값 (N)
        },
    )

    # 6-4b. Feet Yaw Align (Walking 포함, 스윙/공중에서도 양발 yaw 정렬 페널티)
    # - contact(접촉) 여부와 무관하게 yaw 차이를 penalize
    # - turning(회전 보행)에서 학습을 망치지 않도록 |cmd_wz|가 임계값 이상이면 off
    feet_yaw_align_cmd_gated_l2 = RewTerm(
        func=mdp.feet_yaw_align_cmd_gated_l2,
        weight=-3.0, # -2.0 -> -3.0
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "wz_threshold": 0.01,  # rad/s, cmd_wz > 0 이면 즉시 penalty=0
        },
    )

    # 6-6. Feet Clearance Height (발 끌림 방지 - 페널티 방식)
    feet_clearance_penalty = RewTerm(
        func=mdp.feet_clearance_height,
        weight=-0.5,  # -1.50 -> -0.80
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "target_height": 0.07,  # 최소 지면 이격 높이 (7cm)
        },
    )


    # 6-7. Feet Stumble (발 걸림 방지)
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.25,  # -0.25 -> -0.5
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "threshold_ratio": 3.0,  # 수평/수직 힘 비율 임계값
        },
    )

    # 6-9. Contact Momentum (접촉 충격량 - 부드러운 착지)
    contact_momentum = RewTerm(
        func=mdp.contact_momentum,
        weight=-5.0e-4,  # -5.0e-4 -> -6.0e-4 -> -8.0e-4
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
        },
    )



    # === NEW: Phase-based gait schedule + swing clearance profile ==================
    # Enforce alternating contacts with explicit double-support (DS) windows.
    contact_schedule_biped_ds = RewTerm(
        func=mdp.contact_schedule_reward_biped_ds,
        weight=2.0, # 2.0
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "period_steps": 40,
            "ds_ratio": 0.20,
            "contact_threshold": 5.0,
            "cmd_zero_max": 1.0e-3,
        },
    )

    # Penalize if scheduled swing foot doesn't reach the minimum clearance profile.
    # NOTE: this returns a non-negative penalty; use negative weight to penalize.
    swing_clearance_min_profile_penalty = RewTerm(
        func=mdp.swing_clearance_min_profile_penalty,
        weight=-5.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "period_steps": 40,
            "ds_ratio": 0.20,
            "clearance_height": 0.17,
            "contact_threshold": 5.0,
            "cmd_zero_max": 1.0e-3,
        },
    )

    # Spring compliance (pos-match) for impact mitigation (충격량 완화)
    # - At first contact: capture foot z (relative to base_link) as z_d
    # - While in contact: target z = z_d + max(Fz - Fz0, 0) / K, penalize (z - target)^2
    # - Gate: walking only (cmd_vel > 0 or body_vel > velocity_threshold)
    spring_compliance_pos_match = RewTerm(
        func=mdp.spring_compliance_pos_match_reward,
        weight=-8.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "head_body_name": "base_link",
            "k_spring": 9000.0,
            "contact_threshold": 5.0,
            "robot_mass": 73.0,
            "baseline_margin": 1.2,
            "gravity": 9.81,
            "command_name": "base_velocity",
            "velocity_threshold": 0.05,
            "cmd_vel_eps": 1.0e-3,
        },
    )

    # === NEW: Gait Symmetry Rewards (절뚝임 방지) ================================

    # Feet Height Symmetry: 좌우 발 스윙 최대 높이 차이 페널티
    # Compares max swing heights (not instantaneous) to prevent limping.
    # Ported from tocabi (IsaacLab 4.5) to P73 (IsaacLab 5.2).
    feet_height_symmetry_penalty = RewTerm(
        func=mdp.feet_height_symmetry_penalty,
        weight=-7.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "command_name": "base_velocity",
            "velocity_threshold": 0.05,
            "contact_threshold": 5.0,
        },
    )

    # Air Time Variance: 좌우 발의 air/contact time 균등화 페널티
    # Penalizes variance in air and contact times between the two feet.
    # Requires ContactSensor track_air_time=True.
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty_gated,
        weight=-7.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "command_name": "base_velocity",
            "velocity_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.LowerBodyActionsCfg(
        asset_name="robot",
        clip = {".*": (-1.0, 1.0)},
        # Match TOCABI "delta_default" semantics:
        # policy action in [-1, 1] -> Δq in radians after scaling (here: [-0.5, 0.5] rad).
        scale=0.5,
        lower_joint_names=["L_HipRoll_motor", "L_HipPitch_motor", "L_HipYaw_motor", "L_KneeUpper_motor", "L_AnkleM1_motor", "L_AnkleM2_motor",
                           "R_HipRoll_motor", "R_HipPitch_motor", "R_HipYaw_motor", "R_KneeUpper_motor", "R_AnkleM1_motor", "R_AnkleM2_motor"],
        upper_joint_names=["WaistYaw_motor", "WaistRoll_motor", "WaistPitch_motor",
                           "L_ShoulderPitch_motor", "L_ShoulderRoll_motor", "L_ShoulderYaw_motor", "L_ElbowPitch_motor", "L_WristYaw_motor", "L_WristPitch_motor", "L_WristRoll_motor", 
                           "R_ShoulderPitch_motor", "R_ShoulderRoll_motor", "R_ShoulderYaw_motor", "R_ElbowPitch_motor", "R_WristYaw_motor", "R_WristPitch_motor", "R_WristRoll_motor"],
        pd_control=True,

        p_gains = [3000.0, 5000.0, 2000.0, 3700.0, 3200.0, 3200.0, 
                   3000.0, 5000.0, 2000.0, 3700.0, 3200.0, 3200.0, 
                   6000.0, 10000.0, 10000.0, 
                   500.0, 500.0, 500.0, 1500.0, 500.0, 500.0, 500.0, 
                   500.0, 500.0, 500.0, 1500.0, 500.0, 500.0, 500.0],

        d_gains = [50.0, 20.0, 15.0, 30.0, 30.0, 30.0, 
                   50.0, 20.0, 15.0, 30.0, 30.0, 30.0, 
                   100.0, 100.0, 100.0, 
                   40.0, 40.0, 40.0, 40.0, 8.0, 5.0, 5.0,
                   40.0, 40.0, 40.0, 40.0, 8.0, 5.0, 5.0],

        torque_limits= [264, 165, 69, 165, 69, 69,
                        264, 165, 69, 165, 69, 69,
                        110, 110, 110,
                        110, 110, 45.6, 45.6, 23.2, 23.2, 23.2,
                        110, 110, 45.6, 45.6, 23.2, 23.2, 23.2],

        joint_pos_limits = [(-0.43, 0.78), (-1.57, 2.09), (-0.78, 0.78), (-0.0, 1.86), (-0.75, 0.59), (-0.59, 0.75), 
                            (-0.78, 0.43), (-2.09, 1.57), (-0.78, 0.78), (-1.86, 0.0), (-0.59, 0.75), (-0.75, 0.59)],

        rand_motor_scale_range = (0.8, 1.2)
    )

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={
                "period_steps": 40,
                "command_name": "base_velocity",
                "cmd_zero_max": 1.0e-3,
            },
        )  # 1D
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={
                "period_steps": 40,
                "command_name": "base_velocity",
                "cmd_zero_max": 1.0e-3,
            },
        )  # 1D
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["L_HipRoll_motor", "L_HipPitch_motor", "L_HipYaw_motor", "L_KneeUpper_motor", "L_AnkleM1_motor", "L_AnkleM2_motor",
                                 "R_HipRoll_motor", "R_HipPitch_motor", "R_HipYaw_motor", "R_KneeUpper_motor", "R_AnkleM1_motor", "R_AnkleM2_motor"],
                )
            },
            noise=Unoise(n_min=-0.0025, n_max=0.0025),
        )
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["L_HipRoll_motor", "L_HipPitch_motor", "L_HipYaw_motor", "L_KneeUpper_motor", "L_AnkleM1_motor", "L_AnkleM2_motor",
                                 "R_HipRoll_motor", "R_HipPitch_motor", "R_HipYaw_motor", "R_KneeUpper_motor", "R_AnkleM1_motor", "R_AnkleM2_motor"],
                )
            },
            noise=Unoise(n_min=-0.025, n_max=0.025),
        )
        measured_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "L_KneePitch_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                        "R_KneePitch_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        measured_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "L_KneePitch_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                        "R_KneePitch_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
                    ],
                )
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        # actions = ObsTerm(func=mdp.last_action)
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class TargetCfg(ObsGroup):
        """Target observation group for PPOFuture: provides O(t+1) features.

        Notes:
        - This group is stored as O(t+1) inside PPOFuture via `extras["observations"]["target"]`.
        - Keep it relatively compact. (We include height_scan here by design for future-state features.)
        """

        # --- policy-like kinematic features (single-frame, no noise) ---
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # 3D
        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 3D
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # 3D
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={
                "period_steps": 40,
                "command_name": "base_velocity",
                "cmd_zero_max": 1.0e-3,
            },
        )  # 1D
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={
                "period_steps": 40,
                "command_name": "base_velocity",
                "cmd_zero_max": 1.0e-3,
            },
        )  # 1D
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                    ],
                )
            },
        )  # 12D
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                    ],
                )
            },
        )  # 12D
        measured_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "L_KneePitch_Joint",
                        "L_AnklePitch_Joint",
                        "L_AnkleRoll_Joint",
                        "R_KneePitch_Joint",
                        "R_AnklePitch_Joint",
                        "R_AnkleRoll_Joint",
                    ],
                )
            },
        )  # 6D
        measured_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "L_KneePitch_Joint",
                        "L_AnklePitch_Joint",
                        "L_AnkleRoll_Joint",
                        "R_KneePitch_Joint",
                        "R_AnklePitch_Joint",
                        "R_AnkleRoll_Joint",
                    ],
                )
            },
        )  # 6D
        # Height scan (same sensor as policy, but no noise; keep clip for numerical stability)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})  # action_dim

        # --- domain randomization cache (2D) ---
        physics_material = ObsTerm(func=mdp.physics_material_sd)  # 2D (static/dynamic friction)

        # --- physical/randomization parameters we want the future-latent to encode ---
        # Base mass delta (current - default) for base_link
        base_mass_delta = ObsTerm(func=mdp.base_link_mass_delta)  # 1D
        # Base COM offset (current - cached_initial) for base_link
        base_com_offset = ObsTerm(func=mdp.base_link_com_offset)  # 3D
        # Motor joint armature stats: mean/std of (armature/default_armature)
        motor_armature_stats = ObsTerm(func=mdp.motor_joint_armature_stats)  # 2D
        # Motor joint damping stats: mean/std of (damping - default_damping)
        motor_damping_stats = ObsTerm(func=mdp.motor_joint_damping_stats)  # 2D

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observation group for PPOFuture (privileged prefix + value features).

        Layout (prefix for auxiliary supervision):
          - [0:3] gt_vel3       = [v_x, v_y, yaw_rate] in body frame
          - [3:9] gt_foot_force6= [L(Fx,Fy,Fz), R(Fx,Fy,Fz)] in base gravity-aligned (yaw-only) frame
          - [9: ] value features (single-frame, no noise)
        """

        # --- Fixed GT prefix for PPOFuture losses ---
        gt_vel3 = ObsTerm(func=mdp.base_vel_xy_yawrate)  # 3D
        gt_foot_force6 = ObsTerm(
            func=mdp.feet_contact_forces_l_r,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["L_Foot_Link", "R_Foot_Link"],
                    preserve_order=True,
                )
            },
            # Normalize contact forces (N) to O(1) for stable MSE auxiliary loss.
            # P73 is lighter than TOCABI; keep scales conservative and tune later if needed.
            scale=(
                1.0 / 300.0,  # L Fx
                1.0 / 300.0,  # L Fy
                1.0 / 600.0,  # L Fz
                1.0 / 300.0,  # R Fx
                1.0 / 300.0,  # R Fy
                1.0 / 600.0,  # R Fz
            ),
        )  # 6D

        # --- Value features (single-frame, no noise) ---
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # 3D
        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 3D
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  # 3D
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={
                "period_steps": 40,
                "command_name": "base_velocity",
                "cmd_zero_max": 1.0e-3,
            },
        )  # 1D
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={
                "period_steps": 40,
                "command_name": "base_velocity",
                "cmd_zero_max": 1.0e-3,
            },
        )  # 1D
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                    ],
                )
            },
        )  # 12D
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                    ],
                )
            },
        )  # 12D
        measured_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "L_KneePitch_Joint",
                        "L_AnklePitch_Joint",
                        "L_AnkleRoll_Joint",
                        "R_KneePitch_Joint",
                        "R_AnklePitch_Joint",
                        "R_AnkleRoll_Joint",
                    ],
                )
            },
        )  # 6D
        measured_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "L_KneePitch_Joint",
                        "L_AnklePitch_Joint",
                        "L_AnkleRoll_Joint",
                        "R_KneePitch_Joint",
                        "R_AnklePitch_Joint",
                        "R_AnkleRoll_Joint",
                    ],
                )
            },
        )  # 6D
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})  # action_dim

        # --- Match TargetCfg extra terms (privileged value inputs) ---
        # Height scan (no noise; clip for numerical stability)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        # Friction cache (2D)
        physics_material = ObsTerm(func=mdp.physics_material_sd)  # 2D
        # Base mass / COM randomization parameters
        base_mass_delta = ObsTerm(func=mdp.base_link_mass_delta)  # 1D
        base_com_offset = ObsTerm(func=mdp.base_link_com_offset)  # 3D
        # Motor parameter stats
        motor_armature_stats = ObsTerm(func=mdp.motor_joint_armature_stats)  # 2D
        motor_damping_stats = ObsTerm(func=mdp.motor_joint_damping_stats)  # 2D

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    target: TargetCfg = TargetCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # 1. Friction randomization
    # Purpose: Simulate different floor materials
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material_and_cache,  # type: ignore[arg-type]
        mode="startup",
        params={
            # Apply to all bodies (default selection) to avoid regex→ids mismatch on repeated resolves.
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.2, 1.4), # UniFP: [0.3, 2.0]
            "dynamic_friction_range": (0.2, 1.4), # UniFP: [0.3, 2.0]
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 2. Link mass randomization
    # Purpose: Simulate uncertainty in link masses (manufacturing variance, wear, etc.)
    # This applies to ALL robot links (legs, arms, torso) for realistic mass distribution
    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # Randomize all bodies of the robot.
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.8, 1.2),  # UniFP: [0.6, 1.4] × m_default
            "operation": "scale",  # Multiply default mass by random factor in range
        },
    )

    # 3. Base mass randomization
    # Purpose: Simulate carrying a backpack or payload
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # P73 USD uses "base_link" as the root body name.
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-10.0, 20.0),
            "operation": "add",
        },
    )

    # 4. Base COM randomization
    # Purpose: Simulate weight distribution changes
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            # P73 USD uses "base_link" as the root body name.
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.10, 0.10),  # -0.15 -> -0.10
                "y": (-0.10, 0.10),  # -0.15 -> -0.10
                "z": (-0.10, 0.10),  # -0.15 -> -0.10
            },
        },
    )
    # 5. Reset base
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTerm( func=mdp.reset_joints_by_offset, mode="reset",
        params={
            "position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1),
        },
    )
    randomize_armature = EventTerm( func=mdp.randomize_joint_parameters, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_motor"),
            "friction_distribution_params": (0.0, 0.0),
            "armature_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )
    randomize_damping = EventTerm( func=mdp.randomize_actuator_gains, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_motor"),
            "stiffness_distribution_params": (0.0, 0.0),
            "damping_distribution_params": (0.0, 2.9),
            "operation": "add",
        },
    )

    # 9. Push robot
    # Purpose: Train recovery from external disturbances
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 8.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5), # -0.8 -> -0.5 stage2
                "y": (-0.5, 0.5), # -0.8 -> -0.5 stage2
                "yaw": (-0.2, 0.2),
            },
        },
    )



@configclass
class ViewerCfg(ViewerCfg):
    # HD: 1280 x 720; Full HD: 1920 x 1080; 4K: 3840 x 2160
    resolution: tuple[int, int] = (1920, 1080)
    # Default: (7.5, 7.5, 7.5)
    eye: tuple[float, float, float] = (10, 10, 10)


@configclass
class P73RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KangarooRewards = KangarooRewards()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    viewer: ViewerCfg = ViewerCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        #self.sim.dt = 0.002

        # Scene
        self.scene.num_envs = 4096
        self.scene.robot = P73_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.height_scanner.offset.pos = (0.0, 0.0, 0.0)
        self.scene.height_scanner.pattern_cfg.size = [1.2, 1.2]
        # Height scan grid resolution: 1.2/0.15 + 1 = 9 points per axis -> 9*9 = 81 dims
        self.scene.height_scanner.pattern_cfg.resolution = 0.15
        self.scene.height_scanner.debug_vis = True

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # ---------------------------------------------------------------------
        # Phase-gait hyperparameters (centralize for easy tuning)
        # ---------------------------------------------------------------------
        # Keep these synced between:
        # - observations.policy/critic/target gait_phase_sin/cos
        # - rewards.contact_schedule_biped_ds
        # - rewards.swing_clearance_min_profile_penalty
        gait_period_steps = 50 #40
        gait_ds_ratio = 0.20

        # Wire into observations
        self.observations.policy.gait_phase_sin.params["period_steps"] = gait_period_steps
        self.observations.policy.gait_phase_cos.params["period_steps"] = gait_period_steps
        self.observations.critic.gait_phase_sin.params["period_steps"] = gait_period_steps
        self.observations.critic.gait_phase_cos.params["period_steps"] = gait_period_steps
        self.observations.target.gait_phase_sin.params["period_steps"] = gait_period_steps
        self.observations.target.gait_phase_cos.params["period_steps"] = gait_period_steps

        # Wire into rewards
        self.rewards.contact_schedule_biped_ds.params["period_steps"] = gait_period_steps
        self.rewards.contact_schedule_biped_ds.params["ds_ratio"] = gait_ds_ratio
        self.rewards.swing_clearance_min_profile_penalty.params["period_steps"] = gait_period_steps
        self.rewards.swing_clearance_min_profile_penalty.params["ds_ratio"] = gait_ds_ratio
