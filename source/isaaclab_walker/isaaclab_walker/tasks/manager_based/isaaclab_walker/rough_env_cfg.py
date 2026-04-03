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

from isaaclab_walker import P73_CFG  # isort: skip
from isaaclab_walker.assets import P73_ASSETS_DATA_DIR  # isort: skip

import os
_TORQUE_LUT_DIR = os.path.join(P73_ASSETS_DATA_DIR, "p73_walker", "lut")

# =====================================================================
# Walker joint name constants
# =====================================================================
# Lower body: 12 joints (6 per leg), all RL-controlled
_LOWER_JOINT_NAMES = [
    "L_HipRoll_Joint", "L_HipPitch_Joint", "L_HipYaw_Joint",
    "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
    "R_HipRoll_Joint", "R_HipPitch_Joint", "R_HipYaw_Joint",
    "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
]
# Upper body: 1 joint (PD-held at default)
_UPPER_JOINT_NAMES = ["WaistYaw_Joint"]


@configclass
class KangarooRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.8,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    body_height_tracking = RewTerm(
        func=mdp.base_height_l2,
        weight=-3.0,
        params={"target_height": 0.85},
    )

    # === 2. Stability & Safety Rewards ==============================================

    flat_base_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-10.0,
    )

    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.0,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.2,
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                "base_link",
                ".*_Thigh_Link",
                ".*_Knee_Link",
            ]),
            "threshold": 1.0,
        },
    )

    # === 3. Joint Control & Limits ==============================================

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_AnklePitch_Joint", ".*_AnkleRoll_Joint"])},
    )

    all_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*")
        },
    )

    # === 4. Joint Deviation & Limits ==============================================

    joint_deviation_hipyaw = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "R_HipYaw_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.15, # 0.0 -> 0.15
            "deadband_neg": 0.15, # 0.0 -> 0.15
            "stiffness_pos": 0.7,
            "stiffness_neg": 0.7,
        },
    )

    # HipRoll: New URDF — L axis +x, R axis -x (mirrored)
    # Both: pos = abduction(벌림), neg = adduction(오므림) in their own frame
    # → L/R 통합 가능 (같은 부호 = 같은 물리적 방향)
    joint_deviation_hiproll = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipRoll_Joint", "R_HipRoll_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.2,   # 벌림 허용
            "deadband_neg": 0.1,   # 오므림 제한
            "stiffness_pos": 1.5,
            "stiffness_neg": 2.5,
        },
    )

    # Knee: New URDF — L axis +y range [0, 2.56] default=0.35, R axis -y range [-2.56, 0] default=-0.35
    # L/R 축이 mirrored → 같은 부호 방향이 같은 물리적 의미
    # L: e = q - 0.35, R: e = q - (-0.35) = q + 0.35
    # L: e > 0 → q > 0.35 (더 굴곡), e < 0 → q < 0.35 (더 펴짐)
    # R: e > 0 → q > -0.35 (더 펴짐), e < 0 → q < -0.35 (더 굴곡)
    # → 축 방향이 반대이므로 L/R 분리 필요
    # 허용 범위: q ∈ [0.15, 2.0] (L), q ∈ [-2.0, -0.15] (R)
    joint_deviation_left_knee = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-0.60,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_Knee_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 1.65,    # 2.0까지 굴곡 허용 (2.0 - 0.35)
            "deadband_neg": 0.20,    # 0.15까지 펴짐 허용 (0.35 - 0.15)
            "stiffness_pos": 0.2,    # 굴곡은 부드럽게
            "stiffness_neg": 5.0,    # 과신전 강하게 제한
        },
    )

    joint_deviation_right_knee = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-0.60,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["R_Knee_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.20,    # -0.15까지 펴짐 허용 (0.35 - 0.15)
            "deadband_neg": 1.65,    # -2.0까지 굴곡 허용 (2.0 - 0.35)
            "stiffness_pos": 5.0,    # 과신전 강하게 제한
            "stiffness_neg": 0.2,    # 굴곡은 부드럽게
        },
    )

    # AnklePitch: New URDF — L축=+y default=-0.17, R축=-y default=0.17 (mirrored)
    # L: [-1.05, 0.7], R: [-0.7, 1.05]
    # 허용 범위: q ∈ [-0.41, 0.41] (L/R 동일 의도)
    joint_deviation_left_anklepitch = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-1.00,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_AnklePitch_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.58,    # 0.41까지 허용 (0.41 - (-0.17))
            "deadband_neg": 0.24,    # -0.41까지 허용 (-0.17 - (-0.41))
            "stiffness_pos": 3.0,
            "stiffness_neg": 3.0,
        },
    )

    joint_deviation_right_anklepitch = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-1.00,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["R_AnklePitch_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.24,    # 0.41까지 허용 (0.41 - 0.17)
            "deadband_neg": 0.58,    # -0.41까지 허용 (0.17 - (-0.41))
            "stiffness_pos": 3.0,
            "stiffness_neg": 3.0,
        },
    )

    # AnkleRoll: L축=+x, R축=+x (같은 축, 변경 없음), default=0.0
    # L/R: [-0.42, 0.42]
    # 같은 축이지만 대칭 범위 → 통합 가능
    joint_deviation_ankleroll = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-1.00,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_AnkleRoll_Joint", "R_AnkleRoll_Joint"]),
            "target_offset": 0.0,
            "deadband": 0.2,      # ±0.2 rad (~11도) 까지 자유
            "stiffness": 3.0,
        },
    )

    # === 5. Action & Energy Efficiency ==============================================

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )

    action_accel_l2 = RewTerm(
        func=mdp.action_accel_l2,
        weight=-0.003,
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )

    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-4.5e-6,
    )

    stand_still_joint_deviation = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.07,
            "velocity_threshold": 0.07,
            "standing_pose": {
                "L_HipRoll_Joint": 0.0,
                "L_HipPitch_Joint": 0.18,
                "L_HipYaw_Joint": 0.0,
                "L_Knee_Joint": 0.35,
                "L_AnklePitch_Joint": -0.17,
                "L_AnkleRoll_Joint": 0.0,
                "R_HipRoll_Joint": 0.0,
                "R_HipPitch_Joint": -0.18,
                "R_HipYaw_Joint": 0.0,
                "R_Knee_Joint": -0.35,
                "R_AnklePitch_Joint": 0.17,
                "R_AnkleRoll_Joint": 0.0,
                "WaistYaw_Joint": 0.0,
            },
        },
    )

    # === 6. Feet Contact & Stability ==============================================

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_biped,
        weight=2.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.45,
            "velocity_threshold": 0.07,
            "stop_cmd_vel_max": 0.05,
            "yaw_threshold_deg": 5.0,
            "pos_threshold_m": 0.02,
            "stance_width_m": 0.200,
            "contact_threshold": 5.0,
        },
    )

    low_speed_feet_alignment_penalty = RewTerm(
        func=mdp.low_speed_feet_alignment_penalty,
        weight=-0.0,
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
            "yaw_threshold_deg": 5.0,
            "pos_threshold_m": 0.02,
            "stance_width_m": 0.192,
            "contact_threshold": 5.0,
            "yaw_scale": 1.0,
            "pos_scale": 1.0,
            "require_contact": False,
        },
    )

    low_speed_double_support_penalty = RewTerm(
        func=mdp.low_speed_double_support_penalty,
        weight=-0.0,
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

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-7.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
        },
    )

    feet_ground_parallel = RewTerm(
        func=mdp.feet_ground_parallel,
        weight=-7.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "threshold": 1.0,
        },
    )

    feet_parallel = RewTerm(
        func=mdp.feet_parallel,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "threshold": 1.0,
        },
    )

    head_feet_yaw_mismatch_l2 = RewTerm(
        func=mdp.ref_feet_yaw_mismatch_l2,
        weight=-3.0, # -10.0 -> -3.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "reference_body_name": "base_link",
            "threshold": 1.0,
        },
    )

    feet_yaw_align_cmd_gated_l2 = RewTerm(
        func=mdp.feet_yaw_align_cmd_gated_l2,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "wz_threshold": 0.01,
        },
    )

    feet_clearance_penalty = RewTerm(
        func=mdp.feet_clearance_height,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "target_height": 0.07,
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "threshold_ratio": 3.0,
        },
    )

    contact_momentum = RewTerm(
        func=mdp.contact_momentum,
        weight=-4.5e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
        },
    )

    contact_force_limit = RewTerm(
        func=mdp.contact_force_limit_soft_penalty,
        weight=9.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_AnkleRoll_Link"),
            "robot_mass": 60.0,
            "safety_factor": 1.5,
            "std": 200.0,
        },
    )

    contact_schedule_biped_ds = RewTerm(
        func=mdp.contact_schedule_reward_biped_ds,
        weight=5.0,
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

    swing_clearance_min_profile_penalty = RewTerm(
        func=mdp.swing_clearance_min_profile_penalty,
        weight=-20.0,
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
            "robot_mass": 60.3,
            "baseline_margin": 1.2,
            "gravity": 9.81,
            "command_name": "base_velocity",
            "velocity_threshold": 0.05,
            "cmd_vel_eps": 1.0e-3,
        },
    )

    feet_height_symmetry_penalty = RewTerm(
        func=mdp.feet_height_symmetry_penalty,
        weight=-7.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "command_name": "base_velocity",
            "velocity_threshold": 0.07,
            "contact_threshold": 5.0,
        },
    )

    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty_gated,
        weight=-7.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "command_name": "base_velocity",
            "velocity_threshold": 0.07,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.LowerBodyActionsCfg(
        asset_name="robot",
        clip={".*": (-1.0, 1.0)},
        scale=0.5,
        lower_joint_names=_LOWER_JOINT_NAMES,
        upper_joint_names=_UPPER_JOINT_NAMES,
        pd_control=True,

        # PD gains: 12 lower + 1 upper = 13
        # From setting_realrobot_PDgain.yaml
        p_gains=[1536.0, 937.5, 625.0, 747.552, 490.644, 490.104,
                 1536.0, 937.5, 625.0, 747.552, 490.644, 490.104,
                 576.0],

        d_gains=[76.8, 37.5, 12.5, 37.378, 16.355, 5.337,
                 76.8, 37.5, 12.5, 37.378, 16.355, 5.337,
                 19.2],

        torque_limits=[352, 220, 95, 220, 95, 95,
                       352, 220, 95, 220, 95, 95,
                       152],

        # Joint position limits from URDF (L leg then R leg)
        # New URDF: L/R axes are mirrored, limits reflect the new axis directions
        joint_pos_limits=[(-0.58, 0.3), (-1.57, 2.09), (-0.78, 0.78), (0.0, 2.56), (-1.05, 0.7), (-0.42, 0.42),
                          (-0.58, 0.3), (-2.09, 1.57), (-0.78, 0.78), (-2.56, 0.0), (-0.7, 1.05), (-0.42, 0.42)],

        rand_motor_scale_range=(0.8, 1.2),
        torque_lut_dir=_TORQUE_LUT_DIR,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.

        Walker has NO passive joints — all 12 lower-body joints are motor joints.
        Policy obs per frame: 3+3+3+2+12+12+12 = 47D
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.4, n_max=0.4)) # 0.2 -> 0.4
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1)) # 0.05 -> 0.1
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 1.0e-3},
        )
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 1.0e-3},
        )
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)
            },
            noise=Unoise(n_min=-1.0, n_max=1.0),
            clip=(-30.0, 30.0),
            scale=1.0 / 30.0,
        )
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10

    @configclass
    class TargetCfg(ObsGroup):
        """Target observation group for PPOFuture."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 1.0e-3},
        )
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 1.0e-3},
        )
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)},
        )
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)},
            clip=(-30.0, 30.0),
            scale=1.0 / 30.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})

        physics_material = ObsTerm(func=mdp.physics_material_sd)
        base_mass_delta = ObsTerm(func=mdp.base_link_mass_delta)
        base_com_offset = ObsTerm(func=mdp.base_link_com_offset)
        motor_armature_stats = ObsTerm(func=mdp.motor_joint_armature_stats, params={"joint_name_keys": ".*_Joint"})
        motor_damping_stats = ObsTerm(func=mdp.motor_joint_damping_stats, params={"joint_name_keys": ".*_Joint"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observation group for PPOFuture."""

        gt_vel3 = ObsTerm(func=mdp.base_vel_xy_yawrate)
        gt_foot_force6 = ObsTerm(
            func=mdp.feet_contact_forces_l_r,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["L_Foot_Link", "R_Foot_Link"],
                    preserve_order=True,
                )
            },
            scale=(
                1.0 / 300.0, 1.0 / 300.0, 1.0 / 600.0,
                1.0 / 300.0, 1.0 / 300.0, 1.0 / 600.0,
            ),
        )

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 1.0e-3},
        )
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 1.0e-3},
        )
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)},
        )
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)},
            clip=(-30.0, 30.0),
            scale=1.0 / 30.0,
        )
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        physics_material = ObsTerm(func=mdp.physics_material_sd)
        base_mass_delta = ObsTerm(func=mdp.base_link_mass_delta)
        base_com_offset = ObsTerm(func=mdp.base_link_com_offset)
        motor_armature_stats = ObsTerm(func=mdp.motor_joint_armature_stats, params={"joint_name_keys": ".*_Joint"})
        motor_damping_stats = ObsTerm(func=mdp.motor_joint_damping_stats, params={"joint_name_keys": ".*_Joint"})

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

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material_and_cache,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.2, 1.4),
            "dynamic_friction_range": (0.2, 1.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 10.0), # -10.0, 20.0
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "z": (-0.10, 0.10),
            },
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)},
    )
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_Joint"),
            "friction_distribution_params": (0.0, 0.0),
            "armature_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )
    randomize_damping = EventTerm(
        func=mdp.randomize_actuator_gains, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_Joint"),
            "stiffness_distribution_params": (0.0, 0.0),
            "damping_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 8.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class WalkerViewerCfg(ViewerCfg):
    resolution: tuple[int, int] = (1920, 1080)
    eye: tuple[float, float, float] = (10, 10, 10)


@configclass
class P73RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KangarooRewards = KangarooRewards()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    viewer: WalkerViewerCfg = WalkerViewerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005

        # Scene
        self.scene.num_envs = 4096
        self.scene.robot = P73_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.height_scanner.offset.pos = (0.0, 0.0, 0.0)
        self.scene.height_scanner.pattern_cfg.size = [1.2, 1.2]
        self.scene.height_scanner.pattern_cfg.resolution = 0.15
        self.scene.height_scanner.debug_vis = True

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # Phase-gait hyperparameters
        gait_period_steps = 50
        gait_ds_ratio = 0.20

        self.observations.policy.gait_phase_sin.params["period_steps"] = gait_period_steps
        self.observations.policy.gait_phase_cos.params["period_steps"] = gait_period_steps
        self.observations.critic.gait_phase_sin.params["period_steps"] = gait_period_steps
        self.observations.critic.gait_phase_cos.params["period_steps"] = gait_period_steps
        self.observations.target.gait_phase_sin.params["period_steps"] = gait_period_steps
        self.observations.target.gait_phase_cos.params["period_steps"] = gait_period_steps

        self.rewards.contact_schedule_biped_ds.params["period_steps"] = gait_period_steps
        self.rewards.contact_schedule_biped_ds.params["ds_ratio"] = gait_ds_ratio
        self.rewards.swing_clearance_min_profile_penalty.params["period_steps"] = gait_period_steps
        self.rewards.swing_clearance_min_profile_penalty.params["ds_ratio"] = gait_ds_ratio
