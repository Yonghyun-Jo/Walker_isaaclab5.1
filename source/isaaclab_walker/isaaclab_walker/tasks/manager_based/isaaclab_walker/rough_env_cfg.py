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

# Joint names
_LOWER_JOINT_NAMES = [
    "L_HipRoll_Joint", "L_HipPitch_Joint", "L_HipYaw_Joint",
    "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
    "R_HipRoll_Joint", "R_HipPitch_Joint", "R_HipYaw_Joint",
    "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
]
_UPPER_JOINT_NAMES = ["WaistYaw_Joint"]


@configclass
class KangarooRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # === Velocity tracking ===

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
        weight=-12.0,
        params={"target_height": 0.89},
    )

    # === Stability ===

    flat_base_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-20.0,
    )

    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.2,
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

    # === Joint limits ===

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
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*")
        },
    )

    # === Joint deviation ===

    joint_deviation_hipyaw = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint_conditional,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "R_HipYaw_Joint"]),
            "command_name": "base_velocity",
            "command_index": 2,
            "command_threshold": 0.01,
            "base_vel_index": 2,
            "base_vel_threshold": 0.07,
            "target_offset": 0.0,
            "deadband_pos_active": 0.0,
            "deadband_neg_active": 0.0,
            "deadband_pos_inactive": 0.0,
            "deadband_neg_inactive": 0.0,
            "stiffness_pos": 1.5,
            "stiffness_neg": 1.5,
        },
    )

    joint_deviation_hiproll = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint_conditional,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipRoll_Joint", "R_HipRoll_Joint"]),
            "command_name": "base_velocity",
            "command_index": 1,
            "command_threshold": 0.01,
            "base_vel_index": 1,
            "base_vel_threshold": 0.07,
            "target_offset": 0.0,
            "deadband_pos_active": 0.0,
            "deadband_neg_active": 0.0,
            "deadband_pos_inactive": 0.0,
            "deadband_neg_inactive": 0.0,
            "stiffness_pos": 1.5,
            "stiffness_neg": 2.5,
        },
    )

    # Knee: L/R axis mirrored. Allowed q ~ [0.15, 2.0] (L), [-2.0, -0.15] (R).
    joint_deviation_left_knee = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_Knee_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 1.65,
            "deadband_neg": 0.20,
            "stiffness_pos": 0.2,
            "stiffness_neg": 5.0,
        },
    )

    joint_deviation_right_knee = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["R_Knee_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.20,
            "deadband_neg": 1.65,
            "stiffness_pos": 5.0,
            "stiffness_neg": 0.2,
        },
    )

    # AnklePitch: L/R mirrored. Allowed q ~ [-0.41, 0.41].
    joint_deviation_left_anklepitch = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_AnklePitch_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.58,
            "deadband_neg": 0.24,
            "stiffness_pos": 3.0,
            "stiffness_neg": 3.0,
        },
    )

    joint_deviation_right_anklepitch = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["R_AnklePitch_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.24,
            "deadband_neg": 0.58,
            "stiffness_pos": 3.0,
            "stiffness_neg": 3.0,
        },
    )

    joint_deviation_ankleroll = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_AnkleRoll_Joint", "R_AnkleRoll_Joint"]),
            "target_offset": 0.0,
            "deadband": 0.2,
            "stiffness": 3.0,
        },
    )

    # === Smoothness & energy ===

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )

    action_accel_l2 = RewTerm(
        func=mdp.action_accel_l2,
        weight=-0.004,
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5.0e-7,
    )

    # Extra penalty on distal joints with higher qdd.
    dof_acc_distal_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_Knee_Joint",
            ])
        },
    )

    # Torque L2, grouped by tau_max (w ~ 1/tau_max^2). Baseline: Knee/HipPitch.
    dof_torques_hiproll_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "L_HipRoll_Joint", "R_HipRoll_Joint",
            ])
        },
    )

    dof_torques_mid_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_HipPitch_Joint", ".*_Knee_Joint",
            ])
        },
    )

    dof_torques_hipyaw_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-3.0e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipYaw_Joint"])
        },
    )

    dof_torques_ankle_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_AnklePitch_Joint", ".*_AnkleRoll_Joint",
            ])
        },
    )

    # Soft penalty active only when |tau|/tau_max > soft_ratio.
    dof_torques_limit_soft = RewTerm(
        func=mdp.joint_torques_limit_soft_l2,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_HipPitch_Joint", ".*_HipYaw_Joint",
                ".*_Knee_Joint", ".*_AnklePitch_Joint", ".*_AnkleRoll_Joint",
            ]),
            "soft_ratio": 0.8,
        },
    )


    # === Feet & contact ===

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.45,
            "velocity_threshold": 0.01,
            "stop_cmd_vel_max": 0.07,
            "yaw_threshold_deg": 5.0,
            "pos_threshold_m": 0.02,
            "stance_width_m": 0.205,
            "contact_threshold": 5.0,
            "cmd_zero_max": 0.0,
            "grace_steps": 250,
            "push_suppress_threshold": 1.0,
            "push_activates": True,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
        },
    )

    feet_ground_parallel = RewTerm(
        func=mdp.feet_ground_parallel,
        weight=-10.0,
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

    feet_yaw_symmetry_about_base = RewTerm(
        func=mdp.feet_yaw_symmetry_about_base_l2,
        weight=-3.0,
        params={
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
        weight=-15.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
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
            "cmd_zero_max": 0.0,
            "grace_steps": 250,
            "push_suppress_threshold": 1.0,
            "push_activates": True,
        },
    )

    swing_clearance_min_profile_penalty = RewTerm(
        func=mdp.swing_clearance_min_profile_penalty,
        weight=-60.0,
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
            "clearance_height": 0.13,
            "profile_offset": 0.0075,
            "contact_threshold": 5.0,
            "cmd_zero_max": 0.0,
            "grace_steps": 250,
            "push_suppress_threshold": 1.0,
        },
    )

    spring_compliance_pos_match = RewTerm(
        func=mdp.spring_compliance_pos_match_reward,
        weight=-0.0,
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
            "velocity_threshold": 0.0,
            "cmd_vel_eps": 0.0,
        },
    )

    feet_height_symmetry_penalty = RewTerm(
        func=mdp.feet_height_symmetry_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "command_name": "base_velocity",
            "velocity_threshold": 0.0,
            "contact_threshold": 5.0,
        },
    )

    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty_gated,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "command_name": "base_velocity",
            "velocity_threshold": 0.0,
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
        # URDF joint limits, L leg then R leg.
        joint_pos_limits=[(-0.58, 0.3), (-1.57, 2.09), (-0.78, 0.78), (0.0, 2.56), (-1.05, 0.7), (-0.42, 0.42),
                          (-0.58, 0.3), (-2.09, 1.57), (-0.78, 0.78), (-2.56, 0.0), (-0.7, 1.05), (-0.42, 0.42)],
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.4, n_max=0.4))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
        )
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
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
            noise=Unoise(n_min=-1.5, n_max=1.5),
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
        """Target observations for PPOFuture."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
        )
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
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
        push_force = ObsTerm(func=mdp.persistent_push_force, scale=1.0 / 100.0)
        command_delay = ObsTerm(
            func=mdp.actuator_command_delay,
            params={"actuator_name": "walker_motors", "max_delay": 2},
        )
        motor_strength = ObsTerm(func=mdp.actuator_motor_strength, params={"actuator_name": "walker_motors"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for PPOFuture."""

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
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
        )
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
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
        command_delay = ObsTerm(
            func=mdp.actuator_command_delay,
            params={"actuator_name": "walker_motors", "max_delay": 2},
        )
        motor_strength = ObsTerm(func=mdp.actuator_motor_strength, params={"actuator_name": "walker_motors"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

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
            "mass_distribution_params": (-5.0, 10.0),
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
        func=mdp.push_by_persistent_force,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "force_range": {
                "x": (-60.0, 60.0),
                "y": (-60.0, 60.0),
            },
            "duration_s": (0.5, 2.0),
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

        # Gait phase
        gait_period_steps = 60
        gait_ds_ratio = 0.25

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
