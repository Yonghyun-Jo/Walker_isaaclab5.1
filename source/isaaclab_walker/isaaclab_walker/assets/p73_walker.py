import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_walker.actuators import DelayedPDActuatorLUTCfg
from isaaclab_walker.assets import P73_ASSETS_DATA_DIR

import os

_TORQUE_LUT_DIR = os.path.join(P73_ASSETS_DATA_DIR, "p73_walker", "lut")

# Walker initial joint positions (standing pose)
# All joints are directly actuated — no passive/4-bar joints.
# Order: L leg (6) + R leg (6) + WaistYaw (1) = 13 joints
# New URDF: L/R axes are mirrored — opposite sign = same physical motion
init = {
    # Left leg  (HipRoll +X, HipPitch -Y, Knee +Y, AnklePitch +Y)
    "L_HipRoll_Joint": 0.0,
    "L_HipPitch_Joint": 0.18,
    "L_HipYaw_Joint": 0.0,
    "L_Knee_Joint": 0.35,
    "L_AnklePitch_Joint": -0.17,
    "L_AnkleRoll_Joint": 0.0,
    # Right leg (HipRoll -X, HipPitch +Y, Knee -Y, AnklePitch -Y)
    "R_HipRoll_Joint": 0.0,
    "R_HipPitch_Joint": -0.18,
    "R_HipYaw_Joint": 0.0,
    "R_Knee_Joint": -0.35,
    "R_AnklePitch_Joint": 0.17,
    "R_AnkleRoll_Joint": 0.0,
    # Waist
    "WaistYaw_Joint": 0.0,
}


P73_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{P73_ASSETS_DATA_DIR}/p73_walker/p73_walker.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.895),
        joint_pos=init,
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "walker_motors": DelayedPDActuatorLUTCfg(
            joint_names_expr=[".*_Joint"],
            # PD gains (= stiffness / damping) from setting_realrobot_PDgain.yaml
            stiffness={
                ".*_HipRoll_Joint": 1536.0,
                ".*_HipPitch_Joint": 937.5,
                ".*_HipYaw_Joint": 625.0,
                ".*_Knee_Joint": 570.08,
                ".*_AnklePitch_Joint": 463.896,
                ".*_AnkleRoll_Joint": 463.788,
                "WaistYaw_Joint": 576.0,
            },
            damping={
                ".*_HipRoll_Joint": 76.8,
                ".*_HipPitch_Joint": 37.5,
                ".*_HipYaw_Joint": 12.5,
                ".*_Knee_Joint": 28.504,
                ".*_AnklePitch_Joint": 16.0,
                ".*_AnkleRoll_Joint": 16.0,
                "WaistYaw_Joint": 19.2,
            },
            # Effort limits for explicit PD clipping
            effort_limit={
                ".*_HipRoll_Joint": 352.0,
                ".*_HipPitch_Joint": 220.0,
                ".*_HipYaw_Joint": 95.0,
                ".*_Knee_Joint": 220.0,
                ".*_AnklePitch_Joint": 95.0,
                ".*_AnkleRoll_Joint": 95.0,
                "WaistYaw_Joint": 152.0,
            },
            # Simulation-level limits (PhysX solver)
            effort_limit_sim={
                ".*_HipRoll_Joint": 352.0,
                ".*_HipPitch_Joint": 220.0,
                ".*_HipYaw_Joint": 95.0,
                ".*_Knee_Joint": 220.0,
                ".*_AnklePitch_Joint": 95.0,
                ".*_AnkleRoll_Joint": 95.0,
                "WaistYaw_Joint": 152.0,
            },
            velocity_limit_sim={
                ".*_HipRoll_Joint": 4.86,
                ".*_HipPitch_Joint": 7.78,
                ".*_HipYaw_Joint": 11.81,
                ".*_Knee_Joint": 7.78,
                ".*_AnklePitch_Joint": 11.81,
                ".*_AnkleRoll_Joint": 11.81,
                "WaistYaw_Joint": 4.03,
            },
            # Friction parameters from system identification
            viscous_friction={
                "L_HipRoll_Joint": 6.9999,
                "L_HipPitch_Joint": 6.9999,
                "L_HipYaw_Joint": 0.0009,
                "L_Knee_Joint": 6.9997,
                "L_AnklePitch_Joint": 5.1938,
                "L_AnkleRoll_Joint": 6.9999,
                "R_HipRoll_Joint": 6.9999,
                "R_HipPitch_Joint": 6.9999,
                "R_HipYaw_Joint": 3.2707,
                "R_Knee_Joint": 6.9997,
                "R_AnklePitch_Joint": 6.6494,
                "R_AnkleRoll_Joint": 6.9999,
                "WaistYaw_Joint": 4.9717,
            },
            friction={
                "L_HipRoll_Joint": 6.9999,
                "L_HipPitch_Joint": 6.9999,
                "L_HipYaw_Joint": 3.1643,
                "L_Knee_Joint": 2.1674,
                "L_AnklePitch_Joint": 2.3500,
                "L_AnkleRoll_Joint": 4.9901,
                "R_HipRoll_Joint": 6.9999,
                "R_HipPitch_Joint": 6.9999,
                "R_HipYaw_Joint": 0.0003,
                "R_Knee_Joint": 4.6539,
                "R_AnklePitch_Joint": 2.6480,
                "R_AnkleRoll_Joint": 4.2084,
                "WaistYaw_Joint": 2.3793,
            },
            dynamic_friction={
                "L_HipRoll_Joint": 6.9999,
                "L_HipPitch_Joint": 6.9999,
                "L_HipYaw_Joint": 3.1643,
                "L_Knee_Joint": 2.1674,
                "L_AnklePitch_Joint": 2.3500,
                "L_AnkleRoll_Joint": 4.9901,
                "R_HipRoll_Joint": 6.9999,
                "R_HipPitch_Joint": 6.9999,
                "R_HipYaw_Joint": 0.0003,
                "R_Knee_Joint": 4.6539,
                "R_AnklePitch_Joint": 2.6480,
                "R_AnkleRoll_Joint": 4.2084,
                "WaistYaw_Joint": 2.3793,
            },
            armature={
                "L_HipRoll_Joint": 0.96,
                "L_HipPitch_Joint": 0.375,
                "L_HipYaw_Joint": 0.0625,
                "L_Knee_Joint": 0.35630,
                "L_AnklePitch_Joint": 0.12886,
                "L_AnkleRoll_Joint": 0.12883,
                "R_HipRoll_Joint": 0.96,
                "R_HipPitch_Joint": 0.375,
                "R_HipYaw_Joint": 0.0625,
                "R_Knee_Joint": 0.35630,
                "R_AnklePitch_Joint": 0.12886,
                "R_AnkleRoll_Joint": 0.12883,
                "WaistYaw_Joint": 0.16,
            },
            # Command delay: random 0-2 physics steps per env on reset
            min_delay=0,
            max_delay=2,
            # Angle-dependent torque limits for 4-bar linkage joints
            torque_lut_dir=_TORQUE_LUT_DIR,
            # Per-step motor strength randomization
            rand_motor_scale_range=(0.8, 1.2),
        ),
    },
)
