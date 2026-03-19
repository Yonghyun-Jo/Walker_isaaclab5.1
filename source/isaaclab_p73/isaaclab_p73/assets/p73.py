import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_p73.assets import P73_ASSETS_DATA_DIR

import os
init = {
    "L_HipRoll_motor": 0.0,
    "R_HipRoll_motor": 0.0,
    "WaistYaw_motor": 0.0,
    "L_HipPitch_motor": 0.36116683,
    "R_HipPitch_motor": -0.3608843,
    "WaistRoll_motor": 0.0,
    "L_HipYaw_motor": 0.0,
    "R_HipYaw_motor": 0.0,
    "WaistPitch_motor": 0.0,
    "L_KneePitch_Joint": 0.7571845,
    "L_KneeUpper_motor": 0.7855673,
    "R_KneePitch_Joint": -0.75670075,
    "R_KneeUpper_motor": -0.7852487,
    "L_ShoulderPitch_motor": 0.3,
    "R_ShoulderPitch_motor": -0.3,
    "L_AnklePitch_Joint": -0.42094237,
    "L_AnkleM1_motor": -0.32566407,
    "L_AnkleM2_motor": 0.32564506,
    "L_KneeLower_Joint": -0.5823492,
    "R_AnklePitch_Joint": 0.4214624,
    "R_AnkleM1_motor": 0.3258957,
    "R_AnkleM2_motor": -0.32560477,
    "R_KneeLower_Joint": 0.5819463,
    "L_ShoulderRoll_motor": 0.0,
    "R_ShoulderRoll_motor": 0.0,
    "L_AnkleRoll_Joint": -0.00047307153,
    "L_Achilles1_Ball_Joint:0": 0.20761746,
    "L_Achilles1_Ball_Joint:1": 0.33107916,
    "L_Achilles1_Ball_Joint:2": 1.2549043,
    "L_Achilles2_Ball_Joint:0": -0.13269848,
    "L_Achilles2_Ball_Joint:1": 0.3535626,
    "L_Achilles2_Ball_Joint:2": -0.89318097,
    "R_AnkleRoll_Joint": 0.0015726449,
    "R_Achilles1_Ball_Joint:0": -0.5091875,
    "R_Achilles1_Ball_Joint:1": 0.13815664,
    "R_Achilles1_Ball_Joint:2": -3.0788164,
    "R_Achilles2_Ball_Joint:0": 0.22174628,
    "R_Achilles2_Ball_Joint:1": 0.34956872,
    "R_Achilles2_Ball_Joint:2": 1.4934522,
    "L_ShoulderYaw_motor": 0.0,
    "R_ShoulderYaw_motor": 0.0,
    "L_ElbowPitch_motor": -1.27,
    "R_ElbowPitch_motor": 1.27,
    "L_WristYaw_motor": 0.0,
    "R_WristYaw_motor": 0.0,
    "L_WristPitch_motor": 0.0,
    "R_WristPitch_motor": 0.0,
    "L_WristRoll_motor": 0.0,
    "R_WristRoll_motor": 0.0,
}


P73_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{P73_ASSETS_DATA_DIR}/p73_lower_4bar.usd",
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
            enabled_self_collisions= True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.895),
        joint_pos=init,
        # joint_pos={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "p73_motors": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_motor",
            ],
            stiffness={
                ".*_motor": 0.0,
            },
            damping={
                ".*_motor": 0.0,
            },
            effort_limit_sim={
                ".*_HipRoll_motor": 264.0, # 80:1
                ".*_HipPitch_motor": 165.0,
                ".*_HipYaw_motor": 69.0,
                ".*_KneeUpper_motor": 165.0,
                ".*_AnkleM1_motor": 69.0,  # 165.0
                ".*_AnkleM2_motor": 69.0,  # 165.0

                "WaistYaw_motor": 110.0,
                "WaistRoll_motor": 110.0,
                "WaistPitch_motor": 110.0,

                ".*_ShoulderPitch_motor": 110.0,
                ".*_ShoulderRoll_motor": 110.0,
                ".*_ShoulderYaw_motor": 45.6,
                ".*_ElbowPitch_motor": 45.6,
                ".*_WristYaw_motor": 23.2,
                ".*_WristPitch_motor": 23.2,
                ".*_WristRoll_motor": 23.2,
            },
            velocity_limit_sim={
                ".*_HipRoll_motor": 4.86, # 80:1
                ".*_HipPitch_motor": 7.78, # 4.03
                ".*_HipYaw_motor": 11.81, # 4.03
                ".*_KneeUpper_motor": 7.78, # 4.03
                ".*_AnkleM1_motor": 11.81, # 4.03
                ".*_AnkleM2_motor": 11.81, # 4.03

                "WaistYaw_motor": 4.03,
                "WaistRoll_motor": 4.03,
                "WaistPitch_motor": 4.03,

                ".*_ShoulderPitch_motor": 4.03,
                ".*_ShoulderRoll_motor": 4.03,
                ".*_ShoulderYaw_motor": 4.03,
                ".*_ElbowPitch_motor": 4.03,
                ".*_WristYaw_motor": 4.03,
                ".*_WristPitch_motor": 4.03,
                ".*_WristRoll_motor": 4.03,
            },
            armature={
                "L_HipRoll_motor": 0.96, # 80:1
                "L_HipPitch_motor": 0.375, #1.5
                "L_HipYaw_motor": 0.0625, #0.25
                "L_KneeUpper_motor": 0.375, #1.5
                "L_AnkleM1_motor": 0.0625, #0.13
                "L_AnkleM2_motor": 0.0625, #0.13

                "R_HipRoll_motor": 0.375, #1.5
                "R_HipPitch_motor": 0.375, #1.5
                "R_HipYaw_motor": 0.0625, #0.25
                "R_KneeUpper_motor": 0.375, #1.5
                "R_AnkleM1_motor": 0.0625, #0.13
                "R_AnkleM2_motor": 0.0625, #0.13

                "WaistYaw_motor": 0.25,
                "WaistRoll_motor": 0.25,
                "WaistPitch_motor": 0.25,

                "L_ShoulderPitch_motor": 0.25,
                "L_ShoulderRoll_motor": 0.25,
                "L_ShoulderYaw_motor": 0.25,
                "L_ElbowPitch_motor": 0.13,
                "L_WristYaw_motor": 0.038,
                "L_WristPitch_motor": 0.038,
                "L_WristRoll_motor": 0.038,

                "R_ShoulderPitch_motor": 0.25,
                "R_ShoulderRoll_motor": 0.25,
                "R_ShoulderYaw_motor": 0.25,
                "R_ElbowPitch_motor": 0.13,
                "R_WristYaw_motor": 0.038,
                "R_WristPitch_motor": 0.038,
                "R_WristRoll_motor": 0.038,
            },
        ),
    },
)


# damping = [
#     0.0248, 0.0248, 0.0248, 0.0248, 0.0248, 0.0161,
#     0.0248, 0.0248, 0.0248, 0.0248, 0.0248, 0.0161,
        
#     0.0417, 0.0417, 0.0417,

#     0.0148, 0.0148, 0.0148, 0.0148, 0.0047, 0.0047, 0.0029, 0.0029,
        
#     0.0029, 0.0029,
        
#     0.0148, 0.0148, 0.0148, 0.0148, 0.0047, 0.0047, 0.0029, 0.0029,
# ]