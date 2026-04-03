# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class P73RslRlPpoActorCriticFutureCfg(RslRlPpoActorCriticCfg):
    """Config schema for TOCABI-style ActorCriticAdaptationFuture in IsaacLab 5.1.

    Keep only type schema/class binding here and define concrete hyper-parameters
    in the runner config below to avoid duplicated sources of truth.
    """

    class_name: str = "ActorCriticAdaptationFuture"
    encoder_hidden_dims: list[int] = MISSING
    latent_dim: int = MISSING
    num_single_obs: int = MISSING
    target_obs_dim: int = MISSING
    target_encoder_hidden_dims: list[int] = MISSING
    future_dim: int = MISSING


@configclass
class P73RslRlPpoFutureAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Config schema for PPOFuture auxiliary losses."""

    class_name: str = "PPOFuture"
    w_vel: float = MISSING
    w_foot: float = MISSING
    w_future: float = MISSING
    target_obs_dim: int = MISSING


@configclass
class P73RoughPPORunnerFutureCfg(RslRlOnPolicyRunnerCfg):
    """PPOFuture runner config (TOCABI-style latent supervision) for Walker.

    Structure parity with TOCABI framework:
    - history encoder -> latent
    - latent prefix supervision (vel/foot/future)
    - O(t+1) target encoder matching

    Walker differences vs p73:
    - No 4-bar linkage, all 12 lower-body joints directly actuated
    - No measured (passive) joints → policy obs 47D instead of 59D
    - No upper-body arms → only WaistYaw as PD-held upper body
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "walker_flat"
    empirical_normalization = True
    logger = "wandb"
    wandb_project = "walker_flat"

    # Actor-Critic with encoder + target-encoder
    # Single source of truth for architecture hyper-parameters.
    #
    # Walker single-frame policy dim:
    #   base_ang_vel(3) + projected_gravity(3) + cmd(3) + gait_phase(2)
    #   + motor_pos(12) + motor_vel(12) + actions(12) = 47D
    #
    # Walker target obs dim:
    #   base_ang_vel(3) + projected_gravity(3) + cmd(3) + gait_phase(2)
    #   + motor_pos(12) + motor_vel(12) + height_scan(81) + actions(12)
    #   + physics_material(2) + base_mass_delta(1) + base_com_offset(3)
    #   + motor_armature_stats(2) + motor_damping_stats(2) = 138D
    policy = P73RslRlPpoActorCriticFutureCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        # Encoder / latent
        encoder_hidden_dims=[512, 512, 256],
        latent_dim=64,
        num_single_obs=47,
        target_obs_dim=138,
        target_encoder_hidden_dims=[128],
        future_dim=30,
    )

    # PPOFuture algorithm config (aux losses + PPO core)
    algorithm = P73RslRlPpoFutureAlgorithmCfg(
        value_loss_coef=5.0,
        use_clipped_value_loss=True,
        clip_param=0.16,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Auxiliary latent supervision weights
        w_vel=1.0,
        w_foot=1.0,
        w_future=1.0,
        target_obs_dim=138,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=1.0,
            data_augmentation_func=(
                "isaaclab_walker.tasks.manager_based.isaaclab_walker.mdp.symmetry:"
                "p73_data_augmentation_lowerbody_mirror"
            ),
        ),
    )
