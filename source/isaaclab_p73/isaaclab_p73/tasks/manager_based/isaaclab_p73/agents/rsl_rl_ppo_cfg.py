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
    """PPOFuture runner config (TOCABI-style latent supervision) for P73.

    Structure parity with TOCABI framework:
    - history encoder -> latent
    - latent prefix supervision (vel/foot/future)
    - O(t+1) target encoder matching

    P73 differences:
    - lower-body 12D actuation policy
    - height map usage in target/critic privileged features
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "p73_flat_CF"
    empirical_normalization = True
    logger = "wandb"
    wandb_project = "p73_flat_CF"

    # Actor-Critic with encoder + target-encoder
    # Single source of truth for architecture hyper-parameters.
    policy = P73RslRlPpoActorCriticFutureCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        # Encoder / latent (TOCABI-style)
        encoder_hidden_dims=[512, 512, 256],
        latent_dim=64,
        # P73 single-frame policy dim:
        # base_ang_vel(3)+projected_gravity(3)+cmd(3)+gait_phase(2)+motor_pos(12)+motor_vel(12)
        # +measured_pos(6)+measured_vel(6)+actions(12) = 59
        # = 59D after adding gait_phase_sin/cos (2D)
        num_single_obs=59,
        # O(t+1) target features (includes gait_phase_sin/cos): 150D total
        target_obs_dim=150,
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
        target_obs_dim=150,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=1.0,
            data_augmentation_func=(
                "isaaclab_p73.tasks.manager_based.isaaclab_p73.mdp.symmetry:"
                "p73_data_augmentation_lowerbody_mirror"
            ),
        ),
    )
