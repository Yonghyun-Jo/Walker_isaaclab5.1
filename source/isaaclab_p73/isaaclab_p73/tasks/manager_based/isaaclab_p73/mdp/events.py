# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for P73 tasks (IsaacLab 5.1).

This file adds a startup physics material randomization term that also caches
per-environment (static_friction, dynamic_friction) so it can be used as a
2D observation term (TargetCfg).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_rigid_body_material_and_cache(ManagerTermBase):  # noqa: N801
    """Startup physics material randomization with a 2D cache for Target observation.

    Wraps IsaacLab's `randomize_rigid_body_material` event term but additionally stores:
      env._physics_material_sd: (num_envs, 2) = [static_friction, dynamic_friction]

    Cached values are the mean over shapes in the selected asset for each environment.
    """

    def __init__(self, cfg, env: "ManagerBasedEnv"):
        super().__init__(cfg=cfg, env=env)
        # defer import to avoid heavy dependency import at module import time
        from isaaclab.envs.mdp.events import randomize_rigid_body_material as _Base

        self._base = _Base(cast(Any, cfg), env)
        self._env = env

        # initialize cache so observation is always defined
        self._env.__dict__["_physics_material_sd"] = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor | None,
        static_friction_range,
        dynamic_friction_range,
        restitution_range,
        num_buckets,
        asset_cfg,
        make_consistent: bool = False,
    ):
        # Apply base randomization
        self._base(
            env=env,
            env_ids=env_ids,
            static_friction_range=static_friction_range,
            dynamic_friction_range=dynamic_friction_range,
            restitution_range=restitution_range,
            num_buckets=num_buckets,
            asset_cfg=asset_cfg,
            make_consistent=make_consistent,
        )

        # Read back and cache mean(static,dynamic) per env.
        # get_material_properties(): (num_envs, total_shapes, 3) on CPU
        materials = self._base.asset.root_physx_view.get_material_properties()
        if env_ids is None:
            env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids_cpu = env_ids.cpu()

        sd_mean_cpu = materials[env_ids_cpu, :, :2].mean(dim=1)  # (num_env_ids, 2)

        cache = self._env.__dict__.get("_physics_material_sd", None)
        if not isinstance(cache, torch.Tensor) or cache.shape != (env.num_envs, 2):
            cache = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)
            self._env.__dict__["_physics_material_sd"] = cache

        cache[env_ids_cpu.to(device=env.device, dtype=torch.long)] = sd_mean_cpu.to(device=env.device, dtype=torch.float32)

