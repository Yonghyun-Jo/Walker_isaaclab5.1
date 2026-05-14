# 260420 Reward Shaping Summary

- **Base height 비대칭화**: `body_height_tracking` (L2, -18) 제거 → `body_height_below_target_l2` (-30, target 0.89) + `body_height_target_exp` (+8, std 0.03) 양수 보상으로 교체. 크라우치만 처벌·정상 자세는 보상, 폭주 없음.
- **Knee deadband 축소**: `joint_deviation_left/right_knee` `deadband_pos/neg` 1.65 → 1.0 (무릎 굴곡 최대 ~77°). 기하학적으로 base height 낮춤 경로 차단.
- **Tier A 분할**: `action_rate_quiet` 삭제 → `action_rate_hiproll` (-0.02) + `action_rate_hipyaw` (-0.035). real p99에서 HipYaw가 HipRoll의 1.5× jerky.
- **AnklePitch/Knee 상향**: `action_rate_anklepitch` -0.12 → **-0.18** (real p99이 0.087 → 0.21로 2.3× 악화), `action_rate_knee` -0.25 → **-0.30** (여전히 최다 jerker).
- **수정 파일**: `mdp/rewards.py` (`base_height_below_target_l2`, `base_height_target_exp` 신규), `rough_env_cfg.py` (body_height term·knee deadband·action_rate tier).
