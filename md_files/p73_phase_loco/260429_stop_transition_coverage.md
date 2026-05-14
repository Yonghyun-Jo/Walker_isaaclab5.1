# 260429 Stop-Transition Coverage & Grace Step Tuning

## 동기

- 사용자 관찰: cmd→0 시 정책이 “현재 자세 그대로 즉시 정지”. 오른발이 앞으로, 왼발이 뒤로 비대칭인 상태로도 그대로 멈춰버림.
- 본래 목표: 멈춤 신호 직후 한두 번 더 발을 디뎌서 자연스럽게 안정 자세로 settle 한 뒤 정지.
- 분석 결과 ([rewards.py:1098-1164](../../source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/mdp/rewards.py#L1098)) grace 메커니즘 자체는 정상 동작하지만, **학습 커버리지가 너무 얇고**(walk→stop transition 빈도 ~4%/resample) **grace 윈도우가 7초로 과도**해서 사용자 의도와 mismatch.

## 변경

- `rel_standing_envs`: 0.02 → **0.15** (`__post_init__`에서 override). transition 빈도 약 6.5× ↑ → walk↔stop 사이클 학습량 보강.
- `grace_steps`: 350 → **150** (50Hz 기준 7s → **3s**). period_steps=40 기준 약 3.75 cycle, 발 디딤 ~7회 분량으로 정렬용으로 충분하면서 사용자 의도(짧은 settling) 와 정합.
  - 적용된 3개 reward 항: `feet_air_time_biped`, `contact_schedule_biped_ds`, `swing_clearance_min_profile_penalty`.

## 모니터링 지표

- 평가 시 cmd=0 transition 직후 양발 정지 위치 비대칭 (`|x_L_base - x_R_base|`) 의 평균/분산 — 줄어드는지 확인.
- `feet_air_time_biped`, `contact_schedule_biped_ds` reward curve 가 standing 비중 증가로 살짝 내려갈 수 있음. tracking reward 와 fall rate 영향 체크.
- transition 직후 1~3s 구간 평균 stride 횟수 (현재는 사실상 0).

## 향후 후속 후보 (미적용)

- `low_speed_feet_alignment_penalty` 등록 + `|x_L - x_R|` x-대칭 항 추가 (standing branch 자세 shaping).
- `tocabi_should_walk_stop_align`의 `pos_bad`에 x-대칭 조건 추가 → 비대칭 시 standing branch 진입 자체 차단.
- `cmd_zero_max` reward·observation 통일 (0.0 → 0.01).

## 수정 파일

- `source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/rough_env_cfg.py`
  - `feet_air_time` (L428), `contact_schedule_biped_ds` (L557), `swing_clearance_min_profile_penalty` (L584): `grace_steps` 350 → 150.
  - `__post_init__` (L949): `rel_standing_envs = 0.15` 추가.

## 적용 브랜치

- `actuator_net_lim_nvi`
- `custom_regulate_actionrate`
