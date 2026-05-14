# 260429 — Misalign-Activated Schedule (contact_schedule / swing_clearance)

## 변경 요약

`feet_air_time_biped`은 이미 `tocabi_should_walk_stop_align` 통해 *stop intent + 정렬 불량* 상황에서
"걷기 분기"로 들어가 발 공중 보상을 주고 있었지만, `contact_schedule_reward_biped_ds` 와
`swing_clearance_min_profile_penalty` 두 리워드는 cmd≈0 이면 무조건 OFF 상태였다.
그 결과 멈췄을 때 발이 정렬 위치에서 벗어나 있어도 phase-based 스텝 보상/스윙 클리어런스 페널티가
모두 0이 되어, 정책이 어색한 자세로 그대로 멈춰버리는 케이스가 있었다.

이 변경으로 두 리워드도 *멈춤 + 정렬 불량* 상황에서 스케줄을 강제로 켜서, 발이 원하는 평행 자세에
들어올 때까지 계속 발을 구르도록 한다.

## 구현

`mdp/rewards.py::_tocabi_is_schedule_active_from_cmd` 에 옵션 트리거 경로 추가:

- `misalign_activates: bool` 플래그
- `align_sensor_cfg`, `align_asset_cfg`, `stop_cmd_vel_max`,
  `yaw_threshold_deg`, `pos_threshold_m`, `stance_width_m`, `x_sep_target_m`, `align_contact_threshold`

활성 시:
1. `_tocabi_compute_feet_yaw_rel_and_stance_metrics` 로 yaw/stance 메트릭 계산
2. `align_bad = yaw_bad ∨ pos_bad`  (각 임계값과 비교)
3. `stop_intent = cmd_vel < stop_cmd_vel_max`
4. `misalign_trigger = stop_intent ∧ align_bad ∧ ¬pushed`
5. `trigger |= misalign_trigger` (push 모드와 OR 결합)

푸시 중에는 기존 `push_activates` / `push_suppress` 의미가 우선되도록
`& ~pushed` 로 마스킹 (suppress 모드의 안전성 유지).

`contact_schedule_reward_biped_ds` 와 `swing_clearance_min_profile_penalty` 양쪽에
동일한 misalign 파라미터 셋을 노출 (config에서 켜고 끌 수 있게).

## Config 변경 (`rough_env_cfg.py`)

`feet_air_time_biped` 와 동일한 정렬 임계값을 두 리워드에도 적용:

| Param              | Value |
|--------------------|-------|
| `misalign_activates` | True  |
| `stop_cmd_vel_max`   | 0.07  |
| `yaw_threshold_deg`  | 5.0   |
| `pos_threshold_m`    | 0.02  |
| `stance_width_m`     | 0.205 |
| `x_sep_target_m`     | 0.0   |

추가로 `contact_schedule_biped_ds` 에 `asset_cfg=SceneEntityCfg("robot")` 를 명시적으로 추가
(misalign 메트릭 계산 시 robot articulation 참조에 사용).

## 기대 동작

- cmd ≠ 0: 기존과 동일 (걷기 보상 ON)
- cmd ≈ 0 ∧ 정렬 OK: 기존과 동일 (걷기 보상 OFF → 스탠스 보상 우세)
- **cmd ≈ 0 ∧ 정렬 불량: NEW** — contact_schedule/swing_clearance 모두 ON →
  로봇이 정렬을 맞출 때까지 계속 스텝
- 푸시 중: push_activates/suppress 의미 그대로 (변동 없음)

## 모니터링 지표

- `Episode_Reward/contact_schedule_biped_ds` : 정지 정렬 진행 중인 환경에서 0이 아닌 값으로 유지되는지
- `Episode_Reward/swing_clearance_min_profile_penalty` : 같은 상황에서 페널티가 살아있는지
- `Episode_Reward/feet_air_time` (정지 단계): 변경 전 대비 양상
- 정성 지표: stop 직후 발 위치가 어긋나 있을 때 robot이 추가 스텝으로 정렬을 마무리하는가

## 후속 결정 — `grace_steps` 제거

`misalign_activates` 가 *발이 정렬될 때까지* 능동적으로 schedule을 유지하므로,
기존의 고정 시간 grace 윈도우(`grace_steps=150`)는 redundant. 오히려 정렬이 이미
끝났음에도 150 step 동안 stance 보상으로의 전환이 지연되는 단점이 있었다.

세 리워드 모두 `grace_steps`를 `0` 으로 변경:
- `feet_air_time_biped` (이미 `tocabi_should_walk_stop_align`이 align_bad를 OR-in)
- `contact_schedule_biped_ds` (`misalign_activates`로 대체)
- `swing_clearance_min_profile_penalty` (`misalign_activates`로 대체)

`push_activates` 트리거는 그대로 유지 (push 회복 stepping용 별도 경로).

## 수정 파일

- `source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/mdp/rewards.py`
- `source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/rough_env_cfg.py`
