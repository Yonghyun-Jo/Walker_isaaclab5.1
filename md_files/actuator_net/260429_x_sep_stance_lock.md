# 260429 - 정지 정렬에 x_sep(전후 평행 스탠스) 조건 추가

## 배경
- 기존 `tocabi_should_walk_stop_align` / `feet_air_time_biped` / `low_speed_feet_alignment_penalty`의 정렬 판정은
  `(x_mid, y_mid, y_sep)` 세 메트릭만 사용.
- `x_mid = 0.5*(x_L + x_R)` 만 보다 보니 한 발이 앞 / 한 발이 뒤로 어긋난 staggered stance여도
  중점(x_mid)이 0이면 정렬 통과로 간주되어 stance branch로 넘어감.
- 결과적으로 "발이 정렬되지 않았을 때 air time을 켜서 발 위치를 잡게 한다"는 본래 의도가
  staggered 자세에서는 작동하지 않음.

## 변경 내용
1. `_tocabi_compute_feet_yaw_rel_and_stance_metrics`
   - 반환 튜플에 `x_sep = |x_r[0] - x_r[1]|` 추가 (5-튜플 → 6-튜플).
2. `tocabi_should_walk_stop_align`
   - 새 파라미터 `x_sep_target_m: float = 0.0` 추가.
   - 정렬 판정에 `x_sep_err = |x_sep - x_sep_target_m| > pos_threshold_m` 조건 추가.
   - 기본값 0.0 = 양발이 base 기준 같은 x선상(평행 스탠스).
3. `feet_air_time_biped`
   - 새 파라미터 `x_sep_target_m: float = 0.0` 추가, `tocabi_should_walk_stop_align`로 그대로 전달.
4. `low_speed_feet_alignment_penalty`
   - 새 파라미터 `x_sep_target_m: float = 0.0` 추가.
   - 페널티 항에 `relu(x_sep_err - pos_threshold_m)` 추가.
5. `rough_env_cfg.py`의 `feet_air_time` 파라미터에 `"x_sep_target_m": 0.0` 명시.

## 의미
- 정지 시 `is_aligned`가 되려면 이제 다음을 모두 만족:
  - yaw 정렬 (각 발 vs base, `yaw_threshold_deg`).
  - 양발 중점이 base 원점 부근 (`mid_err <= pos_threshold_m`).
  - 측방 간격이 `stance_width_m` 부근 (`width_err <= pos_threshold_m`).
  - **(신규)** 전후 간격이 `x_sep_target_m` 부근 (`x_sep_err <= pos_threshold_m`).
- 한 발 앞 / 한 발 뒤의 staggered stance는 더 이상 정렬로 인정되지 않으며,
  walking branch가 유지되어 발을 다시 디뎌 평행 스탠스로 모이게 됨.

## 부작용 / 모니터링 포인트
- 정렬 임계가 더 빡빡해진 형태(조건 +1)이므로, `pos_threshold_m=0.02` 그대로면
  walking↔standing 게이트가 자주 깜빡일 가능성 → 학습 초반 stance branch 보상이 잘 활성화되는지 모니터.
- 필요 시 `pos_threshold_m`을 0.025~0.03으로 살짝 풀거나 `x_sep_target_m`을 자연 수렴값으로 측정 후 사용.

## 적용 브랜치
- `actuator_net_lim_nvi` (현재)
- `custom_regulate_actionrate` (동일 패치 적용 예정)
