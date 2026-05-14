# Real → Sim Analysis (2026-04-20)

실로봇 로그 6개를 합산(≈316k rows, ≈85% moving)해 per-joint 통계를 뽑고,
`rough_env_cfg.py`에서 역-반영 가능한 항목을 정리.

## 1. 데이터 개요

**소스 파일** (`/home/piene/ros2_ws/src/p73_cc/logs/`)
- `realrobot_260420_173557.csv` (77k)
- `realrobot_260420_175257.csv` (41k)
- `realrobot_260420_183206.csv` (56k)
- `realrobot_260420_183421.csv` (60k)
- `realrobot_260420_183702.csv` (41k)
- `realrobot_260420_185028.csv` (41k)

**샘플링**: 데이터 1 kHz, policy 50 Hz. 아래 |Δa|/|Δ²a| 통계는 policy rate(20-stride 다운샘플), qdot/qddot/tau는 1 kHz. moving = `|cmd_xy|>0.07 OR |v_xy|>0.1`.

**동작 영역**
| 지표 | p50 | p90 | max |
|---|---|---|---|
| cmd_xy 크기 | 0.10 | 0.20 | 0.80 |
| lin_vel_xy | 0.17 | 0.35 | 1.10 |
| ang_vel_xyz | 0.29 | 0.71 | 4.42 |
| proj_grav_xy (base 기울기) | 0.031 | 0.057 | 0.342 |

→ 저속(0.1~0.2 m/s) 중심, 최대 ~0.8. base 기울기는 대체로 <3.3°.

## 2. 핵심 발견

### 2-1. Ankle 토크 포화 — *가장 심각한 real2sim 신호*

**|τ|/τ_max ≥ 0.9 체류 비율 (moving 중)**

| Joint | L | R |
|---|---|---|
| HipRoll (352 Nm) | 0.02% | 0.00% |
| HipPitch (220) | 4.14% | 6.70% |
| HipYaw (95) | **14.57%** | 6.07% |
| Knee (220) | **17.55%** | **21.38%** |
| AnklePitch (95) | **21.63%** | **24.20%** |
| AnkleRoll (95) | **24.47%** | **23.43%** |

현재 `dof_torques_limit_soft` (soft_ratio=0.8, weight=-1.5) 와
`dof_torques_anklepitch_limit_soft` (0.7, -2.5)로는 부족. 실로봇이 4개 원위(Ankle*/Knee) 모두 steady-state ≥20% 포화. 시뮬에서 이 신호를 더 강하게 주어야 real에서 덜 포화됨.

### 2-2. Action rate |Δa| 현황 vs 현재 Tier 가중치

| Joint | |Δa| p99 | 현재 Tier | 현재 weight | 관측 |
|---|---|---|---|---|
| HipRoll L/R | 0.041 / 0.040 | A | -0.02 | 제일 조용 ✓ |
| HipYaw L/R | 0.065 / 0.056 | A | -0.02 | HipRoll 대비 1.5× |
| AnkleRoll L/R | 0.067 / 0.074 | B | -0.06 | 타당 |
| HipPitch L/R | 0.115 / 0.117 | B | -0.06 | 타당 |
| Knee L/R | **0.248 / 0.195** | D | -0.25 | 여전히 가장 거침 |
| AnklePitch L/R | **0.202 / 0.211** | C | -0.12 | **예전 0.087 → 지금 0.21 (2.3×)** |

→ **AnklePitch 가 Knee 수준으로 거칠어짐**. Tier C weight 부족.

### 2-3. Action acceleration |Δ²a|

| Joint | p99 (L/R) | 비고 |
|---|---|---|
| AnklePitch | **0.211 / 0.300** | 압도적 1위, 발목 ID 진동 가능성 |
| Knee | 0.133 / 0.160 | 2위 |
| HipPitch | 0.068 / 0.085 | |
| AnkleRoll | 0.058 / 0.053 | |
| HipYaw / HipRoll | ≤ 0.055 | |

현재 `action_accel_l2`는 **모든 joint 단일 weight=-0.004**. AnklePitch/Knee 한정 가중 추가가 효과적.

### 2-4. 관절 가속도 |qddot| (1 kHz FD)

| Joint | p99 (L/R, rad/s²) | 비고 |
|---|---|---|
| AnkleRoll | **195 / 216** | 최고 |
| Knee | 184 / 127 (L 비대칭↑) | |
| AnklePitch | 168 / 148 | |
| HipPitch | 71 / 68 | |
| HipYaw | 91 / 76 | |
| HipRoll | 24 / 22 | |

현재 `dof_acc_distal_l2` (Knee/AnklePitch/AnkleRoll, -1.5e-6) 는 세 원위 joint 동일 가중. 실측은 AnkleRoll이 가장 큼 → 추가 세분화 가능.

### 2-5. L/R 비대칭

- **L_HipYaw 14.6% vs R_HipYaw 6.1%** (>0.9 tau_max) → 2.4배 차이
- L_Knee |Δa| p99 0.25 vs R_Knee 0.20 (25% 차이)
- R_AnklePitch |Δ²a| p99 0.30 vs L 0.21 (43% 차이)

policy가 좌/우 다른 전략 사용 중. 대칭화 제약 필요 (symmetry loss 혹은 mirror data augmentation).

## 3. 제안 — `rough_env_cfg.py` 수정안

우선순위별(효과 큰 순). 각 항목에 현재 → 제안 weight.

### P0. Ankle 토크 soft-limit 재설정 (real 포화 직격)

**이유**: 원위 4개 joint가 ≥90% tau_max에서 20%+ 시간 체류. `soft_ratio=0.8`은 활성 구간을 놓친다.

변경안 a — AnklePitch 전용 강화 & soft_ratio 낮춤:
```python
dof_torques_anklepitch_limit_soft:
    weight: -2.5 → -5.0
    soft_ratio: 0.7 → 0.6
```

변경안 b — AnkleRoll 전용 항 신설:
```python
dof_torques_ankleroll_limit_soft (신규):
    weight: -4.0
    soft_ratio: 0.6
    joint_names: [".*_AnkleRoll_Joint"]
```

변경안 c — 기존 `dof_torques_limit_soft`는 AnklePitch/AnkleRoll 빼고 Knee/HipPitch/HipYaw만, soft_ratio=0.7로 당김:
```python
dof_torques_limit_soft:
    weight: -1.5 → -2.5
    soft_ratio: 0.8 → 0.7
    joint_names: [".*_HipPitch_Joint", ".*_HipYaw_Joint", ".*_Knee_Joint"]
```

### P1. Action rate Tier 재조정 (AnklePitch 급상승 반영)

```python
action_rate_anklepitch:  # Tier C
    weight: -0.12 → -0.22   # p99가 2.3× 뛰었으므로 비례 증가

action_rate_knee:        # Tier D
    weight: -0.25 → -0.30   # 소폭 증가

# Tier A 분리:
action_rate_hiproll (신규): weight -0.01, indices [0, 6]
action_rate_hipyaw  (신규): weight -0.04, indices [2, 8]   # HipRoll 대비 1.5× 거침
```

### P2. Action accel per-joint 보강

현재 전 joint 공통 `action_accel_l2 = -0.004` 유지하되, 추가로:
```python
action_accel_anklepitch (신규):  # |Δ²a| p99가 압도적 1위
    weight: -0.015
    indices: [4, 10]

action_accel_knee (신규):
    weight: -0.008
    indices: [3, 9]
```

### P3. qddot 세분화 (AnkleRoll 분리)

`dof_acc_distal_l2`를 둘로 나누어 AnkleRoll에 더 큰 가중:
```python
dof_acc_knee_anklepitch_l2:
    weight: -1.5e-6 (기존 유지)
    joint_names: [".*_Knee_Joint", ".*_AnklePitch_Joint"]

dof_acc_ankleroll_l2 (신규):
    weight: -3.0e-6              # qddot p99 ~210, 1.5× 더 크므로 2× 가중
    joint_names: [".*_AnkleRoll_Joint"]
```

### P4. L/R 대칭성 제약 (선택)

HipYaw/Knee/AnklePitch L/R 비대칭이 커서 real 성능에 직접 영향. 이미 `symmetry.py`가 있으니:
- symmetry loss 가중을 키우거나
- `action_rate_asymmetry_l2` 신규: `(a_L - mirror(a_R))²` 계열 reward 추가

현재는 별도 항목 없이 symmetry_data_augmentation 의존인지 확인 필요. (`source/.../mdp/symmetry.py` 참조)

## 4. 변경을 *하지 않을* 항목

- `body_height_*`, knee deadband: 방금(오늘) 건드렸으니 1차 학습 결과 보고 결정.
- `feet_air_time`, `swing_clearance_min_profile_penalty`: real 로그로 직접 검증 어려움(발 접지 force 따로 필요).
- `flat_base_orientation_l2`: proj_grav p90=0.057 (≈3.3°), 현재 -20 weight 적절.
- `termination_penalty`, `track_lin_vel_*`: 보행 품질 관련, real 데이터로 역산 불가.

## 5. 적용 순서 제안

1. **P0만 먼저 적용** (Ankle 토크 soft-limit 3개). 가장 큰 real2sim gap.
2. 11k iter 돌려 테스트. plot_motor_torque_12d 로 sim의 tau 포화 프로파일 확인.
3. real과 비교해 여전히 포화 비중이 높으면 P1 (action rate Tier 재조정) 추가.
4. P2/P3은 smoothness 미세튜닝 단계에서.
5. P4는 sim 성능 수렴 후 real deploy 전 마지막 단계.

## 6. 재현

```bash
python /tmp/analyze_real_260420.py
```
(스크립트는 `/tmp/analyze_real_260420.py` — 6개 CSV 하드코딩.
재사용 원하면 `md_files/real2sim/analyze_real.py`로 복사 권장.)
