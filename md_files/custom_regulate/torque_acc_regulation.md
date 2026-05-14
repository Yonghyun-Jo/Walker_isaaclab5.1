# Per-Joint Torque / Acceleration Regularization — 근거·대안·학습 영향 정리

> Walker(P73, 약 60kg)의 `rough_env_cfg.py` 수정 근거를 실측 데이터와 학계 관례 기준으로 재검증한다.
> 실측 데이터 출처:
> - `/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260417_171433.csv` (57,069 steps)
> - `/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260417_171933.csv` (39,946 steps)
> - 합계 97k steps, 1 kHz 샘플링, 명령 활성 구간 62k steps (63.9%)

---

## 0. TL;DR

| 항목 | 값 | 핵심 근거 |
|---|---|---|
| `dof_torques_hiproll_l2` | **-2.0e-6** | τ_max=352 Nm, 실측 80% 포화율 ~5%에 불과 → 기존 -3.5e-6 약간 과함 |
| `dof_torques_mid_l2` (HipPitch, Knee) | **-5.5e-6** | τ_max=220 Nm, baseline, 17~20% 포화 |
| `dof_torques_small_l2` (HipYaw, Ankle*) | **-3.0e-5** | τ_max=95 Nm, (220/95)² ≈ 5.4배 → saturation-normalized 동일 |
| `dof_torques_limit_soft` | **-1.5** | `(|τ|/τ_max - 0.8)₊²` soft penalty. 포화만 직접 타겟 |
| `dof_acc_l2` | **-5.0e-7** (2×) | 기존 -2.5e-7은 Σqdd² 대비 기여 -0.009/step으로 과소 |
| `dof_acc_distal_l2` | **-1.5e-6** | Ankle/Knee qdd p99이 Hip 대비 2~4배 높음 |
| WaistYaw torque | **제거** | 실측 max 17 Nm << τ_max=152 Nm (11%) |

---

## 1. 각 수치 근거 수식 재검증

### 1.1 실측 토크 분포 (command-active, n=62,016)

| Joint | τ_max | E\[|τ|\] | p95 | p99 | max | **sat > 80%** | **sat > 90%** |
|---|---|---|---|---|---|---|---|
| L_HipRoll | 352 | 43.6 | 135 | 185 | 262 | 0.00% | 0.00% |
| R_HipRoll | 352 | 72.4 | 291 | 338 | 352 | 5.93% | 1.89% |
| L_HipPitch | 220 | 62.5 | 220 | 220 | 220 | 17.61% | 16.39% |
| R_HipPitch | 220 | 60.5 | 220 | 220 | 220 | 17.06% | 16.20% |
| L_HipYaw | 95 | 17.6 | 58 | 65 | 93 | 0.11% | 0.06% |
| R_HipYaw | 95 | 25.0 | 95 | 95 | 95 | **13.96%** | **12.22%** |
| L_Knee | 220 | 69.8 | 220 | 220 | 220 | 20.27% | 18.17% |
| R_Knee | 220 | 71.1 | 220 | 220 | 220 | 17.39% | 15.96% |
| L_AnklePitch | 95 | 51.1 | 95 | 95 | 95 | **35.61%** | **32.59%** |
| R_AnklePitch | 95 | 53.4 | 95 | 95 | 95 | **39.86%** | **34.56%** |
| L_AnkleRoll | 95 | 16.9 | 60 | 70 | 83 | 0.08% | 0.00% |
| R_AnkleRoll | 95 | 18.7 | 95 | 95 | 95 | **8.89%** | **8.31%** |
| WaistYaw | 152 | 0.9 | 1.8 | 3.7 | 17 | 0.00% | 0.00% |

포화가 심한 그룹: **AnklePitch(35~40%) > Knee/HipPitch(17~20%) > R_HipYaw(14%)**. WaistYaw는 사실상 무활동.

### 1.2 L2 가중치 유도 (saturation-normalized)

**원칙**: 각 관절이 포화 시 동일한 페널티를 받도록, `w_i · τ_i_max² = 상수`.

Baseline을 HipPitch/Knee(τ_max=220, w=5.5e-6)로 선택하면:

```
w_i = 5.5e-6 × (220 / τ_i_max)²
```

| Group | τ_max | 계산 | 결과 |
|---|---|---|---|
| HipRoll | 352 | 5.5e-6 × (220/352)² = 5.5e-6 × 0.391 | **2.15e-6** → `-2.0e-6` 선택 |
| HipPitch, Knee | 220 | baseline | **5.5e-6** |
| HipYaw, Ankle* | 95 | 5.5e-6 × (220/95)² = 5.5e-6 × 5.36 | **2.95e-5** → `-3.0e-5` 선택 |

**왜 '포화 시 동일 페널티'인가?** 관절 크기가 다른데 L2를 단일 weight로 걸면 `τ²` 항이 큰 관절(큰 모터)만 사실상 규제되고 작은 관절은 100% 포화해도 벌칙이 작다. 실측에서 Ankle(95Nm)이 40% 포화인데 기여가 작은 이유가 바로 이것.

단, `τ_max²`만으로 정규화하면 **작은 관절에 너무 큰 페널티**가 생겨 dynamic motion이 위축될 수 있다 (cost_small / cost_baseline ≈ 5.4배). 그래서 다음 1.3의 **soft-limit 항이 같이 들어가 있어야** L2 그룹의 가중치를 과하게 높이지 않고도 saturation을 줄일 수 있다.

### 1.3 Soft torque-limit 항 (`joint_torques_limit_soft_l2`)

**정의**:
```
overage_i = max(0, |τ_i| / τ_max_i − 0.8)
reward    = − Σ_i overage_i²
```

**실측 overage² p99 ≈ 0.04 per 포화 joint, 평균 ~0.005**. 활성 구간에서 포화 joint가 보통 3~5개 → 합 p95 ≈ 0.1~0.3 per step.

weight = **-1.5** → 포화 시 per-step 기여 -0.15~-0.45 (교정 강함).

**장점 (L2 대비)**:
- 정상 토크 구간(|τ|/τ_max < 0.8)에서는 gradient 0 → dynamic 동작 자유
- `τ_max`로 정규화되어 관절 크기 무관 동일 스케일
- 포화율이라는 "물리적 제약"을 직접 optimizer에게 전달

**학계 전례**:
- IsaacLab core의 `applied_torque_limits` 함수(`IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:233`): `applied_torque - computed_torque`의 음수 영역을 페널티화. 우리 버전은 `soft_ratio=0.8`로 더 이른 시점부터 개입.
- Unitree rl_gym H1: `dof_pos_limits` (위치 리밋 버전), torques 항은 0
- Booster Gym: `torques: -2e-4` 단일항만 사용 (saturation 명시 없음)

### 1.4 Acceleration 가중치 유도

**실측 qdd 통계 (rad/s², 활성 구간 n=62,014)**:

| Joint | p95 | p99 | max |
|---|---|---|---|
| L_HipRoll / R_HipRoll | 15~18 | 28~32 | 124~128 |
| L/R_HipPitch | 66~69 | 110~115 | 245~378 |
| L/R_HipYaw | 64~65 | 108~110 | 389~534 |
| L/R_Knee | 129~132 | 222~248 | 632~715 |
| **L/R_AnklePitch** | **166~175** | **399~404** | **1207~1214** |
| **L/R_AnkleRoll** | **153~160** | **278~282** | **932~1006** |
| WaistYaw | 13 | 20 | 70 |

**Σqdd² 통계**: mean=3.56e4, p95=1.64e5, max=1.72e6

기존 weight `-2.5e-7`의 per-step 기여 = `2.5e-7 × 3.56e4 = 0.0089`. 이는 `dof_torques_l2`(~0.35), `action_rate_l2`(~0.02) 대비 **10~40배 작아 사실상 비활성**.

**선택**:
- `dof_acc_l2`: -5.0e-7 (2×) → per-step 기여 ~0.018
- `dof_acc_distal_l2`: -1.5e-6 (추가, Knee/Ankle에만) → per-step 기여 ~0.05 (distal만 qdd 합 ~3.3e4)

**Distal 가중치 (1.5e-6) 선정 수식**:
Ankle/Knee의 p99 qdd² ≈ 400² + 280² + 250² ≈ 3.5e5.
목표 per-step 기여 0.05 → w = 0.05 / 3.5e5 ≈ **1.4e-6**. `-1.5e-6` 선택.

### 1.5 WaistYaw 제거 근거

- 실측 max τ = 17 Nm, τ_max = 152 Nm → 11% 사용
- 기존 weight(-5.5e-6) × E\[τ²\] (1.7) = 9.4e-6/step → 전체 torque penalty 0.35의 **0.003%**
- 학습 신호로서 무의미 → 제거로 식 단순화

---

## 2. 개별 제한이 학습에 해로울 수 있는가?

### 2.1 우려 포인트

| 우려 | 메커니즘 | 실제 영향 |
|---|---|---|
| **Reward scale 과소 관리** | 10개 넘는 항의 weight balance가 안 맞으면 dominant term만 학습됨 | 실측에 따라 per-step 기여도를 비교해 weight 선택했으므로 제어됨 (§1.2, §1.3) |
| **Ankle에 너무 강한 페널티** | `dof_torques_small_l2 = -3e-5`는 HipPitch 대비 5.4배 | Ankle이 "움직이지 않는 쪽"으로 기울 수 있음. **dynamic push-off 감소 → 보행 속도/에너지 저하** 가능 |
| **Reward hacking** | soft-limit 항 때문에 policy가 |τ|/τ_max = 0.8 경계에서 "떠는" 거동을 학습할 수 있음 | 희박. `clamp(min=0)`이 부드러워 boundary에서 gradient continuous |
| **Distal acc 항 추가로 stiff gait** | Ankle/Knee가 과도하게 부드러워져 foot clearance/stance timing이 깨질 수 있음 | 실측 qdd p99 근거로 선정했으므로 "정상 동작"은 건드리지 않음. 하지만 경계 주의 |
| **항 개수 증가 (hyper-param inflation)** | 7개 → 10개 RewTerm. 튜닝 search space 커짐 | 각 항이 물리적 의미가 분리돼 있어 경험적으로는 통합된 항보다 **디버깅 쉬움** |

### 2.2 학습 안정성 체크리스트 (초기 1000 iter 확인)

1. **tracking_lin_vel 보상 > 1.0** 유지되는가?
   - 아니면 `dof_torques_small_l2` weight를 -3e-5 → -1.5e-5로 완화
2. **feet_air_time 보상 > 0.5** 유지되는가?
   - 아니면 distal acc/torque 과규제 의심
3. **특정 관절 τ가 항상 0 근처로 붙는가?** (Ankle이 "굳은" 상태)
   - 아니면 해당 관절 weight 완화 또는 soft-limit ratio를 0.8 → 0.9로 완화
4. **mirror_loss / L/R 보상 비대칭**이 심화되는가?
   - 아니면 Ankle 페널티가 한쪽만 억제하는 가능성 → `feet_air_time_variance` 강화

### 2.3 기대 효과 (성공 시)

- 실측 40% Ankle saturation → **10% 이하로 감소**
- Knee/HipPitch 17~20% → **5~10% 수준**
- qdd p99 Ankle 400 → **200 이하**로 smoothness 개선
- Torque reward의 joint별 기여 std 감소 → balanced gait

---

## 3. 학계에서는 어떻게 접근하는가?

### 3.1 Approach A — **단일 L2 with uniform weight** (legged_gym 스타일)

**대표**: Rudin et al. 2022 (*Learning to Walk in Minutes*), Unitree rl_gym (H1/G1), 대부분의 초창기 rough-terrain 프레임워크.

```yaml
torques:     -1e-5     # 모든 관절 동일
dof_vel:     0 또는 -1e-3
dof_acc:     -2.5e-7
action_rate: -0.01
```

- **장점**: 간결, hyper-param 적음, domain randomization과 결합 시 sim2real 경험 많음
- **단점**: 관절 크기 차이 큰 humanoid에서 작은 관절 saturation이 무시됨 (실측으로 검증한 우리 상황이 대표 사례)
- **논문에서 흔한 보완**: domain randomization의 `actuator_strength_range=[0.8, 1.2]`로 "다양한 motor" 경험

→ 출처: [legged_gym (ETH RSL)](https://github.com/leggedrobotics/legged_gym), [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym)

### 3.2 Approach B — **τ_max 정규화 L2** (humanoid-gym, Booster Gym 스타일)

**대표**: Humanoid-Gym (XBot-L, 80kg), Booster Gym (T1, 30kg).

```yaml
torques:      -1e-5 ~ -2e-4    # 전체 weight 공유
torque_tiredness: -1e-2        # Booster: saturation-aware variant
dof_vel:      -1e-4 ~ -5e-4
```

- Booster Gym은 `torque_tiredness`라는 **시간적분 토크**를 별도 규제 (motor 과열 모델)
- 우리 Approach A보다 진일보, 그러나 여전히 단일 weight로 모든 관절 묶음

→ 출처: [Humanoid-Gym (arXiv:2404.05695)](https://github.com/roboterax/humanoid-gym), [Booster Gym (arXiv:2506.15132)](https://github.com/BoosterRobotics/booster_gym)

### 3.3 Approach C — **Per-joint 개별 weight** (우리 현재 접근)

**대표 사례**: 대규모 humanoid 논문에서는 드물지만, **quadruped에서 ANYmal / Spot / Mini Cheetah의 production 코드**에 관찰됨. Foot/Hip 그룹을 분리해서 잡는 것이 흔하다.

특히 `hip_pos` (HipYaw를 중립 근처로 붙이는 항)는 ANYmal/H1 config에 공통으로 존재 — 기능적으로 "관절별 개별 규제"의 일종.

- **장점**: 실측 기반 튜닝 가능, 해석성 높음
- **단점**: 항 수 증가, per-joint 튜닝에 더 많은 실험 필요

### 3.4 Approach D — **Constrained RL (CRL)** — reward에서 빼고 constraint로

**대표 논문**:
- *Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion* (RA-L 2024, Lee et al.)
- *Actuator-Constrained RL for High-Speed Quadrupedal Locomotion* (arXiv 2023)
- *Stage-Wise Reward Shaping for Acrobatic Robots* (2024)

**아이디어**: 토크/속도 한계는 reward weight로 쓰지 말고 **CPO/FOCOPS 같은 constrained optimizer의 hard constraint**로 넘겨라. 이유:
> "Designing reward terms and tuning their coefficients is time-consuming, as engineers need to satisfy robot physical constraints (torque, velocity limits) while designing 10+ reward terms."

- **장점**: weight 튜닝 불필요. 위반 시 Lagrangian multiplier가 자동 조절
- **단점**: rsl_rl/legged_gym 프레임워크에 CPO 미구현. 도입 시 알고리즘 교체 필요
- **우리 상황**: 당장 도입은 부담. 하지만 **soft-limit 항(우리 `dof_torques_limit_soft`)이 CRL의 Lagrangian 완화 버전과 사실상 동일 형태**임

### 3.5 Approach E — **Reward annealing / curriculum**

**대표**: *Learning Gentle Humanoid Locomotion* (SoFTA, 2024), Science Robotics 2024 *Real-World Humanoid Locomotion*.

```
weight(t) = w_init × (1 - t/T_anneal) + w_final × (t/T_anneal)
```

- 초기에는 regularization 약하게 → 빠른 탐색
- 후기에는 강하게 → deploy 품질 향상
- **우리 "외발잡이" 문제의 주된 재발 방지책**: regularization을 한 번에 올리지 말고 **500~1000 iter에 걸쳐 ramp-up**하는 것이 가장 안전

→ 출처: [Real-world humanoid locomotion (Science Robotics 2024)](https://www.science.org/doi/10.1126/scirobotics.adi9579)

### 3.6 Approach F — **Actuator-in-the-loop RL**

**대표**: *Actuator-Constrained RL for High-Speed Quadrupedal Locomotion* (MIT, 2023).

Motor의 torque-speed MOR(Motor Operating Region)을 시뮬레이션 안에서 enforce. 우리 프로젝트의 `DelayedPDActuatorLUTCfg` + `angle_dependent_effort_limit` (이미 md_files에 존재)가 같은 철학.

- 우리는 이미 이 접근을 부분적으로 적용 중 → reward에서 torque limit을 더 강하게 걸 필요가 상대적으로 적어야 함
- 그럼에도 실측에서 포화가 많이 관찰된다는 것은 **LUT가 있음에도 policy가 limit까지 밀어붙이는 상황** → soft-limit reward가 여전히 유효

---

## 4. 접근 비교표 & 우리 선택 위치

| Approach | 복잡도 | 튜닝 민감도 | 실측 근거 필요 | humanoid 적합성 | 우리 채택 여부 |
|---|---|---|---|---|---|
| A. 단일 L2 | ★ | ★★★★ | 낮음 | 중 (작은 관절 무시) | ✕ |
| B. τ_max 정규화 단일 | ★★ | ★★★ | 중 | 중상 | △ (부분 채택) |
| **C. Per-joint 그룹 분리** | **★★★** | **★★** | **중상** | **상** | **● (주 채택)** |
| C'. + Soft torque-limit | ★★★ | ★★ | 중상 | 상 | ● (주 채택) |
| D. Constrained RL | ★★★★ | ★ | 낮음 | 상 | ✕ (인프라 부담) |
| E. Reward annealing | ★★ | ★★ | 낮음 | 상 | △ (다음 단계 권장) |
| F. Actuator-in-loop | ★★★★ | ★★ | 높음 | 상 | ● (이미 부분 적용 중) |

**우리 전략**: C(per-joint) + C'(soft-limit) + F(actuator LUT, 이미 적용). D의 효과를 C'로 근사하는 형태.

---

## 5. 권장 다음 단계

1. **현재 수정안으로 학습 1000 iter 돌려 §2.2 체크리스트 확인**
2. Tracking reward 저하 또는 Ankle 굳음 관찰되면:
   - `dof_torques_small_l2` weight를 -3e-5 → **-1.5e-5** 로 완화
   - 또는 `dof_torques_limit_soft.soft_ratio`를 0.8 → **0.85**로 완화
3. 성공 시: **annealing schedule 도입** (Approach E) — 초기 500 iter는 weight × 0.3, 이후 선형 증가
4. L/R 비대칭(실측에서 관찰됨, 특히 HipRoll/HipYaw) 지속 시: `mirror_loss_coeff` 1.0 → **2.0** + `air_time_variance` -20 → **-40**

---

## 참고문헌

- Rudin, N., Hoeller, D., Reist, P., Hutter, M. (2022). *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*. CoRL.
- [legged_gym (ETH RSL)](https://github.com/leggedrobotics/legged_gym)
- [Unitree RL Gym (H1, G1)](https://github.com/unitreerobotics/unitree_rl_gym)
- [Humanoid-Gym (roboterax)](https://github.com/roboterax/humanoid-gym) — [arXiv:2404.05695](https://arxiv.org/abs/2404.05695)
- [Booster Gym (T1)](https://github.com/BoosterRobotics/booster_gym) — [arXiv:2506.15132](https://arxiv.org/abs/2506.15132)
- Lee et al. *Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion* — [arXiv:2308.12517](https://arxiv.org/abs/2308.12517)
- *Actuator-Constrained RL for High-Speed Quadrupedal Locomotion* — [arXiv:2312.17507](https://arxiv.org/abs/2312.17507)
- *Real-World Humanoid Locomotion with Reinforcement Learning* — [Science Robotics 2024](https://www.science.org/doi/10.1126/scirobotics.adi9579) / [arXiv:2303.03381](https://arxiv.org/abs/2303.03381)
- *Stage-Wise Reward Shaping for Acrobatic Robots* — [arXiv:2409.15755](https://arxiv.org/abs/2409.15755)
- *Learning Gentle Humanoid Locomotion (SoFTA)* — [paper PDF](https://lecar-lab.github.io/SoFTA/resources/Main-Paper.pdf)
- IsaacLab core reward functions — `IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py`
- 내부 분석 스크립트: `/tmp/analyze_real_robot.py` (본 분석 재현 가능)
