# PPOFuture + ActorCriticAdaptationFuture 아키텍처 상세 설명

> 작성 기준 코드: `agents/rsl_rl_ppo_cfg.py`, `algorithms/rsl_rl/ac_future.py`, `algorithms/rsl_rl/ppo_future.py`, `tasks/.../rough_env_cfg.py`
> 작성일: 2026-02-25

---

## 1. 한 줄 요약

**관측 히스토리(noisy) → Encoder → Latent** 안에서 세 가지를 동시에 학습:

| Latent slice | 무엇을 추정하나 | Ground Truth 출처 |
|---|---|---|
| `[0:3]` vel3 | body frame 선속도(vx, vy) + yaw rate | Critic obs GT prefix |
| `[3:9]` foot6 | 왼발/오른발 GRF (3D each) | Critic obs GT prefix |
| `[9:39]` future30 | O(t+1)의 압축 표현 | target_encoder(TargetCfg) |

→ 이 구조가 **Denoising World Model**: 노이즈 낀 히스토리에서 노이즈 없는 미래 상태를 예측.

---

## 2. 전체 아키텍처 다이어그램

```
┌──────────────────────────────────────────────────────────────────┐
│                         시뮬레이션 환경 (Isaac Lab)               │
│                                                                   │
│  ┌─────────────────────────────────────────┐                     │
│  │            PolicyCfg (noisy)             │                     │
│  │  history_length = 5, 59D × 5 = 295D     │                     │
│  │  [ang_vel(3), grav(3), cmd(3),           │                     │
│  │   gait_sin(1), gait_cos(1),              │                     │
│  │   motor_pos(12), motor_vel(12),          │                     │
│  │   meas_pos(6), meas_vel(6), action(12)] │                     │
│  └─────────────┬───────────────────────────┘                     │
│                │ 295D (noisy obs history)                         │
│                ▼                                                   │
│  ┌──────────────────────────┐                                     │
│  │    Encoder (MLP)         │  [512 → 512 → 256 → 64]            │
│  │  history → latent 64D    │                                     │
│  └──────────────┬───────────┘                                     │
│                 │ latent (64D)                                     │
│    ┌────────────┼──────────────────────────────┐                  │
│    │            │                              │                  │
│    ▼            ▼                              ▼                  │
│  [0:3]       [3:9]                         [9:39]                │
│  vel3_pred   foot6_pred                    future_pred(30D)      │
│    │            │                              │                  │
│    │(MSE)       │(MSE)                         │(MSE)            │
│    ▼            ▼                              ▼                  │
│  gt_vel3     gt_foot6        target_encoder(TargetCfg O(t+1))    │
│  (CriticCfg) (CriticCfg)                   (30D)                 │
│                                                                   │
│  ┌────────────────────────┐                                       │
│  │   Actor (MLP)          │  current_obs(59D) + latent(64D)       │
│  │  [256 → 256 → 256]     │  → 123D → 12D action                 │
│  └────────────────────────┘                                       │
│                                                                   │
│  ┌────────────────────────┐                                       │
│  │   Critic (MLP)         │  CriticCfg (159D, privileged)        │
│  │  [256 → 256 → 256]     │  → 1D value                          │
│  └────────────────────────┘                                       │
│                                                                   │
│  ┌────────────────────────┐                                       │
│  │  Target Encoder (MLP)  │  TargetCfg(t+1) (150D)              │
│  │  [128 → 30]            │  → future 30D (teacher signal)       │
│  └────────────────────────┘                                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 관측 그룹 상세

### 3.1 PolicyCfg — 정책 입력 (history_length=5, noisy)

| 항목 | 함수 | 차원 | 노이즈 | 비고 |
|---|---|---|---|---|
| `base_ang_vel` | `mdp.base_ang_vel` | 3D | ±0.2 | IMU로 측정 가능 |
| `projected_gravity` | `mdp.projected_gravity` | 3D | ±0.05 | IMU로 측정 가능 |
| `velocity_commands` | `mdp.generated_commands` | 3D | 없음 | 명령 속도 [vx, vy, wz] |
| `gait_phase_sin` | `mdp.gait_phase_sin` | 1D | 없음 | sin(2π × phase) |
| `gait_phase_cos` | `mdp.gait_phase_cos` | 1D | 없음 | cos(2π × phase), cmd≈0이면 0 |
| `motor_joint_pos` | `mdp.joint_pos_ordered_rel` | 12D | ±0.0025 | **모터** 관절 위치(default 대비 오프셋) |
| `motor_joint_vel` | `mdp.joint_vel_ordered` | 12D | ±0.025 | **모터** 관절 속도 |
| `measured_joint_pos` | `mdp.joint_pos_ordered` | 6D | ±0.01 | **패시브 링크** (KneePitch, AnklePitch/Roll × L/R) |
| `measured_joint_vel` | `mdp.joint_vel_ordered` | 6D | ±0.1 | **패시브 링크** 속도 |
| `actions` | `mdp.last_processed_action` | 12D | 없음 | 마지막 스텝의 처리된 액션 |

**단일 프레임 = 59D**, `history_length=5` → **총 295D** (Encoder 입력)

> **왜 패시브 링크를 따로 측정하나?**
> P73은 4-bar linkage 구조로 모터(KneeUpper)가 KneePitch를 구동하지만 KneePitch는 독립 관절이다.
> 실제 로봇에서는 인코더를 통해 양쪽 모두 읽을 수 있으므로 모터와 패시브를 분리해 관측한다.

---

### 3.2 TargetCfg — O(t+1) 교사 신호 (history_length=1, noise=False)

Target Encoder의 입력. **이 그룹은 학습 시에만 사용**되고, 추론 시에는 전혀 쓰이지 않는다.

| 항목 | 차원 | 비고 |
|---|---|---|
| `base_ang_vel` | 3D | 노이즈 없는 클린 값 |
| `projected_gravity` | 3D | 노이즈 없는 클린 값 |
| `velocity_commands` | 3D | |
| `gait_phase_sin/cos` | 2D | |
| `motor_joint_pos` (12개) | 12D | 노이즈 없음 |
| `motor_joint_vel` (12개) | 12D | 노이즈 없음 |
| `measured_joint_pos` (6개) | 6D | 패시브 링크 |
| `measured_joint_vel` (6개) | 6D | 패시브 링크 |
| `height_scan` | **81D** | 1.2m×1.2m, 0.15m 해상도 → 9×9 |
| `actions` | 12D | |
| `physics_material` | 2D | (static, dynamic) 마찰계수 캐시 |
| `base_mass_delta` | 1D | base_link mass - default mass |
| `base_com_offset` | 3D | base_link COM 오프셋 (xyz) |
| `motor_armature_stats` | 2D | armature randomization의 mean/std |
| `motor_damping_stats` | 2D | damping randomization의 mean/std |
| **합계** | **150D** | `target_obs_dim=150` ✓ |

> **핵심 통찰**: Target Encoder는 **미래 상태 + DR 파라미터**를 하나의 30D 벡터로 압축한다.
> Encoder가 이 30D를 히스토리만으로 맞추도록 학습하면, 자연스럽게 DR 파라미터를
> (마찰, 질량, COM, armature, damping) **암묵적으로 추론**하게 된다.

---

### 3.3 CriticCfg — 특권 관측 (privileged, 학습 시에만 사용)

**앞 9D가 반드시 고정**: PPOFuture가 이 인덱스를 하드코딩으로 슬라이싱한다.

| 인덱스 | 항목 | 차원 | 비고 |
|---|---|---|---|
| `[0:3]` | `gt_vel3` | 3D | [vx, vy, yaw_rate] body frame, **GT supervision** |
| `[3:9]` | `gt_foot_force6` | 6D | L/R GRF yaw-aligned, scale=(1/300, 1/300, 1/600) × 2 |
| `[9:12]` | `base_ang_vel` | 3D | 노이즈 없음 |
| `[12:15]` | `projected_gravity` | 3D | |
| `[15:18]` | `velocity_commands` | 3D | |
| `[18:20]` | `gait_phase_sin/cos` | 2D | |
| `[20:32]` | `motor_joint_pos` | 12D | |
| `[32:44]` | `motor_joint_vel` | 12D | |
| `[44:50]` | `measured_joint_pos` | 6D | |
| `[50:56]` | `measured_joint_vel` | 6D | |
| `[56:68]` | `actions` | 12D | |
| `[68:149]` | `height_scan` | 81D | |
| `[149:151]` | `physics_material` | 2D | |
| `[151]` | `base_mass_delta` | 1D | |
| `[152:155]` | `base_com_offset` | 3D | |
| `[155:157]` | `motor_armature_stats` | 2D | |
| `[157:159]` | `motor_damping_stats` | 2D | |
| **합계** | | **159D** | |

> **왜 CriticCfg와 TargetCfg가 거의 동일한가?**
> Critic이 privileged 정보로 value를 정확히 추정하고, 동시에 그 prefix([0:9])가
> Encoder의 latent prefix를 직접적으로 supervise하는 이중 역할을 한다.

---

## 4. 신경망 컴포넌트 상세

### 4.1 Encoder (History → Latent)

```python
MLP: 295D → 512 → 512 → 256 → 64D (latent)
활성화: ELU
Empirical Normalization: 입력에 적용
```

- 입력: flattened history (5 프레임 × 59D)
- 출력: latent 64D (구조적으로 supervised)
- **학습 목표**: latent가 vel3, foot6, future를 모두 포함하면서도 Actor에게 유용한 표현을 제공

### 4.2 Target Encoder (O(t+1) → Future)

```python
MLP: 150D → 128 → 30D (future)
활성화: ELU
```

- 입력: TargetCfg (다음 스텝의 특권 관측, t+1)
- 출력: future_pred의 학습 타겟
- **이 네트워크의 출력은 Encoder 학습 시 `.detach()`되어 사용됨** (teacher는 gradient를 받지 않음)
- 실제로는 Target Encoder도 gradient를 받지 않도록 `detach()` 후 MSE → **Target Encoder도 간접적으로 학습됨**

  > **주의**: `future_tgt = policy.encode_target(target_next_batch).detach()` 이므로
  > `L_future` gradient는 Encoder만 업데이트하고, Target Encoder는 역전파 차단.
  > 하지만 Target Encoder는 다른 loss(없음)로는 직접 학습되지 않는다.
  > → **Target Encoder는 실질적으로 학습되지 않는 고정 teacher가 아니라**, value/surrogate loss를 통해
  > Critic obs를 간접 처리하는 경로도 없으므로, **실제로는 Target Encoder의 파라미터가 학습되지 않음**.
  > 만약 Target Encoder가 점진적으로 개선되길 원한다면 별도 loss가 필요하다.

### 4.3 Actor

```python
MLP: (59 + 64)D → 256 → 256 → 256 → 12D
활성화: ELU
노이즈: log_std 방식 (학습 가능한 log σ)
```

- 입력: `current_obs(t)` (히스토리의 마지막 59D) + `latent(64D)` = 123D
- 출력: 12D 모터 관절 위치 명령 ([-0.5, 0.5] rad 오프셋)
- **왜 전체 히스토리를 쓰지 않나**: latent가 이미 히스토리를 요약하므로, Actor는 현재 프레임 + 잠재 표현만 본다

### 4.4 Critic

```python
MLP: 159D → 256 → 256 → 256 → 1D
활성화: ELU
```

- 입력: CriticCfg (완전한 특권 정보, history_length=1)
- 출력: state value V(s)
- **학습 시에만 존재**: 배포 시 Critic은 사용되지 않음

---

## 5. Latent 벡터 레이아웃 (64D)

```
latent = [vel3 | foot6 | future30 | free25]
          0   3  3    9  9      39  39    64
```

| 슬라이스 | 이름 | 차원 | Supervision |
|---|---|---|---|
| `[0:3]` | `vel3_pred` | 3D | MSE vs `gt_vel3` (CriticCfg[0:3]) |
| `[3:9]` | `foot6_pred` | 6D | MSE vs `gt_foot_force6` (CriticCfg[3:9]) |
| `[9:39]` | `future_pred` | 30D | MSE vs `target_encoder(O(t+1))` |
| `[39:64]` | (자유) | 25D | 직접 supervision 없음, PPO로만 학습 |

> **"free" 구간의 역할**: 명시적으로 supervision 받지 않아도, Actor가 좋은 행동을 하려면
> 추가적인 문맥 정보가 필요하므로 이 구간이 자동으로 활용된다. 실험적으로 이 구간은
> 보행 위상, 안정성 관련 정보를 암묵적으로 인코딩하는 경향이 있다.

---

## 6. 손실 함수 (PPOFuture.update)

### 총 Loss

```
L_total = L_surrogate + λ_value × L_value - λ_entropy × H
        + w_vel × L_vel + w_foot × L_foot + w_future × L_future
        [+ λ_mirror × L_symmetry]
```

현재 설정값:
```python
w_vel    = 1.0
w_foot   = 1.0
w_future = 1.0
value_loss_coef = 5.0
entropy_coef    = 0.008
mirror_loss_coeff = 1.0
```

### 보조 손실 상세

#### L_vel (속도 추정 손실)
```python
vel3_pred = latent[:, 0:3]                        # Encoder 출력 슬라이스
vel3_gt   = critic_obs[:, 0:3].detach()           # [vx, vy, yaw_rate] 실제값
L_vel = MSE(vel3_pred, vel3_gt)
```
- **sim-to-real 의미**: 실제 로봇에는 IMU는 있지만 직접적인 선속도 센서가 없다.
  Encoder가 히스토리에서 속도를 추론하도록 훈련되면, 실제 배포 시에도 같은 방식으로 추정 가능.

#### L_foot (GRF 추정 손실)
```python
foot6_pred = latent[:, 3:9]
foot6_gt   = critic_obs[:, 3:9].detach()          # 정규화된 GRF (÷300, ÷600)
L_foot = MSE(foot6_pred, foot6_gt)
```
- **sim-to-real 의미**: 발에 force 센서가 없어도 Encoder가 GRF를 추정.
  발이 지면에 닿아 있는지, 체중이 어느 발에 실리는지를 히스토리에서 파악.
- **정규화**: Fz는 체중(600N) 기준, Fx/Fy는 마찰력(300N) 기준

#### L_future (미래 상태 예측 손실)
```python
target_next = storage.target_observations_next[t]  # O(t+1) — 다음 스텝에서 수집
future_tgt  = policy.encode_target(target_next).detach()  # 30D teacher signal
future_pred = latent[:, 9:39]
L_future = MSE(future_pred, future_tgt)
```
- **"Denoising" 효과**: Encoder는 `t` 시점의 **noisy** 히스토리에서 `t+1`의 **clean** 미래를 예측.
  노이즈를 거슬러 올라가 실제 물리 상태를 추정하므로 "denoising"이라 부른다.
- **World Model 효과**: O(t+1)에는 height_scan, DR 파라미터(마찰, 질량, COM, armature, damping)가 포함.
  Encoder가 이 정보를 히스토리에서 복원하려면 **환경의 물리 파라미터를 내부적으로 모델링**해야 함.

#### L_symmetry (미러 대칭 손실)
```python
# 현재 obs를 좌우 미러링 → 대칭 obs 생성
# Actor의 mean action이 미러 대칭적이어야 한다는 제약
L_sym = MSE(actor(mirror(obs)), mirror(actor(obs)))
```
- `use_data_augmentation=False`, `use_mirror_loss=True`
- PPOFuture에서는 data augmentation path는 비활성화, mirror loss만 사용
- `p73_data_augmentation_lowerbody_mirror` 함수로 좌/우 관절 인덱스 교환

---

## 7. 학습 흐름 (Rollout → Update)

### Rollout 수집 (매 스텝)

```
1. env.step()
   - obs = {policy: [...], critic: [...], target: [...]}
   - target은 항상 t 시점에서 수집되지만,
     PPOFuture는 이를 "다음 스텝의 target_obs_next"로 저장

2. PPOFuture.process_env_step(obs, rewards, dones, extras):
   - transition.target_observations_next = obs["target"]  ← O(t+1) 역할!
   - (이전 스텝에서 저장한 target이 현재 스텝 학습 시 "미래"가 됨)

3. RolloutStorageFuture: transition → 버퍼에 누적
   - num_envs=4096, num_steps_per_env=24
   - 총 배치: 4096 × 24 = 98,304 transitions
```

> **타임스텝 오프셋 주의**: `target_obs_next[t]`는 실제로 스텝 t에서 관측한 TargetCfg다.
> 이것이 "t+1의 미래"가 되는 이유는, obs 계산이 스텝 실행 *후* (다음 상태)에서 이루어지기 때문.
> 즉, `env.step()` → new obs → `target_obs_next[t] = obs["target"]`이 t+1 상태다.

### 미니배치 업데이트 (매 PPO epoch)

```
epochs=5, mini_batches=4 → 20번 업데이트 per rollout

각 미니배치:
1. policy.act(obs_batch) → distribution 재계산
2. policy.get_latent(obs_batch) → 64D latent
3. split_latent → (vel3, foot6, future)
4. L_vel + L_foot 계산 (vs critic_obs GT prefix)
5. policy.encode_target(target_next_batch).detach() → 30D target
6. L_future 계산
7. surrogate + value + entropy + aux 합산 → backward
```

---

## 8. Sim-to-Real 메커니즘 심층 분석

### 8.1 도메인 랜덤화 (EventCfg)

| DR 항목 | 범위 | 모드 | Latent에서 추정? |
|---|---|---|---|
| 바닥 마찰 (static, dynamic) | [0.2, 1.4] | startup | O (physics_material, 2D) |
| 링크 질량 스케일 | [0.8, 1.2] × default | startup | △ (간접적) |
| Base 질량 추가 | [-10, +20] kg | startup | O (base_mass_delta, 1D) |
| Upper body 질량 스케일 | 1.0 (고정 가능) | startup | △ |
| Base COM 오프셋 | [-10, +10] cm × 3축 | startup | O (base_com_offset, 3D) |
| 모터 armature 스케일 | [0.6, 1.4] × default | reset | O (motor_armature_stats, 2D) |
| 모터 damping 추가 | [0, 2.9] Nm/(rad/s) | reset | O (motor_damping_stats, 2D) |
| Push 외란 | ±0.5 m/s × 2축 | interval 8s | X (암묵적으로만) |

> **"O"인 항목**: Target Encoder가 직접 이 값을 입력받아 30D로 압축 → Encoder가 추론 학습
> **"△"인 항목**: 직접 관측되지 않지만 운동학/동역학 패턴에서 간접 유추 가능

### 8.2 배포 시나리오 (Deploy)

```
실제 로봇에서:
  PolicyCfg 항목 → 실제 센서로 수집 (IMU, 관절 인코더)
  TargetCfg    → 사용 안 함 (privileged)
  CriticCfg    → 사용 안 함 (privileged)

Encoder → latent (64D)
  [0:3]   vel3: 학습된 추정값 사용 (speed estimator 역할)
  [3:9]   foot6: 학습된 추정값 사용 (GRF estimator 역할)
  [9:39]  future: 환경 동역학 잠재 표현 (implicit system ID)
  [39:64] free: 추가 문맥

Actor → 12D 모터 명령
```

---

## 9. 게이트 보행 위상 (Gait Phase)

```python
phase = (episode_length_buf % period_steps) / period_steps  # [0, 1)
gait_phase_sin = sin(2π × phase)
gait_phase_cos = cos(2π × phase)

# cmd ≈ 0 (서 있을 때) → phase = 0 강제
is_standing = |vx_cmd| + |vy_cmd| + |wz_cmd| ≤ 1e-3
```

- `period_steps=40` (env step 기준) × `sim.dt=0.005` × `decimation=4` = 40 × 0.02s = **0.8초 보행 주기**
- Policy, Critic, Target 모두 동일한 period_steps 사용 (동기화 필수)
- `contact_schedule_reward_biped_ds`와 동기화: ds_ratio=0.20 → 20% double support

---

## 10. 액션 구조 (LowerBodyActionsCfg)

```
action_dim = 12 (하체 모터만)
policy output → clip [-1, 1] → scale × 0.5 → Δq [rad] → q_target = q_default + Δq
PD 제어: τ = Kp(q_target - q) - Kd × q̇
```

| 관절 | Kp | Kd | 토크 한계 |
|---|---|---|---|
| HipRoll | 3000 | 50 | 264 Nm |
| HipPitch | 5000 | 20 | 264 Nm |
| HipYaw | 2000 | 15 | 110.4 Nm |
| KneeUpper | 3700 | 30 | 264 Nm |
| AnkleM1 | 3200 | 30 | 110.4 Nm |
| AnkleM2 | 3200 | 30 | 110.4 Nm |

상체(17 DOF)는 `upper_joint_names`에 정의되어 있지만 기본 위치 유지 (정책이 제어하지 않음).
`rand_motor_scale_range=(0.8, 1.2)`: 학습 시 모터 출력 스케일 무작위화.

---

## 11. 대칭 손실 (Symmetry Mirror Loss)

```python
# p73_data_augmentation_lowerbody_mirror 함수:
# 관측: L/R 관절 인덱스 교환, HipRoll/HipYaw 부호 반전, cmd vy/wz 부호 반전
# 액션: L/R 교환, 해당 관절 부호 반전

L_sym = MSE(
    actor(mirror_obs),          # 미러 obs에 대한 action mean
    mirror(actor(original_obs)) # 원본 action의 미러 버전
)
```

이 loss는 Policy가 좌/우 대칭적으로 행동하도록 강제하여:
1. 데이터 효율 2배
2. 좌우 갈지자 걸음 방지
3. Sim-to-Real 이전성 향상

---

## 12. 핵심 차원 요약

| 컴포넌트 | 입력 차원 | 출력 차원 |
|---|---|---|
| Encoder | 295D (59×5 history) | 64D (latent) |
| Target Encoder | 150D (O(t+1)) | 30D (future target) |
| Actor | 123D (59+64) | 12D (action) |
| Critic | 159D (privileged) | 1D (value) |
| Latent vel3 slice | 64D 중 0:3 | 3D |
| Latent foot6 slice | 64D 중 3:9 | 6D |
| Latent future slice | 64D 중 9:39 | 30D |
| Latent free slice | 64D 중 39:64 | 25D |

**num_single_obs = 59** (설정에서 명시적으로 지정 필수)

---

## 13. 내가 놓치기 쉬운 포인트들

### 13.1 target_obs_next의 시간 오프셋
`PPOFuture.process_env_step`에서 `obs["target"]`를 `target_observations_next`로 저장.
이 obs는 env.step() 이후의 상태이므로 실제로 t+1 상태다. 자연스러운 인과관계.

### 13.2 Target Encoder는 학습이 안 된다
`future_tgt = policy.encode_target(target_next_batch).detach()` 때문에
Target Encoder의 파라미터는 어떤 loss gradient도 받지 않는다.
즉, Target Encoder는 **초기화 상태에서 고정된 랜덤 매핑**이다.
→ 이는 의도적일 수 있다: 안정적인 target 분포 제공 (EMA 타겟과 유사한 효과)
→ 또는 미래 개선점: Target Encoder를 별도 supervised loss로 학습

### 13.3 Critic obs prefix 인덱스가 하드코딩
`ppo_future.py`에서:
```python
vel3_gt  = critic_obs_batch[:, 0:3].detach()
foot6_gt = critic_obs_batch[:, 3:9].detach()
```
CriticCfg의 첫 두 항목이 `gt_vel3`, `gt_foot_force6` 순서여야 한다.
순서를 바꾸면 auxiliary loss가 완전히 깨진다.

### 13.4 GRF 정규화 스케일
`gt_foot_force6`에 `scale=(1/300, 1/300, 1/600) × 2`가 적용됨.
→ Latent의 foot6 슬라이스도 이 정규화된 스케일의 값을 예측해야 함.
→ vel3에는 정규화 없음: body velocity는 이미 O(1)

### 13.5 armature randomization이 reset 모드
`randomize_armature`가 `mode="reset"` → **에피소드마다** armature가 새로 랜덤화.
`physics_material`, `add_base_mass`는 `mode="startup"` → **학습 전체에서 한 번**.
따라서 armature stats는 에피소드 중간에도 변하지 않지만, 에피소드 단위로 변한다.

### 13.6 empirical_normalization
`empirical_normalization=True` (Runner 설정) → Encoder 입력에 RunningMean/Std 정규화.
Critic에도 별도 normalization 가능 (`critic_obs_normalization`).

### 13.7 clip_param과 future loss의 상호작용
PPO의 `clip_param=0.16`은 ratio clipping에만 적용.
Auxiliary losses에는 clipping이 없으므로 `w_vel/w_foot/w_future` 가중치가 너무 크면
gradient를 지배해 PPO 학습이 불안정해질 수 있다.
현재 모두 1.0으로 설정되어 있음.

---

## 14. 관련 파일 경로

```
source/isaaclab_p73/isaaclab_p73/
├── algorithms/rsl_rl/
│   ├── ac_future.py          # ActorCriticAdaptationFuture 네트워크
│   └── ppo_future.py         # PPOFuture 알고리즘 + RolloutStorageFuture
└── tasks/manager_based/isaaclab_p73/
    ├── rough_env_cfg.py       # 환경 설정 (Obs/Reward/Event)
    ├── agents/
    │   └── rsl_rl_ppo_cfg.py  # 하이퍼파라미터 설정
    └── mdp/
        ├── observations.py    # gait_phase, GRF, DR stats 함수
        ├── rewards.py         # 보상 함수
        ├── events.py          # DR 이벤트
        ├── actions.py         # LowerBodyActionsCfg
        └── symmetry.py        # 미러 대칭 함수
```

---

## 15. 이 구조와 유사한 선행 연구

| 연구 | 공통점 | 차이점 |
|---|---|---|
| **RMA** (Kumar et al., 2021) | 환경 encoder + 적응 | RMA는 별도 adaptation phase 필요 |
| **DreamWaQ** (Nahrendra et al., 2023) | history encoder → latent | DreamWaQ는 RSSM 기반 확률적 모델 |
| **Priv. Est.** (Ji et al., 2022) | latent supervision with privileged | 우리는 future matching도 추가 |
| **RLPD / DroQ** | - | off-policy, 다른 패러다임 |

이 구조의 핵심 차별점: **미래 상태 예측(future matching) + 즉각적 물리량 추정(vel, GRF) 결합**으로
단순 privileged distillation보다 더 풍부한 감독 신호를 제공한다.
