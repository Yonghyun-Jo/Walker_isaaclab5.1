---
date: 2026-05-12
project: isaaclab_walker
branch: actuator_net_lim_nvi (기반, 새 branch 생성 예정)
tags: [plan, height_command, crouch, isaaclab_walker]
---

# Height Command (pos_z) 추가 계획

## 목표
현재 `[vel_x, vel_y, ang_z]` 3D command에 **pos_z (base height)** command를 추가하여,
로봇이 "몸을 숙이기/높이 유지" 동작을 명령으로 제어할 수 있도록 한다.

---

## 1. Command 추가: pos_z ✅ 구현 완료

### 설계 결정: Z축은 vel이 아닌 pos command
- X/Y는 무한 공간에서 속도 유지가 목표 → vel command가 자연스러움
- Z(높이)는 유한 범위에서 특정 높이 도달 후 유지가 목표 → **pos command가 적합**
- vel_z command를 주면 목표 높이 도달 후 0으로 전환하는 타이밍이 어려움
- pos_z는 목표 도달 후 reward가 자연스럽게 유지됨
- vel_z는 command가 아닌 **latent supervision**으로 활용 (높이 변화 동역학 인지용)

### 구현 내용

**신규 파일: `mdp/height_command.py`**
- `UniformHeightCommand` — CommandTerm 상속, `(num_envs, 1)` shape
- `UniformHeightCommandCfg` — `ranges.height`, `default_height`, `rel_default_height_envs`

**`rough_env_cfg.py` `__post_init__`에 등록:**
```python
self.commands.base_height = mdp.UniformHeightCommandCfg(
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    default_height=0.89,
    rel_default_height_envs=0.5,
    ranges=mdp.UniformHeightCommandCfg.Ranges(height=(0.55, 0.89)),
)
```

**Policy/Target/Critic 3개 obs group에 `height_command` (1D) 추가:**
- Policy obs: 47D → **48D**

---

## 2. Height Reward: command-conditional로 변경 ✅ 구현 완료

### 기존 (고정 target=0.89) → 신규 (command tracking)

| 기존 | 신규 | 변경 |
|------|------|------|
| `body_height_below_target_l2` (w=-30) | `track_height_below_l2` (w=-30) | target → h_cmd |
| `body_height_target_exp` (w=3, std=0.03) | `track_height_exp` (w=3, std=**0.05**) | target → h_cmd, std 확장 |

**구현된 함수 (`rewards.py`):**
- `track_height_command_exp(command_name, std)` — bell curve centered at h_cmd
- `track_height_command_below_l2(command_name)` — one-sided L2, h < h_cmd일 때만

**주의사항:**
- `std`를 0.03→**0.05**로 넓힘 (높이 범위가 넓어지므로 gradient 도달 범위 확대)
- **curriculum 고려**: 처음에는 height range를 좁게 (0.80~0.89) → 점차 넓히기 (0.55~0.89)

---

## 3. Flat Orientation: 숙일 때 상체 기울기 허용 ✅ 구현 완료

### 구현: 방안 A (height-adaptive pitch 허용)

**`flat_orientation_height_adaptive_l2` (`rewards.py`):**
```python
# height command에 비례하여 pitch 허용량 선형 보간
ratio = clamp((0.89 - cmd_h) / (0.89 - 0.55), 0, 1)
allowed_pitch = ratio * max_allowed_pitch  # 0~0.3 rad

# pitch: deadband 초과분만 페널티
pitch_violation = clamp(|grav_b_y| - allowed_pitch, min=0)
# roll: 항상 페널티 (좌우 안정성)
return roll² + pitch_violation²
```

**`rough_env_cfg.py`에서 교체:**
```python
flat_base_orientation_l2 = RewTerm(
    func=mdp.flat_orientation_height_adaptive_l2,
    weight=-30.0,
    params={
        "command_name": "base_height",
        "max_allowed_pitch": 0.3,     # ~17° at deepest crouch
        "default_height": 0.89,
        "min_height": 0.55,
    },
)
```

**동작:**
- `cmd_h=0.89` (서기) → pitch 허용 0 → 기존과 동일
- `cmd_h=0.55` (깊은 숙이기) → pitch 허용 0.3rad (~17°)
- roll(좌우)은 항상 제한

---

## 4. Joint Range 조정 (Knee / AnklePitch) ✅ 구현 완료 (정적 확장)

### 변경 내용

**Knee (L/R):**
| 항목 | 기존 | 변경 |
|------|------|------|
| `deadband_pos` (굴곡) | 1.0 (q=1.35까지) | **1.5** (q=1.85까지) |
| `deadband_neg` (펴짐) | 0.20 | 0.20 (유지) |

**AnklePitch (L/R):**
| 항목 | 기존 | 변경 |
|------|------|------|
| `deadband_pos` (dorsiflexion) | 0.58 | **0.75** |
| `deadband_neg` (plantarflexion) | 0.24 | **0.40** |

모든 stiffness 값은 유지. 문제 발생시 height command에 연동하는 conditional 방식으로 전환 예정.

---

## 5. Latent Supervision 확장: vel4 + height1 ✅ 구현 완료 (Option C)

### 변경된 구조
```
기존: latent = [vel3(vx,vy,wz), foot6, future(30D), free(25D)]
변경: latent = [vel4(vx,vy,wz,vz), foot6, height1(h), future(30D), free(23D)]
                ├── MSE(vel4_pred, gt_vel4)       ← critic_obs[:, 0:4]
                ├── MSE(foot6_pred, gt_foot6)     ← critic_obs[:, 4:10]
                ├── MSE(height1_pred, gt_height1) ← critic_obs[:, 10:11]
                └── MSE(future_pred, target_enc)  ← target encoder
```

### Z축 command vs supervision 설계 근거
- **Command (pos_z)**: policy에게 "어디로 갈지" 알려줌 → height_command
- **Latent supervision (vel_z + pos_z)**: encoder가 "현재 높이와 변화율" 학습
  - pos_z만: 높이는 알지만 올라가는/내려가는 중인지 구분 불가
  - vel_z만: 변화율은 알지만 목표 대비 현재 위치 불명
  - **둘 다**: 높이 상태를 완전히 파악 → 가장 강력

### 수정 파일 상세

| 파일 | 변경 |
|------|------|
| `mdp/observations.py` | `base_vel_xy_z_yawrate` (4D), `base_height_1d` (1D) 추가 |
| `rough_env_cfg.py` CriticCfg | `gt_vel3` → `gt_vel4`, `gt_height1` 추가 |
| `ac_future.py` | `vel_dim=4`, `height_dim=1`, `split_latent` 4-tuple 반환 |
| `ppo_future.py` | `w_height` 추가, `height_loss` 계산/로깅 |
| `rsl_rl_ppo_cfg.py` | `w_height=1.0`, dims/scales 업데이트 |

### Dimension 변경 요약

| 항목 | 기존 | 변경 |
|------|------|------|
| Policy obs (single frame) | 47D | **48D** (+height_cmd) |
| Target obs | 154D | **155D** (+height_cmd) |
| Critic obs | 160D | **163D** (+vel_z, +height1, +height_cmd) |
| Latent supervised prefix | 9D (vel3+foot6) | **11D** (vel4+foot6+height1) |
| Latent free space | 25D | **23D** |
| `num_single_obs` | 47 | **48** |

---

## 6. 구현 순서

### Phase 1~3: 통합 구현 ✅ 완료
- [x] `UniformHeightCommand` 구현 (command term)
- [x] Policy/Target/Critic observation에 height command 추가
- [x] Height reward를 command-conditional로 수정
- [x] `flat_orientation_height_adaptive_l2` 구현
- [x] Knee/Ankle deadband 확장 (정적)
- [x] Latent에 vel4 + height1 supervision 추가 (Option C)
- [x] `rsl_rl_ppo_cfg.py` dim/scale 업데이트

### Phase 4: 검증 (다음 단계)
- [ ] **고정 높이(0.89)로 학습** → 기존 성능 regression 없는지 확인
  - `rel_default_height_envs=1.0`으로 설정하면 사실상 기존과 동일
- [ ] Height range를 좁게 시작 (0.80~0.89) → 학습 확인
- [ ] Height range 확대 (0.55~0.89) → 숙이는 동작 확인

### Phase 5: 실로봇 배포
- [ ] ONNX export 시 height command 포함 확인
- [ ] Teleop/joystick에 height 축 매핑
- [ ] 실로봇 안전 범위 설정 (높이 하한)

---

## 7. 전체 수정 파일 목록

| 파일 | 변경 내용 | 상태 |
|------|-----------|------|
| `mdp/height_command.py` | 신규 — UniformHeightCommand, UniformHeightCommandCfg | ✅ |
| `mdp/__init__.py` | height_command export 추가 | ✅ |
| `mdp/observations.py` | `base_vel_xy_z_yawrate` (4D), `base_height_1d` (1D) | ✅ |
| `mdp/rewards.py` | `track_height_command_exp`, `track_height_command_below_l2`, `flat_orientation_height_adaptive_l2` | ✅ |
| `rough_env_cfg.py` | command 등록, obs 추가, reward 교체, joint deadband 확장 | ✅ |
| `algorithms/rsl_rl/ac_future.py` | `vel_dim=4`, `height_dim=1`, `split_latent` 4-tuple | ✅ |
| `algorithms/rsl_rl/ppo_future.py` | `w_height`, `height_loss` 계산/로깅 | ✅ |
| `agents/rsl_rl_ppo_cfg.py` | dims, scales, `w_height=1.0` | ✅ |

---

## 8. 리스크 및 고려사항

1. **학습 불안정**: height range가 넓으면 reward landscape가 복잡해짐 → curriculum 필수
2. **넘어짐 증가**: 낮은 자세에서 ZMP 마진 감소 → 넘어짐 페널티 유지/강화
3. **Gait 패턴 변화**: 숙인 자세에서는 보폭/주기가 달라져야 함 → gait reward도 adaptive 고려
4. **Sim2Real gap**: 낮은 자세에서 관절 토크가 크게 증가 → motor strength DR 확인
5. **Orientation vs Height coupling**: orientation을 너무 풀면 학습 초기에 이상한 자세 학습 가능
