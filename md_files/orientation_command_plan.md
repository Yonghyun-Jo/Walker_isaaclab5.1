---
date: 2026-05-13
project: isaaclab_walker
branch: acnet_zcmd (확장 예정)
tags: [plan, orientation_command, roll_pitch_yaw, isaaclab_walker, pose_command]
---

# Base Pose Command (height + RPY) 통합 계획

## 목표
기존 `height_command` (1D)를 **4D pose command [height, roll, pitch, yaw]**로 확장.
Latent supervision도 height1 → pose4로 확장하여 encoder가 자세 상태를 완전히 인코딩.
**Pitch부터 점진적으로** 활성화하고, roll/yaw는 command 구조만 미리 만들어두되 학습 초반에는 0 고정.

## 현재 구조

| 항목 | 현재 | 변경 후 |
|------|------|---------|
| Command | height(1D) | **pose [h, r, p, y] (4D)** |
| Policy obs | 48D | **51D** (+3: roll, pitch, yaw) |
| Critic obs | 163D | **166D** (+3) |
| Target obs | 155D | **158D** (+3) |
| Latent supervised | vel4+foot6+height1=11D | **vel4+foot6+pose4=14D** (+3) |
| Latent free space | 23D | **20D** (-3) |

---

## 설계 결정

### 왜 4D 통합 (방안 C)?
- Height와 orientation은 물리적으로 coupling (숙이면 pitch도 바뀜)
- 하나의 command term으로 관리하면 resampling 동기화 자연스러움
- Latent에서도 `gt_pose4`로 한 블록으로 supervision → 깔끔

### 점진적 활성화 전략
1. **Phase 1 (현재)**: height만 학습 (roll=0, pitch=0, yaw=0 고정)
2. **Phase 2**: pitch 활성화 (roll=0, yaw=0 고정) → 림보/줍기
3. **Phase 3**: roll + yaw 활성화 → 완전 자세 제어

`rel_default_pose_envs` 비율로 조절:
- Phase 1: roll/pitch/yaw 100% default(0) → 사실상 height만
- Phase 2: pitch range [-0.3, 0.1], roll/yaw 100% default(0)
- Phase 3: 전체 range 활성화

---

## 구현 계획

### 1. Command: UniformBasePoseCommand (4D)

기존 `UniformHeightCommand`를 `UniformBasePoseCommand`로 교체:

```python
class UniformBasePoseCommand(CommandTerm):
    """Target base pose: [height, roll, pitch, yaw] (4D)."""
    # command shape: (num_envs, 4)

class UniformBasePoseCommandCfg(CommandTermCfg):
    class Ranges:
        height: tuple[float, float] = (0.55, 0.89)
        roll:   tuple[float, float] = (-0.2, 0.2)
        pitch:  tuple[float, float] = (-0.3, 0.1)
        yaw:    tuple[float, float] = (-0.3, 0.3)

    default_pose: tuple[float, float, float, float] = (0.89, 0.0, 0.0, 0.0)
    # 각 축별 default 비율 (학습 초반 roll/yaw 고정용)
    rel_default_height_envs: float = 0.5
    rel_default_roll_envs: float = 1.0    # Phase 1-2: 100% default
    rel_default_pitch_envs: float = 1.0   # Phase 1: 100% → Phase 2: 0.5
    rel_default_yaw_envs: float = 1.0     # Phase 1-2: 100% default
```

### 2. Observation

```
Policy obs: 51D (기존 48D에서 height 1D → pose 4D, +3D)
[0:3]   ang_vel_b (3)
[3:6]   projected_gravity_b (3)
[6:9]   velocity_commands [vx, vy, wz] (3)
[9:13]  pose_command [h, roll, pitch, yaw] (4)  ← 기존 height(1D) 확장
[13:15] gait_phase [sin, cos] (2)
[15:27] joint_pos_rel (12)
[27:39] joint_vel_scaled (12)
[39:51] last_action (12)
```

### 3. Reward 수정

**`flat_orientation_height_adaptive_l2` → `orientation_pose_command_tracking`으로 교체:**

```python
def orientation_pose_command_tracking(env, command_name, std_pitch=0.05, std_roll=0.05, ...):
    """
    Roll/Pitch: pose command tracking (bell curve)
    command[1]=roll, command[2]=pitch
    roll=0, pitch=0일 때 기존 flat_orientation과 동일 동작
    """
    grav_b = asset.data.projected_gravity_b  # (N, 3)
    cmd = env.command_manager.get_command(command_name)
    roll_cmd = cmd[:, 1]   # target roll (projected gravity x)
    pitch_cmd = cmd[:, 2]  # target pitch (projected gravity y)

    # projected gravity에서 target과의 차이
    roll_error = grav_b[:, 0] - roll_cmd_to_gravity_x(roll_cmd)
    pitch_error = grav_b[:, 1] - pitch_cmd_to_gravity_y(pitch_cmd)

    roll_reward = exp(-(roll_error / std_roll)^2)
    pitch_reward = exp(-(pitch_error / std_pitch)^2)

    return roll_reward + pitch_reward
```

**Height tracking reward**: 기존 유지 (command_name만 변경)
```python
track_height_exp: command[:, 0] (height 슬롯)
track_height_below_l2: command[:, 0]
```

### 4. Symmetry (51D)

```python
single_frame_dim = 51
# [9:13] pose_command [h, roll, pitch, yaw]
pose_cmd = obs[:, start+9 : start+13]
pose_cmd_flipped = pose_cmd * torch.tensor([+1, -1, +1, -1])
# height: +1 (symmetric)
# roll:   -1 (좌우 반전)
# pitch:  +1 (symmetric)
# yaw:    -1 (좌우 반전)
```

### 5. Latent Supervision: height1 → pose4

**변경 전:**
```
latent = [vel4, foot6, height1, future30, free23] = 64D
         critic_obs[:, 0:4]  [:, 4:10]  [:, 10:11]
```

**변경 후:**
```
latent = [vel4, foot6, pose4, future30, free20] = 64D
         critic_obs[:, 0:4]  [:, 4:10]  [:, 10:14]
```

- `ac_future.py`: `height_dim=1` → `pose_dim=4`
- `split_latent`: `(vel4, foot6, pose4, future)` 4-tuple 유지 (이름만 변경)
- `ppo_future.py`: `height_loss` → `pose_loss`, `w_height` → `w_pose`
- Critic obs에 `gt_pose4` = [height, roll_actual, pitch_actual, yaw_actual]

**gt_pose4 observation 함수:**
```python
def base_pose_4d(env, asset_cfg=...):
    """[height, roll, pitch, yaw] in world frame (4D)."""
    h = asset.data.root_pos_w[:, 2:3]          # height
    rpy = euler_xyz_from_quat(root_quat_w)      # (roll, pitch, yaw)
    return torch.cat([h, rpy[0:1], rpy[1:2], rpy[2:3]], dim=-1)
```

### 6. Critic Obs

```
Critic prefix: [vel4(4), foot6(6), pose4(4)] = 14D  (기존 11D에서 +3)
이후: base_ang_vel(3) + gravity(3) + vel_cmd(3) + pose_cmd(4) + gait(2) + ...
```

### 7. Dimension 요약

| 항목 | 현재 (height) | 변경 후 (pose) | 차이 |
|------|:---:|:---:|:---:|
| num_single_obs | 48 | **51** | +3 |
| Policy obs (×10) | 480 | **510** | +30 |
| Target obs | 155 | **158** | +3 |
| Critic obs | 163 | **166** | +3 |
| Latent supervised prefix | 11D | **14D** | +3 |
| Latent free space | 23D | **20D** | -3 |
| latent_dim | 64 | 64 | 0 (충분) |

---

## 수정 파일

| 파일 | 변경 |
|------|------|
| `mdp/height_command.py` | → `mdp/pose_command.py` (4D로 확장) |
| `mdp/__init__.py` | export 변경 |
| `mdp/observations.py` | `base_pose_4d()` 추가 |
| `mdp/rewards.py` | `orientation_pose_command_tracking()` 추가 |
| `mdp/symmetry.py` | 51D, pose [+1,-1,+1,-1] |
| `rough_env_cfg.py` | command 교체, obs 수정, reward 수정 |
| `flat_env_cfg.py` | pose range 설정 |
| `ac_future.py` | `height_dim=1` → `pose_dim=4`, split_latent |
| `ppo_future.py` | `w_height` → `w_pose`, `pose_loss` |
| `rsl_rl_ppo_cfg.py` | dims (51/158/166), scales, `w_pose` |
| `p73_cc/src/cc.cpp` | obs 51D, pose command 4D |
| `p73_cc/include/cc.h` | `target_pose_[4]` |
| `p73_cc/scripts/walker_teleop.py` | pitch/roll/yaw 조절 키 |

---

## 점진적 활성화 순서

### Phase 1: 구조만 4D로 확장, height만 활성
- [h, r=0, p=0, y=0] — `rel_default_{roll,pitch,yaw}_envs = 1.0`
- 기존 height 학습과 동일 동작 확인 (regression test)
- latent에서 pose4 supervision 시작 (roll/pitch/yaw는 0 근처)

### Phase 2: Pitch 활성화
- `rel_default_pitch_envs = 0.5` → pitch range [-0.3, 0.1]
- `orientation_pose_command_tracking`에서 pitch tracking 활성
- 림보/숙이기 동작 검증
- roll/yaw는 여전히 0 고정

### Phase 3: Roll + Yaw 활성화
- `rel_default_roll_envs = 0.7`, `rel_default_yaw_envs = 0.7`
- range를 보수적으로 시작 (roll ±0.1, yaw ±0.15)
- 점차 확대

### Phase 4: 실로봇 배포
- p73_cc obs 51D 반영
- Teleop에 pitch/roll/yaw 조절 추가
- 안전 범위 설정

---

## RPY 좌표계 기준 선택

### 후보

| 기준 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **Global frame** | 세계 좌표계 기준 Euler angles | 직관적 (pitch=0 → 수평) | 로봇 이동 방향과 무관하게 고정, yaw가 heading과 섞임 |
| **Gravity-aligned yaw-only frame** | roll/pitch는 중력 기준, yaw는 로봇 heading 기준 | roll/pitch가 "기울기" 의미 명확, yaw는 로봇 기준 | IsaacLab의 projected_gravity_b와 자연스럽게 대응 |
| **Body frame** | 완전히 로봇 local | 센서 직접 매핑 | 직관성 떨어짐 |

### 추천: Gravity-aligned yaw-only frame (추천)

**이유:**
- IsaacLab의 `projected_gravity_b`가 이미 이 frame 기준 (body frame에 투영된 중력 벡터)
- `projected_gravity_b[:, 0]` ≈ `sin(roll)`, `projected_gravity_b[:, 1]` ≈ `sin(pitch)` (small angle)
- 로봇이 어느 방향을 향하든 "앞으로 숙이기" = pitch, "옆으로 기울기" = roll
- 실로봇 IMU도 이 방식으로 roll/pitch를 출력
- **Command가 heading-independent**: 로봇이 북쪽을 보든 남쪽을 보든 pitch=0.2는 "앞으로 0.2rad 숙이기"

**Yaw 기준:**
- Yaw는 gravity-aligned frame에서의 "heading 대비 상대 회전"
- 현재 WaistYaw_Joint이 이 역할을 하지만, base 전체의 yaw는 heading과 coupling
- **실용적 선택**: yaw command는 WaistYaw_Joint의 target offset으로 사용
  - 즉 "현재 heading 기준으로 상체를 ±0.3rad 돌리기"

### Command → Reward 매핑

```
roll_cmd  → target projected_gravity_b[:, 0] ≈ sin(roll_cmd)
pitch_cmd → target projected_gravity_b[:, 1] ≈ sin(pitch_cmd)
yaw_cmd   → target WaistYaw_Joint offset (또는 base heading offset)
```

Small angle에서 `projected_gravity ≈ sin(angle) ≈ angle`이므로,
command를 **radian 단위**로 주고 reward에서 `projected_gravity`와 직접 비교 가능.

---

## GUI (Isaac Sim + p73_cc teleop)

### Isaac Sim GUI (`unified_command_gui_p73.py`)

현재: `[vx, vy, wz]` 슬라이더 + `[height]` 슬라이더

**추가:**
```
=== Base Pose Command ===
Height (m):    [0.55 ←Crouch ====|======== Stand→ 0.89]  0.89
Roll (rad):    [-0.2 ←Left   ====|======== Right→ 0.2]   0.00
Pitch (rad):   [-0.3 ←Forward===|======== Back→  0.1]   0.00
Yaw (rad):     [-0.3 ←Left   ====|======== Right→ 0.3]   0.00
```

- 기존 Height 슬라이더를 "Base Pose Command" 섹션으로 통합
- Roll/Pitch/Yaw 슬라이더 3개 추가
- Reset 시 default pose (0.89, 0, 0, 0) 복귀
- `env.command_manager.get_command("base_pose")[:, 0:4]`로 직접 write

### p73_cc Teleop (`walker_teleop.py`)

현재: `r/f` (height), `LB/RB` (height)

**확장:**
```
키보드:
  r/f     : height 올리기/내리기 (기존)
  t/g     : pitch 앞으로/뒤로   (신규)
  (roll/yaw는 Phase 3에서 추가)

조이스틱:
  LB/RB   : height (기존)
  LT/RT   : pitch 앞으로/뒤로  (신규, 트리거 아날로그)
  (roll/yaw는 Phase 3에서 추가)
```

**Twist 메시지 매핑:**
```
Twist.linear.x  → vel_x
Twist.linear.y  → vel_y
Twist.linear.z  → height (기존)
Twist.angular.x → roll_cmd   (신규)
Twist.angular.y → pitch_cmd  (신규)
Twist.angular.z → vel_yaw (기존, velocity)
```

주의: `angular.z`는 이미 velocity yaw로 사용 중.
Orientation yaw command는 별도 토픽이 필요하거나, 미사용 필드가 없음.
→ **Phase 3에서 yaw 추가 시** 별도 토픽 `/p73/cmd_pose` (geometry_msgs/Pose) 고려.
→ 또는 Twist 2개 publish: `/p73/cmd_vel` + `/p73/cmd_pose`

---

## 주의: Yaw latent supervision 좌표계 문제 ⚠️

현재 `gt_pose4`의 yaw는 `heading_w` (world frame 절대 heading)를 사용하고 있음.
이는 로봇이 이동하면 yaw 값이 계속 바뀌어서, latent supervision target이 비정상적으로 큼.

**문제**:
- yaw command는 "heading 대비 상대 offset" (예: 0.3rad만 돌려)
- gt_pose4의 yaw는 절대 heading (예: 1.5rad, 3.0rad, ...)
- MSE loss에서 둘의 scale 차이가 너무 큼

**해결 방안 (Phase 3 진입 시 검토)**:
1. gt_pose4의 yaw를 WaistYaw joint angle로 대체 (상대적)
2. 또는 gt_pose4의 yaw를 command와의 error로 대체
3. 또는 yaw를 latent supervision에서 제외하고 pose3(h,r,p)만 사용

**Phase 1-2에서는**: yaw command가 0 고정이고 heading_w가 상대적으로 작은 범위에서 변하므로
큰 문제 없음. Phase 3 진입 전에 반드시 수정 필요.

---

## Reward 구성 (최종, 검증 완료)

### Task tracking rewards (모두 exp bell-curve, POSITIVE weight)
| Term | Weight | Std | Command Source | 비고 |
|------|--------|-----|----------------|------|
| `track_lin_vel_xy_exp` | +3.5 | 0.5 | base_velocity | vx, vy |
| `track_ang_vel_z_exp` | +2.0 | 0.5 | base_velocity | wz |
| `track_height_exp` | +3.0 | 0.05 | base_pose[:, 0] | height |
| `track_roll_exp` | +3.0 | 0.05 | base_pose[:, 1] | roll (Phase 1: cmd=0) |
| `track_pitch_exp` | +3.0 | 0.05 | base_pose[:, 2] | pitch (Phase 1: cmd=0) |
| `track_yaw_exp` | +1.0 | 0.1 | base_pose[:, 3] | yaw (Phase 1: cmd=0) |

**기존 `flat_orientation_l2` (w=-30) 제거됨** → exp tracking만으로 대체.
Phase 1에서 roll/pitch/yaw cmd=0이면 `exp(-err²/std²)` 자체가 flat 보상 역할.
벌점(-30) 대비 신호 약하지만, Phase 2 활성화 시 유연성 확보.

---

## 리스크

1. **학습 폭발**: 4D command 동시 학습은 난이도 높음
   → Phase별 점진적 활성화로 대응
2. **전복 위험**: roll/pitch 과하면 즉시 넘어짐
   → command range 보수적 시작, termination 조건
3. **Orientation 신호 약화**: 기존 L2 벌점(-30) 제거 → exp만으로 부족할 수 있음
   → Phase 1 학습 결과 보고 벌점 추가 여부 결정
4. **Latent 용량**: free space 23→20D (충분하지만 모니터링 필요)
5. **Yaw latent supervision**: heading_w 절대값 문제 → Phase 3 전 수정 필수 (위 참조)
6. **Sim2Real**: 기울어진 자세에서 IMU drift 영향 커짐
