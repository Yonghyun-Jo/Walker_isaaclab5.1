# Angle-Dependent Effort Limit (4-Bar Linkage Torque LUT)

## 1. 배경

P73 walker의 Knee, Ankle 관절은 4-bar linkage 구조로 motor와 joint가 직결되지 않는다.
이 구조에서는 motor torque가 joint torque로 변환될 때 **전달비(transmission ratio)가 joint 각도에 따라 비선형적으로 변한다**.

기존 구현에서는 `effort_limit_sim`에 고정값(예: Knee=220 Nm)을 사용했으나,
실제로는 각도에 따라 ~144~274 Nm 범위로 변한다.

### 핵심 수식: Virtual Work Principle (가상일의 원리)

Motor와 joint의 순간 power가 동일하다는 조건에서 유도:

```
τ_motor × dθ_motor = τ_joint × dθ_joint
→ τ_joint_limit = τ_motor_limit × |dθ_motor / dθ_joint|
```

| Joint | 구조 | 수식 |
|-------|------|------|
| Knee | 1-DOF 4-bar | `τ_j = τ_m × \|dθ_m/dθ_j\|` (scalar 전달비) |
| Ankle | 2-DOF coupled 4-bar | `[τ_pitch, τ_roll]ᵀ = Gᵀ × [τ_m1, τ_m2]ᵀ` (2×2 Jacobian) |

`dθ_m/dθ_j`는 linkage 기하학에 의해 결정되며, 이 값이 CSV에 담겨 있다.


## 2. 파일 구조

```
assets/data/p73_walker/lut/
├── L_knee_lut_new.csv      # L Knee: 1000 rows
├── R_knee_lut_new.csv      # R Knee: 1000 rows
├── L_ankle_lut_new.csv     # L Ankle: 15456 rows (184 pitch × 84 roll)
└── R_ankle_lut_new.csv     # R Ankle: 15456 rows (184 pitch × 84 roll)

tasks/manager_based/isaaclab_walker/mdp/
├── torque_lut.py           # KneeTorqueLUT, AnkleTorqueLUT 클래스
└── actions.py              # LowerBodyActions에 LUT 통합 (수정)

tasks/manager_based/isaaclab_walker/
└── rough_env_cfg.py        # torque_lut_dir 경로 설정 (수정)
```


## 3. CSV 포맷

### Knee CSV (`L_knee_lut_new.csv`, `R_knee_lut_new.csv`)

| 컬럼 | 설명 |
|------|------|
| `joint_angle_rad` | joint 각도 [rad]. L: 0→1.57, R: 0→-1.57 |
| `motor_angle_rad` | 대응하는 motor 각도 [rad] |
| `transmission_ratio` | `\|dθ_motor/dθ_joint\|` |
| `torque_limit_nm` | `motor_limit(220) × transmission_ratio` |

- 1000개 데이터 포인트, 약 0.00157 rad 간격

### Ankle CSV (`L_ankle_lut_new.csv`, `R_ankle_lut_new.csv`)

| 컬럼 | 설명 |
|------|------|
| `pitch_rad` | ankle pitch 각도 [rad] |
| `roll_rad` | ankle roll 각도 [rad] |
| `tau_pitch_nm` | 해당 (pitch, roll)에서의 pitch torque limit [Nm] |
| `tau_roll_nm` | 해당 (pitch, roll)에서의 roll torque limit [Nm] |

- 184(pitch) × 84(roll) = 15456개 격자점
- L: pitch [-1.047, 0.785], roll [-0.419, 0.419]
- R: pitch [-0.785, 1.047], roll [-0.419, 0.419]


## 4. 코드 구조

### 4-1. `torque_lut.py` — LUT 클래스

```python
class KneeTorqueLUT:
    """1-DOF LUT: joint_angle → torque_limit (linear interpolation)"""
    def __init__(self, csv_path, device)  # CSV 로드 → GPU tensor 변환
    def query(self, joint_angle) -> torque_limit  # (num_envs,) → (num_envs,)

class AnkleTorqueLUT:
    """2-DOF LUT: (pitch, roll) → (tau_pitch, tau_roll) (bilinear interpolation)"""
    def __init__(self, csv_path, device)  # CSV 로드 → 2D grid GPU tensor 변환
    def query(self, pitch, roll) -> (tau_pitch, tau_roll)  # (num_envs,) each
```

**Interpolation 방식:**
- Knee: `torch.searchsorted`로 인접 knot 2개를 찾고, linear interpolation
- Ankle: pitch/roll 각 축에서 인접 knot을 찾고, 4개 꼭짓점에서 bilinear interpolation
- 모든 연산이 GPU batch 처리 → 성능 overhead 무시 가능

### 4-2. `actions.py` — LowerBodyActions 수정 사항

**추가된 Config 필드:**
```python
class LowerBodyActionsCfg:
    torque_lut_dir: str | None = None  # None이면 기존 고정 limit 사용
```

**추가된 메서드:**
```python
def _get_lower_torque_limits(self) -> torch.Tensor:
    """(num_envs, num_lower) shape의 torque limit 반환.
    LUT가 있으면 knee/ankle은 현재 각도 기반 동적 limit,
    나머지 joint(Hip 등)은 기존 고정값 유지."""
```

**변경된 흐름 (`apply_actions`):**

```
기존: tau_lower = clamp(tau, -fixed_limit, fixed_limit)
변경: tau_lower = clamp(tau, -dynamic_limit, dynamic_limit)
          └─ dynamic_limit = _get_lower_torque_limits()
                 ├─ HipRoll/HipPitch/HipYaw: 고정값 (352/220/95)
                 ├─ Knee: KneeTorqueLUT.query(현재 각도) → ~144~274 Nm
                 └─ Ankle: AnkleTorqueLUT.query(pitch, roll) → 가변
```

### 4-3. `rough_env_cfg.py` — 설정

```python
_TORQUE_LUT_DIR = os.path.join(P73_ASSETS_DATA_DIR, "p73_walker", "lut")

class ActionsCfg:
    joint_pos = mdp.LowerBodyActionsCfg(
        ...
        torque_lut_dir=_TORQUE_LUT_DIR,  # 이 한 줄 추가
    )
```


## 5. 동작 흐름 (매 simulation step)

```
1. policy → action (normalized -1~1)
2. PD controller → raw torque 계산
3. _get_lower_torque_limits() 호출:
   a. 기본값으로 static limits 복사 (num_envs, 12)
   b. joint_pos에서 현재 knee/ankle 각도 읽기
   c. Knee: 1D LUT interpolation → limits[:, knee_idx] 덮어쓰기
   d. Ankle: 2D LUT interpolation → limits[:, ankle_pitch/roll_idx] 덮어쓰기
4. torch.clamp(tau, -dynamic_limits, dynamic_limits)
5. motor strength randomization 적용
6. set_joint_effort_target() → PhysX에 전달
```


## 6. 대표 수치 (검증용)

### L_Knee

| joint_angle_rad | torque_limit_nm |
|-----------------|-----------------|
| 0.000 (0 deg)   | 249.88 |
| 0.350 (20 deg)  | 225.89 |
| 0.785 (45 deg)  | 195.91 |
| 1.000 (57 deg)  | 183.28 |
| 1.570 (90 deg)  | 144.48 |

### L_Ankle (roll=0 기준)

| pitch_rad | tau_pitch_nm | tau_roll_nm |
|-----------|-------------|-------------|
| -0.170    | 122.29      | 75.07       |
| 0.000     | 107.54      | 74.32       |
| 0.300     | 81.14       | 69.29       |


## 7. ON/OFF 전환

```python
# ON (angle-dependent)
torque_lut_dir=_TORQUE_LUT_DIR,

# OFF (기존 고정값으로 복귀)
torque_lut_dir=None,
```

`effort_limit_sim`의 값은 그대로 유지한다 (PhysX 레벨 safety clamp 역할).
실질적 angle-dependent clamp는 `apply_actions()` 내부의 `torch.clamp`에서 처리된다.
