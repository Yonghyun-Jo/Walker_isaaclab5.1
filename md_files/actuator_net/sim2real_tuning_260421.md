# Actuator Net Sim2Real 튜닝 작업 기록 (260421)

LSTM actuator net 의 sim 배포와 실기 학습 스펙 사이의 간극을 좁히기 위해 수행한
조사 · 측정 · 코드 수정 내역.

> **주의 (260421 반영)**: 초기 계획에 있던 `sensor_noise_*` (actuator net 입력에 실기
> 수준의 잡음을 주입하는 항목) 은 **학계 표준이 아님**을 확인하고 **구현을 되돌렸음**.
> Hwangbo 2019 (ANYmal), Rudin 2022 (legged_gym), Isaac Lab 기본 `ActuatorNetLSTM`
> 전부 actuator net 입력에 noise 를 주입하지 않음. Noise 는 policy observation 에만
> 적용되는 것이 표준. 자세한 근거는 §11 참조.
>
> 따라서 실제 반영된 변경은 **`delay` 고정화** 와 **`rand_motor_scale_range` 축소**
> 두 가지만. §4 (sensor_noise_*) 는 "시도했다가 되돌린 기록" 으로 남겨둠.

## 관련 파일

- 수정 대상:
  - `source/isaaclab_walker/isaaclab_walker/actuators.py` (`ActuatorNetLSTMWalker`, `ActuatorNetLSTMWalkerCfg`)
  - `source/isaaclab_walker/isaaclab_walker/assets/p73_walker.py` (`P73_CFG.actuators["walker_motors"]`)
- 참조:
  - `source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/rough_env_cfg.py` (ObsTerm, EventCfg)
  - `/home/piene/ros2_ws/src/p73_cc/src/cc.cpp` (실기 controller)
  - `/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_*.csv` (5 세션 실기 데이터, 1 kHz)
  - chirp eval 스크립트 (pace-sim2real) — `update_time_lags`, `update_encoder_bias` 호출부
- 관련 기존 문서:
  - `md_files/actuator_net/lstm_actuator_review.md`

---

## 1. 배경 — 이번 작업의 출발점

기존에 actuator net 을 켠 상태로 학습한 policy (`model_13200.pt`) 는 sim 에서
다음 증상을 보였음.

| 증상 | 증거 (acnet_sim_13200.csv vs pd_sim_15000.csv) |
|---|---|
| HipRoll 토크 폭주 | |τ| p99 = 53.7 → 247.7 Nm (4.6×) |
| HipRoll 24 Hz Nyquist 발진 | FFT peak @ 24.3/24.0/25.0 Hz (Nyquist = 25 Hz) |
| Yaw tracking 붕괴 | err_wz std 0.088 → 0.517, track_ang_z_exp 평균 0.97 → 0.57 |
| 전 관절 |Δτ| 2–9× 증가 | 특히 HipRoll/HipPitch 가 극심 |

원인 조사 결과 여러 sim↔학습 spec 불일치가 누적되어 있었음. 이 중 이번 턴에서
즉시 해결 가능한 것들만 반영. 나머지는 §8 에 미결 사항으로 남김.

---

## 2. 조사 결과 — LSTM 입·출력 규약 재확인

사용자의 의심("LSTM 에 raw q 가 들어가는 것 아닌가?")을 해소하기 위해
네 소스를 교차 검증:

| 소스 | 코드 | feature 0 |
|---|---|---|
| IsaacLab stock `ActuatorNetLSTM.compute()` (`actuator_net.py:81`) | `(control_action.joint_positions - joint_pos).flatten()` | **pos_err** |
| IsaacLab stock `ActuatorNetMLP.compute()` (`actuator_net.py:152`) | `control_action.joint_positions - joint_pos` | **pos_err** |
| `ActuatorNetLSTMWalker.compute()` (현재 배포, `actuators.py:269`) | `control_action.joint_positions - joint_pos_noisy` | **pos_err** |
| 실기 `cc.cpp:842, 863` | `rd_.q_desired(i) - q_noise_(i)` | **pos_err** |
| Chirp eval (pace-sim2real) | stock 클래스 호출 → 내부에서 자동 pos_err | **pos_err** |

**결론**: 네 경로 모두 `[pos_err, vel]` 2-D 입력 convention (Hutter ANYmal 방식).
학습도 동일 convention 이라고 볼 근거가 매우 강함.

### Policy obs 파이프라인은 actuator 와 완전 분리
`rough_env_cfg.py` 의 `motor_joint_pos`, `motor_joint_vel` 에 붙은
`Unoise(±0.01)`, `Unoise(±1.5)`, `scale=1/30`, `clip(-30,30)` 은
**policy 입력에만** 적용되고 actuator/LSTM 에는 한 푼도 도달하지 않음.

`articulation.py:1886-1890` 에서 `actuator.compute(control_action, joint_pos=self._data.joint_pos, joint_vel=self._data.joint_vel)` 로 호출될 때
`self._data.joint_pos/vel` 은 **PhysX 원본 값** 이기 때문.

→ 실기 학습 시 LSTM 이 본 sensor noise 를 sim 에 재현하려면 **actuator 내부에서
별도 주입** 필요.

---

## 3. 실기 5 개 CSV 분석 — 주입할 noise 정량화

### 대상
```
realrobot_260420_185028.csv   (1000 Hz, static=5.3s, moving=20.8s)
realrobot_260420_183702.csv   (1000 Hz, static=2.1s, moving=38.6s)
realrobot_260420_183421.csv   (1000 Hz, static=3.5s, moving=25.2s)
realrobot_260420_183206.csv   (1000 Hz, static=20.0s, moving=22.5s)
realrobot_260420_173557.csv   (1008 Hz, static=4.2s, moving=72.7s)
```

### 측정 방법
- Cmd-velocity 기반으로 static/moving 구간 분할
- 50 ms moving-average 를 빼서 HF residual 추출 (`res = x - MA(x, 50 ms)`)
- Residual 의 per-joint std 계산
- 5 파일 간 **MAX** 를 취해 worst-case 커버

### 중요한 발견: "static raw std" 는 부적절
- 서보 hold oscillation 이 함께 찍힘 → q̇ noise 가 **과대 평가**
- 예: L_Knee static raw std = 0.78 rad/s 였음 (85028 세션)
- 같은 joint 의 moving HF residual std = 0.21 rad/s 정도

**MOVING HF residual MAX 가 가장 방어 가능한 기준**: 동작 중 실제 잡음 성분만 격리.

### 측정 결과 (5 파일 MOVING HF residual std MAX)

| Joint | σ_pos [rad] | σ_vel [rad/s] |
|---|---|---|
| L_HipRoll   | 0.00128 | 0.0551 |
| L_HipPitch  | 0.00166 | 0.0828 |
| L_HipYaw    | 0.00158 | 0.1634 |
| L_Knee      | 0.00411 | 0.2089 |
| L_AnklePitch| 0.00321 | 0.2775 |
| L_AnkleRoll | 0.00261 | 0.3266 |
| R_HipRoll   | 0.00126 | 0.0452 |
| R_HipPitch  | 0.00154 | 0.0820 |
| R_HipYaw    | 0.00164 | 0.1487 |
| R_Knee      | 0.00295 | 0.1367 |
| R_AnklePitch| 0.00252 | 0.2392 |
| R_AnkleRoll | 0.00246 | 0.3492 |

### Uniform half-range 변환
`Unoise([-a, a])` 의 std = `a/√3` 이므로 variance-matched 를 위해 `a = σ × √3`.
L/R 쌍은 안전한 쪽인 MAX 로 대칭화.

---

## 4. 변경 사항 ①: `sensor_noise_pos` / `sensor_noise_vel` 추가

### `actuators.py`

1. `ActuatorNetLSTMWalkerCfg` 에 필드 2 개 신설:
   ```python
   sensor_noise_pos: dict[str, float] | float | None = None
   sensor_noise_vel: dict[str, float] | float | None = None
   ```
   dict (관절별) / float (전 관절 공통) / None (비활성) 허용.

2. `ActuatorNetLSTMWalker.__init__` 에 dict → tensor 변환 헬퍼 추가:
   ```python
   def _resolve_joint_scalar(self, val):
       if val is None: return None
       if isinstance(val, dict):
           arr = [float(val[jn]) for jn in self.joint_names]
           return torch.tensor(arr, device=self._device, dtype=torch.float32).unsqueeze(0)
       return torch.full((1, self.num_joints), float(val), device=self._device, dtype=torch.float32)
   ```

3. `compute()` 의 delay buffer 적용 직후, pos_err 계산 직전에 주입:
   ```python
   if self._sensor_noise_pos is not None:
       n = (torch.rand_like(joint_pos) * 2.0 - 1.0) * self._sensor_noise_pos
       joint_pos_noisy = joint_pos + n
   else:
       joint_pos_noisy = joint_pos
   if self._sensor_noise_vel is not None:
       n = (torch.rand_like(joint_vel) * 2.0 - 1.0) * self._sensor_noise_vel
       joint_vel_noisy = joint_vel + n
   else:
       joint_vel_noisy = joint_vel

   pos_err = control_action.joint_positions - joint_pos_noisy
   # LSTM 에는 joint_vel_noisy 도 함께 투입
   ```

### `p73_walker.py` — 관절별 값 (5-CSV MOVING HF residual MAX × √3, L/R 대칭)

```python
sensor_noise_pos={                          sensor_noise_vel={
    "L_HipRoll_Joint":    0.002,                "L_HipRoll_Joint":    0.10,
    "L_HipPitch_Joint":   0.003,                "L_HipPitch_Joint":   0.15,
    "L_HipYaw_Joint":     0.003,                "L_HipYaw_Joint":     0.30,
    "L_Knee_Joint":       0.007,                "L_Knee_Joint":       0.36,
    "L_AnklePitch_Joint": 0.006,                "L_AnklePitch_Joint": 0.50,
    "L_AnkleRoll_Joint":  0.005,                "L_AnkleRoll_Joint":  0.60,
    "R_HipRoll_Joint":    0.002,                "R_HipRoll_Joint":    0.10,
    "R_HipPitch_Joint":   0.003,                "R_HipPitch_Joint":   0.15,
    "R_HipYaw_Joint":     0.003,                "R_HipYaw_Joint":     0.30,
    "R_Knee_Joint":       0.007,                "R_Knee_Joint":       0.36,
    "R_AnklePitch_Joint": 0.006,                "R_AnklePitch_Joint": 0.50,
    "R_AnkleRoll_Joint":  0.005,                "R_AnkleRoll_Joint":  0.60,
},                                          },
```

### 원래 추정치(첫 시도)와의 차이

중간에 첫 번째 파일(`185028`) static stretch 만으로 추정했을 때는:
```
pos: HipRoll 0.001, HipPitch 0.001, HipYaw 0.002, Knee 0.002, AnklePitch 0.0014, AnkleRoll 0.0035
vel: HipRoll 0.28,  HipPitch 0.67,  HipYaw 0.99,  Knee 1.06,  AnklePitch 0.63,   AnkleRoll 1.02
```

이 값은 **pos 과소 / vel 과대** 였음:
- `185028` 은 5 파일 중 정지 중 서보 진동이 유난히 큰 세션 (outlier)
- static raw std 는 서보 hold 진동까지 포함 → q̇ noise 과대
- pos 은 1 개 파일만 보니 세션 편차를 놓침

5-CSV MOVING HF residual MAX 로 바꾸면서 pos 1.4–4× ↑, vel 0.22–0.79× ↓.

---

## 5. 변경 사항 ②: Command delay 를 고정 = 1 로

### 근거
Chirp eval 스크립트:
```python
time_lag = torch.tensor([[round(0.5598)]], dtype=torch.int, device=device)  # = 1
articulation.actuators[drive_type].update_time_lags(time_lag)
```
LSTM 가중치는 **delay = 1 physics step** 조건에서 검증됨.

### Before → After
```python
# p73_walker.py
# Before
min_delay=0, max_delay=2,   # uniform [0,1,2] 샘플링, 1/3 envs 가 LSTM 이 못 본 delay

# After
min_delay=1, max_delay=1,   # 고정 = LSTM training/eval 과 일치
```

### 효과
- 4096 envs 전체가 동일한 delay 분포에서 동작 → OOD 요인 하나 제거
- Policy 강건성은 base mass / CoM / physics material / push_robot 에서 이미 충분히 확보

---

## 6. 변경 사항 ③: `rand_motor_scale_range` 축소

### 근거
Actuator net 이 실기 데이터에서 학습 → **실기 모터 동역학(토크 상수, 기어비,
마찰, 드라이버 response, stiction, backlash)** 을 이미 내재. 런타임 ±20% 스케일을
또 곱하면 **이중 변동성** → policy 가 과도하게 약한 제어를 학습.

"Actuator net 이 흡수하지 못하는 실제 변동성" 의 현실적 범위:
- 다른 개체 간 모터 편차: ±5 % 내외
- 배터리 전압 / 온도 drift: ±수 %
- 장기 마모: 수 % 내외

→ **±5% (0.95, 1.05)** 가 적절.

### Before → After
```python
# Before
rand_motor_scale_range=(0.8, 1.2),

# After
rand_motor_scale_range=(0.95, 1.05),
```

---

## 7. 변경하지 않은 사항 & 이유

| 항목 | 위치 | 이유 |
|---|---|---|
| ObsTerm noise (`±0.01`, `±1.5`) / scale / clip | `rough_env_cfg.py:650-666` | Policy obs 전용 파이프라인. Actuator 와 분리됨. 이 값은 **policy robustness DR 목적**으로 설계된 것. 실기 sensor noise 와 별개. |
| `randomize_armature (0.6, 1.4)` | `rough_env_cfg.py:860` | Armature = PhysX rotor inertia (토크 → 가속도 gain). Actuator net 의 torque 출력 이후 sim 동역학 파라미터라 별개. 실기 추정치 불확실 → DR 유지. |
| `randomize_damping (stiffness=0, damping=0)` | `rough_env_cfg.py:870` | 이미 `(0.0, 0.0)` 로 disable. Actuator net 이 damping 을 내재하므로 올바른 상태. |
| Hidden state persistence | `actuators.py:281-282` | 학습 시 rollout 구조 확인 필요. stateless 로 바꿔야 할지 아직 불확실. `lstm_actuator_review.md` 참조. |
| Encoder bias 보정 | `actuators.py:269` | chirp eval 에서 `update_encoder_bias` 호출 사실은 확인. 학습 시에도 사용됐는지는 training code 확인 전까지 유보. |

---

## 8. 미결 사항 (향후 작업)

1. **Hidden state 처리 검증**
   - 학습 루프가 sequence-level rollout 인지, sliding-window per-sample 인지 확인
   - `/home/piene/Github_Code/actuatornet/actuator_net/train_lstm.py` 열어서 `hidden = ...`
     초기화 시점 및 BPTT 구조 파악
   - stateful 이 맞다면 현재 구현 OK. stateless / short-window 면 수정 필요.

2. **Encoder bias 값 확인**
   - Chirp eval `bias_full = [0.1, 0.0084, -0.0833, ...]` 이 관절 mount offset 인지 확인
   - Training data 전처리에서 실제로 뺐는지 여부 확인
   - 그렇다면 `ActuatorNetLSTMWalkerCfg.encoder_bias` 신설 필요

3. **Impulse / step response 로 LSTM 자체 안정성 테스트**
   - `.pt` 파일 12 개를 독립적으로 로드
   - q̇ = 0 을 수 초간 유지 → torque 가 수렴하는지 발진하는지
   - impulse 자극 → 감쇠 시정수 측정
   - 발진하면 architecture level 문제 → reward tuning 으로는 해결 불가

4. **주입 noise 효과 검증 (학습 시작 직후)**
   - 현재 `model_13200.pt` 로 재학습 없이 play
   - `tools/collect_metric_data_csv.py` 로 CSV 수집
   - 기존 `acnet_sim_13200.csv` 와 비교:
     - HipRoll 24 Hz peak 완화 여부
     - HipRoll |τ| p99 감소 여부
     - err_wz 정상화 여부
   - 증상 해소되면 옵션 A 만으로 충분. 여전하면 §8-1,2 진행.

5. **Armature 실기 추정**
   - 실기 중력보상 + 정지 실험에서 effective rotor inertia 역산
   - `(0.6, 1.4)` 범위를 좁힐 수 있는지 검토

---

## 9. 검증 절차 (학습 전 5 분 체크)

학습 시작 전에 반드시 실행:

```bash
python scripts/tools/p73_command_control/play_with_teleop_p73.py \
  --task=Walker-Flat-Play --num_envs=1 \
  --checkpoint=logs/rsl_rl/walker_flat/2026-04-21_03-42-58/model_13200.pt \
  --control_mode gui --real-time 2>&1 \
  | python tools/collect_metric_data_csv.py \
      --out logs/metric/acnet_noise_fixed_delay_13200.csv \
      --dt 0.02 --warmup_s 3 --collect_s 30
```

이전 `acnet_sim_13200.csv` 와 diff 비교. 다음을 지표로 삼음:

| 지표 | 기대 변화 |
|---|---|
| HipRoll FFT peak @ 24 Hz | 감쇠 또는 사라짐 |
| HipRoll |τ| p99 | 247 Nm → 현저히 감소 (100 이하 바람직) |
| HipRoll |Δτ| p99 | 329 Nm → 100 이하로 감소 |
| err_wz std | 0.52 → 0.2 이하 |
| `track_ang_z_exp` 평균 | 0.57 → 0.8 이상 |

위 지표가 유의미하게 호전되면 reward tuning 단계로 진입 가능.
호전되지 않으면 §8-1 (hidden state), §8-2 (encoder bias) 순으로 진단.

---

## 10. 적용 파일 변경 요약 (diff 관점)

### `source/isaaclab_walker/isaaclab_walker/actuators.py`
- `ActuatorNetLSTMWalkerCfg` 에 `sensor_noise_pos`, `sensor_noise_vel` 필드 추가 (default=None → 기존 동작 호환)
- `ActuatorNetLSTMWalker.__init__` 에 `_resolve_joint_scalar` 헬퍼 + 두 필드 tensor 등록
- `ActuatorNetLSTMWalker.compute()` 에 uniform noise 주입 블록 추가 (delay 버퍼 이후, pos_err 계산 이전)

### `source/isaaclab_walker/isaaclab_walker/assets/p73_walker.py`
- `"walker_motors": ActuatorNetLSTMWalkerCfg(...)` 에
  - `min_delay`: 0 → 1
  - `max_delay`: 2 → 1
  - `rand_motor_scale_range`: (0.8, 1.2) → (0.95, 1.05)
  - `sensor_noise_pos=...` (관절별 dict 12 entry)
  - `sensor_noise_vel=...` (관절별 dict 12 entry)

### `source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/rough_env_cfg.py`
- **변경 없음** (ObsTerm, EventCfg, Rewards 모두 유지)

### `tools/collect_metric_data_csv.py`
- 이미 존재 (actuator_net_lim / custom_regulate_actionrate 양쪽 브랜치에 커밋됨)
- 변경 없음, 검증용으로 재사용

---

## 부록 A. 분석 재현 스크립트 (5 CSV 처리)

```python
import pandas as pd, numpy as np

FILES = [
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_185028.csv",
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_183702.csv",
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_183421.csv",
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_183206.csv",
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_173557.csv",
]
JOINTS = ["L_HipRoll","L_HipPitch","L_HipYaw","L_Knee","L_AnklePitch","L_AnkleRoll",
          "R_HipRoll","R_HipPitch","R_HipYaw","R_Knee","R_AnklePitch","R_AnkleRoll"]

def ma_hp(x, win):
    k = np.ones(win)/win
    return x - np.convolve(x, k, mode='same')

def longest_run(mask):
    idx = np.where(mask)[0]
    if len(idx)==0: return None, None
    runs, s, e = [], idx[0], idx[0]
    for k in idx[1:]:
        if k==e+1: e=k
        else: runs.append((s,e)); s=k; e=k
    runs.append((s,e))
    runs.sort(key=lambda r: r[1]-r[0], reverse=True)
    return runs[0]

pos_by = {j: [] for j in JOINTS}
vel_by = {j: [] for j in JOINTS}
for fp in FILES:
    df = pd.read_csv(fp)
    dt = np.median(np.diff(df['time'].values))
    win = max(5, int(round(0.05/dt)))
    moving = (np.abs(df['cmd_vx'])>0.05) | (np.abs(df['cmd_vy'])>0.05) | (np.abs(df['cmd_vyaw'])>0.05)
    s, e = longest_run(moving.values)
    for j, lab in enumerate(JOINTS):
        x = df[f'q_raw_{j}'].values[s:e+1]
        v = df[f'qdot_{j}'].values[s:e+1]
        pos_by[lab].append(float(np.std(ma_hp(x, win)[win:-win])))
        vel_by[lab].append(float(np.std(ma_hp(v, win)[win:-win])))

print(f"{'Joint':<13} {'pos_MAX':>10} {'pos*sqrt3':>12} {'vel_MAX':>10} {'vel*sqrt3':>12}")
for lab in JOINTS:
    pmx, vmx = max(pos_by[lab]), max(vel_by[lab])
    print(f"{lab:<13} {pmx:10.5f} {pmx*np.sqrt(3):12.5f} {vmx:10.4f} {vmx*np.sqrt(3):12.4f}")
```

---

## 부록 B. 주요 증거 스니펫

### 실기 controller 에서 pos_err 사용 (`cc.cpp:842, 863-872`)
```cpp
double pos_err = rd_.q_desired(i) - q_noise_(i);        // 100 Hz history shift
double pos_err_now = rd_.q_desired(i) - q_noise_(i);    // 1 kHz forward pass

anet_input << pos_err_now,
              anet_pos_err_hist_[i][0],   // -10 ms
              anet_pos_err_hist_[i][1],   // -20 ms
              vel_now,
              anet_vel_hist_[i][0],
              anet_vel_hist_[i][1];
```

### IsaacLab stock `ActuatorNetLSTM.compute()` 에서 pos_err 사용
```python
self.sea_input[:, 0, 0] = (control_action.joint_positions - joint_pos).flatten()
self.sea_input[:, 0, 1] = joint_vel.flatten()
```

### Chirp eval 의 delay = 1 고정
```python
time_lag = torch.tensor([[round(0.5598)]], dtype=torch.int, device=device)
articulation.actuators[drive_type].update_time_lags(time_lag)
```

### Articulation 이 actuator.compute 에 raw physics 값 전달 (`articulation.py:1886-1890`)
```python
control_action = actuator.compute(
    control_action,
    joint_pos=self._data.joint_pos[:, actuator.joint_indices],   # PhysX raw
    joint_vel=self._data.joint_vel[:, actuator.joint_indices],   # PhysX raw
)
```

---

*작성: 2026-04-21. 다음 단계는 §9 의 5 분 체크로 증상 호전 여부 확인.*

---

## 11. 부기: `sensor_noise_*` 를 되돌린 근거 (학계 조사 결과)

§4 에서 구현했던 "실기 5 CSV 기반 sensor noise 주입" 은 학계 조사 이후 **되돌림**.
이 단락은 향후 같은 실수를 하지 않기 위한 기록.

### 조사 대상 문헌
1. Hwangbo et al. 2019 "Learning agile and dynamic motor skills for legged robots" (Science Robotics) — ANYmal 원조
2. Rudin et al. 2022 "Learning to walk in minutes ..." (CoRL) — `legged_gym` / Isaac Gym
3. Miki et al. 2022 "Learning robust perceptive locomotion ..." (Science Robotics)
4. Margolis et al. 2022 "Walk These Ways" (arXiv)
5. Isaac Lab 기본 `ActuatorNetLSTM` / `ActuatorNetMLP` 소스

### 주요 발견
- **어떤 문헌도 actuator net 입력에 noise 를 주입하지 않음**
- Hwangbo 2019: Noise 는 **policy observation** 에만 주입 (예: joint vel obs 에 `U(-0.5, 0.5)` rad/s)
  - Actuator net 학습 데이터 자체는 raw real-robot data (augmentation 없음)
- Rudin 2022 `legged_gym/envs/anymal_c/anymal.py`: `_compute_torques` 가 actuator net 에
  raw simulator state (`dof_vel`, `default_dof_pos - dof_pos`) 를 **직접** 투입. Noise 없음.
- Isaac Lab 기본 `ActuatorNetLSTM.compute()`: noise 주입 코드 없음. Cfg 필드에도 noise 관련
  파라미터 없음 (`pos_scale`, `vel_scale`, `torque_scale`, `input_idx` 만).

### 논리적 근거 (Hwangbo 맥락)
> Actuator net 은 실기 동역학의 **identified model**. 입력에 합성 noise 를 추가하면
> 식별된 동역학을 왜곡하지, 더 현실적으로 만들지 않는다.

즉 학습 시 input/output 쌍이 동일 noise 를 함께 보았기 때문에 net 은 **평균 매핑** 을
학습하고, deployment 시 clean input 을 주면 의미 있는 torque 를 출력하게 되어 있음.
OOD 우려는 perturbation 이 작을 때 well-behaved LSTM 의 interpolation 으로 해소됨.

### 실제 되돌린 변경
1. `p73_walker.py` 의 `sensor_noise_pos={...}`, `sensor_noise_vel={...}` dict 전체 제거
2. `actuators.py` `ActuatorNetLSTMWalker.compute()` 의 noise 주입 블록 제거
3. `actuators.py` `ActuatorNetLSTMWalker.__init__` 의 `_resolve_joint_scalar` 헬퍼 및
   `_sensor_noise_*` 필드 등록 제거
4. `actuators.py` `ActuatorNetLSTMWalkerCfg` 의 `sensor_noise_pos`, `sensor_noise_vel`
   필드 정의 제거

### 되돌린 뒤의 최종 반영 상태
- `min_delay = max_delay = 1` ✓ (유지 — chirp eval 근거 명확)
- `rand_motor_scale_range = (0.95, 1.05)` ✓ (유지 — actuator net 이중 변동성 논리)
- `sensor_noise_*` ✗ (되돌림 — 학계 표준 따름)

### 그래도 24 Hz 발진 원인으로 남은 후보 (우선순위)
1. 🔴 **Encoder bias 미적용** — chirp eval 에서 ±0.1 rad 까지 보정하는데 현재 배포엔 없음.
   Noise (±0.001) 의 100 배 크기라 훨씬 큰 OOD 요인.
2. 🟡 **Hidden state stateful vs 학습 컨벤션 mismatch** — training code 확인 필요
3. 🟢 **Policy 가 LSTM 학습 분포 밖 q̇ 영역 진입** — actuator 로는 못 고침, 재학습 필수

이 세 축 중 (1) 이 가장 큰 OOD 원인일 가능성이 높으므로 다음 턴에는 `encoder_bias`
관련 조사 및 적용이 우선 순위.

### 교훈
- 이론적 가설 (나: "실기 noise 분포 matching 필요") vs 학계 표준 (의외로 "matching 불필요")
- 증거 없는 직관적 판단은 확인 후 조정 필수
- 구현 인프라는 간단한 편이었지만 불필요한 DR 축을 넣는 것은 policy 학습에
  **유익보다는 혼란** 을 줄 가능성 높음
