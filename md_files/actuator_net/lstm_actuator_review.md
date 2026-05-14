# LSTM Actuator 구현 리뷰 (P73 Walker)

학습 스크립트(`actuator_net_comparison.py`) ↔ 배포 구현(`p73_walker.py`, `rough_env_cfg.py`, `actuators.py`) 의 입·출력 규약과 상태 관리가 일치하는지 확인한 기록.

## 관련 파일

- 학습/검증 스크립트 (stateless per-joint 추론):
  - `/home/piene/Github_Code/pace-sim2real/scripts/pace/actuator_net_comparison.py`
- 학습 파이프라인 (실제 LSTM을 학습시킨 코드):
  - `/home/piene/Github_Code/actuatornet/actuator_net/train_lstm.py`
  - `/home/piene/Github_Code/actuatornet/actuator_net/utils.py`
- 배포 구현:
  - `/home/piene/isaaclab5.2/isaaclab_walker/source/isaaclab_walker/isaaclab_walker/actuators.py` (`ActuatorNetLSTMWalker`)
  - `/home/piene/isaaclab5.2/isaaclab_walker/source/isaaclab_walker/isaaclab_walker/assets/p73_walker.py`
  - `/home/piene/isaaclab5.2/isaaclab_walker/source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/rough_env_cfg.py`
- Isaac Lab 기본 구현 (비교용):
  - `/home/piene/isaaclab5.2/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_net.py` (`ActuatorNetLSTM`)
  - `/home/piene/isaaclab5.2/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py` (`DelayedPDActuator` 등)

---

## 1. 학습 코드의 입·출력 규약

`utils.py`

### 입력 feature

`prepare_data_for_joint_group()` LSTM 분기 (line 378-384):

```python
else:   # model_type == "lstm"
    for i in joint_indices:
        xs_parts.append(joint_position_errors[:, i:i+1])   # (T, 1)
        xs_parts.append(joint_velocities[:, i:i+1])        # (T, 1)
    xs = torch.cat(xs_parts, dim=1).unsqueeze(1)   # (T, 1, 2)
    ys = tau_ests[:, joint_indices]                # (T, 1)
    return xs, ys
```

- 조인트별 feature 순서: `[pos_err, vel]`
- `.unsqueeze(1)` → shape `(T, seq_len=1, features=2)`
- **seq_len = 1** — 한 샘플 = 1 tick 짜리 "시퀀스"

### 출력 타깃

`load_experiments()` / `load_single_experiment()` (line 288, 312):

```python
tau_ests[i, :] = np.array(datas[i]["joint_efforts"][1:]) * torque_scaling   # 0.01
```

- 타깃은 실제 토크에 **× 0.01**이 적용된 값
- 따라서 네트워크 출력은 실제 토크의 **1/100 스케일**
- 추론 시에는 출력에 **× 100**을 곱해야 Nm로 복원

### pos_err의 정의

`load_single_experiment()` (line 316):

```python
jpe = torch.tensor(joint_targets - joint_positions, dtype=torch.float)
```

→ `pos_err = 지령값(target) - 측정값(encoder reading)`

---

## 2. "Stateless로 학습됐다"는 주장의 **확정 증거**

LSTM을 시간에 걸쳐 기억을 쌓는 모델로 쓰려면 (a) `seq_len > 1`로 학습하거나 (b) TBPTT로 state를 이월해야 한다. 이 코드는 **둘 다 아니다**.

### 증거 1 — 학습 샘플 하나가 "1 tick"짜리

`prepare_data_for_joint_group()` line 382:

```python
xs = torch.cat(xs_parts, dim=1).unsqueeze(1)   # (T, 1, 2)
```

`.unsqueeze(1)`이 중간에 크기 1짜리 축을 만들어 `(T, seq_len=1, features=2)`로 만듦. 샘플 하나 = `[pos_err, vel]` 두 숫자 = 1 tick 시퀀스. 시간 히스토리가 들어있지 않음. MLP 분기(367-377행)는 history를 concat하지만 LSTM 분기에서는 안 함.

### 증거 2 — 샘플 간 시간 순서가 깨진다

`train_actuator_network()` line 127-130:

```python
dataset = ActuatorDataset({"joint_states": xs, "tau_ests": ys})
train_set, val_set = random_split(dataset, [num_train, num_test])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
```

`random_split`이 T개 시점을 무작위로 쪼개므로 `shuffle=False`여도 배치 안의 64개 샘플이 시간적으로 연속이 아님. 샘플 간에 state를 이을 근거가 없음.

`ActuatorDataset.__getitem__` (line 34-35):

```python
def __getitem__(self, idx):
    return {k: v[idx] for k, v in self.data.items()}
```

→ DataLoader가 B개 모아 `(B, 1, 2)`로 배치화. 각 샘플 `(1, 2)` 한 덩어리 유지.

### 증거 3 — forward 호출 시 state를 안 넘기고 반환값도 버린다

`train_actuator_network()` line 177-200:

```python
for batch in train_loader:
    data = batch['joint_states'].to(device)   # (B, 1, 2)
    if model_type == 'lstm':
        y_pred, _ = model(data)               # state 인자 없음, (h_new, c_new)는 '_'로 버림
    ...
for batch in test_loader:
    data = batch['joint_states'].to(device)
    if model_type == 'lstm':
        y_pred, _ = model(data)               # 여기도 동일
```

- 호출자가 `model(data, state=...)` 형태로 이전 state를 넣는 코드가 없음
- 반환된 `(h_new, c_new)`도 버려짐 → 다음 배치로 이월되는 경로 자체가 없음

### 증거 4 — state가 없으면 forward 내부에서 h=c=0으로 시작

`LSTMModel.forward` (line 96-103):

```python
def forward(self, x, state=None):
    if state is None:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        state = (h0, c0)
    out, (h, c) = self.lstm(x, state)
    out = self.fc(out[:, -1, :])
    return out, (h, c)
```

증거 3에서 state가 항상 None이므로 **학습 내내 모든 forward 호출이 `(h0, c0)=(0, 0)`으로 시작**.

### 종합 결론

> **한 forward 호출당**
> - 입력: `(B, seq_len=1, 2)` — 증거 1, 2
> - 초기 hidden: `(h0, c0) = (0, 0)` — 증거 3, 4
> - LSTM이 timestep 1번만 돌고 `out = fc(h_1)` 반환
> - 다음 호출에서도 `(h, c) = (0, 0)` 재시작 — 증거 3 (`_`로 버림)

네트워크는 학습 도중 **(h_prev, c_prev)가 0이 아닌 경우를 본 적이 없다**. 배포 시에도 매 step h/c를 0으로 리셋해야 학습과 같은 입력 분포가 된다.

### 참고: 일반적인 stateful LSTM 학습 형태 (여기서는 아님)

**방법 A — 긴 시퀀스를 한 번에** (`seq_len > 1`):
```python
xs.shape == (B, 100, 2)    # 100 tick
y_pred, _ = model(xs)      # LSTM이 내부에서 h, c를 100번 갱신
```

**방법 B — TBPTT로 state 이월**:
```python
state = None
for batch in sequential_loader:   # shuffle 금지, 시간순
    y_pred, state = model(batch, state)
    state = (state[0].detach(), state[1].detach())
```

---

## 3. 배포 구현 체크

### `p73_walker.py`

- `_LSTM_NETWORK_FILES`: 관절명 ↔ `p73_lstm_<group>.pt`가 학습 때 `JOINT_GROUPS` 이름(`left_hip_roll`, `left_knee_pitch`, …)과 **전부 일치** ✓
- `torque_scale=100.0` ✓ (학습 `torque_scaling=0.01`의 역)
- `effort_limit`, `velocity_limit_sim`, `armature` 등은 P73 스펙으로 설정됨
- **친구/댐핑 값은 테스트 스크립트와 다름** — 어느 쪽이 최신 sysid 값인지 확인 필요
  - 예: `L_HipYaw` viscous — 테스트 스크립트 1.7339 vs p73_walker 0.0009

### `ActuatorNetLSTMWalker.compute()` (`actuators.py`)

**맞는 부분**
- 입력 feature 순서·shape: `torch.stack([pos_err[:,j], joint_vel[:,j]], dim=-1).unsqueeze(1)` → `(num_envs, 1, 2)` ✓
- 조인트별 별도 네트워크 루프, 가중치 파일 경로 해석 ✓
- 출력 스케일 `torques * self._torque_scale` (기본 100.0) ✓
- LUT(knee/ankle) 기반 effort clipping, 모터 강도 랜덤화는 `DelayedPDActuatorLUT`에서 상속 ✓

**문제점**

#### (1) hidden/cell state가 stateful로 흘러감 — 가장 중요

```python
# compute() 안
h_in = self.sea_hidden_state[j]   # 이전 step의 값 유지
c_in = self.sea_cell_state[j]
out, (h_new, c_new) = net(x, (h_in, c_in))
self.sea_hidden_state[j] = h_new
self.sea_cell_state[j] = c_new
```

`reset(env_ids)`는 env reset 시점에만 0으로 만듬. 학습은 seq_len=1 + 매 샘플 h=c=0 → 배포 중 non-zero h가 들어가면 OOD (out-of-distribution).

**수정 방안**: `compute()` 진입 시 매번 0으로 밀기.
```python
def compute(self, control_action, joint_pos, joint_vel):
    self.sea_hidden_state.zero_()
    self.sea_cell_state.zero_()
    ...
```

#### (2) position/velocity delay 버퍼가 target에만 적용

```python
control_action.joint_positions = self.positions_delay_buffer.compute(...)
...
pos_err = control_action.joint_positions - joint_pos
```

학습 데이터는 이미 실기 command↔encoder 레이턴시를 내장한 상태로 기록됨. 여기서 target만 추가 지연시키면 **레이턴시 double-count** + pos_err 분포 왜곡.

**수정 방안**: LSTM 경로에서 position/velocity delay를 끄거나(`min_delay=max_delay=0`), 최소한 target에만 적용하지 말 것.

#### (3) encoder bias 미적용

`ActuatorNetLSTMWalker`는 `pos_err = target - joint_pos(raw sim)`. 반면 테스트 스크립트 및 `PaceDCMotor`는 `pos_err = target - (joint_pos - encoder_bias)`.

- sim URDF zero = 실기 커맨드 zero 라면 현 구현 OK
- 실기 캘리브레이션 오프셋이 있다면 bias를 빼줘야 함

**수정 방안**: `ActuatorNetLSTMWalkerCfg`에 `encoder_bias` 필드를 두고 `pos_err = control_action.joint_positions - (joint_pos - encoder_bias)`로 맞추기 (테스트 스크립트와 완전히 동일한 경로로).

#### (4) motor strength 랜덤화 × LSTM

```python
self.applied_effort = self.applied_effort * self.motor_strength_scale   # U(0.8, 1.2)
```

LSTM이 이미 실기 출력 분포를 반영하는데 위에 ±20% scale을 또 곱함. Domain randomization 목적이면 유지, LSTM 결과 정확 재현이면 꺼야 함.

### `rough_env_cfg.py`

- `P73_CFG`가 `scene.robot`으로 그대로 들어가고 `actuators["walker_motors"] = ActuatorNetLSTMWalkerCfg(...)`
- policy는 `LowerBodyActionsCfg`로 `q_default + Δ`를 target으로 세팅
- 이 target이 `_apply_actuator_model()` → `ActuatorNetLSTMWalker.compute()`의 `control_action.joint_positions`로 들어감 → **입력 루트 정합** ✓
- `observations.*.motor_joint_vel`은 `1/30` 스케일, `actions`는 `last_processed_action` — LSTM 입력에는 스케일 없이 raw가 들어가므로 무관 ✓
- reward/event 쪽에 LSTM 입출력에 영향 주는 config 없음 ✓

---

## 4. 요약 체크리스트

| 항목 | 학습 규약 | Walker 구현 | 판정 |
|---|---|---|---|
| 입력 dim / 순서 | `(B,1,2)` `[pos_err, vel]` | `(num_envs,1,2)` `[pos_err, vel]` | ✓ |
| 조인트별 분리 네트워크 | 12개 per-joint .pt | 12개 per-joint .pt, 이름 매칭 | ✓ |
| 출력 스케일 | 학습 시 ×0.01 | 배포 시 ×100.0 | ✓ |
| hidden/cell state | 매 샘플 zero (stateless) | env reset 시에만 zero (**stateful**) | ✗ |
| pos_err 기준 프레임 | encoder frame | raw sim frame (encoder_bias 미적용) | ⚠ |
| command delay buffer | 없음 (실기 데이터에 내장) | target에 추가 delay 적용 | ⚠ |
| motor strength rand | 없음 | ±20% 곱 | 의도 확인 필요 |
| LUT clipping | 없음 | 유지 | ✓ (안전장치) |
| 친구/댐핑 sysid 값 | 테스트 스크립트 값 | p73_walker 값(다름) | ⚠ 값 원천 확인 |

---

## 5. 우선순위별 수정 항목

1. **(최우선) hidden state 매 step 0으로 리셋** — `ActuatorNetLSTMWalker.compute()` 맨 앞에 `self.sea_hidden_state.zero_(); self.sea_cell_state.zero_()` 추가. 이걸 안 하면 네트워크가 학습 때 본 적 없는 상태에서 추론하게 됨.

2. **delay 버퍼 재검토** — LSTM 경로에서 position/velocity delay를 끄거나 학습 convention과 맞추기.

3. **encoder bias 경로 추가** — 학습 데이터가 실기 인코더 프레임이라면 `joint_pos`에서 `encoder_bias`를 빼주는 경로를 `ActuatorNetLSTMWalkerCfg`에 명시.

이 세 가지를 반영하면 `actuator_net_comparison.py`의 stateless per-joint 추론과 수치적으로 동일한 경로가 된다.

---

## 부록: LSTM hidden/cell state 개념

LSTM은 매 step 두 개의 내부 "기억"을 들고 다닌다:

- **hidden state `h_t`** — "단기 기억". 이번 step 출력(`out_t`)과 강하게 연결된 상태.
- **cell state `c_t`** — "장기 기억". 게이트를 통해 천천히 변하면서 긴 시퀀스 정보를 보존.

shape: `(num_layers, batch, hidden_dim)`.

한 step 수식:
```
(h_t, c_t) = LSTMCell(x_t, (h_{t-1}, c_{t-1}))
out_t      = Linear(h_t)
```

"stateful 배포"는 step 간 `(h, c)`를 이어가는 방식. "stateless 배포"는 매 step `(h, c)=0`으로 재시작하는 방식. 어느 쪽이 맞는지는 **학습이 어떻게 됐는지**에 따라 결정되며, 이 LSTM은 위 증거들에 의해 stateless로 배포해야 맞다.
