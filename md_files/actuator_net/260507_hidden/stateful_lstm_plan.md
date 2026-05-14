# 260507 - LSTM Actuator Net: Stateless → Stateful 전환 구현 계획

## 배경

새로 학습된 LSTM actuator net은 기존 stateless 방식과 다르게 **hidden state를 누적하는 stateful 방식**으로 학습됨.

### 학습 방식 변경점
- **기존**: seq_len=1, 매 forward마다 h=c=0 (stateless). 사실상 MLP처럼 동작.
- **신규**: window 단위 zero-init + warm-up masking. 학습 시 window 시작에서 h=c=0으로 초기화하되, LSTM이 내부적으로 시간에 걸쳐 h,c를 누적. warm-up 구간(~10 step, 50ms)은 loss에서 masking하여 zero-init 직후의 부정확한 출력이 학습에 영향을 주지 않도록 함.
- **추론**: (h, c)를 step 간 유지. env reset 시에만 0으로 리셋.

### 새 네트워크 스펙
- 파일: 12개 per-joint `.pt` (파일명 동일, 내용물 교체)
- 아키텍처: layers=3, hidden=32, input=2 (전 관절 동일)
- 입력: `[pos_err, vel]` (2D, 기존과 동일)
- 출력 스케일: ×0.01로 학습 → 배포 시 ×100 (기존과 동일)

### Warm-up 특성
- 추론 초반 ~10 step (50ms @200Hz)은 h,c가 0에서 채워지는 warm-up 구간
- 학습 시 이 구간의 loss가 masking되었으므로 정확도가 떨어지는 것은 **의도된 동작**
- env reset 직후 약간의 토크 부정확성이 있을 수 있으나, 자연스럽게 수렴

---

## 수정 대상 파일

1. **`source/isaaclab_walker/isaaclab_walker/actuators.py`** — 핵심 수정
2. **`source/isaaclab_walker/isaaclab_walker/assets/p73_walker.py`** — 변경 없음 (파일명 동일, delay/scale 유지)

---

## 구현 계획

### Step 1: `__init__`에 persistent hidden/cell state 버퍼 할당

현재 상태:
```python
# NOTE on hidden state: ... We therefore do *not* allocate persistent hidden/cell buffers.
```

변경:
```python
# hidden/cell state 버퍼 할당 (per-joint × per-env)
# shape: (num_joints, num_layers, num_envs, hidden_dim)
self._lstm_num_layers = num_layers   # 이미 계산됨 (line 143)
self._lstm_hidden_dim = hidden_dim   # 이미 계산됨 (line 144)
self.sea_hidden_state = torch.zeros(
    self.num_joints, num_layers, self._num_envs, hidden_dim,
    device=self._device,
)
self.sea_cell_state = torch.zeros(
    self.num_joints, num_layers, self._num_envs, hidden_dim,
    device=self._device,
)
```

**왜 `(num_joints, num_layers, num_envs, hidden_dim)` 인가**:
- LSTM forward의 state shape은 `(num_layers, batch, hidden_dim)`
- 관절별 독립 네트워크이므로 첫 축이 joint index
- `self.sea_hidden_state[j]` 로 슬라이싱하면 바로 `(num_layers, num_envs, hidden_dim)`

### Step 2: `reset()`에서 해당 env_ids의 h,c를 0으로 리셋

현재 상태:
```python
# No LSTM hidden state to reset — see __init__ note
```

변경:
```python
self.sea_hidden_state[:, :, env_ids, :] = 0.0
self.sea_cell_state[:, :, env_ids, :] = 0.0
```

### Step 3: `compute()`에서 state를 전달하고 업데이트

현재 상태:
```python
for j, net in enumerate(self.networks):
    x = torch.stack([pos_err[:, j], joint_vel[:, j]], dim=-1).unsqueeze(1)
    out, _ = net(x, None)          # ← stateless
    torques[:, j] = out.squeeze(-1)
```

변경:
```python
for j, net in enumerate(self.networks):
    x = torch.stack([pos_err[:, j], joint_vel[:, j]], dim=-1).unsqueeze(1)
    h_in = self.sea_hidden_state[j]      # (num_layers, num_envs, hidden_dim)
    c_in = self.sea_cell_state[j]
    out, (h_new, c_new) = net(x, (h_in, c_in))   # ← stateful
    self.sea_hidden_state[j] = h_new
    self.sea_cell_state[j] = c_new
    torques[:, j] = out.squeeze(-1)
```

### Step 4: 주석/docstring 업데이트

- `__init__`의 NOTE 블록 수정: stateless → stateful 설명으로 교체
- `reset()`의 주석 수정
- `compute()`의 주석 수정
- 클래스 docstring 상단의 "Hidden state is preserved across physics steps and zeroed on env reset" 은 이미 맞으므로 유지

---

## 변경하지 않는 사항

| 항목 | 이유 |
|---|---|
| 입력 feature `[pos_err, vel]` | 학습 convention 동일 |
| `torque_scale = 100.0` | 동일 |
| `min_delay=1, max_delay=1` | LSTM 학습 조건 동일 |
| `rand_motor_scale_range=(0.95, 1.05)` | DR 목적 유지 |
| LUT clipping | 안전장치, 변경 불필요 |
| BEMF clipping | DCMotor 상속, 변경 불필요 |
| 네트워크 파일명/경로 | 동일 |
| `p73_walker.py` | 파일 자체 변경 불필요 (`.pt` 내용물만 교체됨) |

---

## 메모리 사용량 추정

Per-joint buffer: `num_layers × num_envs × hidden_dim × 4 bytes`
= 3 × 4096 × 32 × 4 = 1.5 MB

전체 (12 joints × h + c): 12 × 2 × 1.5 MB = **~36 MB**
→ 무시할 수 있는 수준.

---

## 검증 포인트

1. **기능 검증**: play 스크립트로 기존 체크포인트 로드 → LSTM 토크 출력이 발산하지 않는지 확인
2. **Warm-up 확인**: env reset 직후 ~10 step 동안 토크가 급변하지 않는지 로그 확인
3. **pace-sim2real 비교**: IsaacLab 배포의 토크 출력과 pace-sim2real 추론 결과가 수치적으로 일치하는지 (같은 입력 시퀀스에 대해)

---

## 참고: 기존 lstm_actuator_review.md와의 관계

기존 리뷰에서 "문제점 (1): hidden state가 stateful로 흘러감"으로 지적했던 사항이
이번 재학습으로 **의도적으로 stateful이 정답**이 됨. 리뷰 문서의 해당 섹션은
더 이상 유효하지 않음 (학습 코드 자체가 변경됨).
