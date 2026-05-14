# Stateful LSTM Warm-up 구간이 RL 학습 성능을 저하시키는 문제 분석

## 1. 문제 현상

Stateless → Stateful LSTM 전환 후 RL policy 학습 성능이 오히려 감소.

### 원인 가설

1. Env reset 시 h,c = 0 → LSTM 관점에서 **warm-up 구간** (~10 step, 50ms)
2. LSTM 학습 시 이 구간은 loss masking → 네트워크가 이 구간에서 **정확할 의무 없음**
3. RL 학습 중 매 에피소드 시작마다 이 구간이 반복 발생
4. Warm-up 동안 actuator 출력이 부정확 → policy가 경험하는 동역학이 불안정
5. Policy가 이에 대응해 **보수적**으로 학습됨 (큰 action 회피, 느린 수렴)

### 규모 추정

- 에피소드 길이: 1000 steps (20초)
- Warm-up: ~10 steps
- 비율: 1% (작아 보이지만...)
- **문제**: 에피소드 최초 10 step은 reward 수집의 "첫인상". 이 구간에서 불안정하면
  policy gradient에 noisy signal이 누적됨. 특히 `falling` termination이 warm-up
  부정확성으로 트리거되면 에피소드 전체가 날아감.
- 추가로 `init_at_random_ep_len=True`로 에피소드 길이가 분산되어, 4096 envs 중
  매 step 일부가 항상 warm-up 구간에 있음.

---

## 2. 학계 선행 연구

### 2.1 R2D2: Burn-in for RNN Hidden State (Kapturowski et al., 2019)

**"Recurrent Experience Replay in Distributed Reinforcement Learning"** (DeepMind)

가장 직접적으로 관련된 연구. Distributed RL에서 RNN policy를 사용할 때 replay
buffer에서 꺼낸 trajectory의 hidden state 초기화 문제를 다룸.

**핵심 기법: Burn-in**
- Trajectory를 replay할 때 처음 `B` step은 **loss 계산 없이** forward만 실행
- 이 구간에서 h,c가 0에서 의미 있는 값으로 수렴
- 수렴 후의 나머지 구간만 loss/gradient 계산에 사용

**우리 문제와의 대응**:
- R2D2의 "replay에서 h=0으로 시작" = 우리의 "env reset에서 h=0으로 시작"
- R2D2의 burn-in = 우리가 필요한 "reset 직후 grace period"
- 차이점: R2D2는 policy RNN의 hidden state, 우리는 actuator net의 hidden state

### 2.2 Hwangbo et al., 2019 (ANYmal Actuator Net)

원조 actuator net 논문. **Stateless** (MLP history window or per-tick LSTM) 사용.
Warm-up 문제 자체가 발생하지 않는 구조였음.

### 2.3 Recurrent Actor-Critic (Ni et al., 2022)

**"Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs"**

RNN policy + value function에서 hidden state 초기화의 영향을 체계적으로 분석.
결론: **zero-init hidden state는 초반 few steps에서 value estimation을 왜곡**하고
policy gradient의 bias를 유발. 해결책으로 burn-in 또는 learned initial state 제안.

### 2.4 Learned Initial Hidden State

여러 연구에서 h₀를 학습 가능한 파라미터로 두는 방법을 제안:
- `h₀ = MLP(initial_observation)` 또는
- `h₀ = learned_parameter` (fixed but optimized)

우리 경우: actuator net의 h₀는 이미 학습 완료된 frozen 모델이므로 직접 수정 불가.
대신 **reset 시 pre-warm**하는 방식으로 유사한 효과를 얻을 수 있음.

---

## 3. 해결 방안 (우선순위 순)

### 방안 A: Reset 시 Hidden State Pre-warm (가장 권장)

**아이디어**: `reset()` 시점에 h,c=0이 아니라, "정지 서 있는 상태"에 해당하는
pre-warmed h,c로 초기화.

**구현**:
```python
def reset(self, env_ids):
    ...
    # h,c를 0으로 리셋한 뒤, neutral input으로 N step forward 실행
    self.sea_hidden_state[:, :, env_ids, :] = 0.0
    self.sea_cell_state[:, :, env_ids, :] = 0.0

    # Pre-warm: 정지 상태 입력 (pos_err=0, vel=0)으로 warm-up 실행
    num_reset = len(env_ids) if not isinstance(env_ids, slice) else self._num_envs
    x_neutral = torch.zeros(num_reset, 1, 2, device=self._device)
    with torch.inference_mode():
        for _ in range(self.cfg.warm_up_steps):       # e.g. 10
            for j, net in enumerate(self.networks):
                h_in = self.sea_hidden_state[j, :, env_ids, :]
                c_in = self.sea_cell_state[j, :, env_ids, :]
                _, (h_new, c_new) = net(x_neutral, (h_in, c_in))
                self.sea_hidden_state[j, :, env_ids, :] = h_new
                self.sea_cell_state[j, :, env_ids, :] = c_new
```

**장점**:
- 에피소드 첫 tick부터 LSTM이 "수렴된" hidden state로 시작
- Policy가 보는 동역학이 처음부터 일관됨
- 코드 변경이 actuator 내부에 국한됨 (reward/env 수정 불필요)
- 학습 시에도 추론 시에도 동일하게 적용

**단점**:
- Reset마다 12 joints × 10 steps = 120 forward pass 추가 (한 번에 reset되는 env 수에 비례)
- "정지 상태"가 실제 reset 초기 상태와 다를 수 있음 (reset_joints에 ±0.1 offset 있음)

**성능 영향 추정**:
- 4096 envs 전체 reset (최초 1회): 12 × 10 × 4096 forward = ~490K forward
  → 실측 필요하지만 0.1~0.5초 수준 (1회성)
- 런타임 중 부분 reset: 평균 ~200 envs/step → 12 × 10 × 200 = 24K forward
  → 무시할 수준

**변형: 학습된 steady-state h,c 사용**:
- 더 정교한 버전: offline에서 실기 데이터(정지 구간)를 LSTM에 흘려서
  수렴된 h,c를 미리 계산 → 상수 텐서로 저장
- `reset()` 시 이 상수를 복사하면 forward pass 0회
- 단, 관절별/자세별로 달라질 수 있어 "평균 standing h,c" 하나로 충분한지 검증 필요

### 방안 B: Reward Grace Period (보조적)

**아이디어**: Reset 직후 N step 동안 reward를 0으로 마스킹.

**구현**: 이미 `episode_length_buf` 를 활용하는 인프라가 `rewards.py`에 있음
(`_tocabi_is_schedule_active_from_cmd`의 `grace_steps` 패턴).

```python
# rough_env_cfg.py 의 reward term들에 공통 적용
warm_up_grace_steps: int = 10  # 또는 cfg로
```

**장점**: 간단, actuator 코드 수정 불필요
**단점**:
- Reward만 마스킹해도 warm-up 중 action은 실행됨 → 물리적 불안정은 여전
- `falling` termination이 warm-up 중 발동하면 여전히 에피소드 손실
- 근본 해결이 아닌 증상 완화

### 방안 C: 학습 시에만 Stateless, 추론 시 Stateful (대안)

**아이디어**: RL 학습 중에는 기존처럼 `net(x, None)` (stateless)로 구동하여
안정적인 학습을 보장하고, sim2real 배포 시에만 stateful로 전환.

**근거**: Stateful LSTM의 이점은 **시간적 맥락**을 활용한 더 정확한 토크 예측.
학습 중에는 약간 부정확하더라도 **일관된** 동역학이 더 중요하고,
실기 배포 시 정확성을 높이는 것이 실익.

```python
# ActuatorNetLSTMWalkerCfg에 추가
stateful: bool = True  # False로 하면 기존 stateless 동작

# compute()에서 분기
if self.cfg.stateful:
    out, (h_new, c_new) = net(x, (h_in, c_in))
    ...
else:
    out, _ = net(x, None)
```

**장점**: 학습 안정성 확보, sim2real gap 축소 가능성
**단점**: 학습과 추론의 동역학이 다름 → 또 다른 sim2real gap 발생 가능

### 방안 D: Warm-up 구간에서 PD fallback

**아이디어**: Reset 직후 N step은 LSTM 대신 단순 PD 제어로 토크 생성.
N step 후 LSTM이 warm-up을 마치면 LSTM으로 전환.

**장점**: Warm-up 중 안정적 토크 보장
**단점**: PD→LSTM 전환 시 토크 불연속 발생 가능, 구현 복잡

---

## 4. 권장 조합

**1순위: 방안 A (Pre-warm)** — 근본 해결. Reset 시 LSTM을 neutral input으로
10 step 돌려서 h,c를 수렴시킨 뒤 에피소드 시작. Policy가 처음부터 일관된
동역학을 경험.

**보조: 방안 B (Grace Period)** — Pre-warm만으로 부족할 경우, 추가로
termination에 grace period를 두어 warm-up 중 `falling`으로 죽지 않도록 보호.
(현재 `falling` threshold = 0.6m, 정상 서있는 높이 0.895m → warm-up 50ms에
0.3m 떨어질 가능성은 낮지만 안전장치로.)

---

## 5. 구현 우선순위

1. **방안 A 구현** → `actuators.py`의 `reset()`에 pre-warm 루프 추가
2. **학습 실행** → 기존 stateless 대비 수렴 속도 비교
3. **부족하면 방안 B 추가** → reward grace period
4. **그래도 부족하면 방안 C 검토** → 학습 stateless / 추론 stateful 분리

---

## 6. 참고문헌

1. Kapturowski et al., "Recurrent Experience Replay in Distributed Reinforcement Learning", ICLR 2019 — burn-in 기법
2. Ni et al., "Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs", NeurIPS 2022 — RNN hidden init 영향 분석
3. Hwangbo et al., "Learning agile and dynamic motor skills for legged robots", Science Robotics 2019 — 원조 actuator net (stateless)
4. Heess et al., "Memory-based control with recurrent neural networks", NIPS 2015 — RNN policy의 hidden state 초기화 논의
