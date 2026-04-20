# ONNX Export 시 주의할 부분

## 핵심 문제: Actor만 Export하면 안 된다

IsaacLab의 기본 `export_policy_as_onnx`는 `policy.actor`만 export한다.
일반적인 Actor-Critic 구조에서는 이것으로 충분하지만,
**ActorCriticAdaptationFuture** 같은 encoder 기반 구조에서는 전체 추론 파이프라인이 빠진다.

---

## 네트워크 구조 비교

### 일반 Actor-Critic (문제 없음)
```
obs → actor → actions
```
- `export_policy_as_onnx`가 actor만 export해도 OK
- 입력 = obs 차원 그대로

### ActorCriticAdaptationFuture (주의 필요)
```
history_obs(470D) → normalizer → encoder → latent(64D)
                                           ↘
                   current_obs(마지막 47D) + latent(64D) → actor → actions(12D)
```
- actor의 입력은 `47 + 64 = 111D`이지만, 이건 중간 표현
- 실제 배포 시 입력은 **470D** (47D × 10 history frames)
- normalizer + encoder가 반드시 포함되어야 함

---

## 차원 정리

| 구성 요소 | 차원 | 설명 |
|-----------|------|------|
| Single-frame obs | 47D | ang_vel(3) + gravity(3) + cmd(3) + gait_phase(2) + motor_pos(12) + motor_vel(12) + actions(12) |
| History length | 10 | `PolicyCfg.__post_init__`에서 `self.history_length = 10` |
| **Full history obs** | **470D** | 47 × 10 = ONNX 입력 차원 |
| Encoder output (latent) | 64D | `latent_dim=64` in config |
| Actor input | 111D | current_obs(47) + latent(64) — 내부 중간값 |
| **Actions (출력)** | **12D** | 12 lower-body joints |

---

## 기본 Exporter가 실패하는 이유

`isaaclab_rl/rsl_rl/exporter.py`의 `_OnnxPolicyExporter`:

```python
# 이 코드가 하는 것:
self.actor = copy.deepcopy(policy.actor)  # actor MLP만 복사
# ...
def forward(self, x):
    return self.actor(self.normalizer(x))  # normalizer → actor (111D → 12D)

# export 시:
obs = torch.zeros(1, self.actor[0].in_features)  # (1, 111) ← 잘못됨!
```

- `actor[0].in_features = 111` (actor MLP의 첫 레이어 입력)
- encoder가 없으므로 latent를 외부에서 계산해서 넣어줘야 하는 구조가 됨
- 배포 환경(로봇)에서는 encoder가 없으므로 사용 불가

---

## 올바른 Export 방식

전체 추론 파이프라인을 하나의 Module로 감싸서 export:

```python
class _FullPipelineExporter(torch.nn.Module):
    def __init__(self, policy_nn):
        super().__init__()
        self.normalizer = copy.deepcopy(policy_nn.actor_obs_normalizer)
        self.encoder = copy.deepcopy(policy_nn.encoder)
        self.actor = copy.deepcopy(policy_nn.actor)
        self.num_single_obs = policy_nn.num_single_obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.normalizer(obs)            # (1, 470) → (1, 470)
        latent = self.encoder(obs)            # (1, 470) → (1, 64)
        current_obs = obs[:, -self.num_single_obs:]  # (1, 47)
        actor_input = torch.cat((current_obs, latent), dim=-1)  # (1, 111)
        return self.actor(actor_input)        # (1, 111) → (1, 12)
```

- 입력: `(1, 470)` — history obs 그대로
- 출력: `(1, 12)` — actions
- `act_inference()`와 동일한 연산

---

## 체크리스트

export 전에 반드시 확인할 것:

1. **Normalizer 포함 여부**: `EmpiricalNormalization`의 running mean/var가 ONNX에 bake-in 되는지 확인
2. **입력 차원 확인**: `num_actor_obs` = `num_single_obs × history_length` (47 × 10 = 470)
3. **current_obs 추출**: `obs[:, -num_single_obs:]` — 마지막 프레임이 현재 obs
4. **Encoder 가중치**: 학습된 encoder 가중치가 정상적으로 복사되었는지
5. **CPU 이동**: export 전에 `.to("cpu")` + `.eval()` 필수
6. **검증**: export 후 ONNX 모델의 입출력 shape 확인
   ```python
   import onnx
   model = onnx.load("policy.onnx")
   # input: [1, 470], output: [1, 12] 이어야 함
   ```

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `scripts/tools/p73_command_control/play_with_teleop_p73.py` | export 호출 (custom full-pipeline export 포함) |
| `source/.../algorithms/rsl_rl/ac_future.py` | `ActorCriticAdaptationFuture` 네트워크 정의 |
| `source/.../agents/rsl_rl_ppo_cfg.py` | 아키텍처 하이퍼파라미터 (latent_dim, num_single_obs 등) |
| `IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py` | 기본 exporter (actor만 export — encoder 구조에 부적합) |
