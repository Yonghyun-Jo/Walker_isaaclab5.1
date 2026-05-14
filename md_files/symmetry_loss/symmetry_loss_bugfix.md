# Mirror-Loss Symmetry Bugfix

## TL;DR

`_flip_lowerbody_12_axis_aware` + `p73_data_augmentation_lowerbody_mirror` 에 **총 5개 부호 버그**가 있었습니다. PPO mirror-loss 알고리즘 자체(`ppo_future.py`)는 정확했으나, 그 안에 주입되는 mirror 함수가 잘못된 부호를 냈기 때문에 `mirror_loss_coeff > 0`로 훈련하면 정책이 **진짜 sagittal mirror symmetry** 가 아닌 **일부 왜곡된 대칭**으로 수렴해왔습니다.

- 수정 파일: `source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/mdp/symmetry.py`
- 수정 지점: 함수 `_flip_lowerbody_12_axis_aware` (joint 3개 부호 반전), 함수 `p73_data_augmentation_lowerbody_mirror` (projected_gravity multiplier, gait_phase 부호 추가)
- 검증 방법: ① 레포 자체 테스트 스크립트 `scripts/tools/check_p73_symmetry.py` 의 hardcoded expectation, ② URDF `p73_walker.urdf` axis 로 돌린 forward-kinematics mirror-consistency 테스트, ③ `projected_gravity` / `gait_phase` 수학적 재유도
- 결과: 수정 전 모든 테스트 FAIL → 수정 후 모든 테스트 PASS (기계정밀도)

---

## 1. 배경: 증상과 가설

학습된 정책이 좌우 대칭이 살짝 아쉬운 현상이 관찰됐습니다. 원인 후보는 세 가지였는데:

1. `mirror_loss_coeff=2.0` 이 너무 낮다 (regularizer 세기 부족)
2. Mirror loss 구현 구조(PPO 쪽)가 틀렸다
3. Mirror 함수 내부의 부호 규칙이 틀렸다

1, 2 는 기존 코드 리뷰로 배제했고 (2는 `ppo_future.py` 구조가 교과서적으로 정확함을 확인), 3번이 유력해서 URDF 기반으로 전면 재검증했습니다.

---

## 2. 검증 결과

### 2.1 레포 자체 테스트가 이미 올바른 규칙을 기대하고 있었음

`scripts/tools/check_p73_symmetry.py` L107-110:

```python
x = torch.arange(12, dtype=torch.float32).view(1, 12) + 1.0  # [1,2,...,12]
y = mod._flip_lowerbody_12_axis_aware(x)
expected = torch.tensor([[7.0, -8.0, -9.0, -10.0, -11.0, -12.0,
                          1.0, -2.0, -3.0, -4.0, -5.0, -6.0]])
assert torch.allclose(y, expected)
print("[PASS] Lower-body mapping: HipRoll swap-only, others swap+negate")
```

이 expected 값을 해석하면:

| slot | expected | 규칙 |
|---|---|---|
| out[0] L_HipRoll | +R_HR (no negate) | **swap only** |
| out[1..5] L_HP..L_AR | −R_x | swap+negate |
| out[6] R_HipRoll | +L_HR | **swap only** |
| out[7..11] | −L_x | swap+negate |

즉 **HipRoll 하나만 swap only, 나머지 5개 모두 swap+negate**.

그러나 이 테스트 파일 L12 의 `REPO_ROOT = Path("/home/piene/p73/isaaclab_walker")` 는 **존재하지 않는 경로**라 walker 레포에서는 한 번도 실행된 적이 없었습니다 (`ls` 확인됨). 이 테스트는 아마 옛날 p73 프로젝트에서 통과했었고, walker로 포팅할 때 `_flip_lowerbody_12_axis_aware` 함수 내부의 3개 부호가 뒤집혔는데 테스트 경로가 끊겨 있어서 CI에서 안 잡혔던 것으로 추정됩니다.

### 2.2 URDF 기반 forward-kinematics 검증

`p73_walker.urdf` 에서 각 joint의 origin·axis 를 읽어 Rodrigues' rotation 으로 FK를 구현하고, 다음을 수치검증했습니다:

**요구 조건**: 어떤 joint 상태 `q = (q_L, q_R)` 이라도
- `FK_L(mirror(q)_L) == sagittal_mirror(FK_R(q_R))`
- `FK_R(mirror(q)_R) == sagittal_mirror(FK_L(q_L))`

(여기서 `sagittal_mirror = diag(1, -1, 1)` — xz-plane reflection)

500개 무작위 샘플에 대한 결과:

| Rule | fails(>1e-9) | max foot position err | max foot orientation err |
|---|---|---|---|
| **현재(버그) 코드** | 500 / 500 | **0.905 m** | 2.60 |
| 수정안 (HipRoll 만 swap only, 나머지 swap+negate) | 0 / 500 | 0.000 | 0.000 |

### 2.3 물리 직관 재확인

각 joint에서 "+θ 가 양쪽 다리에서 어떤 물리적 모션을 만드는가" 를 표로 정리:

| Joint | L axis | R axis | L +θ | R +θ | Mirror 규칙 |
|---|---|---|---|---|---|
| **HipRoll** | +x | −x | **abduction** | **abduction** | swap only (둘 다 같은 물리 방향) |
| HipPitch | −y | +y | leg forward | leg backward | swap + negate |
| HipYaw | −z | −z | internal rot | external rot | swap + negate |
| Knee | +y | −y | flexion | extension | swap + negate |
| AnklePitch | +y | −y | toe down | toe up | swap + negate |
| AnkleRoll | +x | +x | inversion | eversion | swap + negate |

**HipRoll만 특별한 이유**: URDF에서 L/R axis 를 `+x / -x` 로 mirror 해 놓았기 때문. 이 경우 축-벡터 sagittal mirror 기준 `axial_mirror(+x) = -x = R_axis` 가 되어 +θ 가 양쪽에서 같은 물리 방향(abduction)을 낸다. 그래서 대칭 훈련 시 L, R joint 값이 **같은 부호**로 수렴해야 맞고, mirror 시 그냥 값만 swap하면 된다.

나머지 5개 joint는 `axial_mirror(L_axis) ≠ R_axis` 이므로 +θ 가 양쪽에서 반대 물리 방향을 내기 때문에 대칭 훈련 시 L, R 값은 **반대 부호**로 수렴해야 맞고, mirror 시 swap + negate 해야 한다.

### 2.4 projected_gravity 재유도

`projected_gravity = R_wb^T · (0, 0, -1)` (world frame 중력 벡터를 body frame으로 투영).

Roll φ, pitch θ, yaw ψ 일 때:
```
g_body = ( sin θ, -sin φ · cos θ, -cos φ · cos θ )
```

Sagittal mirror 하에서는 roll → -roll, yaw → -yaw, pitch 는 불변. 대입하면:
```
g_body_mirror = ( sin θ, +sin φ · cos θ, -cos φ · cos θ )
```

→ **y 성분만 부호 반전** → multiplier **`[1, -1, 1]`**.

현재 코드 `[-1, 1, 1]` 는 x, y 성분 부호가 모두 반대로 잘못 잡혀 있었음.

### 2.5 gait_phase 재유도

`_tocabi_desired_contact_biped_ds` (mdp/rewards.py L1133-1164) 의 정의에 따르면:
- `phase ∈ [0, 0.5)` → L stance, R swing
- `phase ∈ [0.5, 1.0)` → R stance, L swing

Sagittal mirror 로 L↔R swap 하면 원래 L stance였던 것이 이제 R stance가 된다. 즉 **phase 를 0.5 (= π)만큼 shift**해야 스케줄 정합성이 유지된다.

```
sin(2π (φ + 0.5)) = sin(2π φ + π) = -sin(2π φ)
cos(2π (φ + 0.5)) = cos(2π φ + π) = -cos(2π φ)
```

→ gait_phase_sin, gait_phase_cos **둘 다 부호 반전**. 현재 코드는 그대로 두고 있어 mirror 상태와 gait 스케줄이 엇갈리게 되어 있었음.

---

## 3. 적용된 수정 내역

### 3.1 `_flip_lowerbody_12_axis_aware` 내부 부호 3개 수정

수정 전 (버그):
```python
out[:, 6] = -joint_tensor[:, 0]    # HipRoll:    swap + negate
out[:, 8] = joint_tensor[:, 2]     # HipYaw:     swap only
out[:, 11] = joint_tensor[:, 5]    # AnkleRoll:  swap only
out[:, 0] = -joint_tensor[:, 6]    # (R→L HipRoll)
out[:, 2] = joint_tensor[:, 8]     # (R→L HipYaw)
out[:, 5] = joint_tensor[:, 11]    # (R→L AnkleRoll)
```

수정 후 (올바름):
```python
out[:, 6] = joint_tensor[:, 0]     # HipRoll:    swap only
out[:, 8] = -joint_tensor[:, 2]    # HipYaw:     swap + negate
out[:, 11] = -joint_tensor[:, 5]   # AnkleRoll:  swap + negate
out[:, 0] = joint_tensor[:, 6]     # (R→L HipRoll)
out[:, 2] = -joint_tensor[:, 8]    # (R→L HipYaw)
out[:, 5] = -joint_tensor[:, 11]   # (R→L AnkleRoll)
```

Docstring의 axis table 및 설명 문구도 URDF axial-vector mirror 규칙으로 재작성.

### 3.2 `p73_data_augmentation_lowerbody_mirror` obs flip 수정

수정 전:
```python
projected_gravity_flipped = projected_gravity * torch.tensor([-1.0, 1.0, 1.0], ...)
# gait_phase 는 flip 없이 그대로 concat
```

수정 후:
```python
projected_gravity_flipped = projected_gravity * torch.tensor([1.0, -1.0, 1.0], ...)
gait_phase_flipped = -gait_phase          # 추가
```

그리고 concat block 의 `gait_phase` → `gait_phase_flipped`.

### 3.3 그대로 유지된 부분 (버그 아님, 검증 완료)

- `base_ang_vel_flipped = base_ang_vel * [-1.0, 1.0, -1.0]` ✓ (axial: roll/yaw flip)
- `vel_cmd_flipped = vel_cmd * [1.0, -1.0, -1.0]` ✓ (lin_y flip, yaw rate flip)
- HipPitch, Knee, AnklePitch 의 `swap + negate` ✓
- Mirror loss 알고리즘 자체 (`ppo_future.py`) ✓ — `π(mirror(o))` vs `mirror(π(o))` MSE 구조 정확

---

## 4. 증거: 수정 전후 테스트 비교

### 수정 전 (BUGGY)
```
[Test 1] check_p73_symmetry.py hardcoded expectation:  FAIL (diff=76, 12개 중 6개 원소 불일치)
[Test 2] URDF FK (500 random full-states):            FAIL (500/500, max pos err 0.905 m)
[Test 3] projected_gravity [-1, 1, 1]:                 FAIL (max err 1.02)
[Test 3] gait_phase no flip:                           FAIL (max err 2.00)
```

### 수정 후 (FIXED)
```
[Test 1] check_p73_symmetry.py hardcoded expectation:  PASS (diff=0)
[Test 2] URDF FK (500 random full-states):            PASS (0/500, max pos err 0.000e+00)
[Test 3] projected_gravity [1, -1, 1]:                 PASS (max err 0.000e+00)
[Test 3] gait_phase both × -1:                         PASS (max err ~1e-15)
[Involution] f(f(q)) == q:                             PASS
```

---

## 5. 훈련에 대한 영향

### 수정 전까지 훈련이 "돌긴 돌았던" 이유
Mirror loss 는 **soft regularizer** 이기 때문에 잘못된 mirror 에 대해서도 학습은 진행되고, 정책은 "그 잘못된 mirror 기준의 symmetry" 쪽으로 수렴합니다. 보상 곡선·episode length 같은 거시 metric 은 크게 나빠지지 않음.

### 특히 어긋났던 부분
5 bug 중 HipPitch / Knee / AnklePitch / base_ang_vel / vel_cmd 는 **정상**이었고, 나머지에서만 틀어져 있었습니다:
- **HipRoll (외전/내전)**: swap+negate 로 학습하도록 강요됐는데 실제로는 swap only 가 맞음. → 좌우 외전 방향이 대칭이 아닌 쪽으로 수렴.
- **HipYaw (내회전/외회전)**: swap only 로 학습하도록 강요됐는데 실제로는 swap+negate. → 좌우 hip yaw 방향이 어긋남.
- **AnkleRoll (발목 inversion/eversion)**: 마찬가지로 어긋남.
- **projected_gravity**: roll 감지 성분(y) 대신 forward 성분(x)이 부호 반전되도록 잘못 mirror → 기울어진 자세 인지에 편향.
- **gait_phase**: mirror 상태에서 gait 스케줄과 정책이 반주기 차이. → 발 디딤 타이밍의 좌우 대칭성이 깨짐.

사용자 관찰한 "좌우 대칭이 살짝 아쉬움" 은 정확히 이 패턴과 부합합니다 (pitch-only sagittal motion은 괜찮지만 lateral/yaw 움직임이 미묘하게 어긋남).

### 수정 후 권장 실험 순서
1. **기존 `mirror_loss_coeff=2.0` 유지**한 채 짧게 (~200-500 iter) 돌려보고, HipRoll / HipYaw / AnkleRoll 관련 좌우 대칭 metric, episode reward, lin_vel tracking 이 **최소한 regression 하지 않는지** 확인.
2. 개선이 확인되면 그때 `mirror_loss_coeff`를 3.0 → 4.0 으로 단계적 상향.
3. 기존 체크포인트와의 직접 비교가 필요하면 현재 fix를 별도 브랜치에 커밋해 A/B 비교.

---

## 6. Data Augmentation 미사용 이유 (부록)

현재 `ppo_future.py` L139-143에서 `use_data_augmentation=True` 는 명시적으로 `NotImplementedError` 를 raise 합니다. 이유는 PPOFuture가 여러 obs 그룹(policy 47D / target 155D / critic 161D / target_next 155D)을 동시에 다루는데, 현재 mirror 함수는 **policy obs 만** 처리 가능하기 때문입니다 (`obs_type != "policy"` 는 에러). Data aug 를 풀려면 target_obs, critic_obs, target_obs_next 각각에 대한 mirror 함수를 신규 작성해야 하고, 특히 `height_scan(81D)` 2D 격자 재배열 규칙 · `gt_foot_force6` · `motor_strength(13)` 등 여러 DR 파라미터에 대한 joint-wise mirror 규칙을 확정해야 합니다. 비용 대비 이득이 낮아 mirror-loss 경로를 유지하는 쪽이 합리적입니다.

---

## 7. 후속 조치 권장

### 7.1 `check_p73_symmetry.py` CI 복원
L12 의 `REPO_ROOT = Path("/home/piene/p73/isaaclab_walker")` 를 walker 경로로 수정해서 실행 가능 상태로 만들고, 사전 커밋 훅이나 CI에서 자동 실행되도록 걸어두면 재발 방지됨. 참고: 이 테스트는 involution (`f(f(x)) == x`) 과 1×12 hardcoded expected 만 체크하므로 **물리적 올바름은 커버 못 함**. URDF FK 기반 테스트도 `tests/` 하위에 추가하는 것을 권장.

### 7.2 다른 프로젝트(isaaclab_tocabi, isaaclab_p73 등)에 동일 버그 유무 확인
동일 파일명/함수명 패턴이 다른 레포에도 있으면 같은 포팅 실수가 있을 수 있음:
```
/home/piene/isaaclab5.2/isaaclab_tocabi/source/isaaclab_tocabi/isaaclab_tocabi/algorithms/rsl_rl/ppo_future.py
/home/piene/isaaclab5.2/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/algorithms/rsl_rl/ppo_future.py
```
필요 시 동일한 검증 스크립트로 빠르게 체크 가능.

### 7.3 Mirror_loss_coeff 스케줄링 검토
정책이 완전히 올바른 mirror 를 학습하도록 규제된 뒤에는 coeff 를 점진적으로 낮추는 스케줄(예: linear decay 2.0 → 0.5 over iters)이 tracking reward 에 덜 방해되면서 symmetry 를 유지할 수 있음. 다만 fix 후 경험적으로 관찰한 뒤 결정 권장.

---

## 8. 참고 파일

| 파일 | 역할 |
|---|---|
| `source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/mdp/symmetry.py` | 수정됨 (mirror 함수) |
| `source/isaaclab_walker/isaaclab_walker/assets/data/p73_walker/urdf/p73_walker.urdf` | axis 규칙 유도 근거 (L344-436) |
| `source/isaaclab_walker/isaaclab_walker/algorithms/rsl_rl/ppo_future.py` | mirror_loss 호출자 (변경 없음, 정상) |
| `source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/agents/rsl_rl_ppo_cfg.py` | `symmetry_cfg` 정의 (변경 없음) |
| `source/isaaclab_walker/isaaclab_walker/tasks/manager_based/isaaclab_walker/mdp/rewards.py` | `_tocabi_desired_contact_biped_ds` (gait phase semantics 근거, L1133-1164) |
| `scripts/tools/check_p73_symmetry.py` | 레포 내 기존 symmetry 테스트 (경로 끊겨서 비활성, 복원 권장) |
