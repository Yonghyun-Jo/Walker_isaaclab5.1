# P73 Phase-based Locomotion (Isaac Lab 5.1) 정리

이 문서는 P73 task에 추가된 **phase 기반(phase-based; 위상 기반)** locomotion 로직을 “코드가 실제로 어떻게 계산하는지” 관점에서 정리합니다.

## 0. 큰 그림(데이터 흐름)
- **Phase observation(관측)**: policy/critic/target에 `gait_phase_sin/cos` 2D를 추가
- **Contact schedule reward(접촉 스케줄 보상)**: phase로 **DS/SSP 구간을 고정 분할**해서 “원하는 접촉(desired contact)”을 만들고 실제 접촉과 매칭
- **Swing clearance min-profile penalty(스윙 clearance 최소 프로파일 페널티)**: 스윙 발이 **z축 높이(clearance)** 최소 프로파일을 못 맞추면 페널티

구현 위치:
- Phase obs: `isaaclab_p73/.../mdp/observations.py`
- Phase reward/penalty: `isaaclab_p73/.../mdp/rewards.py`
- Wiring(관측/보상 연결): `isaaclab_p73/.../rough_env_cfg.py`

## 1. Phase가 어떻게 구성되는가?

### 1.1 phase는 “시간 추정”이 아니라 step counter 기반의 고정 주기 인덱스
`gait_phase()`는 `episode_length_buf`(reset 이후 **environment step counter**)로 phase를 계산합니다.

```19:55:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/observations.py
def gait_phase(
    env: "ManagerBasedRLEnv",
    *,
    period_steps: int,
    command_name: str = "base_velocity",
    cmd_zero_max: float = 1.0e-3,
) -> torch.Tensor:
    ...
    # `period_steps` is in *environment steps* (i.e., after decimation).
    ...
    phase = (step_count % int(period_steps)).to(torch.float32) / float(period_steps)
```

즉,
$$
phase = (step\\_count \\bmod period\\_steps) / period\\_steps, \\quad phase\\in[0,1)
$$

여기서 **period_steps는 고정**이므로, phase 기반 schedule의 swing/contact 구간도 기본적으로 **고정 step 길이**가 됩니다.

### 1.2 standing gate(정지 게이트): cmd≈0이면 phase=0으로 고정
cmd가 거의 0인 경우에는 phase를 0으로 출력합니다. 목적은 **standing에서 phase-driven in-place marching(제자리 발 들기) 학습을 방지**하는 것입니다.

```39:55:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/observations.py
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 3)
    cmd_vel = torch.linalg.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])  # (num_envs,)
    is_standing = cmd_vel <= cmd_zero_max
    ...
    phase = torch.where(is_standing.view(-1, 1), torch.zeros_like(phase), phase)
```

### 1.3 policy/critic/target에는 sin/cos(2D)로 넣는다
관측으로는 `phase` 자체가 아니라 `sin(2πphase)`, `cos(2πphase)`를 넣습니다.

```58:79:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/observations.py
def gait_phase_sin(...):
    """Sin(2π * gait_phase)."""
    ...
def gait_phase_cos(...):
    """Cos(2π * gait_phase)."""
```

이 2D 표현은 **phase의 wrap-around(0→1 경계) 불연속 문제**를 완화하는 전형적인 표현입니다.

## 2. swing phase / contact phase는 “고정 시간”인가?
### 2.1 정의는 고정(phase 구간으로 고정 분할)
contact schedule은 `phase01∈[0,1)`을 `ds_ratio`로 **DS(Double Support; 양발 지지)** window를 두고, 나머지를 SSP(Single Support Phase; 단일 지지)로 나눕니다.

```957:988:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/rewards.py
def _tocabi_desired_contact_biped_ds(phase01, *, ds_ratio):
    ...
    # DS around phase=0 and phase=0.5
    # SSP1: Left stance, Right swing
    # SSP2: Right stance, Left swing
```

`ds_ratio` 해석:
- 한 주기 전체에서 DS가 차지하는 총 비율(두 DS window의 합)
- `h = ds_ratio/4`로 두 transition 주변에 DS window를 생성

```975:981:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/rewards.py
    ds_ratio_c = float(max(0.0, min(ds_ratio, 0.49)))
    h = ds_ratio_c / 4.0  # because: 2 transitions × (2h) = ds_ratio
```

### 2.2 실제 “초 단위 시간”은 env step dt에 의해 정해진다
P73 config에서:
- `sim.dt = 0.005`
- `decimation = 4`

```973:978:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/rough_env_cfg.py
        self.decimation = 4
        ...
        self.sim.dt = 0.005
```

일반적으로 environment step은 대략:
$$
env\\_step\\_dt \\approx sim.dt \\times decimation = 0.005 \\times 4 = 0.02s
$$

현재 `period_steps=40`이므로, 한 주기 시간은 대략:
$$
T \\approx 40 \\times 0.02 = 0.8s
$$

> 주의: 정확한 `env.step_dt`는 Isaac Lab 내부 정의를 따르지만, 위 계산이 보통의 의미(“decimation 이후 step”)와 일치합니다.

## 3. Contact schedule reward: 실제 접촉과 “원하는 접촉”을 맞춘다

### 3.1 활성화 gate는 cmd-only
요청대로 “명령 속도(cmd_vel)만”으로 ON/OFF를 결정합니다.

```945:955:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/rewards.py
def _tocabi_is_schedule_active_from_cmd(...):
    """... (cmd-only gate)."""
    cmd = env.command_manager.get_command(command_name)
    cmd_vel = torch.linalg.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    return cmd_vel > cmd_zero_max
```

즉, **cmd≈0이면 schedule 관련 reward/penalty는 0**이 됩니다.

### 3.2 실제 접촉(actual)은 Fz threshold로 판정
ContactSensor의 `net_forces_w[...,2]`(world-frame z force)가 `contact_threshold`보다 크면 contact로 봅니다.

```1024:1039:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/rewards.py
    fz = net_forces_w[:, body_ids, 2]  # (N,2)
    actual = (fz > contact_threshold).to(torch.float32)  # (N,2)
```

### 3.3 reward는 match score (1 - mean(|actual - desired|))
desired와 actual이 다르면 페널티가 커지고, 같으면 1에 가까워집니다.

```1039:1041:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/rewards.py
    match = 1.0 - torch.mean(torch.abs(actual - desired), dim=1)
    match = torch.clamp(match, 0.0, 1.0)
```

## 4. Swing clearance min-profile penalty: “z축 궤적”만 강제한다

여기서 중요한 포인트는 질문하신 대로:
- **x/y 궤적은 추정/강제하지 않음**
- 스윙 발의 **z clearance(높이)**만 최소 프로파일로 강제함

### 4.1 발 높이는 body position의 z만 사용

```1079:1081:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/rewards.py
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].to(torch.float32)  # (N,2)
```

### 4.2 terrain query 없이 stance_z 버퍼로 “지면 기준”을 만든다
접촉 중인 발은 `stance_z = feet_z`로 계속 업데이트하고, clearance는 `feet_z - stance_z`.

```1095:1103:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/rewards.py
    stance_key = "_p73_stance_z_buffer"
    ...
    stance_z = torch.where(is_contact, feet_z, stance_z)
    ...
    clearance = feet_z - stance_z  # (N,2)
```

이 설계는 **RayCaster(레이캐스터; 광선 기반 지면 높이 측정)** 같은 terrain 높이 query 없이도 “최근 접촉 높이 대비”로 clearance를 만들 수 있어 간단합니다.

### 4.3 스윙 구간에서만 페널티(스윙 마스크)
schedule에서 desired contact가 0인 발만 스윙으로 봅니다.

```1069:1072:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/rewards.py
    desired = _tocabi_desired_contact_biped_ds(...)
    swing_mask = desired < 0.5  # (N,2)
```

### 4.4 local swing phase(φ)와 z_ref(최소 높이 프로파일)
SSP 내부에서 local phase `φ∈[0,1]`를 만들고,
$$
z_{ref}(\\phi) = H\\sin(\\pi\\phi)
$$
를 최소 높이로 둡니다.

```1104:1116:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/mdp/rewards.py
    denom = max(1.0e-6, 0.5 - 2.0 * h)
    phi_r = torch.clamp((phase01 - h) / denom, 0.0, 1.0)
    phi_l = torch.clamp((phase01 - (0.5 + h)) / denom, 0.0, 1.0)
    phi_local = torch.stack([phi_l, phi_r], dim=1)  # (N,2) [L,R]

    z_ref = float(clearance_height) * torch.sin(torch.pi * phi_local)  # (N,2)
    per_foot = torch.square(torch.clamp(z_ref - clearance, min=0.0)) * swing_mask.to(torch.float32)
    penalty = torch.mean(per_foot, dim=1)
```

해석:
- `clearance < z_ref`이면 `(z_ref - clearance)^2` 만큼 페널티
- `clearance >= z_ref`이면 0
- 스윙 발만 적용

## 5. rough_env_cfg.py에서 어떻게 연결(wiring)했나?
### 5.1 policy/critic/target에 phase obs를 velocity_commands 뒤에 삽입
Policy 예시:

```470:490:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/rough_env_cfg.py
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase_sin = ObsTerm(func=mdp.gait_phase_sin, params={...})
        gait_phase_cos = ObsTerm(func=mdp.gait_phase_cos, params={...})
```

### 5.2 reward term 2개 추가
contact schedule reward + swing clearance penalty가 `KangarooRewards`에 추가되어 있습니다.

```376:441:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/rough_env_cfg.py
    contact_schedule_biped_ds = RewTerm(func=mdp.contact_schedule_reward_biped_ds, ...)
    swing_clearance_min_profile_penalty = RewTerm(func=mdp.swing_clearance_min_profile_penalty, ...)
```

### 5.3 period_steps/ds_ratio는 __post_init__에서 단일 소스로 중앙화
관측과 보상이 서로 다른 값을 참조하면 바로 깨지므로, `__post_init__`에서 한 번에 바인딩합니다.

```1008:1030:/home/piene/p73/isaaclab_p73/source/isaaclab_p73/isaaclab_p73/tasks/manager_based/isaaclab_p73/rough_env_cfg.py
        gait_period_steps = 40
        gait_ds_ratio = 0.20
        ...
        self.observations.policy.gait_phase_sin.params["period_steps"] = gait_period_steps
        ...
        self.rewards.contact_schedule_biped_ds.params["period_steps"] = gait_period_steps
        self.rewards.contact_schedule_biped_ds.params["ds_ratio"] = gait_ds_ratio
```

## 6. Symmetry / PPOFuture 차원 변화(중요)
phase 2D가 들어가면서 policy single-frame observation dimension이 **57→59**로 변했습니다.
- mirror augmentation slicing도 59 기준으로 업데이트됨
- PPOFuture 설정도 `num_single_obs=59`, `target_obs_dim=150`로 업데이트됨

이 때문에 **기존 checkpoint는 입력 차원 불일치로 그대로 로드가 안 됩니다**(재학습 또는 모델/체크포인트 변환이 필요).

## 7. 튜닝 포인트(실험 시 체크리스트)
- **period_steps**: 보행 주기(초)를 바꾸고 싶으면 period_steps를 바꿉니다. (step_dt에 의해 실제 초가 정해짐)
- **ds_ratio**: DS 구간 비율. 너무 크면 스윙이 짧아져 발이 잘 안 들릴 수 있고, 너무 작으면 착지 안정성이 떨어질 수 있습니다.
- **clearance_height**: 최소 프로파일의 최고점 H. 너무 크면 과도한 무릎/발 들기가 강제될 수 있습니다.
- **contact_threshold**: 접촉 판정 Fz 임계값. 노이즈가 크면 hysteresis(히스테리시스; on/off 임계값 분리)를 고려할 수 있습니다(현재 코드는 단일 threshold).

