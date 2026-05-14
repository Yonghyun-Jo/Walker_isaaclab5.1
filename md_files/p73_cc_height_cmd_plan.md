---
date: 2026-05-12
project: isaaclab_walker / p73_cc
branch: acnet_zcmd (isaaclab) / p73_crouch (ros2_ws)
tags: [plan, sim2real, p73_cc, height_command, ros2]
---

# p73_cc Height Command 반영 계획

## 배경
IsaacLab에서 height command (pos_z)를 추가하여 policy obs가 47D→48D로 변경됨.
p73_crouch branch에서 crouch policy 전용으로 반영. (기존 policy 호환 불필요)

## 현재 p73_cc 상태 (변경 전)

### Observation (47D per frame)
```
[0:3]   ang_vel_b (3)
[3:6]   projected_gravity_b (3)
[6:9]   velocity_commands [vx, vy, wz] (3)
[9:11]  gait_phase [sin, cos] (2)
[11:23] joint_pos_rel (12)
[23:35] joint_vel_clipped_scaled (12)
[35:47] last_action_processed (12)
```

### Action 처리
- `last_action_processed_ = clip(action * 0.5, -1.0, 1.0)`
- `target_pos = q_default + clip(action * 0.5, -1.0, 1.0)`

---

## 변경 사항

### 1. Observation 48D로 확장 ✅
```
[0:3]   ang_vel_b (3)
[3:6]   projected_gravity_b (3)
[6:9]   velocity_commands (3)
[9:10]  height_command (1)      ← 신규
[10:12] gait_phase [sin, cos] (2)
[12:24] joint_pos_rel (12)
[24:36] joint_vel_clipped_scaled (12)
[36:48] last_action_processed (12)
```
- `num_single_obs` = 48
- ONNX input shape 추론: `input_shape / 48 = history_length`

### 2. Height Command 수신 ✅
- `/p73/cmd_vel` Twist.linear.z 활용 (추가 토픽 불필요)
- `target_height_` 멤버 (default: 0.89m)
- velCmdCallback에서 `msg->linear.z` → `target_height_`

### 3. Action Clip [-2, 2] ✅
```cpp
last_action_processed_(i) = minmax_cut(action * 0.5, -2.0, 2.0);
dq = minmax_cut(dq, -2.0, 2.0);
```

### 4. Teleop ✅
- 키보드: r(숙이기, -0.05m) / f(올리기, +0.05m), range [0.55, 0.89]
- 조이스틱: LB(숙이기) / RB(올리기)
- Twist.linear.z로 publish

---

## 수정 파일

| 파일 | 변경 |
|------|------|
| `p73_cc/src/cc.cpp` | obs 48D, height cmd from Twist.linear.z, action clip [-2,2] |
| `p73_cc/include/cc.h` | `target_height_` 멤버, frame 크기 |
| `p73_cc/scripts/walker_teleop.py` | r/f 키보드, LB/RB 조이스틱 height 매핑 |
