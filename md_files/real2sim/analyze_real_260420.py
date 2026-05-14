"""Analyze realrobot_260420_*.csv for real-to-sim reward shaping.

Data runs at 1000 Hz; policy updates at 50 Hz.
Action-rate / action-accel stats are computed at 50 Hz (downsampled).
qdot / qddot / tau stats are at 1000 Hz.
All stats computed over MOVING frames only (cmd_norm > 0.07 OR |v_xy| > 0.1).
"""
import os
import numpy as np
import pandas as pd

CSV_FILES = [
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_173557.csv",
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_175257.csv",
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_183206.csv",
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_183421.csv",
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_183702.csv",
    "/home/piene/ros2_ws/src/p73_cc/logs/realrobot_260420_185028.csv",
]

JOINT_NAMES = [
    "L_HipRoll", "L_HipPitch", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
    "R_HipRoll", "R_HipPitch", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll",
]
TAU_MAX = {
    "HipRoll": 352.0, "HipPitch": 220.0, "HipYaw": 95.0,
    "Knee": 220.0, "AnklePitch": 95.0, "AnkleRoll": 95.0,
}
tau_max_per_joint = np.array([TAU_MAX[n.split("_", 1)[1]] for n in JOINT_NAMES])

# policy rate = 50 Hz, control rate = 1000 Hz -> stride 20
POLICY_STRIDE = 20

# collect into per-joint flat arrays; only MOVING frames
abs_da_all   = [[] for _ in range(12)]   # |Δa| at 50 Hz
abs_dda_all  = [[] for _ in range(12)]   # |Δ²a| at 50 Hz
qdot_all     = [[] for _ in range(12)]
qddot_all    = [[] for _ in range(12)]
tau_all      = [[] for _ in range(12)]
tau_ratio_all= [[] for _ in range(12)]

cmd_norms, lv_norms, av_norms, pg_norms = [], [], [], []

total_moving_kHz = 0
total_rows = 0

for fpath in CSV_FILES:
    print(f"loading {os.path.basename(fpath)}...", flush=True)
    df = pd.read_csv(fpath, low_memory=False).dropna().reset_index(drop=True)
    n = len(df)
    total_rows += n
    if n < 100:
        continue

    t = df["time"].to_numpy()
    dt = np.median(np.diff(t))
    print(f"  rows={n}, dt≈{dt*1000:.2f} ms")

    # moving mask at 1 kHz
    cmd = df[["cmd_vx", "cmd_vy", "cmd_vyaw"]].to_numpy()
    lv = df[["lin_vel_wx", "lin_vel_wy"]].to_numpy()
    cmd_norm_xy = np.linalg.norm(cmd[:, :2], axis=1)
    lv_norm_xy = np.linalg.norm(lv, axis=1)
    moving = (cmd_norm_xy > 0.07) | (lv_norm_xy > 0.1)
    total_moving_kHz += moving.sum()

    cmd_norms.append(cmd_norm_xy[moving])
    lv_norms.append(lv_norm_xy[moving])
    av_norms.append(np.linalg.norm(df[["ang_vel_bx","ang_vel_by","ang_vel_bz"]].to_numpy()[moving], axis=1))
    pg_norms.append(np.linalg.norm(df[["proj_grav_x","proj_grav_y"]].to_numpy()[moving], axis=1))

    # --- action rate/accel at POLICY rate (stride 20) ---
    A = df[[f"action_{i}" for i in range(12)]].to_numpy()
    A_ds = A[::POLICY_STRIDE]  # shape (N/20, 12)
    moving_ds = moving[::POLICY_STRIDE]
    # align diffs to the later sample (a_t - a_{t-1}) -> use mask from sample t
    dA = np.diff(A_ds, axis=0)  # len N-1
    ddA = np.diff(dA, axis=0)   # len N-2
    mask_dA = moving_ds[1:]
    mask_ddA = moving_ds[2:]
    for j in range(12):
        abs_da_all[j].append(np.abs(dA[mask_dA, j]))
        abs_dda_all[j].append(np.abs(ddA[mask_ddA, j]))

    # --- qdot / qddot / tau at 1 kHz, moving frames only ---
    QD = df[[f"qdot_{i}" for i in range(12)]].to_numpy()
    for j in range(12):
        qdot_all[j].append(np.abs(QD[moving, j]))
    QDD = np.diff(QD, axis=0) / dt
    mask_qdd = moving[1:]
    for j in range(12):
        qddot_all[j].append(np.abs(QDD[mask_qdd, j]))

    TAU = df[[f"tau_motor_{i}" for i in range(12)]].to_numpy()
    for j in range(12):
        tau_all[j].append(np.abs(TAU[moving, j]))
        tau_ratio_all[j].append(np.abs(TAU[moving, j]) / tau_max_per_joint[j])

print(f"\n===== aggregate: {total_rows} rows, moving@1kHz: {total_moving_kHz} ({total_moving_kHz/total_rows*100:.1f}%) =====")

def pstats(arrs, scale=1.0):
    a = np.concatenate(arrs) * scale
    if len(a) == 0:
        return None
    return dict(
        mean=a.mean(), std=a.std(),
        p50=np.percentile(a, 50),
        p90=np.percentile(a, 90),
        p95=np.percentile(a, 95),
        p99=np.percentile(a, 99),
        p999=np.percentile(a, 99.9),
        max=a.max(), n=len(a),
    )

def hline(fmt, headers):
    print(" ".join(h.rjust(w) for h, w in zip(headers, fmt)))

def print_tbl(name, stats_list, fmt_str="{:9.4f}"):
    print(f"\n--- {name} (moving frames, per joint) ---")
    cols = ["joint","mean","std","p50","p90","p95","p99","p999","max"]
    print(f"{'idx':<3} {'joint':<14} {'mean':>10} {'std':>10} {'p50':>10} {'p90':>10} {'p95':>10} {'p99':>10} {'p999':>10} {'max':>10}")
    for j in range(12):
        s = stats_list[j]
        vals = " ".join(fmt_str.format(s[k]).rjust(10) for k in ["mean","std","p50","p90","p95","p99","p999","max"])
        print(f"{j:<3} {JOINT_NAMES[j]:<14} {vals}")

cmd_norm_all = np.concatenate(cmd_norms)
lv_norm_all  = np.concatenate(lv_norms)
av_norm_all  = np.concatenate(av_norms)
pg_norm_all  = np.concatenate(pg_norms)
print(f"\ncmd_xy     p50={np.percentile(cmd_norm_all,50):.3f} p90={np.percentile(cmd_norm_all,90):.3f} max={cmd_norm_all.max():.3f}")
print(f"lin_vel_xy p50={np.percentile(lv_norm_all,50):.3f} p90={np.percentile(lv_norm_all,90):.3f} max={lv_norm_all.max():.3f}")
print(f"ang_vel    p50={np.percentile(av_norm_all,50):.3f} p90={np.percentile(av_norm_all,90):.3f} max={av_norm_all.max():.3f}")
print(f"proj_grav  p50={np.percentile(pg_norm_all,50):.4f} p90={np.percentile(pg_norm_all,90):.4f} max={pg_norm_all.max():.4f}")

da_s  = [pstats(abs_da_all[j])  for j in range(12)]
dda_s = [pstats(abs_dda_all[j]) for j in range(12)]
qd_s  = [pstats(qdot_all[j])    for j in range(12)]
qdd_s = [pstats(qddot_all[j])   for j in range(12)]
tau_s = [pstats(tau_all[j])     for j in range(12)]
rat_s = [pstats(tau_ratio_all[j]) for j in range(12)]

print_tbl("|Δa|  (rad, policy rate 50 Hz)", da_s, "{:.4f}")
print_tbl("|Δ²a| (rad, policy rate 50 Hz)", dda_s, "{:.4f}")
print_tbl("|qdot| (rad/s, 1 kHz)", qd_s, "{:.3f}")
print_tbl("|qddot|(rad/s², 1 kHz FD)", qdd_s, "{:.1f}")
print_tbl("|tau_motor| (Nm, 1 kHz)", tau_s, "{:.2f}")
print_tbl("|tau|/tau_max (1 kHz)", rat_s, "{:.4f}")

print("\n--- saturation fractions (moving, 1 kHz) ---")
print(f"{'idx':<3} {'joint':<14} {'>0.5':>8} {'>0.6':>8} {'>0.7':>8} {'>0.8':>8} {'>0.9':>8}")
for j in range(12):
    a = np.concatenate(tau_ratio_all[j])
    def fr(th):
        return (a > th).mean()*100
    print(f"{j:<3} {JOINT_NAMES[j]:<14} {fr(0.5):>7.2f}% {fr(0.6):>7.2f}% {fr(0.7):>7.2f}% {fr(0.8):>7.2f}% {fr(0.9):>7.2f}%")
