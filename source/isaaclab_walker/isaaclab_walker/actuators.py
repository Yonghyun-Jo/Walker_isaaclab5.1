"""Custom actuators for the P73 Walker.

* :class:`ActuatorNetLSTMWalker` — per-joint LSTM actuator network combining:

  - **DC-motor BEMF clipping** (inherits from :class:`isaaclab.actuators.DCMotor`,
    matching NVIDIA's official :class:`isaaclab.actuators.ActuatorNetLSTM`).
    The torque-speed curve clipping reproduces the real motor's back-EMF
    behaviour (peak torque decreases with rotor speed), giving the closed loop
    a velocity-dependent torque envelope without adding fake damping.
  - **Per-joint LSTM networks** (PACE-style) — one TorchScript network per
    joint, since each motor has its own dynamics. Hidden state is preserved
    across physics steps and zeroed on env reset.
  - **Angle-dependent torque LUT** — for knee/ankle joints whose simulation
    representation is the joint-side angle while the real torque limit is
    governed by the 4-bar linkage motor-side geometry. The LUT replaces the
    static ``effort_limit`` in the BEMF clipping step.
  - **Command delay** — manual delay buffers (1–N physics steps) so the LSTM
    sees the same delay distribution it was trained on.
  - **Real motor parameters** — friction / viscous_friction / dynamic_friction
    / armature / saturation_effort / velocity_limit are passed straight from
    system identification through :class:`ActuatorNetLSTMWalkerCfg`.

Compared with :class:`isaaclab.actuators.ActuatorNetLSTM` the changes are
strictly additive: per-joint networks, LUT clipping, delay buffer. The core
torque-speed clipping logic is reused from :class:`DCMotor`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path

import torch
from isaaclab.actuators.actuator_pd import DCMotor, IdealPDActuator
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg
from isaaclab.utils import DelayBuffer
from isaaclab.utils import configclass
from isaaclab.utils.assets import read_file
from isaaclab.utils.types import ArticulationActions

from isaaclab_walker.tasks.manager_based.isaaclab_walker.mdp.torque_lut import (
    AnkleTorqueLUT,
    KneeTorqueLUT,
)


class ActuatorNetLSTMWalker(DCMotor):
    """Per-joint LSTM actuator with DC-motor BEMF clipping, LUT, and command delay.

    Inheritance: ``ActuatorBase → IdealPDActuator → DCMotor → ActuatorNetLSTMWalker``.
    This mirrors NVIDIA's :class:`isaaclab.actuators.ActuatorNetLSTM` (also a
    direct subclass of :class:`DCMotor`) and inherits its BEMF torque-speed
    clipping unchanged.

    Differences from :class:`isaaclab.actuators.ActuatorNetLSTM`:
        * Per-joint TorchScript networks (a Python loop instead of one batched
          forward pass) — joints have distinct motor characteristics.
        * Angle-dependent torque LUT for knee/ankle replaces the static
          ``effort_limit`` inside the BEMF clipping step.
        * Command delay buffers (positions/velocities/efforts) so the LSTM
          input distribution matches its training-time delay.
        * Output torque is multiplied by ``torque_scale`` (training used
          ``torque_scaling=0.01`` so raw outputs are 1/100 of physical Nm).
    """

    cfg: ActuatorNetLSTMWalkerCfg

    def __init__(self, cfg: ActuatorNetLSTMWalkerCfg, *args, **kwargs):
        # ``DCMotor.__init__`` reads ``cfg.saturation_effort`` raw and uses it
        # directly in tensor arithmetic, so it crashes when the user passes a
        # per-joint dict (the standard pattern for ``effort_limit`` etc.). To
        # keep dict-based per-joint configuration we skip ``DCMotor.__init__``,
        # call the grandparent ``IdealPDActuator.__init__`` directly, then
        # replicate the DCMotor setup manually with proper joint-parameter
        # parsing for ``saturation_effort``.
        IdealPDActuator.__init__(self, cfg, *args, **kwargs)

        if cfg.saturation_effort is None:
            raise ValueError("saturation_effort must be provided for ActuatorNetLSTMWalkerCfg.")
        if cfg.velocity_limit is None and cfg.velocity_limit_sim is None:
            raise ValueError("velocity_limit (or velocity_limit_sim) must be provided.")

        # Parse saturation_effort the same way effort_limit / armature etc. are
        # parsed: dict → per-joint tensor of shape (num_envs, num_joints).
        self._saturation_effort = self._parse_joint_parameter(cfg.saturation_effort, torch.inf)
        self._vel_at_effort_lim = self.velocity_limit * (1 + self.effort_limit / self._saturation_effort)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._zeros_effort = torch.zeros_like(self.computed_effort)

        # ----- Command delay buffers (mirrors DelayedPDActuator behaviour) -----
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

        # ----- Angle-dependent torque LUT (4-bar linkage joint-side compensation) -----
        self._use_lut = cfg.torque_lut_dir is not None
        if self._use_lut:
            lut_dir = Path(cfg.torque_lut_dir)
            self._knee_lut_L = KneeTorqueLUT(str(lut_dir / "L_knee_lut_new.csv"), device=self._device)
            self._knee_lut_R = KneeTorqueLUT(str(lut_dir / "R_knee_lut_new.csv"), device=self._device)
            self._ankle_lut_L = AnkleTorqueLUT(str(lut_dir / "L_ankle_lut_new.csv"), device=self._device)
            self._ankle_lut_R = AnkleTorqueLUT(str(lut_dir / "R_ankle_lut_new.csv"), device=self._device)

            # Resolve LUT joint indices within this actuator group.
            name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
            self._lut_knee_L = name_to_idx["L_Knee_Joint"]
            self._lut_knee_R = name_to_idx["R_Knee_Joint"]
            self._lut_ankle_pitch_L = name_to_idx["L_AnklePitch_Joint"]
            self._lut_ankle_roll_L = name_to_idx["L_AnkleRoll_Joint"]
            self._lut_ankle_pitch_R = name_to_idx["R_AnklePitch_Joint"]
            self._lut_ankle_roll_R = name_to_idx["R_AnkleRoll_Joint"]

        # ----- Optional per-step motor strength randomization -----
        self._motor_scale_range = cfg.rand_motor_scale_range
        # Persistent buffer so observations can read the last-used scale.
        self.motor_strength_scale = torch.ones(self._num_envs, self.num_joints, device=self._device)

        # ----- Per-joint LSTM networks -----
        net_dir = Path(cfg.network_dir) if cfg.network_dir else None
        net_paths: list[str] = []
        for jn in self.joint_names:
            rel = cfg.network_files.get(jn)
            if rel is None:
                raise ValueError(
                    f"ActuatorNetLSTMWalker: no network_file specified for joint '{jn}'. "
                    f"Keys in network_files: {list(cfg.network_files.keys())}"
                )
            path = rel if os.path.isabs(rel) else (str(net_dir / rel) if net_dir else rel)
            net_paths.append(path)

        networks: list[torch.nn.Module] = []
        for path in net_paths:
            file_bytes = read_file(path)
            net = torch.jit.load(file_bytes, map_location=self._device).eval()
            networks.append(net)
        self.networks = torch.nn.ModuleList(networks)

        # Sanity-check that all networks share the same LSTM architecture.
        first_lstm_sd = networks[0].lstm.state_dict()
        num_layers = len(first_lstm_sd) // 4
        hidden_dim = first_lstm_sd["weight_hh_l0"].shape[1]
        for j, net in enumerate(networks):
            sd = net.lstm.state_dict()
            if len(sd) // 4 != num_layers or sd["weight_hh_l0"].shape[1] != hidden_dim:
                raise ValueError(
                    f"Network for joint '{self.joint_names[j]}' has mismatched LSTM "
                    f"architecture (expected layers={num_layers}, hidden={hidden_dim})"
                )

        # ----- Persistent LSTM hidden/cell state -----
        # The LSTMs were trained with window-based zero-init + warm-up loss
        # masking: h,c start at zero at the beginning of each window, the LSTM
        # accumulates state across time steps, and the first ~10 steps (~50 ms)
        # are masked from the loss. At inference we therefore preserve (h, c)
        # across physics steps and only zero them on env reset, matching training.
        # Shape per buffer: (num_joints, num_layers, num_envs, hidden_dim).
        self._lstm_num_layers = num_layers
        self._lstm_hidden_dim = hidden_dim
        self.sea_hidden_state = torch.zeros(
            self.num_joints, num_layers, self._num_envs, hidden_dim,
            device=self._device,
        )
        self.sea_cell_state = torch.zeros(
            self.num_joints, num_layers, self._num_envs, hidden_dim,
            device=self._device,
        )

        # ----- PD warm-up blending -----
        self._warm_up_steps = cfg.warm_up_steps
        if self._warm_up_steps > 0 and cfg.warm_up_stiffness is not None:
            self._warm_up_kp = self._resolve_joint_dict(cfg.warm_up_stiffness)
            self._warm_up_kd = self._resolve_joint_dict(cfg.warm_up_damping if cfg.warm_up_damping is not None else 0.0)
            self._warm_up_counter = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        else:
            self._warm_up_kp = None

        # Torque scale that undoes the training pipeline's ``torque_scaling``.
        self._torque_scale = float(cfg.torque_scale)

    """
    Operations.
    """

    def _resolve_joint_dict(self, val: dict[str, float] | float) -> torch.Tensor:
        """Convert a per-joint dict or scalar to a (1, num_joints) tensor."""
        if isinstance(val, dict):
            arr = [float(val[jn]) for jn in self.joint_names]
            return torch.tensor(arr, device=self._device, dtype=torch.float32).unsqueeze(0)
        return torch.full((1, self.num_joints), float(val), device=self._device, dtype=torch.float32)

    def reset(self, env_ids: Sequence[int]):
        # Parent (DCMotor → IdealPDActuator → ActuatorBase) reset.
        super().reset(env_ids)

        # Sample a fresh random delay for the resetting envs (same pattern as
        # :class:`DelayedPDActuator`).
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

        # Zero LSTM hidden/cell state for resetting envs. The first ~10 steps
        # after reset are a warm-up period where h,c build up from zero —
        # this matches training (window zero-init + warm-up loss masking).
        self.sea_hidden_state[:, :, env_ids, :] = 0.0
        self.sea_cell_state[:, :, env_ids, :] = 0.0

        # Reset warm-up counter so PD→LSTM blending restarts for these envs.
        if self._warm_up_kp is not None:
            self._warm_up_counter[env_ids] = 0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Cache joint vel for BEMF clipping (used by parent ``_clip_effort``).
        self._joint_vel[:] = joint_vel

        # 1) Apply command delay (positions / velocities / efforts).
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        # 2) Per-joint LSTM inference (stateful). Input ``[pos_err, vel]``
        #    with seq_len=1; runs once per physics step (200 Hz).
        #    Hidden/cell state is carried over from the previous step so the
        #    LSTM accumulates temporal context — matching the window-based
        #    stateful training. State is zeroed only on env reset.
        pos_err = control_action.joint_positions - joint_pos
        torques = torch.empty_like(pos_err)
        with torch.inference_mode():
            for j, net in enumerate(self.networks):
                x = torch.stack([pos_err[:, j], joint_vel[:, j]], dim=-1).unsqueeze(1)
                h_in = self.sea_hidden_state[j]
                c_in = self.sea_cell_state[j]
                out, (h_new, c_new) = net(x, (h_in, c_in))
                self.sea_hidden_state[j] = h_new
                self.sea_cell_state[j] = c_new
                torques[:, j] = out.view(-1)

        # 3) Recover Nm from training-time scaled output.
        lstm_effort = torques * self._torque_scale

        # 3b) PD warm-up blending: during the first ``warm_up_steps`` after
        #     reset, blend from pure PD towards pure LSTM using a cosine
        #     schedule. The LSTM already ran above (shadow mode) so h,c are
        #     accumulating real observations regardless of blending.
        if self._warm_up_kp is not None:
            # Cosine schedule: λ = 0.5*(1 - cos(π * t/N)), clamped to [0, 1]
            t = self._warm_up_counter.float()  # (num_envs,)
            N = float(self._warm_up_steps)
            lam = torch.where(
                t >= N,
                torch.ones_like(t),
                0.5 * (1.0 - torch.cos(torch.pi * t / N)),
            ).unsqueeze(1)  # (num_envs, 1) for broadcasting over joints

            # PD torque: τ_PD = Kp * pos_err + Kd * (-vel)
            tau_pd = self._warm_up_kp * pos_err + self._warm_up_kd * (-joint_vel)

            self.computed_effort = (1.0 - lam) * tau_pd + lam * lstm_effort
            self._warm_up_counter += 1
        else:
            self.computed_effort = lstm_effort

        # 4) Clipping pipeline:
        #      a) Decide effective per-joint effort_limit.
        #         - Static ``self.effort_limit`` for non-LUT joints.
        #         - LUT(joint_pos) for knee/ankle (4-bar linkage angle-dependent).
        #      b) Apply DC-motor torque-speed curve (BEMF) using the resolved limit.
        if self._use_lut:
            eff_lim = self.effort_limit.clone()
            eff_lim[:, self._lut_knee_L] = self._knee_lut_L.query(joint_pos[:, self._lut_knee_L])
            eff_lim[:, self._lut_knee_R] = self._knee_lut_R.query(joint_pos[:, self._lut_knee_R])
            lp_L, lr_L = self._ankle_lut_L.query(
                joint_pos[:, self._lut_ankle_pitch_L], joint_pos[:, self._lut_ankle_roll_L]
            )
            eff_lim[:, self._lut_ankle_pitch_L] = lp_L
            eff_lim[:, self._lut_ankle_roll_L] = lr_L
            lp_R, lr_R = self._ankle_lut_R.query(
                joint_pos[:, self._lut_ankle_pitch_R], joint_pos[:, self._lut_ankle_roll_R]
            )
            eff_lim[:, self._lut_ankle_pitch_R] = lp_R
            eff_lim[:, self._lut_ankle_roll_R] = lr_R
        else:
            eff_lim = self.effort_limit

        # BEMF (DC motor torque-speed envelope), capped by the resolved continuous limit.
        # Same form as DCMotor._clip_effort but with `eff_lim` instead of `self.effort_limit`.
        clipped_vel = torch.clip(self._joint_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)
        ts_top = self._saturation_effort * (1.0 - clipped_vel / self.velocity_limit)
        ts_bot = self._saturation_effort * (-1.0 - clipped_vel / self.velocity_limit)
        max_effort = torch.clip(ts_top, max=eff_lim)
        min_effort = torch.clip(ts_bot, min=-eff_lim)
        self.applied_effort = torch.clip(self.computed_effort, min=min_effort, max=max_effort)

        # 5) Optional per-step motor strength randomization.
        if self._motor_scale_range is not None:
            self.motor_strength_scale.uniform_(self._motor_scale_range[0], self._motor_scale_range[1])
            self.applied_effort = self.applied_effort * self.motor_strength_scale

        # Write back as explicit effort command (PhysX skips its own PD).
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class ActuatorNetLSTMWalkerCfg(DCMotorCfg):
    """Configuration for :class:`ActuatorNetLSTMWalker`.

    Inherits from :class:`DCMotorCfg` (the same base used by NVIDIA's
    :class:`isaaclab.actuators.ActuatorNetLSTMCfg`). Required additions over the
    DC-motor base:

    - ``saturation_effort`` (from :class:`DCMotorCfg`): motor stall torque
      (peak torque at zero speed). Must be supplied by the user — typical values
      are 1.5–3 × ``effort_limit`` for BLDC servo motors.
    - ``velocity_limit`` / ``velocity_limit_sim``: motor no-load speed.

    Plus walker-specific extras:

    - Per-joint LSTM weights and torque scale.
    - 4-bar linkage angle-dependent torque LUT directory.
    - Command delay (``min_delay``, ``max_delay``) — fixed delay if min == max.
    - Optional per-step motor strength randomization (off by default).

    Note:
        The PD ``stiffness`` and ``damping`` fields inherited from
        :class:`DCMotorCfg` / :class:`IdealPDActuatorCfg` are deliberately
        defaulted to ``None``; the LSTM replaces the PD law entirely.
    """

    class_type: type = ActuatorNetLSTMWalker

    # The network replaces PD — stiffness/damping are unused.
    stiffness = None
    damping = None

    # ---- Per-joint LSTM ----
    network_files: dict[str, str] = MISSING
    """Mapping from joint name → LSTM TorchScript ``.pt`` file path.

    Keys must be **exact** joint names (not regex) and must cover every joint
    the actuator manages. Values can be absolute paths, or relative paths
    resolved against :attr:`network_dir`.
    """

    network_dir: str | None = None
    """Base directory for relative entries in :attr:`network_files`."""

    torque_scale: float = 100.0
    """Scalar multiplier applied to the network output before clipping.

    Training pipeline used ``torque_scaling=0.01`` on the effort target, so raw
    network outputs are ~1/100 of the physical torque. Default 100.0 recovers Nm.
    """

    # ---- 4-bar linkage angle-dependent torque LUT ----
    torque_lut_dir: str | None = None
    """Path to directory containing torque LUT CSVs (knee/ankle). ``None``
    disables angle-dependent limits and falls back to static ``effort_limit``.
    """

    # ---- Command delay (mirrors DelayedPDActuatorCfg fields) ----
    min_delay: int = 0
    """Minimum number of physics steps the command may be delayed."""

    max_delay: int = 0
    """Maximum number of physics steps the command may be delayed."""

    # ---- Per-step motor strength randomization (typically disabled) ----
    rand_motor_scale_range: tuple[float, float] | None = None
    """Uniform range ``(low, high)`` for per-step motor strength randomization.

    ``None`` disables this term. Re-randomizing every physics step injects
    multiplicative white noise on the actuator output and tends to excite
    closed-loop resonances in the LSTM, so the safer behaviour is to either
    leave it disabled or randomize once per environment reset instead.
    """

    # ---- PD warm-up blending for LSTM hidden state convergence ----
    warm_up_steps: int = 0
    """Number of physics steps after env reset during which torque output is
    blended from pure PD towards pure LSTM using a cosine schedule.

    During warm-up the LSTM still runs (shadow mode) so its hidden/cell state
    accumulates real physics observations, but the actual torque applied to the
    simulation is ``(1-λ)*τ_PD + λ*τ_LSTM`` where λ ramps from 0 to 1.
    Set to 0 to disable (pure LSTM from the first tick).
    """

    warm_up_stiffness: dict[str, float] | float | None = None
    """PD position gain (Kp) used during the warm-up blending window.

    Accepts a per-joint dict (exact joint names) or a single float for all
    joints. ``None`` disables PD warm-up even if ``warm_up_steps > 0``.
    """

    warm_up_damping: dict[str, float] | float | None = None
    """PD velocity gain (Kd) used during the warm-up blending window.

    Same format as :attr:`warm_up_stiffness`.
    """
