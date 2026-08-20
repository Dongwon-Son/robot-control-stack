#!/usr/bin/env python3
"""Single-process TAM + Franka control with robot-control-stack.

One machine, one script: connects to the robot with the RCS ``hw.Franka``
class (``ignore_realtime=True`` by default, so no PREEMPT_RT kernel is
required), loads a TAM checkpoint, exports its adaptor into the controller's
``TamHook``, starts the background history-encoder thread
(:class:`rcs_tam.TamDeployment`), and drives a joint-space test motion while
TAM corrects the torque command at 1 kHz inside the RCS control thread.

Example::

    python examples/franka_tam_direct.py --robot panda --ip 192.168.0.52 \
        --ckpt /path/to/tam_checkpoint --motion sine --duration-s 30

Supported checkpoints: applied-torque history (the Panda-specific and DAgger
finetuned checkpoints) and ``base_tam_fusion`` (the fused-input DAgger
checkpoint); the mode is read from the checkpoint. Pass ``--xml`` only if the
checkpoint bundle has no ``robot_model/robot.xml``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def _parse_vec7(raw: str) -> tuple[float, ...]:
    vals = tuple(float(x) for x in raw.split(",") if x.strip())
    if len(vals) == 1:
        vals = vals * 7
    if len(vals) != 7:
        raise argparse.ArgumentTypeError(f"expected 1 or 7 comma-separated values, got {len(vals)}")
    return vals


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", choices=("panda", "fr3"), default="panda")
    p.add_argument("--ip", default="192.168.0.52", help="Franka FCI address.")
    p.add_argument("--ckpt", required=True, help="TAM checkpoint directory (save_dict.pkl or checkpoint_<step>).")
    p.add_argument("--xml", default=None, help="Robot MJCF for the ideal model (default: the checkpoint's robot_model).")
    p.add_argument("--history-torque-mode", choices=("auto", "applied", "base_tam_fusion"), default="auto")
    p.add_argument("--torque-limit", type=_parse_vec7, default=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
                   help="RCS FrankaConfig.torque_limit on the gravity-free command (RCS' own 5 Nm default is too tight).")
    p.add_argument("--residual-clip", type=_parse_vec7, default=(10.0, 10.0, 10.0, 10.0, 8.0, 8.0, 8.0),
                   help="Per-joint clip of the TAM residual in Nm. Start small (e.g. 2) on first runs.")
    p.add_argument("--joint-kp", type=_parse_vec7, default=None, help="RCS joint PD kp override (7 values).")
    p.add_argument("--joint-kd", type=_parse_vec7, default=None, help="RCS joint PD kd override (7 values).")
    p.add_argument("--no-tam", action="store_true", help="Run the motion without enabling TAM (baseline).")
    p.add_argument("--enforce-realtime", action="store_true",
                   help="Require an RT kernel (FrankaConfig.ignore_realtime=False). Default: ignore, single stock machine.")
    p.add_argument("--home", action="store_true", help="move_home() before starting.")
    p.add_argument("--motion", choices=("hold", "sine"), default="sine")
    p.add_argument("--sine-amp-deg", type=float, default=10.0)
    p.add_argument("--sine-period-s", type=float, default=6.0)
    p.add_argument("--policy-rate-hz", type=float, default=20.0)
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--embedding-interval-s", type=float, default=0.2)
    p.add_argument("--attention-history-s", type=float, default=4.0)
    p.add_argument("--status-interval-s", type=float, default=2.0)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    import rcs
    from rcs import common

    if args.robot == "panda":
        from rcs_panda._core import hw

        robot_type = common.RobotType.Panda
        cfg_cls = hw.PandaConfig
    else:
        from rcs_fr3._core import hw

        robot_type = common.RobotType.FR3
        cfg_cls = hw.FR3Config

    meta = rcs.ROBOTS[robot_type]
    robot_cfg = cfg_cls(
        ip=str(args.ip),
        async_control=True,
        ignore_realtime=not args.enforce_realtime,
        torque_limit=np.asarray(args.torque_limit, dtype=float),
        kinematic_model_path=meta.mjcf_model_path,
    )
    robot_cfg.robot_type = robot_type
    robot_cfg.q_home = meta.q_home
    robot_cfg.joint_limits = meta.joint_limits
    robot_cfg.dof = meta.dof
    if args.joint_kp is not None:
        robot_cfg.kp = np.asarray(args.joint_kp, dtype=float)
    if args.joint_kd is not None:
        robot_cfg.kd = np.asarray(args.joint_kd, dtype=float)

    print(f"[example] connecting to {args.robot} at {args.ip} "
          f"(ignore_realtime={not args.enforce_realtime}, torque_limit={list(args.torque_limit)})")
    robot = hw.Franka(robot_cfg)
    if args.home:
        robot.move_home()

    from rcs_tam import TamDeployment

    tam = TamDeployment(
        robot,
        Path(args.ckpt),
        xml_path=args.xml,
        history_torque_mode=args.history_torque_mode,
        attention_history_s=float(args.attention_history_s),
        embedding_interval_s=float(args.embedding_interval_s),
        residual_torque_limits=np.asarray(args.residual_clip, dtype=float),
        enable_after_first_embedding=not args.no_tam,
    )

    # Start the RCS joint controller holding the current configuration; the
    # TamHook begins recording history from the first control tick.
    q0 = np.asarray(robot.get_joint_position(), dtype=float)
    robot.controller_set_joint_position(q0)
    tam.start()
    print(f"[example] joint controller running, q0={np.round(q0, 3).tolist()}; encoder thread started")

    amp = np.deg2rad(float(args.sine_amp_deg)) * np.asarray([1.0, 0.6, 0.8, 0.6, 1.0, 0.8, 1.0])
    period = 1.0 / float(args.policy_rate_hz)
    t0 = time.perf_counter()
    next_cmd = t0
    next_status = t0
    try:
        while time.perf_counter() - t0 < float(args.duration_s):
            now = time.perf_counter()
            if now >= next_cmd:
                if args.motion == "sine":
                    q_ref = q0 + amp * np.sin(2.0 * np.pi * (now - t0) / float(args.sine_period_s))
                else:
                    q_ref = q0
                robot.controller_set_joint_position(q_ref)
                next_cmd += period
            if now >= next_status:
                st = tam.status()
                q_now = np.asarray(robot.get_joint_position(), dtype=float)
                print(
                    f"[example] t={now - t0:6.1f}s mode={st['mode']} tam={'on' if st['enabled'] else 'off'} "
                    f"emb_sent={st['embeddings_sent']} skip={st.get('hook_last_skip_reason', '?')!r} "
                    f"fwd={st.get('hook_adaptor_forward_dt_ms', 0.0):.2f}ms "
                    f"q_err_deg={np.rad2deg(np.max(np.abs(q_now - q_ref))):.2f}"
                )
                next_status += float(args.status_interval_s)
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("[example] interrupted")
    finally:
        tam.stop()
        robot.stop_control_thread()
        print("[example] stopped; final status:", {k: v for k, v in tam.status().items()
                                                   if k in ("mode", "embeddings_sent", "controller_restarts", "last_error")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
