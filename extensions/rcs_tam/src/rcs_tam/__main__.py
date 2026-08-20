"""NUC entrypoint: ``python -m rcs_tam`` — TAM bridge on top of an RCS ``hw.Franka``.

Creates an RCS Franka handle (``rcs_panda`` or ``rcs_fr3`` from the ``tam``
branch), starts the RCS joint controller thread holding the current
configuration, and serves the TAM ZMQ protocol (history PUB :5555, async PULL
:5556, reliable REP :5557) so the TAM workstation stack (``mapping_server.py``,
``HistoryControllerClient``, launchers) works unchanged.  Policies can stream
``target_q`` / Cartesian targets over the bridge or drive the same ``robot``
through the RCS gym API in this process (see ``rcs_tam.env_wrapper``).

Configuration: ``--config`` JSON (see ``rcs_tam.config`` and
``rcs_tam_config.example.json``) plus the flags below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from rcs_tam.bridge import BridgeEndpoints, TamBridge
from rcs_tam.config import load_runtime_config
from rcs_tam.rcs_backend import RcsBackend


def _parse_vec(raw: str, size: int) -> tuple[float, ...]:
    vals = tuple(float(x) for x in raw.split(",") if x.strip())
    if len(vals) == 1:
        vals = vals * size
    if len(vals) != size:
        raise argparse.ArgumentTypeError(f"expected {size} comma-separated values, got {len(vals)}")
    return vals


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m rcs_tam", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", choices=("panda", "fr3"), default="panda", help="RCS extension to use.")
    p.add_argument("--robot-ip", default=None, help="FCI address (default: network.robot_host from the config).")
    p.add_argument("--config", default=None, help="JSON config path (default: RCS_TAM_CONFIG env or ./rcs_tam_config.json).")
    p.add_argument("--host", default=None, help="Bind host for the three ZMQ sockets (default: network.nuc_control_host).")
    p.add_argument("--torque-limit", type=lambda s: _parse_vec(s, 7), default=None,
                   help="RCS FrankaConfig.torque_limit clamp on the gravity-free command, 1 or 7 values (config default 87/87/87/87/12/12/12; RCS' own default 5 Nm is too tight for TAM).")
    p.add_argument("--joint-kp", type=lambda s: _parse_vec(s, 7), default=None, help="RCS joint PD kp (7 values).")
    p.add_argument("--joint-kd", type=lambda s: _parse_vec(s, 7), default=None, help="RCS joint PD kd (7 values).")
    p.add_argument("--adaptor-bin", default=None, help="Optional SimAdaptor .bin to load at startup (stays disabled until enable_adaptor).")
    p.add_argument("--adaptor-torque-limits", type=lambda s: _parse_vec(s, 7), default=None, help="TAM residual clip per joint (Nm).")
    p.add_argument("--no-ideal-model-gravity", action="store_true", help="Set tam ideal_model_has_gravity=False.")
    p.add_argument("--home", action="store_true", help="Move to q_home before starting the controller.")
    p.add_argument("--home-on-reset", action="store_true", help="Also move home on bridge resets.")
    p.add_argument("--no-start-controller", action="store_true", help="Do not start the RCS joint controller thread at startup.")
    p.add_argument("--require-remote-actuation-for-adaptor-enable", action="store_true",
                   help="Queue enable_adaptor until a fresh remote actuation command has been seen (guards against enabling the residual while nothing streams targets).")
    p.add_argument("--ext-force-threshold-n", type=float, default=None)
    p.add_argument("--ext-torque-threshold-nm", type=float, default=None)
    p.add_argument("--print-config", action="store_true", help="Print the resolved configuration and exit (no robot connection).")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_runtime_config(args.config, search_dir=Path(__file__).resolve().parent)
    if args.host:
        cfg.network.nuc_control_host = str(args.host)
        cfg.network.history_endpoint = cfg.network.command_endpoint = cfg.network.request_endpoint = None
    endpoints = BridgeEndpoints.from_network_config(cfg.network)
    robot_ip = args.robot_ip or cfg.network.robot_host
    torque_limit = tuple(args.torque_limit) if args.torque_limit is not None else tuple(float(x) for x in cfg.rcs.torque_limit)
    if len(torque_limit) == 1:
        torque_limit = torque_limit * 7
    ext_force = cfg.safety.ext_force_threshold if args.ext_force_threshold_n is None else args.ext_force_threshold_n
    ext_torque = cfg.safety.ext_torque_threshold if args.ext_torque_threshold_nm is None else args.ext_torque_threshold_nm
    print(f"[rcs_tam] config={cfg.loaded_from or 'defaults'} robot={args.robot} ip={robot_ip} endpoints={endpoints}")
    print(
        f"[rcs_tam] history_len={cfg.timing.history_len} publish_dt={cfg.timing.history_pub_dt} "
        f"loop_dt={cfg.timing.loop_dt} stale_timeout={cfg.timing.remote_command_stale_timeout_s}s "
        f"torque_limit={list(torque_limit)} ext_force={ext_force}N ext_torque={ext_torque}Nm"
    )
    if args.print_config:
        return 0

    import rcs  # type: ignore
    from rcs import common  # type: ignore

    if args.robot == "panda":
        from rcs_panda._core import hw  # type: ignore

        robot_type = common.RobotType.Panda
        cfg_cls = hw.PandaConfig
    else:
        from rcs_fr3._core import hw  # type: ignore

        robot_type = common.RobotType.FR3
        cfg_cls = hw.FR3Config

    robot_meta = rcs.ROBOTS[robot_type]
    robot_cfg = cfg_cls(
        ip=str(robot_ip),
        async_control=True,
        ignore_realtime=False,
        torque_limit=np.asarray(torque_limit, dtype=float),
        kinematic_model_path=robot_meta.mjcf_model_path,
    )
    robot_cfg.robot_type = robot_type
    robot_cfg.q_home = robot_meta.q_home
    robot_cfg.joint_limits = robot_meta.joint_limits
    robot_cfg.dof = robot_meta.dof
    kp = args.joint_kp if args.joint_kp is not None else cfg.rcs.joint_kp
    kd = args.joint_kd if args.joint_kd is not None else cfg.rcs.joint_kd
    if kp is not None:
        robot_cfg.kp = np.asarray(kp, dtype=float)
    if kd is not None:
        robot_cfg.kd = np.asarray(kd, dtype=float)

    robot = hw.Franka(robot_cfg)
    robot.tam_set_ideal_model_has_gravity(bool(cfg.rcs.ideal_model_has_gravity) and not args.no_ideal_model_gravity)
    limits = args.adaptor_torque_limits if args.adaptor_torque_limits is not None else cfg.rcs.adaptor_torque_limits
    if limits is not None:
        robot.tam_set_torque_limits(np.asarray(limits, dtype=float))
    if args.adaptor_bin:
        ok = robot.tam_load_adaptor(str(args.adaptor_bin))
        print(f"[rcs_tam] tam_load_adaptor({args.adaptor_bin}) -> {ok}")
    if args.home:
        robot.move_home()
    if not args.no_start_controller:
        q_now = np.asarray(robot.get_joint_position(), dtype=float)
        robot.controller_set_joint_position(q_now)
        print(f"[rcs_tam] joint controller started, holding q={np.round(q_now, 3).tolist()}")

    backend = RcsBackend(
        robot,
        common_module=common,
        home_on_reset=bool(args.home_on_reset or cfg.rcs.home_on_reset),
        ext_force_threshold_n=float(ext_force),
        ext_torque_threshold_nm=float(ext_torque),
    )
    bridge = TamBridge(
        backend,
        endpoints,
        history_len=int(cfg.timing.history_len),
        publish_dt=float(cfg.timing.history_pub_dt),
        loop_dt=float(cfg.timing.loop_dt),
        remote_command_stale_timeout_s=float(cfg.timing.remote_command_stale_timeout_s),
        require_remote_actuation_for_adaptor_enable=bool(
            args.require_remote_actuation_for_adaptor_enable or cfg.rcs.require_remote_actuation_for_adaptor_enable
        ),
    )
    try:
        bridge.run_forever()
    except KeyboardInterrupt:
        print("[rcs_tam] interrupted")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
