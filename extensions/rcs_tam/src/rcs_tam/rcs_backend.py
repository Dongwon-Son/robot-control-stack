"""robot-control-stack backend: ``hw.Franka`` (``rcs_fr3``/``rcs_panda`` ``tam`` fork) + TAM hook."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional

import numpy as np

from rcs_tam.backend import BridgeBackend, UnsupportedCommand, history_rows_dict_to_samples
from rcs_tam.protocol import CONTROL_MODE_CARTESIAN, CONTROL_MODE_JOINT, normalize_control_mode_name

_UNSUPPORTED_ACTUATION_KEYS = {
    "filter",
    "feedforward",
    "damping_ratio",
    "cartesian_stiffness",
    "cartesian_damping",
    "cartesian_nullspace_stiffness",
    "target_q_nullspace",
}


class RcsBackend(BridgeBackend):
    """Drive an RCS ``hw.Franka`` (async torque controllers) through the TAM bridge protocol.

    Mapping of protocol commands onto RCS:

    * ``target_q``                  -> ``robot.controller_set_joint_position(q)`` (RCS joint PD thread)
    * ``target_position/orientation``-> ``robot.osc_set_cartesian_position(Pose)`` (RCS OSC thread);
      quaternions are xyzw on both sides
    * ``control_mode``              -> stops the running control thread so the next target
      starts the requested controller
    * ``stiffness/damping``         -> written into ``FrankaConfig.kp/kd``; RCS reads gains
      when a control thread starts, so they apply on the next (re)start
    * ``embedding``, ``enable_adaptor``, ``load_simadaptor_bin``, ``ideal_model_has_gravity``
      -> ``robot.tam_*``
    * sysid / external torque prediction / soft block / test disturbances -> unsupported
    """

    name = "rcs"

    def __init__(
        self,
        robot: Any,
        *,
        common_module: Any = None,
        franka_exceptions: Any = None,
        home_on_reset: bool = True,
        ext_force_threshold_n: float = 100.0,
        ext_torque_threshold_nm: float = 50.0,
        default_control_mode: str = CONTROL_MODE_JOINT,
        log=print,
    ) -> None:
        self.robot = robot
        self._common = common_module
        self._franka_exceptions = franka_exceptions
        self.home_on_reset = bool(home_on_reset)
        self.ext_force_threshold_n = float(ext_force_threshold_n)
        self.ext_torque_threshold_nm = float(ext_torque_threshold_nm)
        self.control_mode = normalize_control_mode_name(default_control_mode)
        self._log = log
        self._last_target_q: Optional[np.ndarray] = None
        self._last_target_pose: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._warned_unsupported: set[str] = set()
        self._pending_gains: Dict[str, np.ndarray] = {}
        self.command_count = 0

    # ---- lazy RCS imports -------------------------------------------------- #
    def _pose(self, translation: np.ndarray, quaternion_xyzw: np.ndarray):
        common = self._common
        if common is None:
            import rcs  # type: ignore

            common = rcs.common
            self._common = common
        return common.Pose(translation=np.asarray(translation, dtype=float), quaternion=np.asarray(quaternion_xyzw, dtype=float))

    def _control_exception_types(self) -> tuple:
        if self._franka_exceptions is not None:
            return tuple(self._franka_exceptions)
        types: list = []
        for mod_name in ("rcs_panda._core.hw", "rcs_fr3._core.hw"):
            try:
                mod = __import__(mod_name, fromlist=["exceptions"])
                exc_mod = getattr(mod, "exceptions", None)
                if exc_mod is not None:
                    for name in ("FrankaControlException", "FrankaException"):
                        exc = getattr(exc_mod, name, None)
                        if exc is not None:
                            types.append(exc)
                break
            except Exception:
                continue
        self._franka_exceptions = types
        return tuple(types)

    # ---- history / TAM ---------------------------------------------------- #
    def get_history_samples(self, max_rows: int):
        return history_rows_dict_to_samples(self.robot.tam_get_history(int(max_rows)))

    def set_embedding(self, embedding: np.ndarray) -> None:
        self.robot.tam_set_embedding(np.asarray(embedding, dtype=float).reshape(-1))

    def get_embedding_seq(self) -> int:
        return int(self.robot.tam_get_embedding_seq())

    def enable_adaptor(self, enabled: bool) -> None:
        self.robot.tam_enable(bool(enabled))

    def adaptor_enabled(self) -> bool:
        return bool(self.robot.tam_is_enabled())

    def load_adaptor_path(self, path: str) -> bool:
        return bool(self.robot.tam_load_adaptor(str(path)))

    def set_ideal_model_has_gravity(self, enabled: bool) -> None:
        self.robot.tam_set_ideal_model_has_gravity(bool(enabled))

    def ideal_model_has_gravity(self) -> bool:
        return bool(self.robot.tam_get_ideal_model_has_gravity())

    # ---- actuation ---------------------------------------------------------- #
    def _warn_unsupported(self, key: str) -> None:
        if key not in self._warned_unsupported:
            self._warned_unsupported.add(key)
            self._log(f"[rcs_backend] ignoring unsupported actuation key {key!r} (RCS has no equivalent)")

    def _switch_mode(self, mode: str) -> None:
        mode = normalize_control_mode_name(mode)
        if mode == self.control_mode:
            return
        self.robot.stop_control_thread()
        self.control_mode = mode
        self._log(f"[rcs_backend] control mode: {mode} (control thread stopped; next target restarts it)")

    def _set_joint_target(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).reshape(-1)
        try:
            self.robot.controller_set_joint_position(q)
        except RuntimeError as exc:
            if "Controller type" not in str(exc):
                raise
            # A different controller thread is running (e.g. OSC): RCS requires stopping it first.
            self.robot.stop_control_thread()
            self.robot.controller_set_joint_position(q)
        self._last_target_q = q
        self._last_target_pose = None
        self.control_mode = CONTROL_MODE_JOINT

    def _set_pose_target(self, position: np.ndarray, quaternion_xyzw: np.ndarray) -> None:
        pose = self._pose(position, quaternion_xyzw)
        try:
            self.robot.osc_set_cartesian_position(pose)
        except RuntimeError as exc:
            if "Controller type" not in str(exc):
                raise
            self.robot.stop_control_thread()
            self.robot.osc_set_cartesian_position(pose)
        self._last_target_pose = (np.asarray(position, dtype=float), np.asarray(quaternion_xyzw, dtype=float))
        self._last_target_q = None
        self.control_mode = CONTROL_MODE_CARTESIAN

    def apply_actuation(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        resp: Dict[str, Any] = {}
        unsupported = sorted(k for k in payload if k in _UNSUPPORTED_ACTUATION_KEYS)
        for key in unsupported:
            self._warn_unsupported(key)
        if unsupported:
            resp["unsupported_keys"] = unsupported

        if "control_mode" in payload:
            self._switch_mode(str(payload["control_mode"]))

        if "stiffness" in payload or "damping" in payload:
            cfg = self.robot.get_config()
            if "stiffness" in payload:
                cfg.kp = np.asarray(payload["stiffness"], dtype=float).reshape(-1)
            if "damping" in payload:
                cfg.kd = np.asarray(payload["damping"], dtype=float).reshape(-1)
            self.robot.set_config(cfg)
            resp["gains_apply_on_next_controller_start"] = True
            self._log("[rcs_backend] joint gains stored in FrankaConfig; RCS applies them when a control thread starts")

        if "target_q" in payload:
            self._set_joint_target(np.asarray(payload["target_q"], dtype=float))
            self.command_count += 1
        if "target_position" in payload or "target_orientation" in payload:
            if self._last_target_pose is not None:
                position, quaternion = self._last_target_pose
            else:
                current = self.robot.get_cartesian_position()
                position = np.asarray(current.translation(), dtype=float)
                quaternion = np.asarray(current.rotation_q(), dtype=float)
            if "target_position" in payload:
                position = np.asarray(payload["target_position"], dtype=float).reshape(3)
            if "target_orientation" in payload:
                quaternion = np.asarray(payload["target_orientation"], dtype=float).reshape(4)
                norm = float(np.linalg.norm(quaternion))
                if norm <= 0.0:
                    raise ValueError("target_orientation must be a non-zero quaternion.")
                quaternion = quaternion / norm
            self._set_pose_target(position, quaternion)
            self.command_count += 1
        return resp

    def neutralize_remote_motion(self) -> None:
        # Hold the current joint configuration with the joint controller.
        try:
            q_now = np.asarray(self.robot.get_joint_position(), dtype=float)
        except Exception as exc:
            self._log(f"[rcs_backend] neutralize: could not read joints: {exc}")
            return
        self._set_joint_target(q_now)

    def reset(self) -> None:
        self.robot.stop_control_thread()
        self.robot.reset()  # automatic error recovery
        if self.home_on_reset:
            try:
                self.robot.move_home()
            except Exception as exc:
                self._log(f"[rcs_backend] move_home failed during reset: {exc}")
        self.robot.tam_reset()
        self._last_target_q = None
        self._last_target_pose = None
        self.control_mode = CONTROL_MODE_JOINT

    def safety_check(self) -> Optional[str]:
        try:
            # get_joint_position() rethrows exceptions raised inside the RCS
            # control thread (check_for_background_errors) and stops that thread.
            self.robot.get_joint_position()
            state = self.robot.get_state()
        except Exception as exc:
            if isinstance(exc, self._control_exception_types()):
                return f"franka_control_exception:{exc}"
            return f"state_read_error:{exc}"
        robot_state = getattr(state, "robot_state", None)
        if robot_state is None:
            return None
        try:
            f_ext = np.asarray(robot_state.O_F_ext_hat_K, dtype=float)
            tau_j_d = np.asarray(robot_state.tau_J_d, dtype=float)
        except Exception:
            return None
        f_norm = float(np.linalg.norm(f_ext[:3]))
        tau_norm = float(np.linalg.norm(tau_j_d))
        if f_norm > self.ext_force_threshold_n:
            return f"ext_force f_norm={f_norm:.1f}N"
        if tau_norm > self.ext_torque_threshold_nm:
            return f"ext_torque tau_norm={tau_norm:.1f}Nm"
        return None

    def status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"rcs_control_mode": self.control_mode, "rcs_command_count": self.command_count}
        try:
            st = dict(self.robot.tam_status())
            st.pop("torque_limits", None)
            out.update({f"tam_{k}": v for k, v in st.items()})
        except Exception as exc:  # pragma: no cover
            out["tam_status_error"] = str(exc)
        return out

    def close(self) -> None:
        try:
            self.robot.stop_control_thread()
        except Exception:
            pass


__all__ = ["RcsBackend"]
