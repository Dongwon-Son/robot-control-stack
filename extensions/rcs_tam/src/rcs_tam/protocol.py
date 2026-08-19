"""TAM NUC-bridge wire protocol (ZMQ + JSON).

Message shapes (identical to the TAM reference NUC controller, so the
workstation tooling — mapping server, ``HistoryControllerClient``, experiment
launchers — works unchanged):

* History publish (PUB)::

      {"type": "history", "window": [sample, ...]}        # every publish_dt, newest rows, oldest first
      {"type": "reset", "reason": str, "timestamp": float}

  ``sample`` is built by :func:`sample_dict_from_row` (q, dq, torque
  decomposition, gravity, embedding sequence, validity flags, ...).

* Async commands (PULL; may carry ``command_source``/``command_id`` for
  per-source dedup)::

      {"target_q": [...], "target_dq": [...], "stiffness": [...], "damping": [...],
       "filter": float, "embedding": [...], "feedforward": [...],
       "control_mode": "joint"|"cartesian", "target_position": [...],
       "target_orientation": [x,y,z,w], "enable_adaptor": bool, "reset": bool, ...}

* Reliable commands (REQ/REP)::

      {"cmd": "set_embedding", "embedding": [...]}
      {"cmd": "enable_adaptor", "enabled": bool}
      {"cmd": "load_bin_blob", "name": str, "blob": base64, "enable_after_load": bool}
      {"cmd": "load_bin_path", "path": str}
      {"cmd": "set_ideal_model_has_gravity", "enabled": bool}
      {"cmd": "enable_sysid"|"clear_sysid"|"set_sysid"|"enable_external_torque_prediction"|...}
      {"cmd": "reset"}
      {}                                   # status query

  Replies are ``{"ok": bool, ...status fields...}``; unsupported commands
  answer ``ok=False, unsupported=True``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import zmq

CONTROL_MODE_JOINT = "joint"
CONTROL_MODE_CARTESIAN = "cartesian"

TEST_DISTURBANCE_COMMANDS = {"arm_test_disturbance", "cancel_test_disturbance", "test_disturbance_status"}
TEST_DISTURBANCE_PAYLOAD_KEYS = set(TEST_DISTURBANCE_COMMANDS)

ASYNC_MERGEABLE_KEYS = {
    "control_mode",
    "target_q",
    "target_dq",
    "target_position",
    "target_orientation",
    "target_q_nullspace",
    "stiffness",
    "damping",
    "damping_ratio",
    "filter",
    "cartesian_stiffness",
    "cartesian_damping",
    "cartesian_nullspace_stiffness",
    "embedding",
    "feedforward",
    "enable_adaptor",
    "enable_sysid",
    "enable_external_torque_prediction",
    "enable_external_force_soft_block",
    "external_force_soft_block",
}
ASYNC_METADATA_KEYS = {"command_source", "command_id"}
REMOTE_ACTUATION_KEYS = {
    "control_mode",
    "target_q",
    "target_dq",
    "target_position",
    "target_orientation",
    "target_q_nullspace",
    "stiffness",
    "damping",
    "damping_ratio",
    "filter",
    "cartesian_stiffness",
    "cartesian_damping",
    "cartesian_nullspace_stiffness",
    "feedforward",
    "embedding",
}

ZERO7: List[float] = [0.0] * 7
IDENTITY4: List[List[float]] = np.eye(4).tolist()


@dataclass
class AsyncSourceState:
    last_command_id: int
    last_seen_monotonic: float


def normalize_control_mode_name(mode: Any) -> str:
    if not isinstance(mode, str):
        raise ValueError("control_mode must be 'joint' or 'cartesian'.")
    normalized = mode.strip().lower()
    if normalized not in {CONTROL_MODE_JOINT, CONTROL_MODE_CARTESIAN}:
        raise ValueError("control_mode must be 'joint' or 'cartesian'.")
    return normalized


def normalize_command_message(msg: Dict[str, Any], *, allow_test_disturbance: bool = False) -> Dict[str, Any]:
    """Map a reliable ``{"cmd": ...}`` request (or an async dict) onto payload keys."""
    if not isinstance(msg, dict):
        raise TypeError("Command message must be a dict.")
    if "cmd" not in msg:
        if not allow_test_disturbance and TEST_DISTURBANCE_PAYLOAD_KEYS.intersection(msg):
            raise ValueError("test disturbance commands are accepted only on the reliable request endpoint")
        return dict(msg)

    cmd = msg.get("cmd")
    if cmd in TEST_DISTURBANCE_COMMANDS:
        if not allow_test_disturbance:
            raise ValueError("test disturbance commands are accepted only on the reliable request endpoint")
        if cmd == "arm_test_disturbance":
            return {"arm_test_disturbance": {k: msg.get(k) for k in ("event_id", "tau_nm", "delay_s", "duration_s")}}
        if cmd == "cancel_test_disturbance":
            return {"cancel_test_disturbance": {"event_id": msg.get("event_id")}}
        return {"test_disturbance_status": True}
    if cmd == "load_bin_blob":
        return {
            "simadaptor_bin_name": msg.get("name", "simadaptor.bin"),
            "simadaptor_bin_blob": msg.get("blob"),
            "simadaptor_enable_after_load": bool(msg.get("enable_after_load", False)),
        }
    if cmd == "load_bin_path":
        return {
            "load_simadaptor_bin": msg.get("path"),
            "simadaptor_enable_after_load": bool(msg.get("enable_after_load", False)),
        }
    if cmd == "set_embedding":
        return {"embedding": msg.get("embedding", [])}
    if cmd == "set_control_mode":
        return {"control_mode": msg.get("mode")}
    if cmd == "set_cartesian_target":
        payload: Dict[str, Any] = {}
        if "position" in msg:
            payload["target_position"] = msg.get("position")
        if "orientation" in msg:
            payload["target_orientation"] = msg.get("orientation")
        if "q_nullspace" in msg:
            payload["target_q_nullspace"] = msg.get("q_nullspace")
        return payload
    if cmd == "set_cartesian_gains":
        payload = {}
        if "stiffness" in msg:
            payload["cartesian_stiffness"] = msg.get("stiffness")
        if "damping" in msg:
            payload["cartesian_damping"] = msg.get("damping")
        if "nullspace_stiffness" in msg:
            payload["cartesian_nullspace_stiffness"] = msg.get("nullspace_stiffness")
        if "damping_ratio" in msg:
            payload["damping_ratio"] = msg.get("damping_ratio")
        return payload
    if cmd == "enable_adaptor":
        return {"enable_adaptor": bool(msg.get("enabled", False))}
    if cmd == "enable_sysid":
        return {"enable_sysid": bool(msg.get("enabled", False))}
    if cmd == "enable_external_torque_prediction":
        return {"enable_external_torque_prediction": bool(msg.get("enabled", False))}
    if cmd == "enable_external_force_soft_block":
        return {"enable_external_force_soft_block": bool(msg.get("enabled", True))}
    if cmd == "set_external_force_soft_block":
        params = dict(msg.get("params") or {})
        for key in ("enabled", "threshold_n", "gain", "damping_n_per_mps", "max_force_n"):
            if key in msg:
                params[key] = msg.get(key)
        return {"external_force_soft_block": params}
    if cmd == "set_ideal_model_has_gravity":
        return {"ideal_model_has_gravity": bool(msg.get("enabled", False))}
    if cmd == "clear_sysid":
        return {"clear_sysid": True}
    if cmd == "set_sysid":
        return {"set_sysid": msg.get("params", msg.get("sysid", {}))}
    if cmd == "reset":
        return {"reset": True}
    raise ValueError(f"unknown cmd {cmd}")


# ---- async (PULL) helpers -------------------------------------------------- #


def extract_async_metadata(msg: Dict[str, Any]) -> Optional[tuple[str, int]]:
    source = msg.get("command_source")
    command_id = msg.get("command_id")
    if source is None and command_id is None:
        return None
    if not isinstance(source, str) or not source:
        raise ValueError("Async command requires a non-empty string command_source.")
    if isinstance(command_id, bool) or not isinstance(command_id, int):
        raise ValueError("Async command requires an integer command_id.")
    return source, int(command_id)


def strip_async_metadata(msg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in msg.items() if k not in ASYNC_METADATA_KEYS}


def is_mergeable_async_payload(payload: Dict[str, Any]) -> bool:
    payload_keys = set(payload.keys())
    return bool(payload_keys) and payload_keys.issubset(ASYNC_MERGEABLE_KEYS)


def payload_has_remote_actuation(payload: Dict[str, Any]) -> bool:
    return any(key in payload for key in REMOTE_ACTUATION_KEYS)


def strip_remote_actuation_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    ignored_keys = sorted(REMOTE_ACTUATION_KEYS.intersection(payload.keys()))
    if not ignored_keys:
        return dict(payload), []
    return {k: v for k, v in payload.items() if k not in REMOTE_ACTUATION_KEYS}, ignored_keys


def expire_async_command_sources(
    async_source_states: Dict[str, AsyncSourceState], stale_timeout_s: float, now_monotonic: Optional[float] = None
) -> None:
    if now_monotonic is None:
        now_monotonic = time.monotonic()
    expired = [s for s, st in async_source_states.items() if (now_monotonic - st.last_seen_monotonic) > stale_timeout_s]
    for source in expired:
        del async_source_states[source]


def drain_async_commands(
    pull: zmq.Socket, async_source_states: Dict[str, AsyncSourceState], stale_timeout_s: float, log=print
) -> list[Dict[str, Any]]:
    """Receive every queued async message; drop stale/duplicate command ids per source."""
    accepted: list[Dict[str, Any]] = []
    expire_async_command_sources(async_source_states, stale_timeout_s)
    while True:
        try:
            msg = pull.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        if not isinstance(msg, dict):
            continue
        try:
            meta = extract_async_metadata(msg)
        except Exception as exc:
            log(f"[rcs_tam] Rejecting malformed async metadata: {exc}")
            continue
        if meta is not None:
            source, command_id = meta
            now_monotonic = time.monotonic()
            expire_async_command_sources(async_source_states, stale_timeout_s, now_monotonic)
            last_state = async_source_states.get(source)
            if last_state is not None and command_id <= last_state.last_command_id:
                continue
            async_source_states[source] = AsyncSourceState(last_command_id=command_id, last_seen_monotonic=now_monotonic)
        accepted.append(strip_async_metadata(msg))
    return accepted


def collapse_async_commands(msgs: list[Dict[str, Any]], log=print) -> list[Dict[str, Any]]:
    """Merge consecutive state-like payloads (latest value wins); stop at a reset."""
    collapsed: list[Dict[str, Any]] = []
    pending: Dict[str, Any] = {}
    for msg in msgs:
        try:
            payload = normalize_command_message(msg)
        except Exception as exc:
            log(f"[rcs_tam] Rejecting malformed async command: {exc}")
            continue
        if is_mergeable_async_payload(payload):
            pending.update(payload)
            continue
        if pending:
            collapsed.append(dict(pending))
            pending.clear()
        collapsed.append(payload)
        if payload.get("reset"):
            break
    if pending:
        collapsed.append(dict(pending))
    return collapsed


def drain_pending_async_messages(pull: zmq.Socket) -> int:
    drained = 0
    while True:
        try:
            pull.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        drained += 1
    return drained


# ---- publish helpers ------------------------------------------------------- #


def publish_reset(pub: zmq.Socket, reason: str, **extra: Any) -> None:
    payload: Dict[str, Any] = {"type": "reset", "reason": str(reason), "timestamp": time.time()}
    for key, value in extra.items():
        if value is not None:
            payload[key] = float(value) if isinstance(value, (int, float, np.floating)) else value
    pub.send_json(payload)


def sample_dict_from_row(
    *,
    t: float,
    q: Any,
    dq: Any,
    tau_applied: Any,
    tau_base: Any,
    tau_adaptor_delta: Any,
    tau_commanded: Any,
    tau_measured: Any,
    gravity: Any,
    history_embedding_seq: int,
    adaptor_active: bool,
    valid_for_history: bool,
    synthetic_padding: bool,
    sample_dt_sec: float,
    O_T_EE: Any = None,
    coriolis: Any = None,
) -> Dict[str, Any]:
    """One history sample in the TAM wire format.

    Fields the RCS hook does not produce (external wrench, soft block, test
    disturbance, coriolis) are filled with neutral values so every downstream
    parser sees a complete row; ``tau_cmd`` stays the compatibility alias of the
    final applied torque and ``tau_tam_residual`` of the adaptor residual.
    """
    tau_applied_l = np.asarray(tau_applied, dtype=float).tolist()
    tau_delta_l = np.asarray(tau_adaptor_delta, dtype=float).tolist()
    return {
        "t": float(t),
        "q": np.asarray(q, dtype=float).tolist(),
        "dq": np.asarray(dq, dtype=float).tolist(),
        "O_T_EE": IDENTITY4 if O_T_EE is None else np.asarray(O_T_EE, dtype=float).tolist(),
        "ft_O": [0.0] * 6,
        "external_force_soft_block_active": False,
        "external_force_soft_block_force": [0.0, 0.0, 0.0],
        "external_force_soft_block_tau": list(ZERO7),
        "tau_cmd": tau_applied_l,
        "tau_applied": tau_applied_l,
        "tau_base": np.asarray(tau_base, dtype=float).tolist(),
        "tau_commanded": np.asarray(tau_commanded, dtype=float).tolist(),
        "tau_measured": np.asarray(tau_measured, dtype=float).tolist(),
        "gravity": np.asarray(gravity, dtype=float).tolist(),
        "coriolis": list(ZERO7) if coriolis is None else np.asarray(coriolis, dtype=float).tolist(),
        "tau_adaptor_delta": tau_delta_l,
        "tau_tam_residual": tau_delta_l,
        "history_embedding_seq": int(history_embedding_seq),
        "disturbance_event_id": 0,
        "disturbance_active": False,
        "disturbance_phase": 0,
        "disturbance_tick": 0,
        "tau_test_disturbance": list(ZERO7),
        "tau_pre_disturbance": tau_applied_l,
        "tau_pre_clip": tau_applied_l,
        "tau_clip_delta": list(ZERO7),
        "tau_safety_correction": list(ZERO7),
        "adaptor_active": bool(adaptor_active),
        "valid_for_history": bool(valid_for_history),
        "synthetic_padding": bool(synthetic_padding),
        "publish_ready": True,
        "sample_dt_sec": float(sample_dt_sec),
    }


__all__ = [
    "ASYNC_MERGEABLE_KEYS",
    "ASYNC_METADATA_KEYS",
    "CONTROL_MODE_CARTESIAN",
    "CONTROL_MODE_JOINT",
    "REMOTE_ACTUATION_KEYS",
    "TEST_DISTURBANCE_COMMANDS",
    "AsyncSourceState",
    "collapse_async_commands",
    "drain_async_commands",
    "drain_pending_async_messages",
    "extract_async_metadata",
    "is_mergeable_async_payload",
    "normalize_command_message",
    "normalize_control_mode_name",
    "payload_has_remote_actuation",
    "publish_reset",
    "sample_dict_from_row",
    "strip_async_metadata",
    "strip_remote_actuation_payload",
]
