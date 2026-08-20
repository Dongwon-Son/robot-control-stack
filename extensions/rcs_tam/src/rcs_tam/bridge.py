"""Generic NUC-side TAM bridge: ZMQ PUB/PULL/REP loop over a :class:`BridgeBackend`.

Wire protocol (see ``rcs_tam.protocol``):

* PUB  ``history_bind``  -> ``{"type": "history", "window": [sample, ...]}`` every
  ``publish_dt`` seconds and ``{"type": "reset", "reason": ...}`` on resets;
* PULL ``command_bind``  <- async command dicts (``target_q``, ``embedding``,
  ``enable_adaptor``, ``simadaptor_bin_blob``, ``reset`` ...);
* REP  ``request_bind``  <- reliable ``{"cmd": ...}`` requests, answered with
  ``{"ok": bool, ...status fields...}``.

Commands a backend cannot honour are answered with ``ok=False`` and
``unsupported=True`` (the TAM mapping server already degrades gracefully on
rejected sysid commands).
"""

from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import zmq

from rcs_tam.backend import BridgeBackend, UnsupportedCommand
from rcs_tam.protocol import (
    CONTROL_MODE_JOINT,
    REMOTE_ACTUATION_KEYS,
    AsyncSourceState,
    collapse_async_commands,
    drain_async_commands,
    drain_pending_async_messages,
    normalize_command_message,
    normalize_control_mode_name,
    payload_has_remote_actuation,
    publish_reset,
    strip_remote_actuation_payload,
)

ACTUATION_KEYS = set(REMOTE_ACTUATION_KEYS) - {"embedding"}
SYSID_KEYS = {"enable_sysid", "clear_sysid", "set_sysid"}
TEST_DISTURBANCE_KEYS = {"arm_test_disturbance", "cancel_test_disturbance", "test_disturbance_status"}


@dataclass
class BridgeEndpoints:
    history_bind: str = "tcp://192.168.1.101:5555"
    command_bind: str = "tcp://192.168.1.101:5556"
    request_bind: str = "tcp://192.168.1.101:5557"

    @classmethod
    def from_host(cls, host: str, *, history_port: int = 5555, command_port: int = 5556, request_port: int = 5557) -> "BridgeEndpoints":
        return cls(
            history_bind=f"tcp://{host}:{history_port}",
            command_bind=f"tcp://{host}:{command_port}",
            request_bind=f"tcp://{host}:{request_port}",
        )

    @classmethod
    def from_network_config(cls, network: Any) -> "BridgeEndpoints":
        """Build from a config object exposing ``history_bind``/``command_bind``/``request_bind``."""
        return cls(
            history_bind=str(network.history_bind),
            command_bind=str(network.command_bind),
            request_bind=str(network.request_bind),
        )


class TamBridge:
    def __init__(
        self,
        backend: BridgeBackend,
        endpoints: BridgeEndpoints,
        *,
        history_len: int = 50,
        publish_dt: float = 0.01,
        loop_dt: float = 0.005,
        remote_command_stale_timeout_s: float = 2.0,
        startup_remote_actuation_ignore_s: float = 1.0,
        require_remote_actuation_for_adaptor_enable: bool = False,
        adaptor_enable_remote_fresh_timeout_s: float = 0.5,
        status_print_interval_s: float = 2.0,
        context: Optional[zmq.Context] = None,
        log=print,
    ) -> None:
        self.backend = backend
        self.endpoints = endpoints
        self.history_len = int(history_len)
        self.publish_dt = float(publish_dt)
        self.loop_dt = float(loop_dt)
        self.remote_command_stale_timeout_s = float(remote_command_stale_timeout_s)
        self.startup_remote_actuation_ignore_s = float(startup_remote_actuation_ignore_s)
        self.require_remote_actuation_for_adaptor_enable = bool(require_remote_actuation_for_adaptor_enable)
        self.adaptor_enable_remote_fresh_timeout_s = float(adaptor_enable_remote_fresh_timeout_s)
        self.status_print_interval_s = float(status_print_interval_s)
        self._log = log
        self._ctx = context or zmq.Context.instance()
        self._own_ctx = context is None
        self._pub: Optional[zmq.Socket] = None
        self._pull: Optional[zmq.Socket] = None
        self._rep: Optional[zmq.Socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.control_mode = CONTROL_MODE_JOINT
        self.sysid_enabled = False
        self.external_torque_prediction_enabled = False
        self._async_source_states: Dict[str, AsyncSourceState] = {}
        self._remote_actuation_armed = False
        self._remote_actuation_last_seen = 0.0
        self._pending_adaptor_enable = False
        self._startup_ignore_until = 0.0
        self.publish_count = 0
        self.reset_count = 0
        self.last_error: Optional[str] = None
        self.stats: Dict[str, Any] = {"publishes": 0, "async_commands": 0, "reliable_requests": 0, "resets": 0}

    # ------------------------------------------------------------------ #
    # life cycle
    # ------------------------------------------------------------------ #
    def bind(self) -> None:
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(self.endpoints.history_bind)
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.bind(self.endpoints.command_bind)
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(self.endpoints.request_bind)
        self._log(
            f"[rcs_tam] backend={self.backend.name} history={self.endpoints.history_bind} "
            f"command={self.endpoints.command_bind} request={self.endpoints.request_bind}"
        )

    def start(self) -> None:
        """Run the loop in a daemon thread (bind happens in the caller thread)."""
        if self._thread is not None:
            return
        self.bind()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rcs_tam_bridge", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._close_sockets()

    def run_forever(self) -> None:
        self.bind()
        try:
            self._run()
        finally:
            self._close_sockets()

    def _close_sockets(self) -> None:
        for sock in (self._pub, self._pull, self._rep):
            if sock is not None:
                sock.close(linger=0)
        self._pub = self._pull = self._rep = None

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        assert self._pub is not None and self._pull is not None and self._rep is not None
        poller = zmq.Poller()
        poller.register(self._pull, zmq.POLLIN)
        poller.register(self._rep, zmq.POLLIN)
        self._startup_ignore_until = time.monotonic() + self.startup_remote_actuation_ignore_s
        t_prev_pub = time.time()
        t_prev_loop = time.time()
        t_prev_status = 0.0
        while not self._stop.is_set():
            now_mono = time.monotonic()
            try:
                reason = self.backend.safety_check()
            except Exception as exc:  # pragma: no cover - defensive
                reason = f"safety_check_error:{exc}"
            if reason:
                self._do_reset(reason)
                t_prev_loop = time.time()
                continue

            socks = dict(poller.poll(timeout=0))
            if self._pull in socks and socks[self._pull] == zmq.POLLIN:
                msgs = drain_async_commands(self._pull, self._async_source_states, self.remote_command_stale_timeout_s, log=self._log)
                reset_requested = False
                for msg in collapse_async_commands(msgs, log=self._log):
                    self.stats["async_commands"] += 1
                    try:
                        _, msg_reset = self._dispatch(msg, reliable=False)
                    except Exception as exc:
                        self.last_error = str(exc)
                        self._log(f"[rcs_tam] Failed to apply async command: {exc}")
                        continue
                    if msg_reset:
                        reset_requested = True
                        break
                if reset_requested:
                    self._do_reset("command")
                    t_prev_pub = t_prev_loop = time.time()
                    continue

            if self._rep in socks and socks[self._rep] == zmq.POLLIN:
                try:
                    req = self._rep.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    req = None
                if req is not None:
                    self.stats["reliable_requests"] += 1
                    resp: Dict[str, Any] = {"ok": True}
                    reset_requested = False
                    try:
                        extra, reset_requested = self._dispatch(req, reliable=True)
                        resp.update(extra)
                    except UnsupportedCommand as exc:
                        resp = {"ok": False, "unsupported": True, "error": str(exc)}
                        resp.update(self._status_fields())
                    except Exception as exc:
                        self.last_error = str(exc)
                        resp = {"ok": False, "error": str(exc)}
                    if reset_requested:
                        self._do_reset("command")
                        resp["reset"] = True
                        resp["control_mode"] = self.control_mode
                    try:
                        self._rep.send_json(resp)
                    except Exception as exc:  # pragma: no cover
                        self._log(f"[rcs_tam] Failed to answer request: {exc}")

            if self._remote_actuation_armed:
                idle = now_mono - self._remote_actuation_last_seen
                if idle > self.remote_command_stale_timeout_s:
                    self._log(f"[rcs_tam] Neutralizing stale remote command stream: idle_for={idle:.3f}s")
                    try:
                        if self.backend.adaptor_enabled():
                            self.backend.enable_adaptor(False)
                            self._log("[rcs_tam] adaptor enabled: False (stale remote command stream)")
                        self.backend.neutralize_remote_motion()
                    except Exception as exc:
                        self._log(f"[rcs_tam] neutralize failed: {exc}")
                    self._async_source_states.clear()
                    drain_pending_async_messages(self._pull)
                    self._remote_actuation_armed = False
                    self._remote_actuation_last_seen = 0.0
                    publish_reset(self._pub, "stale_remote_command")

            if time.time() - t_prev_pub >= self.publish_dt:
                self._publish_history()
                t_prev_pub = time.time()
                if t_prev_pub - t_prev_status >= self.status_print_interval_s and self.status_print_interval_s > 0:
                    self._print_status()
                    t_prev_status = t_prev_pub

            elapsed = time.time() - t_prev_loop
            if elapsed < self.loop_dt:
                time.sleep(self.loop_dt - elapsed)
            t_prev_loop = time.time()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _publish_history(self) -> None:
        assert self._pub is not None
        try:
            window = self.backend.get_history_samples(self.history_len)
        except Exception as exc:
            self.last_error = str(exc)
            self._log(f"[rcs_tam] get_history_samples failed: {exc}")
            return
        self._pub.send_json({"type": "history", "window": window})
        self.publish_count += 1
        self.stats["publishes"] += 1

    def _print_status(self) -> None:
        try:
            st = self.backend.status()
        except Exception:
            st = {}
        self._log(
            f"[rcs_tam] backend={self.backend.name} adaptor={'on' if self._adaptor_enabled_safe() else 'off'} "
            f"embedding_seq={self._embedding_seq_safe()} publishes={self.stats['publishes']} "
            f"async={self.stats['async_commands']} reliable={self.stats['reliable_requests']} "
            + " ".join(f"{k}={v}" for k, v in st.items() if k in ("last_skip_reason", "adaptor_forward_dt_ms", "history_size"))
        )

    def _adaptor_enabled_safe(self) -> bool:
        try:
            return bool(self.backend.adaptor_enabled())
        except Exception:
            return False

    def _embedding_seq_safe(self) -> int:
        try:
            return int(self.backend.get_embedding_seq())
        except Exception:
            return -1

    def _status_fields(self) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "backend": self.backend.name,
            "adaptor_enabled": self._adaptor_enabled_safe(),
            "sysid_enabled": bool(self.sysid_enabled),
            "external_torque_prediction_enabled": bool(self.external_torque_prediction_enabled),
            "control_mode": self.control_mode,
            "history_embedding_seq": self._embedding_seq_safe(),
        }
        try:
            fields["ideal_model_has_gravity"] = bool(self.backend.ideal_model_has_gravity())
        except Exception:
            pass
        try:
            fields.update(self.backend.status())
        except Exception:
            pass
        return fields

    def _do_reset(self, reason: str) -> None:
        assert self._pub is not None and self._pull is not None
        self._log(f"[rcs_tam] Resetting controller: {reason}")
        try:
            self.backend.reset()
        except Exception as exc:
            self.last_error = str(exc)
            self._log(f"[rcs_tam] backend reset failed: {exc}")
        self.control_mode = CONTROL_MODE_JOINT
        self._async_source_states.clear()
        self._remote_actuation_armed = False
        self._remote_actuation_last_seen = 0.0
        self._pending_adaptor_enable = False
        drained = drain_pending_async_messages(self._pull)
        if drained:
            self._log(f"[rcs_tam] Discarded {drained} queued async command(s) after reset.")
        publish_reset(self._pub, reason)
        self.reset_count += 1
        self.stats["resets"] += 1

    def _load_blob(self, blob_b64: Any, name: Any) -> str:
        if not isinstance(blob_b64, str) or not blob_b64:
            raise ValueError("simadaptor_bin_blob must be a non-empty base64 string.")
        raw = base64.b64decode(blob_b64)
        safe_name = os.path.basename(str(name or "simadaptor.bin")) or "simadaptor.bin"
        target_dir = os.path.join(tempfile.gettempdir(), "rcs_tam_adaptors")
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, safe_name)
        with open(path, "wb") as f:
            f.write(raw)
        if not self.backend.load_adaptor_path(path):
            raise RuntimeError(f"controller rejected adaptor bin {path}")
        return path

    def _dispatch(self, msg: Mapping[str, Any], *, reliable: bool) -> tuple[Dict[str, Any], bool]:
        """Apply one command message; returns (reply fields, reset_requested)."""
        payload = normalize_command_message(dict(msg), allow_test_disturbance=reliable)
        resp: Dict[str, Any] = {}
        now_mono = time.monotonic()

        if TEST_DISTURBANCE_KEYS.intersection(payload):
            raise UnsupportedCommand("test disturbances are not supported by this backend")

        if now_mono < self._startup_ignore_until:
            payload, ignored = strip_remote_actuation_payload(payload)
            if ignored:
                self._log("[rcs_tam] Ignoring startup remote actuation command(s): " + ", ".join(ignored))
                resp["startup_motion_ignored"] = True
                resp["ignored_remote_actuation_keys"] = ignored

        remote_actuation_seen = payload_has_remote_actuation(payload)
        actuation = {k: v for k, v in payload.items() if k in ACTUATION_KEYS}
        if "control_mode" in actuation:
            self.control_mode = normalize_control_mode_name(actuation["control_mode"])
        if actuation:
            resp.update(self.backend.apply_actuation(actuation) or {})
        if remote_actuation_seen:
            self._remote_actuation_armed = True
            self._remote_actuation_last_seen = now_mono

        if "embedding" in payload:
            embedding = np.asarray(payload["embedding"], dtype=float).reshape(-1)
            self.backend.set_embedding(embedding)
            resp["embedding_size"] = int(embedding.size)

        if "ideal_model_has_gravity" in payload:
            enabled = bool(payload["ideal_model_has_gravity"])
            self.backend.set_ideal_model_has_gravity(enabled)
            resp["ideal_model_has_gravity"] = enabled

        if "load_simadaptor_bin" in payload:
            path = str(payload["load_simadaptor_bin"])
            if not self.backend.load_adaptor_path(path):
                raise RuntimeError(f"controller rejected adaptor bin {path}")
            self.backend.enable_adaptor(bool(payload.get("simadaptor_enable_after_load", False)))
            resp["path"] = path

        if "simadaptor_bin_blob" in payload:
            path = self._load_blob(payload.get("simadaptor_bin_blob"), payload.get("simadaptor_bin_name"))
            self.backend.enable_adaptor(bool(payload.get("simadaptor_enable_after_load", False)))
            resp["path"] = path

        if "enable_adaptor" in payload:
            requested = bool(payload["enable_adaptor"])
            fresh = (
                self._remote_actuation_armed
                and (now_mono - self._remote_actuation_last_seen) <= self.adaptor_enable_remote_fresh_timeout_s
            )
            if requested and self.require_remote_actuation_for_adaptor_enable and not (fresh or remote_actuation_seen):
                self._pending_adaptor_enable = True
                resp["adaptor_enable_pending"] = True
                resp["adaptor_enable_blocked_reason"] = "remote_actuation_not_fresh"
            else:
                self.backend.enable_adaptor(requested)
                self._pending_adaptor_enable = False
                resp["adaptor_enable_pending"] = False
                self._log(f"[rcs_tam] adaptor enabled: {requested}")

        if "enable_sysid" in payload:
            self.backend.enable_sysid(bool(payload["enable_sysid"]))
            self.sysid_enabled = bool(payload["enable_sysid"])
        if payload.get("clear_sysid"):
            self.backend.clear_sysid()
        if "set_sysid" in payload:
            params = payload["set_sysid"]
            if not isinstance(params, Mapping):
                raise ValueError("set_sysid expects a dict.")
            self.backend.set_sysid(params)
            resp["sysid_keys"] = sorted(params.keys())
        if "enable_external_torque_prediction" in payload:
            enabled = bool(payload["enable_external_torque_prediction"])
            self.backend.enable_external_torque_prediction(enabled)
            self.external_torque_prediction_enabled = enabled
        if "enable_external_force_soft_block" in payload or "external_force_soft_block" in payload:
            raise UnsupportedCommand("external force soft block is not supported by this backend")

        reset_requested = bool(payload.get("reset", False))
        resp.update(self._status_fields())
        return resp, reset_requested


__all__ = ["BridgeEndpoints", "TamBridge"]
