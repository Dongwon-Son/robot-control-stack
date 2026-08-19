from __future__ import annotations

import base64
import socket
import time
import unittest
from typing import Any, Dict, List, Mapping

try:
    import numpy as np
    import zmq
except ModuleNotFoundError as exc:  # pragma: no cover - lightweight source-only environments
    raise unittest.SkipTest(f"bridge runtime dependencies unavailable: {exc}") from exc

from rcs_tam.backend import BridgeBackend, UnsupportedCommand, history_rows_dict_to_samples
from rcs_tam.bridge import BridgeEndpoints, TamBridge


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class FakeBackend(BridgeBackend):
    name = "fake"

    def __init__(self) -> None:
        self.embedding = None
        self.embedding_seq = 0
        self.enabled = False
        self.loaded_paths: List[str] = []
        self.gravity_flag = True
        self.actuation: List[Dict[str, Any]] = []
        self.neutralized = 0
        self.resets = 0
        self.rows = 0

    def get_history_samples(self, max_rows: int):
        n = min(int(max_rows), 5)
        self.rows += 1
        rows = {
            "t": np.arange(n) * 1e-3 + self.rows,
            "q": np.ones((n, 7)) * 0.1,
            "dq": np.zeros((n, 7)),
            "tau_applied": np.ones((n, 7)),
            "tau_base": np.ones((n, 7)) * 0.9,
            "tau_adaptor_delta": np.ones((n, 7)) * 0.1,
            "tau_commanded": np.zeros((n, 7)),
            "tau_measured": np.zeros((n, 7)),
            "gravity": np.ones((n, 7)) * 2.0,
            "history_embedding_seq": np.full(n, self.embedding_seq, dtype=np.int64),
            "adaptor_active": np.full(n, self.enabled),
            "valid_for_history": np.ones(n, dtype=bool),
            "synthetic_padding": np.zeros(n, dtype=bool),
            "sample_dt_sec": np.full(n, 1e-3),
        }
        return history_rows_dict_to_samples(rows)

    def set_embedding(self, embedding):
        self.embedding = np.asarray(embedding, dtype=float)
        self.embedding_seq += 1

    def get_embedding_seq(self) -> int:
        return self.embedding_seq

    def enable_adaptor(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def adaptor_enabled(self) -> bool:
        return self.enabled

    def load_adaptor_path(self, path: str) -> bool:
        self.loaded_paths.append(path)
        return True

    def set_ideal_model_has_gravity(self, enabled: bool) -> None:
        self.gravity_flag = bool(enabled)

    def ideal_model_has_gravity(self) -> bool:
        return self.gravity_flag

    def apply_actuation(self, payload: Mapping[str, Any]):
        self.actuation.append(dict(payload))
        return {"applied": sorted(payload.keys())}

    def neutralize_remote_motion(self) -> None:
        self.neutralized += 1

    def reset(self) -> None:
        self.resets += 1

    def status(self):
        return {"fake_status": True}


class TamBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ctx = zmq.Context()
        cls.endpoints = BridgeEndpoints.from_host(
            "127.0.0.1", history_port=_free_port(), command_port=_free_port(), request_port=_free_port()
        )
        cls.backend = FakeBackend()
        cls.bridge = TamBridge(
            cls.backend,
            cls.endpoints,
            publish_dt=0.01,
            loop_dt=0.002,
            startup_remote_actuation_ignore_s=0.0,
            remote_command_stale_timeout_s=0.4,
            status_print_interval_s=0.0,
            context=cls.ctx,
            log=lambda *_: None,
        )
        cls.bridge.start()
        cls.sub = cls.ctx.socket(zmq.SUB)
        cls.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        cls.sub.connect(cls.endpoints.history_bind)
        cls.push = cls.ctx.socket(zmq.PUSH)
        cls.push.connect(cls.endpoints.command_bind)
        cls.req = cls.ctx.socket(zmq.REQ)
        cls.req.setsockopt(zmq.RCVTIMEO, 3000)
        cls.req.connect(cls.endpoints.request_bind)
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.bridge.stop()
        for s in (cls.sub, cls.push, cls.req):
            s.close(linger=0)
        cls.ctx.term()

    def _recv_pub(self, msg_type: str, timeout_s: float = 2.0) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                msg = self.sub.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.005)
                continue
            if msg.get("type") == msg_type:
                return msg
        raise AssertionError(f"no {msg_type!r} message received")

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.req.send_json(payload)
        return self.req.recv_json()

    def test_01_history_publish_shape(self) -> None:
        msg = self._recv_pub("history")
        self.assertIn("window", msg)
        self.assertGreater(len(msg["window"]), 0)
        sample = msg["window"][-1]
        for key in ("t", "q", "dq", "tau_cmd", "tau_applied", "tau_base", "tau_adaptor_delta", "tau_tam_residual",
                    "gravity", "history_embedding_seq", "valid_for_history", "publish_ready", "sample_dt_sec"):
            self.assertIn(key, sample)
        self.assertEqual(sample["tau_cmd"], sample["tau_applied"])
        self.assertEqual(sample["tau_tam_residual"], sample["tau_adaptor_delta"])
        self.assertTrue(sample["publish_ready"])

    def test_02_reliable_embedding_and_enable(self) -> None:
        resp = self._request({"cmd": "set_embedding", "embedding": [0.5] * 4})
        self.assertTrue(resp["ok"], resp)
        self.assertEqual(resp["embedding_size"], 4)
        self.assertEqual(self.backend.embedding_seq, 1)
        self.assertEqual(resp["history_embedding_seq"], 1)
        self.assertEqual(resp["backend"], "fake")
        resp = self._request({"cmd": "enable_adaptor", "enabled": True})
        self.assertTrue(resp["ok"], resp)
        self.assertTrue(self.backend.enabled)
        self.assertTrue(resp["adaptor_enabled"])
        self.assertFalse(resp["adaptor_enable_pending"])
        resp = self._request({"cmd": "set_ideal_model_has_gravity", "enabled": False})
        self.assertTrue(resp["ok"], resp)
        self.assertFalse(self.backend.gravity_flag)
        self.assertFalse(resp["ideal_model_has_gravity"])

    def test_03_load_bin_blob(self) -> None:
        blob = base64.b64encode(b"\x01\x02\x03fake").decode("ascii")
        resp = self._request({"cmd": "load_bin_blob", "name": "unit.bin", "blob": blob, "enable_after_load": True})
        self.assertTrue(resp["ok"], resp)
        self.assertTrue(self.backend.loaded_paths)
        with open(self.backend.loaded_paths[-1], "rb") as f:
            self.assertEqual(f.read(), b"\x01\x02\x03fake")
        self.assertTrue(self.backend.enabled)

    def test_04_unsupported_commands(self) -> None:
        resp = self._request({"cmd": "enable_sysid", "enabled": True})
        self.assertFalse(resp["ok"])
        self.assertTrue(resp.get("unsupported"))
        resp = self._request({"cmd": "arm_test_disturbance", "event_id": 1, "tau_nm": [0] * 7, "delay_s": 0, "duration_s": 0.1})
        self.assertFalse(resp["ok"])
        self.assertTrue(resp.get("unsupported"))
        # Disabling/clearing absent features is a no-op success (mapping-server prepare relies on it).
        resp = self._request({"cmd": "clear_sysid"})
        self.assertTrue(resp["ok"], resp)
        resp = self._request({"cmd": "enable_sysid", "enabled": False})
        self.assertTrue(resp["ok"], resp)
        resp = self._request({"cmd": "enable_external_torque_prediction", "enabled": False})
        self.assertTrue(resp["ok"], resp)
        resp = self._request({"cmd": "enable_external_torque_prediction", "enabled": True})
        self.assertFalse(resp["ok"])
        resp = self._request({"cmd": "set_sysid", "params": {"torque_scale": [1.0] * 14}})
        self.assertFalse(resp["ok"])
        # Status query is an empty payload.
        resp = self._request({})
        self.assertTrue(resp["ok"], resp)
        self.assertEqual(resp["backend"], "fake")
        self.assertIn("control_mode", resp)
        self.assertTrue(resp["fake_status"])

    def test_05_async_actuation_and_stale_neutralize(self) -> None:
        before = len(self.backend.actuation)
        self.push.send_json({"target_q": [0.0] * 7, "target_dq": [0.0] * 7, "command_source": "unit", "command_id": 1})
        deadline = time.time() + 2.0
        while time.time() < deadline and len(self.backend.actuation) == before:
            time.sleep(0.01)
        self.assertGreater(len(self.backend.actuation), before)
        self.assertIn("target_q", self.backend.actuation[-1])
        # No new remote actuation -> stale timeout neutralizes and publishes a reset.
        msg = self._recv_pub("reset", timeout_s=3.0)
        self.assertEqual(msg["reason"], "stale_remote_command")
        self.assertGreaterEqual(self.backend.neutralized, 1)
        self.assertFalse(self.backend.enabled)  # adaptor disabled on stale stream

    def test_06_reset_command(self) -> None:
        resets_before = self.backend.resets
        resp = self._request({"cmd": "reset"})
        self.assertTrue(resp["ok"], resp)
        self.assertTrue(resp.get("reset"))
        self.assertEqual(self.backend.resets, resets_before + 1)
        msg = self._recv_pub("reset", timeout_s=2.0)
        self.assertEqual(msg["reason"], "command")


class HistoryConversionTest(unittest.TestCase):
    def test_rows_dict_to_samples(self) -> None:
        rows = {
            "t": np.asarray([0.001, 0.002]),
            "q": np.arange(14).reshape(2, 7),
            "dq": np.zeros((2, 7)),
            "tau_applied": np.ones((2, 7)),
        }
        samples = history_rows_dict_to_samples(rows)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[1]["q"][6], 13.0)
        self.assertEqual(samples[0]["tau_base"], samples[0]["tau_applied"])
        self.assertEqual(samples[0]["history_embedding_seq"], 0)
        self.assertEqual(len(samples[0]["O_T_EE"]), 4)
        self.assertEqual(history_rows_dict_to_samples({"t": np.zeros(0)}), [])


if __name__ == "__main__":
    unittest.main()
