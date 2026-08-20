from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rcs_tam.config import load_runtime_config
from rcs_tam.protocol import collapse_async_commands, normalize_command_message, normalize_control_mode_name


class ProtocolTest(unittest.TestCase):
    def test_normalize_reliable_commands(self) -> None:
        self.assertEqual(normalize_command_message({"cmd": "set_embedding", "embedding": [1, 2]}), {"embedding": [1, 2]})
        self.assertEqual(normalize_command_message({"cmd": "enable_adaptor", "enabled": True}), {"enable_adaptor": True})
        self.assertEqual(
            normalize_command_message({"cmd": "load_bin_blob", "name": "a.bin", "blob": "QUJD", "enable_after_load": True}),
            {"simadaptor_bin_name": "a.bin", "simadaptor_bin_blob": "QUJD", "simadaptor_enable_after_load": True},
        )
        self.assertEqual(normalize_command_message({"cmd": "reset"}), {"reset": True})
        self.assertEqual(normalize_command_message({"target_q": [0] * 7}), {"target_q": [0] * 7})
        with self.assertRaises(ValueError):
            normalize_command_message({"cmd": "nope"})
        with self.assertRaises(ValueError):
            normalize_command_message({"cmd": "arm_test_disturbance"})  # reliable-only
        self.assertIn("arm_test_disturbance", normalize_command_message({"cmd": "arm_test_disturbance"}, allow_test_disturbance=True))

    def test_collapse_merges_state_payloads_and_stops_at_reset(self) -> None:
        msgs = [
            {"target_q": [1] * 7},
            {"embedding": [0.5]},
            {"target_q": [2] * 7},
            {"reset": True},
            {"target_q": [3] * 7},
        ]
        out = collapse_async_commands(msgs, log=lambda *_: None)
        self.assertEqual(out[0], {"target_q": [2] * 7, "embedding": [0.5]})
        self.assertEqual(out[1], {"reset": True})
        self.assertEqual(len(out), 2)

    def test_control_mode_names(self) -> None:
        self.assertEqual(normalize_control_mode_name(" Cartesian "), "cartesian")
        with self.assertRaises(ValueError):
            normalize_control_mode_name("osc")


class ConfigTest(unittest.TestCase):
    def test_config_json_and_strict_schema(self) -> None:
        with TemporaryDirectory() as d:
            path = Path(d) / "rcs_tam_config.json"
            path.write_text(
                json.dumps(
                    {
                        "network": {"nuc_control_host": "10.0.0.5", "robot_host": "10.0.1.2", "history_port": 6555},
                        "timing": {"history_pub_dt": 0.02, "history_len": 40},
                        "safety": {"ext_force_threshold": 60.0},
                        "rcs": {"torque_limit": [50] * 7, "adaptor_torque_limits": [5] * 7},
                    }
                )
            )
            cfg = load_runtime_config(str(path))
            self.assertEqual(cfg.network.history_bind, "tcp://10.0.0.5:6555")
            self.assertEqual(cfg.network.command_bind, "tcp://10.0.0.5:5556")
            self.assertEqual(cfg.network.robot_host, "10.0.1.2")
            self.assertEqual(cfg.timing.history_len, 40)
            self.assertEqual(cfg.safety.ext_force_threshold, 60.0)
            self.assertEqual(cfg.rcs.torque_limit, [50] * 7)
            self.assertEqual(cfg.rcs.adaptor_torque_limits, [5] * 7)
            self.assertTrue(cfg.loaded_from.endswith("rcs_tam_config.json"))

            path.write_text(json.dumps({"network": {"nuc_control_host": "10.0.0.5"}, "fast_bridge": {"enabled": True}}))
            with self.assertRaisesRegex(ValueError, "unknown rcs_tam config block 'fast_bridge'"):
                load_runtime_config(str(path))
            path.write_text(json.dumps({"timing": {"history_pub_dt": 0.02, "loop_dtt": 1.0}}))
            with self.assertRaisesRegex(ValueError, "unknown rcs_tam config key timing.'loop_dtt'"):
                load_runtime_config(str(path))

    def test_example_config_parses(self) -> None:
        example = Path(__file__).resolve().parents[1] / "rcs_tam_config.example.json"
        cfg = load_runtime_config(str(example))
        self.assertEqual(cfg.network.history_bind, "tcp://192.168.1.101:5555")
        self.assertEqual(cfg.rcs.torque_limit, [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


if __name__ == "__main__":
    unittest.main()
