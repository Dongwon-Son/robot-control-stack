from __future__ import annotations

import sys
import time
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rcs_tam.runtime import (  # noqa: E402
    HISTORY_TORQUE_MODE_APPLIED,
    HISTORY_TORQUE_MODE_BASE_TAM_FUSION,
    TamDeployment,
    resolve_history_torque_mode,
)


def _fake_inf(*, dagger_mode=None, cfg_mode=None, fusion=False):
    inf = types.SimpleNamespace()
    inf.dagger_cfg = types.SimpleNamespace(history_torque_mode=dagger_mode) if dagger_mode else None
    inf.cfg = types.SimpleNamespace(history_torque_mode=cfg_mode)
    inf._simadaptor_params = {"adaptor": {}, "hist": {}}
    if fusion:
        inf._simadaptor_params["history_fusion"] = {"kernel": np.zeros((3, 1)), "bias": np.zeros(1)}
    return inf


class ResolverTest(unittest.TestCase):
    def test_defaults_to_applied(self) -> None:
        self.assertEqual(resolve_history_torque_mode(_fake_inf()), HISTORY_TORQUE_MODE_APPLIED)

    def test_fusion_params_imply_fused(self) -> None:
        self.assertEqual(
            resolve_history_torque_mode(_fake_inf(fusion=True)), HISTORY_TORQUE_MODE_BASE_TAM_FUSION
        )

    def test_dagger_cfg_wins(self) -> None:
        inf = _fake_inf(dagger_mode="base_tam_fusion", fusion=True)
        self.assertEqual(resolve_history_torque_mode(inf), HISTORY_TORQUE_MODE_BASE_TAM_FUSION)
        inf = _fake_inf(dagger_mode="applied", fusion=True)
        self.assertEqual(resolve_history_torque_mode(inf), HISTORY_TORQUE_MODE_APPLIED)

    def test_fused_request_without_weights_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_history_torque_mode(_fake_inf(), "base_tam_fusion")
        with self.assertRaises(ValueError):
            resolve_history_torque_mode(_fake_inf(), "nope")


class _FakeRuntime:
    """Emits an embedding every ``every`` pushes; records inputs."""

    def __init__(self, every: int = 1) -> None:
        self.every = int(every)
        self.pushes = 0
        self.resets = 0
        self.last_kwargs = None

    def push_window(self, t, q, dq, tau, **kwargs):
        self.pushes += 1
        self.last_kwargs = dict(kwargs, n=len(np.asarray(t).reshape(-1)))
        if self.pushes % self.every == 0:
            return np.full((1, 4), float(self.pushes), dtype=np.float32)
        return None

    def reset(self) -> None:
        self.resets += 1


class _FakeRobot:
    def __init__(self) -> None:
        self.embeddings: list[np.ndarray] = []
        self.enabled = False
        self.t0 = 0.0
        self.rows_t = np.arange(50) * 1e-3

    def tam_get_history(self, max_rows: int):
        n = min(int(max_rows), self.rows_t.size)
        t = self.rows_t[-n:]
        z7 = np.zeros((n, 7))
        return {
            "t": t,
            "q": z7 + 0.1,
            "dq": z7,
            "tau_applied": z7 + 1.0,
            "tau_base": z7 + 0.9,
            "tau_adaptor_delta": z7 + 0.1,
            "gravity": z7 + 2.0,
            "valid_for_history": np.ones(n, dtype=bool),
        }

    def advance(self, dt: float = 0.05) -> None:
        self.rows_t = self.rows_t + dt

    def restart(self) -> None:
        self.rows_t = np.arange(50) * 1e-3

    def tam_set_embedding(self, z) -> None:
        self.embeddings.append(np.asarray(z, dtype=float).copy())

    def tam_enable(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def tam_set_torque_limits(self, limits) -> None:
        self.limits = np.asarray(limits, dtype=float)

    def tam_set_enable_ramp_s(self, seconds: float) -> None:
        self.ramp = float(seconds)

    def tam_status(self):
        return {"loaded": True, "enabled": self.enabled, "last_skip_reason": ""}


def _make_deployment(robot, mode, applied, base=None, tam=None, fusion=None, **kw):
    inf = _fake_inf(fusion=mode == HISTORY_TORQUE_MODE_BASE_TAM_FUSION)
    return TamDeployment(
        robot,
        "unused-ckpt",
        runtimes=(inf, mode, applied, base, tam, fusion),
        log=lambda *_: None,
        **kw,
    )


class DeploymentTest(unittest.TestCase):
    def test_applied_mode_streams_and_enables(self) -> None:
        robot = _FakeRobot()
        applied = _FakeRuntime(every=1)
        dep = _make_deployment(
            robot,
            HISTORY_TORQUE_MODE_APPLIED,
            applied,
            embedding_interval_s=0.0,
            min_patches_before_send=2,
            residual_torque_limits=np.full(7, 2.0),
        )
        self.assertFalse(robot.enabled)
        dep._step()  # patch 1: below min_patches, nothing sent
        self.assertEqual(len(robot.embeddings), 0)
        robot.advance()
        dep._step()  # patch 2: sent + enabled
        self.assertEqual(len(robot.embeddings), 1)
        self.assertTrue(robot.enabled)
        self.assertEqual(robot.embeddings[0].shape, (4,))  # flattened
        self.assertEqual(applied.last_kwargs["n"], 50)
        np.testing.assert_allclose(robot.limits, np.full(7, 2.0))
        self.assertEqual(dep.status()["mode"], "applied")

    def test_embedding_interval_gates_sends(self) -> None:
        robot = _FakeRobot()
        dep = _make_deployment(
            robot,
            HISTORY_TORQUE_MODE_APPLIED,
            _FakeRuntime(every=1),
            embedding_interval_s=10.0,
            min_patches_before_send=0,
        )
        dep._step()
        robot.advance()
        dep._step()
        self.assertEqual(len(robot.embeddings), 1)  # second send gated by interval

    def test_fused_mode_requires_all_streams_and_fuses(self) -> None:
        robot = _FakeRobot()
        applied, base, tam = _FakeRuntime(1), _FakeRuntime(1), _FakeRuntime(2)
        fusion = {"kernel": np.eye(12, 4, dtype=np.float32), "bias": np.zeros(4, dtype=np.float32)}
        dep = _make_deployment(
            robot,
            HISTORY_TORQUE_MODE_BASE_TAM_FUSION,
            applied,
            base,
            tam,
            fusion,
            embedding_interval_s=0.0,
            min_patches_before_send=0,
        )
        dep._step()  # tam runtime emits nothing yet -> no embedding
        self.assertEqual(len(robot.embeddings), 0)
        robot.advance()
        dep._step()  # all three emit -> fused embedding sent
        self.assertEqual(len(robot.embeddings), 1)
        self.assertTrue(robot.enabled)
        self.assertTrue(base.last_kwargs["n"] == tam.last_kwargs["n"] == 50)
        self.assertTrue(tam.last_kwargs.get("tau_is_model_space"))

    def test_controller_restart_resets_runtimes_and_disables(self) -> None:
        robot = _FakeRobot()
        applied = _FakeRuntime(every=2)  # like the real encoder, not every poll yields a token
        dep = _make_deployment(
            robot,
            HISTORY_TORQUE_MODE_APPLIED,
            applied,
            embedding_interval_s=0.0,
            min_patches_before_send=0,
        )
        robot.advance(2.0)
        dep._step()  # push 1: no embedding yet
        dep._step()  # push 2: embedding -> enabled
        self.assertTrue(robot.enabled)
        robot.restart()  # controller thread restart: timestamps jump back
        dep._step()  # push 3: reset detected, no fresh embedding yet
        self.assertEqual(applied.resets, 1)
        self.assertFalse(robot.enabled)  # disabled until a fresh embedding arrives
        self.assertEqual(dep.status()["controller_restarts"], 1)
        robot.advance(5.0)
        dep._step()  # push 4: fresh embedding -> re-enabled
        self.assertTrue(robot.enabled)

    def test_thread_start_stop(self) -> None:
        robot = _FakeRobot()
        dep = _make_deployment(
            robot,
            HISTORY_TORQUE_MODE_APPLIED,
            _FakeRuntime(every=1),
            embedding_interval_s=0.0,
            min_patches_before_send=0,
            poll_period_s=0.005,
        )
        dep.start()
        deadline = time.time() + 2.0
        while time.time() < deadline and not robot.embeddings:
            robot.advance(0.01)
            time.sleep(0.01)
        dep.stop()
        self.assertTrue(robot.embeddings)
        self.assertFalse(robot.enabled)  # stop() disables


if __name__ == "__main__":
    unittest.main()
