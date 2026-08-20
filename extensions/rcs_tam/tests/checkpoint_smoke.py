#!/usr/bin/env python3
"""Robot-free end-to-end smoke for a real TAM checkpoint (no pytest, run manually).

Exercises the exact production path with the real C++ ``hw.TamHook`` but a
fake robot: the checkpoint is loaded, its adaptor exported and loaded into the
hook, synthetic 1 kHz joint/torque rows are fed through
``TamHook.apply``/``finalize_row``, and :class:`rcs_tam.TamDeployment` streams
the recorded history through the JAX encoder back into the hook. Passes when
embeddings flow, the hook runs the adaptor (empty skip reason), and the
residual is finite and non-zero.

Usage (rcs_panda or rcs_fr3 built from this branch)::

    python tests/checkpoint_smoke.py --ckpt <tam checkpoint dir>   # --xml overrides the packaged ideal-model MJCF
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class HookRobot:
    """Duck-typed robot exposing the ``tam_*`` API on a bare ``hw.TamHook``."""

    def __init__(self, hook) -> None:
        self.hook = hook

    def tam_load_adaptor(self, path: str) -> bool:
        return bool(self.hook.load_adaptor(str(path)))

    def tam_set_embedding(self, z) -> None:
        self.hook.set_embedding(np.asarray(z, dtype=float))

    def tam_enable(self, enabled: bool) -> None:
        self.hook.enable(bool(enabled))

    def tam_set_torque_limits(self, limits) -> None:
        self.hook.set_torque_limits(np.asarray(limits, dtype=float))

    def tam_set_enable_ramp_s(self, seconds: float) -> None:
        self.hook.set_enable_ramp_s(float(seconds))

    def tam_get_history(self, max_rows: int):
        return self.hook.get_history(int(max_rows))

    def tam_status(self):
        return dict(self.hook.status())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--xml", default=None)
    parser.add_argument("--history-torque-mode", default="auto")
    parser.add_argument("--warmup-s", type=float, default=6.0, help="History fed before waiting for the first embedding.")
    parser.add_argument("--drain-timeout-s", type=float, default=600.0, help="Max wait for the first embedding (CPU encoders are slow).")
    parser.add_argument("--active-s", type=float, default=5.0, help="History fed with the adaptor enabled (residual statistics).")
    parser.add_argument("--extension", choices=("panda", "fr3"), default="panda")
    args = parser.parse_args(argv)

    if args.extension == "panda":
        from rcs_panda._core import hw
    else:
        from rcs_fr3._core import hw

    from rcs_tam import TamDeployment

    hook = hw.TamHook(8192)
    hook.on_control_start()
    robot = HookRobot(hook)
    tam = TamDeployment(
        robot,
        args.ckpt,
        xml_path=args.xml,
        history_torque_mode=args.history_torque_mode,
        min_patches_before_send=1,
        embedding_interval_s=0.2,
        residual_torque_limits=np.asarray([10.0, 10.0, 10.0, 10.0, 8.0, 8.0, 8.0]),
        enable_ramp_s=0.2,
    )
    tam.start()

    home = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    amp = np.deg2rad(12.0) * np.asarray([1.0, 0.6, 0.8, 0.6, 1.0, 0.8, 1.0])
    dt = 1e-3
    delta_max = 0.0
    delta_active_ticks = 0
    tick = 0

    def feed(duration_s: float, count_active: bool) -> bool:
        nonlocal delta_max, delta_active_ticks, tick
        t_start = time.perf_counter()
        for i in range(int(duration_s / dt)):
            t = tick * dt
            tick += 1
            q = home + amp * np.sin(2 * np.pi * t / 4.0)
            dq = amp * (2 * np.pi / 4.0) * np.cos(2 * np.pi * t / 4.0)
            gravity = 3.0 * np.sin(q) + 1.0
            tau_base = 30.0 * amp * np.sin(2 * np.pi * t / 4.0 + 0.3)  # gravity-free command
            delta = np.asarray(hook.apply(dt, q, dq, tau_base, gravity, np.zeros(7), tau_base + gravity))
            hook.finalize_row(tau_base + delta)
            if not np.all(np.isfinite(delta)):
                print("FAIL: non-finite residual", delta)
                return False
            if count_active and np.any(delta != 0.0):
                delta_active_ticks += 1
                delta_max = max(delta_max, float(np.max(np.abs(delta))))
            lag = t_start + i * dt - time.perf_counter()
            if lag > 0:
                time.sleep(min(lag, 0.001))
        return True

    # Phase 1: build up history, then wait for the encoder to deliver the
    # first embedding (fast on GPU, up to minutes for the fused mode on CPU).
    if not feed(args.warmup_s, count_active=False):
        return 1
    deadline = time.perf_counter() + float(args.drain_timeout_s)
    while time.perf_counter() < deadline and tam.stats["embeddings_sent"] < 1 and tam.stats["last_error"] is None:
        time.sleep(0.5)
    if tam.stats["embeddings_sent"] < 1:
        tam.stop()
        print(f"[smoke] FAIL: no embedding within {args.drain_timeout_s:.0f}s "
              f"(last_error={tam.stats['last_error']!r})")
        return 1
    # Phase 2: adaptor enabled — measure the residual on fresh history.
    if not feed(args.active_s, count_active=True):
        return 1
    tam.stop()
    st = tam.status()
    n_active = int(args.active_s / dt)
    print(
        f"[smoke] mode={st['mode']} embeddings_sent={st['embeddings_sent']} restarts={st['controller_restarts']} "
        f"skip={st.get('hook_last_skip_reason')!r} forward_ms={st.get('hook_adaptor_forward_dt_ms'):.3f} "
        f"delta_active_ticks={delta_active_ticks}/{n_active} delta_max={delta_max:.3f} Nm"
    )
    ok = (
        st["embeddings_sent"] >= 1
        and st.get("hook_last_skip_reason") == ""
        and delta_active_ticks > int(0.5 * n_active)
        and 0.0 < delta_max <= 10.0
        and st["last_error"] is None
    )
    print("[smoke] PASS" if ok else "[smoke] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import os
    import sys as _sys

    rc = main()
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(rc)  # skip interpreter teardown (JAX/mjx atexit can abort)
