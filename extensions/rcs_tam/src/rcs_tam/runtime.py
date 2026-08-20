"""Single-process TAM deployment for robot-control-stack.

Everything runs on one machine, in one Python process:

* ``hw.Franka`` (``rcs_panda`` / ``rcs_fr3``) executes the 1 kHz torque
  controllers; the built-in ``TamHook`` applies the TAM residual and records
  the history (``robot.tam_*``).
* :class:`TamDeployment` (this module) loads a TAM checkpoint, exports the
  adaptor to the controller ``.bin``, and runs a background thread that reads
  ``robot.tam_get_history()``, streams it through the JAX history encoder, and
  writes the latent embedding back via ``robot.tam_set_embedding()``.

No realtime kernel is required for the encoder side; the robot connection can
be opened with ``FrankaConfig.ignore_realtime=True`` so the whole process runs
on a stock kernel (the 1 kHz callback then runs without RT scheduling — keep
the machine lightly loaded).

Supported checkpoints (``history_torque_mode`` is resolved from the
checkpoint): plain applied-torque history (the Panda-specific and DAgger
finetuned checkpoints) and ``base_tam_fusion`` (the DAgger checkpoint with
fused applied/base/residual history inputs). The ideal model always includes
gravity; the vendored inference code enforces this and the controller-side
hook always feeds ``tau + gravity`` to the adaptor.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

HISTORY_TORQUE_MODE_AUTO = "auto"
HISTORY_TORQUE_MODE_APPLIED = "applied"
HISTORY_TORQUE_MODE_BASE_TAM_FUSION = "base_tam_fusion"
HISTORY_TORQUE_MODE_CHOICES = (
    HISTORY_TORQUE_MODE_AUTO,
    HISTORY_TORQUE_MODE_APPLIED,
    HISTORY_TORQUE_MODE_BASE_TAM_FUSION,
)


def _checkpoint_history_torque_mode(inf: Any) -> str:
    dagger_cfg = getattr(inf, "dagger_cfg", None)
    dagger_mode = getattr(dagger_cfg, "history_torque_mode", None)
    if dagger_mode:
        return str(dagger_mode).strip()
    cfg_mode = getattr(getattr(inf, "cfg", None), "history_torque_mode", None)
    if cfg_mode:
        return str(cfg_mode).strip()
    if _has_history_fusion(inf):
        return HISTORY_TORQUE_MODE_BASE_TAM_FUSION
    return HISTORY_TORQUE_MODE_APPLIED


def _has_history_fusion(inf: Any) -> bool:
    params = getattr(inf, "_simadaptor_params", {}) or {}
    try:
        return "history_fusion" in params
    except Exception:
        return False


def _history_fusion_params(inf: Any) -> Any:
    params = getattr(inf, "_simadaptor_params", {}) or {}
    try:
        return params["history_fusion"]
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint requested base_tam_fusion history, but history_fusion "
            "parameters are missing."
        ) from exc


def resolve_history_torque_mode(inf: Any, requested: str = HISTORY_TORQUE_MODE_AUTO) -> str:
    requested = str(requested or HISTORY_TORQUE_MODE_AUTO).strip()
    if requested not in HISTORY_TORQUE_MODE_CHOICES:
        raise ValueError(
            f"Unsupported history torque mode {requested!r}; choose one of {HISTORY_TORQUE_MODE_CHOICES}."
        )
    cfg_mode = _checkpoint_history_torque_mode(inf)
    mode = cfg_mode if requested == HISTORY_TORQUE_MODE_AUTO else requested
    if mode == HISTORY_TORQUE_MODE_BASE_TAM_FUSION:
        if not _has_history_fusion(inf):
            raise RuntimeError(
                "base_tam_fusion history requires checkpoint params['history_fusion']; "
                f"checkpoint history_torque_mode={cfg_mode!r} but no fusion weights were found."
            )
        return HISTORY_TORQUE_MODE_BASE_TAM_FUSION
    return HISTORY_TORQUE_MODE_APPLIED


class TamDeployment:
    """Load a TAM checkpoint and keep a ``hw.Franka``'s TamHook fed with embeddings."""

    def __init__(
        self,
        robot: Any,
        ckpt_path: str | Path,
        *,
        xml_path: str | Path | None = None,  # default: the packaged panda_pandagripper.xml
        history_torque_mode: str = HISTORY_TORQUE_MODE_AUTO,
        attention_history_s: float = 4.0,
        expected_dt: float = 1e-3,
        embedding_interval_s: float = 0.2,
        min_patches_before_send: int = 2,
        poll_period_s: float = 0.02,
        history_rows: int = 2000,
        enable_after_first_embedding: bool = True,
        residual_torque_limits: Optional[np.ndarray] = None,
        enable_ramp_s: Optional[float] = None,
        bin_path: str | Path | None = None,
        jax_cache_dir: str | Path | None = None,
        runtimes: Any = None,
        log=print,
    ) -> None:
        self.robot = robot
        self._log = log
        self.embedding_interval_s = float(embedding_interval_s)
        self.min_patches_before_send = int(min_patches_before_send)
        self.poll_period_s = float(poll_period_s)
        self.history_rows = int(history_rows)
        self.enable_after_first_embedding = bool(enable_after_first_embedding)
        self.expected_dt = float(expected_dt)

        if runtimes is not None:
            # Test hook: (inf, mode, applied_runtime, base_runtime, tam_runtime, fusion_params)
            (self.inf, self.mode, self._applied, self._base, self._tam, self._fusion_params) = runtimes
        else:
            from rcs_tam.assets import default_panda_xml
            from rcs_tam.simadaptor.deploy.history_runtime import RealTimeHistoryAdaptor

            if xml_path is None:
                # Packaged training MJCF of the supported Panda checkpoints.
                xml_path = default_panda_xml()
                self._log(f"[rcs_tam] ideal-model xml: packaged {xml_path}")
            self._applied = RealTimeHistoryAdaptor(
                simadaptor_ckpt_path=str(ckpt_path),
                xml_path=str(xml_path),
                expected_dt=self.expected_dt,
                attention_history_s=float(attention_history_s),
                jax_cache_dir=jax_cache_dir,
            )
            self.inf = self._applied.inf
            self.mode = resolve_history_torque_mode(self.inf, history_torque_mode)
            self._base = None
            self._tam = None
            self._fusion_params = None
            if self.mode == HISTORY_TORQUE_MODE_BASE_TAM_FUSION:
                self._fusion_params = _history_fusion_params(self.inf)
                self._base = RealTimeHistoryAdaptor(
                    sim_inf=self.inf,
                    runtime_bundle=self._applied.runtime_bundle,
                    expected_dt=self.expected_dt,
                    attention_history_s=float(attention_history_s),
                    jax_cache_dir=jax_cache_dir,
                )
                self._tam = RealTimeHistoryAdaptor(
                    sim_inf=self.inf,
                    runtime_bundle=self._applied.runtime_bundle,
                    expected_dt=self.expected_dt,
                    attention_history_s=float(attention_history_s),
                    jax_cache_dir=jax_cache_dir,
                )
        self._log(f"[rcs_tam] history_torque_mode={self.mode}")

        # Export the adaptor into the controller hook.
        if runtimes is None:
            if bin_path is None:
                bin_dir = Path(tempfile.gettempdir()) / "rcs_tam_adaptors"
                bin_dir.mkdir(parents=True, exist_ok=True)
                bin_path = bin_dir / f"{Path(str(ckpt_path)).name}.bin"
            self.bin_path = Path(bin_path)
            self.inf.export_simadaptor_weights_cpp(self.bin_path)
            if not bool(self.robot.tam_load_adaptor(str(self.bin_path))):
                raise RuntimeError(f"controller rejected adaptor bin {self.bin_path}")
            self._log(f"[rcs_tam] adaptor bin loaded: {self.bin_path}")
        else:
            self.bin_path = None
        if residual_torque_limits is not None:
            self.robot.tam_set_torque_limits(np.asarray(residual_torque_limits, dtype=float).reshape(7))
        if enable_ramp_s is not None:
            self.robot.tam_set_enable_ramp_s(float(enable_ramp_s))
        self.robot.tam_enable(False)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_row_t: Optional[float] = None
        self._patches_since_reset = 0
        self._last_send_wall: Optional[float] = None
        self._pending_embedding: Optional[np.ndarray] = None
        self.enabled = False
        self.stats: Dict[str, Any] = {
            "windows": 0,
            "rows_seen": 0,
            "embeddings": 0,
            "embeddings_sent": 0,
            "controller_restarts": 0,
            "last_error": None,
        }

    # ------------------------------------------------------------------ #
    def _reset_runtimes(self, reason: str) -> None:
        self._log(f"[rcs_tam] resetting history runtimes: {reason}")
        for runtime in (self._applied, self._base, self._tam):
            if runtime is not None:
                runtime.reset()
        self._last_row_t = None
        self._patches_since_reset = 0
        self._pending_embedding = None
        self._last_send_wall = None
        self.stats["controller_restarts"] += 1
        if self.enable_after_first_embedding and self.enabled:
            # The hook cleared its embedding on the controller restart; disable
            # until a fresh embedding has been delivered.
            self.robot.tam_enable(False)
            self.enabled = False

    def _push_window(self, rows: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        t = np.asarray(rows["t"], dtype=np.float64).reshape(-1)
        if t.size == 0:
            return None
        if self._last_row_t is not None and float(t[-1]) < self._last_row_t - 0.5:
            self._reset_runtimes(f"controller time went backwards ({t[-1]:.3f} < {self._last_row_t:.3f})")
        self._last_row_t = float(t[-1])
        q = np.asarray(rows["q"], dtype=np.float32)
        dq = np.asarray(rows["dq"], dtype=np.float32)
        tau_applied = np.asarray(rows["tau_applied"], dtype=np.float32)
        gravity = np.asarray(rows["gravity"], dtype=np.float32)
        keep = np.asarray(rows["valid_for_history"], dtype=np.float32).reshape(-1)
        self.stats["rows_seen"] += int(t.size)
        if self.mode == HISTORY_TORQUE_MODE_APPLIED:
            return self._applied.push_window(
                t, q, dq, tau_applied, gravity=gravity, keep_mask=keep
            )
        tau_base = np.asarray(rows["tau_base"], dtype=np.float32)
        tau_delta = np.asarray(rows["tau_adaptor_delta"], dtype=np.float32)
        applied_emb = self._applied.push_window(
            t, q, dq, tau_applied, gravity=gravity, raw_tau=tau_applied, keep_mask=keep
        )
        base_emb = self._base.push_window(
            t, q, dq, tau_base, gravity=gravity, raw_tau=tau_applied, keep_mask=keep
        )
        tam_emb = self._tam.push_window(
            t, q, dq, tau_delta, tau_is_model_space=True, raw_tau=tau_applied, keep_mask=keep
        )
        if applied_emb is None or base_emb is None or tam_emb is None:
            return None
        from rcs_tam.simadaptor.deploy.runtime_common import apply_history_fusion

        return apply_history_fusion(self._fusion_params, applied_emb, base_emb, tam_emb)

    def _step(self) -> None:
        rows = self.robot.tam_get_history(self.history_rows)
        self.stats["windows"] += 1
        emb = self._push_window(rows)
        now = time.perf_counter()
        if emb is not None:
            self._pending_embedding = np.asarray(emb, dtype=np.float32)
            self._patches_since_reset += 1
            self.stats["embeddings"] += 1
        if self._pending_embedding is None:
            return
        if self._patches_since_reset < self.min_patches_before_send:
            return
        if self._last_send_wall is not None and (now - self._last_send_wall) < self.embedding_interval_s:
            return
        from rcs_tam.simadaptor.deploy.runtime_common import flatten_history_embedding_for_transport

        flat = flatten_history_embedding_for_transport(self._pending_embedding)
        self.robot.tam_set_embedding(np.asarray(flat, dtype=np.float64))
        self.stats["embeddings_sent"] += 1
        self._last_send_wall = now
        self._pending_embedding = None
        if self.enable_after_first_embedding and not self.enabled:
            self.robot.tam_enable(True)
            self.enabled = True
            self._log("[rcs_tam] adaptor enabled after first embedding")

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._step()
            except Exception as exc:  # keep the encoder loop alive
                self.stats["last_error"] = str(exc)
                self._log(f"[rcs_tam] encoder step failed: {exc}")
                time.sleep(0.1)
            elapsed = time.perf_counter() - t0
            if elapsed < self.poll_period_s:
                time.sleep(self.poll_period_s - elapsed)

    # ------------------------------------------------------------------ #
    def start(self) -> "TamDeployment":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="rcs_tam_encoder", daemon=True)
            self._thread.start()
        return self

    def stop(self, *, disable: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if disable:
            try:
                self.robot.tam_enable(False)
            except Exception:
                pass
            self.enabled = False

    def status(self) -> Dict[str, Any]:
        out = dict(self.stats)
        out["mode"] = self.mode
        out["enabled"] = self.enabled
        try:
            hook = dict(self.robot.tam_status())
            hook.pop("torque_limits", None)
            out.update({f"hook_{k}": v for k, v in hook.items()})
        except Exception as exc:  # pragma: no cover
            out["hook_status_error"] = str(exc)
        return out


__all__ = [
    "HISTORY_TORQUE_MODE_APPLIED",
    "HISTORY_TORQUE_MODE_AUTO",
    "HISTORY_TORQUE_MODE_BASE_TAM_FUSION",
    "TamDeployment",
    "resolve_history_torque_mode",
]
