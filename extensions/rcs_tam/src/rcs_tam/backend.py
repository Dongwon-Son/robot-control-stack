"""Controller backend interface for :class:`rcs_tam.bridge.TamBridge`."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from rcs_tam.protocol import sample_dict_from_row

HistoryRowsDict = Mapping[str, Any]


class UnsupportedCommand(Exception):
    """Raised by a backend for protocol commands it cannot honour."""


class BridgeBackend:
    """What the generic bridge needs from a controller.

    Implementations must be thread-safe with respect to the bridge thread
    (all calls happen from the bridge loop; the controller's realtime thread
    is the backend's own business).
    """

    name: str = "abstract"

    # ---- history / TAM state ------------------------------------------- #
    def get_history_samples(self, max_rows: int) -> List[Dict[str, Any]]:
        """Newest publish-ready rows, oldest first, in wire format."""
        raise NotImplementedError

    def set_embedding(self, embedding: np.ndarray) -> None:
        raise NotImplementedError

    def get_embedding_seq(self) -> int:
        raise NotImplementedError

    def enable_adaptor(self, enabled: bool) -> None:
        raise NotImplementedError

    def adaptor_enabled(self) -> bool:
        raise NotImplementedError

    def load_adaptor_path(self, path: str) -> bool:
        raise NotImplementedError

    def set_ideal_model_has_gravity(self, enabled: bool) -> None:
        raise NotImplementedError

    def ideal_model_has_gravity(self) -> bool:
        raise NotImplementedError

    # ---- actuation ------------------------------------------------------- #
    def apply_actuation(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Apply the actuation keys of a normalized command payload.

        Keys follow ``rcs_tam.protocol.REMOTE_ACTUATION_KEYS``
        (``control_mode``, ``target_q``, ``target_dq``, ``target_position``,
        ``target_orientation``, ``stiffness``, ``damping``, ``filter``,
        ``feedforward`` ...).  Return a dict merged into the reply; raise
        :class:`UnsupportedCommand` for keys the backend cannot honour.
        """
        raise NotImplementedError

    def neutralize_remote_motion(self) -> None:
        """Hold the current pose after the remote command stream went stale."""
        raise NotImplementedError

    def reset(self) -> None:
        """Full controller reset (stop, move home, restart)."""
        raise NotImplementedError

    # ---- optional capabilities ------------------------------------------ #
    # Disabling / clearing a feature the backend does not have is a harmless
    # no-op (the TAM mapping server issues these during prepare and treats a
    # rejection as "controller not ready"); enabling it is unsupported.
    def enable_sysid(self, enabled: bool) -> None:
        if bool(enabled):
            raise UnsupportedCommand("sysid is not supported by this backend")

    def clear_sysid(self) -> None:
        return None

    def set_sysid(self, params: Mapping[str, Any]) -> None:
        raise UnsupportedCommand("sysid is not supported by this backend")

    def enable_external_torque_prediction(self, enabled: bool) -> None:
        if bool(enabled):
            raise UnsupportedCommand("external torque prediction is not supported by this backend")

    def status(self) -> Dict[str, Any]:
        """Extra status fields merged into every reply (``backend`` is added by the bridge)."""
        return {}

    def safety_check(self) -> Optional[str]:
        """Return a reason string when the backend wants a controller reset, else None."""
        return None

    def close(self) -> None:
        pass


def history_rows_dict_to_samples(rows: HistoryRowsDict) -> List[Dict[str, Any]]:
    """Convert an RCS-style ``tam_get_history`` dict of arrays into wire samples."""
    t = np.asarray(rows["t"], dtype=float).reshape(-1)
    n = int(t.shape[0])
    if n == 0:
        return []

    def _arr(key: str, default: Optional[np.ndarray] = None) -> np.ndarray:
        if key in rows:
            return np.asarray(rows[key])
        if default is None:
            raise KeyError(key)
        return default

    zeros = np.zeros((n, 7), dtype=float)
    q = _arr("q")
    dq = _arr("dq")
    tau_applied = _arr("tau_applied")
    tau_base = _arr("tau_base", tau_applied)
    tau_delta = _arr("tau_adaptor_delta", zeros)
    tau_commanded = _arr("tau_commanded", zeros)
    tau_measured = _arr("tau_measured", zeros)
    gravity = _arr("gravity", zeros)
    emb_seq = _arr("history_embedding_seq", np.zeros(n, dtype=np.int64))
    adaptor_active = _arr("adaptor_active", np.zeros(n, dtype=bool))
    valid = _arr("valid_for_history", np.ones(n, dtype=bool))
    padding = _arr("synthetic_padding", np.zeros(n, dtype=bool))
    sample_dt = _arr("sample_dt_sec", np.full(n, 1e-3))
    o_t_ee = rows.get("O_T_EE") if isinstance(rows, Mapping) else None
    coriolis = rows.get("coriolis") if isinstance(rows, Mapping) else None

    samples: List[Dict[str, Any]] = []
    for i in range(n):
        samples.append(
            sample_dict_from_row(
                t=float(t[i]),
                q=q[i],
                dq=dq[i],
                tau_applied=tau_applied[i],
                tau_base=tau_base[i],
                tau_adaptor_delta=tau_delta[i],
                tau_commanded=tau_commanded[i],
                tau_measured=tau_measured[i],
                gravity=gravity[i],
                history_embedding_seq=int(emb_seq[i]),
                adaptor_active=bool(adaptor_active[i]),
                valid_for_history=bool(valid[i]),
                synthetic_padding=bool(padding[i]),
                sample_dt_sec=float(sample_dt[i]),
                O_T_EE=None if o_t_ee is None else np.asarray(o_t_ee)[i],
                coriolis=None if coriolis is None else np.asarray(coriolis)[i],
            )
        )
    return samples


__all__ = ["BridgeBackend", "HistoryRowsDict", "UnsupportedCommand", "history_rows_dict_to_samples"]
