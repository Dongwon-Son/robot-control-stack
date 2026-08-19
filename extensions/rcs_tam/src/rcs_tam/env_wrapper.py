"""Gymnasium wrapper that runs the TAM bridge next to an RCS hardware env (policy via gym, TAM over ZMQ)."""

from __future__ import annotations

from typing import Any, Optional

try:  # gymnasium is an RCS dependency, not a bridge dependency
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover
    gym = None  # type: ignore

from rcs_tam.bridge import BridgeEndpoints, TamBridge
from rcs_tam.rcs_backend import RcsBackend


def _require_gym():
    if gym is None:
        raise ModuleNotFoundError("gymnasium is required for TamBridgeWrapper")
    return gym


class TamBridgeWrapper((gym.Wrapper if gym is not None else object)):  # type: ignore[misc]
    """Start ``TamBridge(RcsBackend(env.robot))`` when constructed, stop it on ``close()``.

    Usage (NUC, inside the RCS process)::

        env = RCSPandaConfigEnvCreator().create_env(cfg)   # async_control=True
        env = TamBridgeWrapper(env, BridgeEndpoints.from_host("192.168.1.101"))
        ...                                                # normal gym loop / vlagents client
        env.close()

    The workstation runs the unchanged ``mapping_server.py`` against the bridge.
    """

    def __init__(
        self,
        env: Any,
        endpoints: BridgeEndpoints,
        *,
        robot: Any = None,
        home_on_reset: bool = False,
        bridge_kwargs: Optional[dict] = None,
        backend_kwargs: Optional[dict] = None,
    ) -> None:
        _require_gym()
        super().__init__(env)
        if robot is None:
            robot = env.get_wrapper_attr("robot")
        backend_kwargs = dict(backend_kwargs or {})
        backend_kwargs.setdefault("home_on_reset", home_on_reset)
        self.backend = RcsBackend(robot, **backend_kwargs)
        self.bridge = TamBridge(self.backend, endpoints, **dict(bridge_kwargs or {}))
        self.bridge.start()

    def close(self) -> None:
        try:
            self.bridge.stop()
        finally:
            super().close()


__all__ = ["TamBridgeWrapper"]
