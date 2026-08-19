"""Configuration for ``python -m rcs_tam`` (JSON file + environment overrides).

The JSON schema is a subset of the TAM reference NUC controller config so an
existing ``history_controller_config.json`` can be reused unchanged (unknown
blocks/keys are ignored)::

    {
      "network": {"workstation_host": "192.168.1.100", "nuc_control_host": "192.168.1.101",
                  "robot_host": "192.168.0.52", "history_port": 5555, "command_port": 5556,
                  "request_port": 5557},
      "timing":  {"history_pub_dt": 0.01, "loop_dt": 0.005, "history_len": 50,
                  "remote_command_stale_timeout_s": 2.0},
      "safety":  {"ext_force_threshold": 100.0, "ext_torque_threshold": 50.0},
      "rcs":     {"torque_limit": [87,87,87,87,12,12,12], "adaptor_torque_limits": [10,10,10,10,8,8,8],
                  "joint_kp": null, "joint_kd": null, "ideal_model_has_gravity": true}
    }

Environment overrides: ``RCS_TAM_CONFIG`` (path), ``RCS_TAM_NUC_CONTROL_HOST``,
``RCS_TAM_ROBOT_HOST`` (the ``PANDA_NUC_CONTROL_HOST`` / ``PANDA_ROBOT_HOST``
names of the reference controller are honoured too).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

CONFIG_ENV = "RCS_TAM_CONFIG"
DEFAULT_CONFIG_NAMES = ("rcs_tam_config.json", "history_controller_config.json")


@dataclass
class NetworkConfig:
    workstation_host: str = "192.168.1.100"
    nuc_control_host: str = "192.168.1.101"
    robot_host: str = "192.168.0.52"
    history_port: int = 5555
    command_port: int = 5556
    request_port: int = 5557
    history_endpoint: Optional[str] = None
    command_endpoint: Optional[str] = None
    request_endpoint: Optional[str] = None

    @property
    def history_bind(self) -> str:
        return self.history_endpoint or f"tcp://{self.nuc_control_host}:{self.history_port}"

    @property
    def command_bind(self) -> str:
        return self.command_endpoint or f"tcp://{self.nuc_control_host}:{self.command_port}"

    @property
    def request_bind(self) -> str:
        return self.request_endpoint or f"tcp://{self.nuc_control_host}:{self.request_port}"


@dataclass
class TimingConfig:
    history_pub_dt: float = 0.01
    loop_dt: float = 0.005
    history_len: int = 50
    remote_command_stale_timeout_s: float = 2.0


@dataclass
class SafetyConfig:
    ext_force_threshold: float = 100.0
    ext_torque_threshold: float = 50.0


@dataclass
class RcsConfig:
    torque_limit: list[float] = field(default_factory=lambda: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    adaptor_torque_limits: Optional[list[float]] = None
    joint_kp: Optional[list[float]] = None
    joint_kd: Optional[list[float]] = None
    ideal_model_has_gravity: bool = True
    home_on_reset: bool = False
    require_remote_actuation_for_adaptor_enable: bool = False


@dataclass
class RuntimeConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    rcs: RcsConfig = field(default_factory=RcsConfig)
    loaded_from: Optional[str] = None


def _merge(obj: Any, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            continue  # unknown keys (e.g. reference-controller blocks) are ignored
        setattr(obj, key, value)


def load_runtime_config(path: Optional[str] = None, *, search_dir: Optional[Path] = None) -> RuntimeConfig:
    cfg = RuntimeConfig()
    config_path = path or os.environ.get(CONFIG_ENV)
    if config_path is None:
        for base in filter(None, (search_dir, Path.cwd())):
            for name in DEFAULT_CONFIG_NAMES:
                cand = Path(base) / name
                if cand.is_file():
                    config_path = str(cand)
                    break
            if config_path:
                break
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            raise TypeError("rcs_tam config must be a JSON object.")
        for block, target in (("network", cfg.network), ("timing", cfg.timing), ("safety", cfg.safety), ("rcs", cfg.rcs)):
            if isinstance(data.get(block), Mapping):
                _merge(target, data[block])
        cfg.loaded_from = str(Path(config_path).expanduser())
    env = os.environ
    cfg.network.nuc_control_host = env.get("RCS_TAM_NUC_CONTROL_HOST", env.get("PANDA_NUC_CONTROL_HOST", cfg.network.nuc_control_host))
    cfg.network.robot_host = env.get("RCS_TAM_ROBOT_HOST", env.get("PANDA_ROBOT_HOST", cfg.network.robot_host))
    cfg.network.workstation_host = env.get("RCS_TAM_WORKSTATION_HOST", env.get("PANDA_WORKSTATION_HOST", cfg.network.workstation_host))
    for attr, names in (
        ("history_endpoint", ("RCS_TAM_HISTORY_ENDPOINT", "PANDA_HISTORY_ENDPOINT")),
        ("command_endpoint", ("RCS_TAM_COMMAND_ENDPOINT", "PANDA_COMMAND_ENDPOINT")),
        ("request_endpoint", ("RCS_TAM_REQUEST_ENDPOINT", "PANDA_REQUEST_ENDPOINT")),
    ):
        for name in names:
            if env.get(name):
                setattr(cfg.network, attr, env[name])
                break
    return cfg


__all__ = ["NetworkConfig", "RcsConfig", "RuntimeConfig", "SafetyConfig", "TimingConfig", "load_runtime_config"]
