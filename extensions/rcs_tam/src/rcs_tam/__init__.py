"""TAM (Torque Adaptation Module) for robot-control-stack — single-process deployment.

``hw.Franka`` (rcs_panda / rcs_fr3, this fork) applies the TAM residual at
1 kHz through its built-in ``TamHook``; :class:`TamDeployment` runs the JAX
history encoder in the same Python process and keeps the hook's latent
embedding fresh. See ``examples/franka_tam_direct.py`` for the single-script
usage and ``rcs_tam.simadaptor`` for the vendored TAM inference code.
"""

import os as _os

# The encoder shares the GPU with the user's policy in one process; without
# this JAX preallocates ~75% of device memory at first use and starves a
# non-JAX policy (e.g. torch). An explicit user setting always wins. Must be
# set before JAX initializes its backend, hence here at package import.
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from rcs_tam.runtime import (  # noqa: E402,F401
    HISTORY_TORQUE_MODE_APPLIED,
    HISTORY_TORQUE_MODE_AUTO,
    HISTORY_TORQUE_MODE_BASE_TAM_FUSION,
    TamDeployment,
    resolve_history_torque_mode,
)

__all__ = [
    "HISTORY_TORQUE_MODE_APPLIED",
    "HISTORY_TORQUE_MODE_AUTO",
    "HISTORY_TORQUE_MODE_BASE_TAM_FUSION",
    "TamDeployment",
    "resolve_history_torque_mode",
]
