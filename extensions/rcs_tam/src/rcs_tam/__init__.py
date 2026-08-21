"""TAM (Torque Adaptation Module) for robot-control-stack — single-process deployment.

``hw.Franka`` (rcs_panda / rcs_fr3, this fork) applies the TAM residual at
1 kHz through its built-in ``TamHook``; :class:`TamDeployment` runs the JAX
history encoder in the same Python process and keeps the hook's latent
embedding fresh. See ``examples/franka_tam_direct.py`` for the single-script
usage and ``rcs_tam.simadaptor`` for the vendored TAM inference code.
"""

import os as _os

# The encoder shares the GPU with the user's policy; JAX preallocating ~75%
# of device memory at first use starves everything else, so it is always
# disabled here (before JAX initializes its backend). The encoder's own
# allocations are small; on-demand allocation costs it nothing.
_os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
