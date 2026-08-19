"""RCS TAM bridge: serve the TAM history/embedding ZMQ protocol on top of the ``TamHook`` of ``hw.Franka``.

Modules:

* ``protocol``    -- wire protocol helpers (history sample format, async command
                     dedup/merge, reliable command normalization, reset publish)
* ``backend``     -- ``BridgeBackend`` interface + helpers
* ``bridge``      -- ``TamBridge``: PUB/PULL/REP loop over a backend
* ``rcs_backend`` -- ``RcsBackend``: ``hw.Franka`` (rcs_panda / rcs_fr3, ``tam`` branch)
* ``env_wrapper`` -- ``TamBridgeWrapper``: run the bridge next to an RCS gym env
* ``config``      -- JSON/env configuration (network, timing, safety)
* ``__main__``    -- ``python -m rcs_tam`` NUC entrypoint

The workstation side (history encoder, mapping server, clients) lives in the TAM
repository and is unchanged.
"""

from rcs_tam.backend import BridgeBackend, UnsupportedCommand, history_rows_dict_to_samples  # noqa: F401
from rcs_tam.bridge import BridgeEndpoints, TamBridge  # noqa: F401

__all__ = [
    "BridgeBackend",
    "BridgeEndpoints",
    "TamBridge",
    "UnsupportedCommand",
    "history_rows_dict_to_samples",
]
