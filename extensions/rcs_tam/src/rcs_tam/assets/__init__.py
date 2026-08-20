"""Packaged ideal-model assets.

``panda_pandagripper.xml`` (plus its meshes) is the exact MJCF the supported
Panda TAM checkpoints were trained against; it is the default ideal model for
:class:`rcs_tam.TamDeployment` when no explicit ``xml_path`` is given.
"""

from __future__ import annotations

from pathlib import Path

PANDA_PANDAGRIPPER_XML = Path(__file__).resolve().parent / "franka_panda" / "panda_pandagripper.xml"


def default_panda_xml() -> Path:
    if not PANDA_PANDAGRIPPER_XML.is_file():
        raise FileNotFoundError(f"packaged ideal-model XML missing: {PANDA_PANDAGRIPPER_XML}")
    return PANDA_PANDAGRIPPER_XML


__all__ = ["PANDA_PANDAGRIPPER_XML", "default_panda_xml"]
