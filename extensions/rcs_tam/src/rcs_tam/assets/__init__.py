"""Packaged ideal-model assets.

``panda_pandagripper.xml`` (plus its meshes) is the exact MJCF the supported
Panda TAM checkpoints were trained against; it is the default ideal model for
:class:`rcs_tam.TamDeployment` when no explicit ``xml_path`` is given.
"""

from __future__ import annotations

from pathlib import Path

PANDA_PANDAGRIPPER_XML = Path(__file__).resolve().parent / "franka_panda" / "panda_pandagripper.xml"

# DAgger-finetuned applied-torque checkpoint (step 8850) — the default TAM
# checkpoint for deployment. The loader reads the sibling save_dict.pkl.
DAGGER_APPLIED_CKPT = (
    Path(__file__).resolve().parent / "checkpoints" / "dagger_applied_8850" / "checkpoint_8850"
)


def default_checkpoint() -> Path:
    if not DAGGER_APPLIED_CKPT.is_dir():
        raise FileNotFoundError(f"packaged TAM checkpoint missing: {DAGGER_APPLIED_CKPT}")
    return DAGGER_APPLIED_CKPT


def default_panda_xml() -> Path:
    if not PANDA_PANDAGRIPPER_XML.is_file():
        raise FileNotFoundError(f"packaged ideal-model XML missing: {PANDA_PANDAGRIPPER_XML}")
    return PANDA_PANDAGRIPPER_XML


__all__ = ["DAGGER_APPLIED_CKPT", "PANDA_PANDAGRIPPER_XML", "default_checkpoint", "default_panda_xml"]
