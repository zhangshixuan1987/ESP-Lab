"""Shared filesystem paths for ESP-Lab diagnostics."""

from __future__ import annotations

import os
from pathlib import Path


S2D_DIAG_ROOT = Path(
    os.environ.get("ESP_LAB_S2D_DIAG_ROOT", "/global/cfs/cdirs/e3sm/zhan391/s2d_diag")
)

E3SMLE_DIAG_DIR = S2D_DIAG_ROOT / "E3SMLE"
CESM_SMYLE_DIAG_DIR = S2D_DIAG_ROOT / "CESM-SMYLE"
NMME_DIAG_DIR = S2D_DIAG_ROOT / "NMME"
MODES_VARIABILITY_DIAG_DIR = S2D_DIAG_ROOT / "modes_variability"


__all__ = [
    "S2D_DIAG_ROOT",
    "E3SMLE_DIAG_DIR",
    "CESM_SMYLE_DIAG_DIR",
    "NMME_DIAG_DIR",
    "MODES_VARIABILITY_DIAG_DIR",
]
