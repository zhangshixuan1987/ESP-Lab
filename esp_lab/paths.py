"""Shared filesystem paths for ESP-Lab diagnostics."""

from __future__ import annotations

import os
from pathlib import Path


S2D_DIAG_ROOT = Path(
    os.environ.get("ESP_LAB_S2D_DIAG_ROOT", "/global/cfs/cdirs/e3sm/S2S2D/s2d_diag")
)

E3SMLE_DIAG_DIR = S2D_DIAG_ROOT
CESM_SMYLE_DIAG_DIR = S2D_DIAG_ROOT / "CESM-SMYLE"
HADISST2_DIAG_DIR = S2D_DIAG_ROOT / "HadISST2"
NMME_DIAG_DIR = S2D_DIAG_ROOT / "NMME"
NMME_FIXED_DIR = NMME_DIAG_DIR / "fixed"
# Modes-of-variability products are stored below
# S2D_DIAG_ROOT / <source-or-case> / "modes_variability".
MODES_VARIABILITY_DIAG_DIR = S2D_DIAG_ROOT


__all__ = [
    "S2D_DIAG_ROOT",
    "E3SMLE_DIAG_DIR",
    "CESM_SMYLE_DIAG_DIR",
    "HADISST2_DIAG_DIR",
    "NMME_DIAG_DIR",
    "NMME_FIXED_DIR",
    "MODES_VARIABILITY_DIAG_DIR",
]
