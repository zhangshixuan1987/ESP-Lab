"""Shared filesystem paths for ESP-Lab diagnostics."""

from __future__ import annotations

import os
from pathlib import Path


S2D_DIAG_ROOT = Path(
    os.environ.get("ESP_LAB_S2D_DIAG_ROOT", "/global/cfs/cdirs/e3sm/S2S2D/s2d_diag")
)
FIGURE_OUTDIR = Path(
    os.environ.get(
        "ESP_LAB_FIGURE_OUTDIR",
        "/global/cfs/cdirs/e3sm/www/zhan391/esp-lab_diag",
    )
)

E3SMLE_DIAG_DIR = S2D_DIAG_ROOT
CESM_SMYLE_DIAG_DIR = S2D_DIAG_ROOT / "CESM-SMYLE"
HADISST2_DIAG_DIR = S2D_DIAG_ROOT / "HadISST2"
NMME_DIAG_DIR = S2D_DIAG_ROOT / "NMME"
NMME_FIXED_DIR = NMME_DIAG_DIR / "fixed"
# Modes-of-variability products are stored below
# S2D_DIAG_ROOT / <source-or-case> / "modes_variability".
MODES_VARIABILITY_DIAG_DIR = S2D_DIAG_ROOT
MULTIMODEL_DIAG_DIR = S2D_DIAG_ROOT / "multimodel"


def leadtime_acc_dir(
    source: str,
    *parts: str,
    root: str | Path | None = None,
) -> Path:
    """Return a source-first lead-time ACC diagnostic directory.

    Examples are ``<root>/<case>/leadtime_acc/inputs/land/H2OSOI`` and
    ``<root>/<case>/leadtime_acc/skill/atm/TREFHT``. Passing ``root`` keeps
    notebook and command-line configurations explicit and machine portable.
    """
    path = Path(root) if root is not None else S2D_DIAG_ROOT
    path = path / str(source) / "leadtime_acc"
    for part in parts:
        path = path / str(part)
    return path


def diagnostic_dir(
    source: str,
    diagnostic: str,
    *parts: str,
    root: str | Path | None = None,
) -> Path:
    """Return ``<root>/<source>/<diagnostic>/<parts...>``."""
    path = Path(root) if root is not None else S2D_DIAG_ROOT
    path = path / str(source) / str(diagnostic)
    for part in parts:
        path = path / str(part)
    return path


def multimodel_diagnostic_dir(
    diagnostic: str,
    *parts: str,
    root: str | Path | None = None,
) -> Path:
    """Return a canonical directory for cross-experiment diagnostics."""
    return diagnostic_dir("multimodel", diagnostic, *parts, root=root)


def figure_output_dir(
    diagnostic: str,
    *parts: str,
    root: str | Path | None = None,
) -> Path:
    """Return a diagnostic-specific directory below the public figure root."""
    path = Path(root) if root is not None else FIGURE_OUTDIR
    path = path / str(diagnostic)
    for part in parts:
        path = path / str(part)
    return path


__all__ = [
    "S2D_DIAG_ROOT",
    "FIGURE_OUTDIR",
    "E3SMLE_DIAG_DIR",
    "CESM_SMYLE_DIAG_DIR",
    "HADISST2_DIAG_DIR",
    "NMME_DIAG_DIR",
    "NMME_FIXED_DIR",
    "MODES_VARIABILITY_DIAG_DIR",
    "MULTIMODEL_DIAG_DIR",
    "leadtime_acc_dir",
    "diagnostic_dir",
    "multimodel_diagnostic_dir",
    "figure_output_dir",
]
