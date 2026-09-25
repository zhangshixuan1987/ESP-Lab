"""Environment-overridable defaults for machine-specific ESP-Lab roots.

Notebooks previously hardcoded one person's NERSC paths (for example
``/global/cfs/cdirs/e3sm/www/zhan391/e3sm-s2d_diag``) as literal defaults,
which meant no other user or account could run them without manual
find-replace. These helpers keep the same defaults for the existing
workflow but let any of them be overridden with an ``ESP_LAB_*``
environment variable, so notebooks stay runnable out of the box while
remaining portable.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULTS = {
    "S2D_DIAG_ROOT": "/global/cfs/cdirs/e3sm/S2S2D/s2d_diag",
    "RAW_MODEL_ROOT": "/global/cfs/cdirs/e3sm/S2S2D/post_process",
    "FIGURE_ROOT": "/global/cfs/cdirs/e3sm/www/zhan391/e3sm-s2d_diag",
    "OBS_ROOT": "/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series",
    "DATA_ROOT": "/global/cfs/cdirs/e3sm/zhan391/data",
}


def _resolve(name: str) -> Path:
    return Path(os.environ.get(f"ESP_LAB_{name}", _DEFAULTS[name]))


def s2d_diag_root() -> Path:
    """Root for staged inputs, skill caches, and other s2d_diag outputs."""
    return _resolve("S2D_DIAG_ROOT")


def raw_model_root() -> Path:
    """Root for raw E3SM post-processed model archives."""
    return _resolve("RAW_MODEL_ROOT")


def figure_root() -> Path:
    """Root for published figures and the diagnostics gallery."""
    return _resolve("FIGURE_ROOT")


def obs_root() -> Path:
    """Root for e3sm_diags observational time-series inputs."""
    return _resolve("OBS_ROOT")


def data_root() -> Path:
    """Root for supplemental reference datasets (e.g. snow/soil/TWS products)."""
    return _resolve("DATA_ROOT")


__all__ = [
    "s2d_diag_root",
    "raw_model_root",
    "figure_root",
    "obs_root",
    "data_root",
]
