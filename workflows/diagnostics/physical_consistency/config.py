"""
config.py
=========
Authoritative configuration for the S2D Physical-Consistency diagnostics.

Diagnostics included:
  - Flux partitioning: Evaporative Fraction (EF) & Bowen Ratio (BR)
  - Land-atmosphere coupling: Soil Moisture -> LHFLX & TREFHT
  - Precip-soil-moisture lag response: PRECT -> ΔSM (lags 0..7 days)
  - Ocean-atmosphere coupling: SST - TREFHT contrast
  - Apparent surface energy residual: FSNS - FLNS - LHFLX - SHFLX
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent

import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.daily_core import DEFAULT_DAILY_EXPERIMENT_SPECS, DEFAULT_DAILY_WINDOW_DEFS
from esp_lab.diagnostics.monthly_core import DEFAULT_WINDOW_DEFS as DEFAULT_MONTHLY_WINDOW_DEFS
from esp_lab.paths import figure_output_dir, multimodel_diagnostic_dir

DEFAULT_OUTPUT_ROOT = multimodel_diagnostic_dir("physical_consistency")
DEFAULT_FIGURE_OUTDIR = figure_output_dir("physical_consistency")

DATA_DIR = os.environ.get(
    "ESP_LAB_PHYSICAL_DATA_DIR",
    "/global/cfs/cdirs/e3sm/S2S2D/post_process",
)

INIT_YEARS:  List[int] = list(range(1980, 1987))
INIT_MONTHS: List[int] = [5, 11]
MEMBERS:     List[str] = [f"EN{i:02d}" for i in range(10)]

# Required physical fields
PHYSICAL_FIELDS = [
    "LHFLX", "SHFLX", "TREFHT", "PRECT", "H2OSOI", "TS", "FSNS", "FLNS"
]

DAILY_WINDOWS = DEFAULT_DAILY_WINDOW_DEFS
MONTHLY_WINDOWS = DEFAULT_MONTHLY_WINDOW_DEFS
