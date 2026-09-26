"""
config.py
=========
Unified S2D Workflow Configuration.

Defines:
  - Experiment cohort (JRA55_FOSIRL, Reanalysis)
  - Explicit member lists
  - Initialization years (1980–1986) & months (May, November)
  - Multi-source readiness flags (daily_hindcast, monthly_hindcast, observations, historical)
  - Branch toggles (field_drift, physical_consistency, model_attractor)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent

import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.daily_core import DEFAULT_DAILY_EXPERIMENT_SPECS, RECOMMENDED_DAILY_WINDOW_DEFS
from esp_lab.diagnostics.monthly_core import DEFAULT_WINDOW_DEFS as DEFAULT_MONTHLY_WINDOW_DEFS
from esp_lab import env_paths
from esp_lab.paths import multimodel_diagnostic_dir

PATH_SETTINGS = {
    "s2d_diag_root": env_paths.s2d_diag_root(),
    "figure_outdir": env_paths.drift_figure_root(),
}

DEFAULT_OUTPUT_ROOT = multimodel_diagnostic_dir(
    "unified_diagnostics", root=PATH_SETTINGS["s2d_diag_root"]
)
# Figures are published flat under the drift figure root, as in s2d_skill.
DEFAULT_FIGURE_OUTDIR = PATH_SETTINGS["figure_outdir"]

DATA_DIR = os.environ.get(
    "ESP_LAB_UNIFIED_DATA_DIR",
    str(env_paths.raw_model_root()),
)

# Explicit cohort definition
EXPERIMENTS = DEFAULT_DAILY_EXPERIMENT_SPECS
INIT_YEARS  = list(range(1980, 1987))
INIT_MONTHS = [5, 11]
MEMBERS     = [f"EN{i:02d}" for i in range(10)]

DAILY_WINDOWS   = RECOMMENDED_DAILY_WINDOW_DEFS
MONTHLY_WINDOWS = DEFAULT_MONTHLY_WINDOW_DEFS

# Multi-source readiness status container
READINESS_FLAGS = {
    "daily_hindcast":          "NOT_CHECKED",
    "monthly_hindcast":        "NOT_CHECKED",
    "observations":            "NOT_CHECKED",
    "e3sm_historical_daily":   "NOT_CHECKED",
    "e3sm_historical_monthly": "NOT_CHECKED",
}
