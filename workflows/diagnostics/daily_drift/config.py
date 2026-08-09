"""
config.py
=========
Authoritative cohort definition for the daily S2D spatial analysis.

Settings:
  - JRA55_FOSIRL vs Reanalysis
  - Init years: 1980–1986
  - Init months: May (5) and November (11)
  - Members: EN00..EN09
  - Lead days: 1..84
  - Baseline day: 1
  - Windows: week_1 (1..7), weeks_2_3 (8..21), weeks_4_6 (22..42), weeks_7_12 (43..84)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent

import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.daily_core import (
    DEFAULT_DAILY_EXPERIMENT_SPECS,
    DEFAULT_DAILY_WINDOW_DEFS,
    DailyDriftConfig,
    DailyVariableSpec,
    convert_kelvin_to_celsius_if_needed,
    convert_prect_to_mmday,
    no_conversion,
)
from esp_lab.paths import figure_output_dir, multimodel_diagnostic_dir

DEFAULT_OUTPUT_ROOT = multimodel_diagnostic_dir(
    "leadtime_drift", "atm", "daily_spatial"
)
DEFAULT_FIGURE_OUTDIR = figure_output_dir(
    "leadtime_drift", "atm", "daily_spatial"
)

DATA_DIR = os.environ.get(
    "ESP_LAB_DAILY_DATA_DIR",
    "/global/cfs/cdirs/e3sm/S2S2D/post_process",
)

INIT_YEARS:  List[int] = list(range(1980, 1987))   # 1980–1986
INIT_MONTHS: List[int] = [5, 11]                    # May and November
MEMBERS:     List[str] = [f"EN{i:02d}" for i in range(10)]   # EN00–EN09
LEAD_DAYS:   List[int] = list(range(1, 85))          # days 1–84

DAILY_WINDOW_DEFS: Dict[str, Tuple[int, int]] = DEFAULT_DAILY_WINDOW_DEFS

# Daily variables metadata
DAILY_VARIABLE_SPECS: List[DailyVariableSpec] = [
    DailyVariableSpec(
        native_field="TREFHT",
        plot_name="2-m air temperature",
        plot_units="degC",
        obs_product="ERA5_daily",
        obs_var="tas",
        mask="none",
        model_convert=convert_kelvin_to_celsius_if_needed,
        obs_convert=convert_kelvin_to_celsius_if_needed,
        daily_aggregation="mean",
    ),
    DailyVariableSpec(
        native_field="PRECT",
        plot_name="Precipitation",
        plot_units="mm/day",
        obs_product=None,  # to be configured when daily precip obs selected
        obs_var="pr",
        mask="none",
        model_convert=convert_prect_to_mmday,
        obs_convert=no_conversion,
        daily_aggregation="mean",
    ),
    DailyVariableSpec(
        native_field="LHFLX",
        plot_name="Latent heat flux",
        plot_units="W/m²",
        obs_product=None,
        obs_var=None,
        mask="none",
        model_convert=no_conversion,
        obs_convert=no_conversion,
        daily_aggregation="mean",
    ),
    DailyVariableSpec(
        native_field="SHFLX",
        plot_name="Sensible heat flux",
        plot_units="W/m²",
        obs_product=None,
        obs_var=None,
        mask="none",
        model_convert=no_conversion,
        obs_convert=no_conversion,
        daily_aggregation="mean",
    ),
]


def build_daily_config(
    pilot_only: bool = True,
    pilot_years: Optional[List[int]] = None,
    pilot_months: Optional[List[int]] = None,
    variable_filter: Optional[List[str]] = None,
    gate_strict: bool = False,
    data_dir: Optional[str] = None,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> DailyDriftConfig:
    py = pilot_years or [1980, 1981, 1982]
    pm = pilot_months or [5]

    variables = DAILY_VARIABLE_SPECS
    if variable_filter:
        variables = [v for v in variables if v.native_field in variable_filter]
    if not variables:
        raise ValueError(f"variable_filter={variable_filter} produced no variables.")

    return DailyDriftConfig(
        experiments=DEFAULT_DAILY_EXPERIMENT_SPECS,
        init_years=INIT_YEARS,
        init_months=INIT_MONTHS,
        members=MEMBERS,
        lead_days=LEAD_DAYS,
        window_defs=DAILY_WINDOW_DEFS,
        variables=variables,
        data_dir=data_dir or DATA_DIR,
        baseline_day=1,
        pilot_only=pilot_only,
        pilot_years=py,
        pilot_months=pm,
        gate_strict=gate_strict,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        freq_tag="day",
        output_root=str(output_root),
    )
