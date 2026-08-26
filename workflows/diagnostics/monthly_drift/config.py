"""
config.py
=========
Authoritative cohort definition for the monthly S2D spatial analysis.

Edit this file to change:
  - experiments, case prefixes
  - init years / months
  - members
  - primary variables and their conversions
  - window definitions
  - data paths

All other workflow modules import from here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Locate package root for relative imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent

import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.monthly_core import (
    DEFAULT_EXPERIMENT_SPECS,
    DEFAULT_WINDOW_DEFS,
    MonthlyConfig,
    VariableConversionSpec,
    WindowDef,
    convert_kelvin_to_celsius,
    convert_prect_to_mmday,
    no_conversion,
)
from esp_lab.paths import figure_output_dir, multimodel_diagnostic_dir


# ===========================================================================
# Canonical cohort definition
# ===========================================================================

DATA_DIR = os.environ.get(
    "ESP_LAB_MONTHLY_DATA_DIR",
    "/global/cfs/cdirs/e3sm/S2S2D/post_process",
)

# Durable diagnostic products and public figures live outside the checkout.
PATH_SETTINGS = {
    "s2d_diag_root": Path("/global/cfs/cdirs/e3sm/S2S2D/s2d_diag"),
    "figure_outdir": Path("/global/cfs/cdirs/e3sm/www/zhan391/esp-lab_diag"),
}

DEFAULT_OUTPUT_ROOT = multimodel_diagnostic_dir(
    "leadtime_drift",
    "atm",
    "monthly_spatial",
    root=PATH_SETTINGS["s2d_diag_root"],
)
DEFAULT_FIGURE_OUTDIR = figure_output_dir(
    "leadtime_drift",
    "atm",
    "monthly_spatial",
    root=PATH_SETTINGS["figure_outdir"],
)

INIT_YEARS:  List[int] = list(range(1980, 1987))   # 1980–1986
INIT_MONTHS: List[int] = [5, 11]                    # May and November
MEMBERS:     List[str] = [f"EN{i:02d}" for i in range(10)]   # EN00–EN09
LEADS:       List[int] = list(range(1, 25))          # leads 1–24

MONTHLY_WINDOW_DEFS: Dict[str, Tuple[int, int]] = DEFAULT_WINDOW_DEFS

# Variables — extend this list for TREFHT, LHFLX, SHFLX, soil moisture
VARIABLE_SPECS: List[VariableConversionSpec] = [
    VariableConversionSpec(
        native_name="PRECT",
        plot_name="Total precipitation",
        plot_units="mm/day",
        mask_type="none",
        obs_product=None,           # set obs_product="GPCP" once obs path is configured
        obs_var=None,
        model_convert=convert_prect_to_mmday,
        obs_convert=no_conversion,
    ),
    VariableConversionSpec(
        native_name="TREFHT",
        plot_name="Near-surface air temperature",
        plot_units="degC",
        mask_type="none",
        obs_product=None,
        obs_var=None,
        model_convert=convert_kelvin_to_celsius,
        obs_convert=no_conversion,
    ),
    VariableConversionSpec(
        native_name="LHFLX",
        plot_name="Latent heat flux",
        plot_units="W/m²",
        mask_type="none",
        obs_product=None,
        obs_var=None,
        model_convert=no_conversion,
        obs_convert=no_conversion,
    ),
    VariableConversionSpec(
        native_name="SHFLX",
        plot_name="Sensible heat flux",
        plot_units="W/m²",
        mask_type="none",
        obs_product=None,
        obs_var=None,
        model_convert=no_conversion,
        obs_convert=no_conversion,
    ),
]

# Observation file paths (set when obs data are available)
OBS_PATHS: Dict[str, Optional[str]] = {
    "GPCP":  None,   # e.g. "/global/cfs/cdirs/e3sm/obs_data/GPCP_mon_2.5deg.nc"
}


# ===========================================================================
# Config builder
# ===========================================================================

def build_monthly_config(
    pilot_only: bool = True,
    pilot_years: Optional[List[int]] = None,
    pilot_months: Optional[List[int]] = None,
    variable_filter: Optional[List[str]] = None,
    gate_strict: bool = False,
    data_dir: Optional[str] = None,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> MonthlyConfig:
    """Build a MonthlyConfig with sensible defaults.

    Parameters
    ----------
    pilot_only:
        If True, use pilot_years/pilot_months only.
    pilot_years:
        Defaults to [1980, 1981, 1982].
    pilot_months:
        Defaults to [5] (May only).
    variable_filter:
        Restrict to these variable native names.  None = all.
    gate_strict:
        If True, inventory gate failures raise SystemExit(1).
    data_dir:
        Override the global DATA_DIR.
    """
    py = pilot_years or [1980, 1981, 1982]
    pm = pilot_months or [5]

    variables = VARIABLE_SPECS
    if variable_filter:
        variables = [v for v in variables if v.native_name in variable_filter]
    if not variables:
        raise ValueError(f"variable_filter={variable_filter} produced no variables.")

    return MonthlyConfig(
        experiments=DEFAULT_EXPERIMENT_SPECS,
        init_years=INIT_YEARS,
        init_months=INIT_MONTHS,
        members=MEMBERS,
        leads=LEADS,
        window_defs=MONTHLY_WINDOW_DEFS,
        variables=variables,
        data_dir=data_dir or DATA_DIR,
        realm="atm",
        grid="180x360_aave",
        freq="monthly",
        ts_split="2yr",
        baseline_lead=1,
        pilot_only=pilot_only,
        pilot_years=py,
        pilot_months=pm,
        gate_strict=gate_strict,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        output_root=str(output_root),
    )


def load_config_yaml(yaml_path: Path) -> MonthlyConfig:
    """Load MonthlyConfig from a YAML override file.

    YAML keys match build_monthly_config kwargs:
        pilot_only, pilot_years, pilot_months, variable_filter,
        gate_strict, data_dir, n_bootstrap, bootstrap_seed, output_root.

    Missing keys fall back to the canonical defaults.
    """
    with open(yaml_path) as fh:
        overrides = yaml.safe_load(fh) or {}
    return build_monthly_config(**{k: v for k, v in overrides.items()
                                   if k in build_monthly_config.__code__.co_varnames})
