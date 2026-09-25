"""Diagnostic modules for ESP-Lab evaluation workflows."""

from . import config
from . import initial_shock
from . import initial_shock_error
from . import native_eli
from . import regional
from . import s2d
from . import skill
from . import sst_index
from . import web
from .config import (
    DEFAULT_CLIMATOLOGY_END_YEAR,
    DEFAULT_CLIMATOLOGY_START_YEAR,
    DEFAULT_CLIMATOLOGY_YEARS,
    S2DConfig,
)
from .initial_shock import (
    align_observation_months,
    compute_initial_shock_index,
    plot_normalized_change,
    plot_std_ratio,
)
from .initial_shock_error import (
    compute_initial_shock_error_index,
    plot_error_heatmap,
)
from .s2d import S2DDiagnostics
from .skill import compute_skill
from .web import discover_workflow_figures, generate_diagnostics_webpage

__all__ = [
    "config",
    "initial_shock",
    "initial_shock_error",
    "native_eli",
    "regional",
    "s2d",
    "skill",
    "sst_index",
    "web",
    "DEFAULT_CLIMATOLOGY_START_YEAR",
    "DEFAULT_CLIMATOLOGY_END_YEAR",
    "DEFAULT_CLIMATOLOGY_YEARS",
    "S2DConfig",
    "S2DDiagnostics",
    "align_observation_months",
    "compute_initial_shock_index",
    "plot_normalized_change",
    "plot_std_ratio",
    "compute_initial_shock_error_index",
    "plot_error_heatmap",
    "compute_skill",
    "discover_workflow_figures",
    "generate_diagnostics_webpage",
]
