import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _configure_conda_geospatial_data_paths():
    """Provide GDAL/PROJ data paths when a Conda kernel omits activation vars."""
    share = Path(sys.prefix) / "share"
    candidates = {
        "GDAL_DATA": share / "gdal",
        "PROJ_DATA": share / "proj",
    }
    for variable, path in candidates.items():
        if variable not in os.environ and path.is_dir():
            os.environ[variable] = str(path)


_configure_conda_geospatial_data_paths()

from . import data_access_smyle
from . import data_access_e3sm
from . import data_access_cesm_smyle
from . import diagnostics
from . import land_skill
from . import psl_skill
from . import paths
from .data_access_smyle import get_monthly_data as get_monthly_data_smyle
from .data_access_smyle import preprocessor as preprocessor_smyle
from .data_access_e3sm import get_monthly_data as get_monthly_data_e3sm
from .data_access_e3sm import preprocessor as preprocessor_e3sm
from .data_access_cesm_smyle import get_monthly_data as get_monthly_data_cesm_smyle
from .data_access_cesm_smyle import preprocessor as preprocessor_cesm_smyle
from .stats import leadtime_skill_seas_resamp

__all__ = [
    "data_access_smyle",
    "data_access_e3sm",
    "data_access_cesm_smyle",
    "diagnostics",
    "land_skill",
    "psl_skill",
    "paths",
    "get_monthly_data_smyle",
    "preprocessor_smyle",
    "get_monthly_data_e3sm",
    "preprocessor_e3sm",
    "get_monthly_data_cesm_smyle",
    "preprocessor_cesm_smyle",
    "leadtime_skill_seas_resamp",
]

try:
    __version__ = version("esp_lab")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"  # pragma: no cover
