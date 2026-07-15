from importlib.metadata import PackageNotFoundError, version

from . import data_access_smyle
from . import data_access_e3sm
from . import data_access_cesm_smyle
from . import diagnostics
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
