from dataclasses import dataclass, field as dc_field
from typing import Optional, Sequence, Dict, Union


DEFAULT_CLIMATOLOGY_START_YEAR = 1981
DEFAULT_CLIMATOLOGY_END_YEAR = 2010
DEFAULT_CLIMATOLOGY_YEARS = (
    DEFAULT_CLIMATOLOGY_START_YEAR,
    DEFAULT_CLIMATOLOGY_END_YEAR,
)


@dataclass
class S2DConfig:
    # -----------------------------
    # Core variable / experiment
    # -----------------------------
    field: str = "TS"
    case_prefix: str = ""
    data_dir: str = ""

    members: Optional[Sequence[str]] = None
    nlead: int = 24

    # -----------------------------
    # Initialization settings
    # -----------------------------
    init_months: Sequence[int] = (5, 11)

    # e.g. {5: np.arange(...), 11: np.arange(...)}
    init_years: Optional[Dict[int, Union[Sequence[int], object]]] = None

    # -----------------------------
    # Region
    # -----------------------------
    region: Sequence[float] = (-170.0, -120.0, -5.0, 5.0)
    region_name: str = "Nino3.4"

    # -----------------------------
    # Climatology (drift / skill)
    # -----------------------------
    climy0: int = DEFAULT_CLIMATOLOGY_START_YEAR
    climy1: int = DEFAULT_CLIMATOLOGY_END_YEAR

    # -----------------------------
    # Output
    # -----------------------------
    outdir: str = ""
    force_rewrite: bool = False

    # -----------------------------
    # Data access options
    # -----------------------------
    realm: str = "atm"
    grid: str = "180x360_aave"
    freq: str = "monthly"
    ts_split: str = "2yr"

    engine: Optional[str] = None
    chunks: Dict = dc_field(default_factory=dict)

    require_all_members: bool = True
    verify_field_name: bool = True
    verify_coverage: bool = True

    # -----------------------------
    # Execution / Dask behavior
    # -----------------------------
    persist_intermediate: bool = False
    load_regional_series: bool = True

    # -----------------------------
    # Variable-specific processing
    # -----------------------------
    convert_ts_to_degC: bool = True

    # -----------------------------
    # Derived / optional settings
    # -----------------------------
    seasonal_nlead: int = 8   # default for 24 monthly leads

    # -----------------------------
    # Validation hook
    # -----------------------------
    def __post_init__(self):
        if self.members is None or len(self.members) == 0:
            raise ValueError("members must be provided and non-empty")

        if self.init_years is None:
            raise ValueError("init_years must be provided")

        if not isinstance(self.init_months, (list, tuple)):
            raise ValueError("init_months must be list/tuple")

        for m in self.init_months:
            if m not in self.init_years:
                raise ValueError(f"init_years missing entry for month {m}")

        if len(self.region) != 4:
            raise ValueError("region must be [lon_w, lon_e, lat_s, lat_n]")

        if self.climy1 < self.climy0:
            raise ValueError("climy1 must be >= climy0")

        if self.nlead <= 0:
            raise ValueError("nlead must be positive")
