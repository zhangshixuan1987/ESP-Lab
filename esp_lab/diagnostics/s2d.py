from typing import Dict, Any, Optional, Tuple

import dask
import xarray as xr

from esp_lab.utils.sst_utils import normalize_sst_to_degc

from .regional import build_landmask, compute_weights, compute_regional_mean
from .drift import remove_model_drift
from .skill import compute_skill

try:
    from esp_lab.utils.dask_utils import maybe_persist, maybe_load
except ImportError:
    def maybe_persist(obj, do_persist=True):
        return obj.persist() if do_persist else obj

    def maybe_load(obj, do_load=True):
        return obj.load() if do_load else obj


class S2DDiagnostics:
    """
    Standard S2D regional diagnostics pipeline.

    This class currently focuses on:
      - loading model monthly data
      - computing monthly and seasonal regional indices
      - returning time coordinates and useful metadata

    It is intentionally xarray-first and Dask-friendly.
    """

    def __init__(self, cfg, data_access, obs_access, stats, spatial, cal):
        self.cfg = cfg
        self.data_access = data_access
        self.obs_access = obs_access
        self.stats = stats
        self.spatial = spatial
        self.cal = cal

        # Cache the land-sea mask and spatial weights — they depend only on the
        # lat/lon grid, which is the same for every init month.
        self._mask_cache: Optional[tuple] = None

        self._validate_config()

    # ------------------------------------------------------------------
    # validation / config helpers
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        required_attrs = [
            "field",
            "data_dir",
            "case_prefix",
            "members",
            "nlead",
            "init_months",
            "init_years",
            "region",
        ]
        for attr in required_attrs:
            if not hasattr(self.cfg, attr):
                raise AttributeError(f"cfg is missing required attribute: '{attr}'")

        if self.cfg.field is None or str(self.cfg.field).strip() == "":
            raise ValueError("cfg.field must be a non-empty string")

        if not isinstance(self.cfg.members, (list, tuple)) or len(self.cfg.members) == 0:
            raise ValueError("cfg.members must be a non-empty list/tuple")

        if not isinstance(self.cfg.init_months, (list, tuple)) or len(self.cfg.init_months) == 0:
            raise ValueError("cfg.init_months must be a non-empty list/tuple")

        if not isinstance(self.cfg.init_years, dict):
            raise ValueError("cfg.init_years must be a dict like {5: years, 11: years}")

        for month in self.cfg.init_months:
            if month not in self.cfg.init_years:
                raise ValueError(f"cfg.init_years is missing entry for init month {month}")

        if not isinstance(self.cfg.region, (list, tuple)) or len(self.cfg.region) != 4:
            raise ValueError("cfg.region must be [lon_w, lon_e, lat_s, lat_n]")

        if int(self.cfg.nlead) <= 0:
            raise ValueError("cfg.nlead must be positive")

    def _get_cfg(self, name: str, default):
        return getattr(self.cfg, name, default)

    @property
    def realm(self):
        return self._get_cfg("realm", "atm")

    @property
    def grid(self):
        return self._get_cfg("grid", "180x360_aave")

    @property
    def freq(self):
        return self._get_cfg("freq", "monthly")

    @property
    def ts_split(self):
        return self._get_cfg("ts_split", "2yr")

    @property
    def require_all_members(self):
        return self._get_cfg("require_all_members", True)

    @property
    def verify_field_name(self):
        return self._get_cfg("verify_field_name", True)

    @property
    def verify_coverage(self):
        return self._get_cfg("verify_coverage", True)

    @property
    def engine(self):
        return self._get_cfg("engine", None)

    @property
    def chunks(self):
        return self._get_cfg("chunks", {})

    @property
    def persist_intermediate(self):
        return self._get_cfg("persist_intermediate", False)

    @property
    def load_regional_series(self):
        return self._get_cfg("load_regional_series", True)

    @property
    def convert_ts_to_degC(self):
        return self._get_cfg("convert_ts_to_degC", True)

    # ------------------------------------------------------------------
    # core loaders / preprocessors
    # ------------------------------------------------------------------
    def load_model(self, init_month: int) -> xr.Dataset:
        years = self.cfg.init_years[init_month]
        init_tags = self.data_access.build_init_tags(years, init_month)

        kwargs = dict(
            data_dir=self.cfg.data_dir,
            case_prefix=self.cfg.case_prefix,
            members=self.cfg.members,
            init_tags=init_tags,
            field=self.cfg.field,
            nlead=self.cfg.nlead,
            chunks=self.chunks,
            realm=self.realm,
            grid=self.grid,
            freq=self.freq,
            ts_split=self.ts_split,
            require_all_members=self.require_all_members,
            verify_field_name=self.verify_field_name,
            verify_coverage=self.verify_coverage,
        )
        if self.engine is not None:
            kwargs["engine"] = self.engine

        ds = self.data_access.get_monthly_data(**kwargs)

        if self.cfg.field not in ds.data_vars:
            raise KeyError(
                f"Loaded dataset does not contain requested field '{self.cfg.field}'. "
                f"Available variables: {list(ds.data_vars)}"
            )

        ds = self._postprocess_model(ds)
        self._validate_model_dataset(ds, stage=f"monthly init={init_month}")
        return ds

    def _postprocess_model(self, ds: xr.Dataset) -> xr.Dataset:
        ds = ds.copy()

        if self.cfg.field == "TS" and self.convert_ts_to_degC:
            ds[self.cfg.field] = normalize_sst_to_degc(ds[self.cfg.field])

        return ds

    def _validate_model_dataset(self, ds: xr.Dataset, stage: str = "") -> None:
        da = ds[self.cfg.field]

        required_dims = {"Y", "L"}
        missing_dims = required_dims - set(da.dims)
        if missing_dims:
            raise ValueError(
                f"Field '{self.cfg.field}' is missing required dims {missing_dims} at stage '{stage}'. "
                f"Found dims: {da.dims}"
            )

        for dim in ("lat", "lon"):
            if dim not in da.dims and dim not in da.coords:
                raise ValueError(
                    f"Field '{self.cfg.field}' must include '{dim}' as a dim or coord at stage '{stage}'."
                )

        if "time" not in ds:
            raise ValueError(f"Dataset is missing 'time' variable at stage '{stage}'")

    def _get_or_build_mask(
        self, da: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Return (landmask, oceanmask, weights), computing once and caching."""
        if self._mask_cache is None:
            landmask = build_landmask(da, self.spatial)
            oceanmask = ~landmask.fillna(0).astype(bool)
            weights = compute_weights(
                self.data_access,
                da,
                self.cfg.region,
                oceanmask,
            )
            self._mask_cache = (landmask, oceanmask, weights)
        return self._mask_cache

    def compute_regional_index(
        self, ds: xr.Dataset
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        da = ds[self.cfg.field]
        landmask, oceanmask, weights = self._get_or_build_mask(da)

        reg = compute_regional_mean(da, weights)
        reg = maybe_load(reg, do_load=self.load_regional_series)

        return reg, weights, landmask, oceanmask

    def compute_seasonal_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        ds_seas = self.cal.mon_to_seas_dask(ds)
        ds_seas = maybe_persist(ds_seas, do_persist=self.persist_intermediate)

        self._validate_model_dataset(ds_seas, stage="seasonal")
        return ds_seas

    # ------------------------------------------------------------------
    # per-init execution
    # ------------------------------------------------------------------
    def run_single_init(self, init_month: int) -> Dict[str, Any]:
        ds = self.load_model(init_month)

        # monthly index
        reg_mon, weights, landmask, oceanmask = self.compute_regional_index(ds)

        # seasonal index
        ds_seas = self.compute_seasonal_dataset(ds)
        reg_seas = compute_regional_mean(ds_seas[self.cfg.field], weights)
        reg_seas = maybe_load(reg_seas, do_load=self.load_regional_series)

        out = {
            "mon": reg_mon,
            "seas": reg_seas,
            "time_mon": ds["time"],
            "time_seas": ds_seas["time"],
            "weights": weights,
            "landmask": landmask,
            "oceanmask": oceanmask,
            "ds_mon": ds,
            "ds_seas": ds_seas,
        }

        return out

    # ------------------------------------------------------------------
    # all-init execution
    # ------------------------------------------------------------------
    def run(self) -> Dict[int, Dict[str, Any]]:
        """
        Run diagnostics for all init months.

        Strategy
        --------
        Phase 1 — open all datasets and build **lazy** regional-mean graphs.
          * The land-sea mask and spatial weights are computed once (on the
            first iteration) and reused via ``_get_or_build_mask``.
          * The seasonal rolling-mean is persisted on the Dask cluster so it
            is computed in the background while the next init month is set up.
        Phase 2 — a **single** ``dask.compute`` call materialises every
          regional mean at once, letting the Dask scheduler parallelise
          across init months and optimise the combined task graph.
        """
        months = self.cfg.init_months

        # ---- Phase 1: build lazy graphs --------------------------------
        lazy_mons: Dict[int, xr.DataArray] = {}
        lazy_seas: Dict[int, xr.DataArray] = {}
        metadata:  Dict[int, Dict[str, Any]] = {}

        for init_month in months:
            ds  = self.load_model(init_month)
            da  = ds[self.cfg.field]

            landmask, oceanmask, weights = self._get_or_build_mask(da)

            lazy_mons[init_month] = compute_regional_mean(da, weights)

            ds_seas = self.compute_seasonal_dataset(ds)   # may .persist()
            lazy_seas[init_month] = compute_regional_mean(
                ds_seas[self.cfg.field], weights
            )

            metadata[init_month] = {
                "time_mon":  ds["time"].load(),
                "time_seas": ds_seas["time"].load(),
                "weights":   weights,
                "landmask":  landmask,
                "oceanmask": oceanmask,
                "ds_mon":    ds,
                "ds_seas":   ds_seas,
            }

        # ---- Phase 2: compute everything in one Dask pass --------------
        all_mons = [lazy_mons[m] for m in months]
        all_seas = [lazy_seas[m] for m in months]
        computed = dask.compute(*all_mons, *all_seas)

        n = len(months)
        results: Dict[int, Dict[str, Any]] = {}
        for i, m in enumerate(months):
            results[m] = dict(metadata[m])
            results[m]["mon"]  = computed[i]
            results[m]["seas"] = computed[n + i]

        return results

    # ------------------------------------------------------------------
    # optional downstream helpers
    # ------------------------------------------------------------------
    def remove_drift_for_results(self, results: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Apply lead-time-dependent drift removal to monthly and seasonal regional series.

        Requires cfg.climy0 and cfg.climy1.
        """
        if not hasattr(self.cfg, "climy0") or not hasattr(self.cfg, "climy1"):
            raise AttributeError("cfg must define climy0 and climy1 for drift removal")

        out = {}

        for init_month, res in results.items():
            if "mon" not in res or "seas" not in res:
                raise KeyError(
                    f"Results for init month {init_month} must contain 'mon' and 'seas'. "
                    "Run run() or run_single_init() first."
                )
            mon_dd, mon_drift = remove_model_drift(
                self.stats,
                res["mon"],
                res["time_mon"],
                self.cfg.climy0,
                self.cfg.climy1,
            )
            seas_dd, seas_drift = remove_model_drift(
                self.stats,
                res["seas"],
                res["time_seas"],
                self.cfg.climy0,
                self.cfg.climy1,
            )

            tmp = dict(res)
            tmp["mon_dd"] = mon_dd
            tmp["mon_drift"] = mon_drift
            tmp["seas_dd"] = seas_dd
            tmp["seas_drift"] = seas_drift
            out[init_month] = tmp

        return out

    def compute_skill_for_results(
        self,
        results: Dict[int, Dict[str, Any]],
        obs_mon: xr.DataArray,
        obs_seas: xr.DataArray,
        seasonal_nlead: int = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute monthly and seasonal skill for drift-corrected results.

        Expects results to already contain:
          - mon_dd
          - seas_dd
        """
        if not hasattr(self.cfg, "climy0") or not hasattr(self.cfg, "climy1"):
            raise AttributeError("cfg must define climy0 and climy1 for skill computation")

        if seasonal_nlead is None:
            # often 24 monthly leads -> 8 seasonal leads
            seasonal_nlead = self._get_cfg("seasonal_nlead", 8)

        out = {}

        for init_month, res in results.items():
            if "mon_dd" not in res or "seas_dd" not in res:
                raise KeyError(
                    f"Results for init month {init_month} must contain 'mon_dd' and 'seas_dd'. "
                    "Run remove_drift_for_results() first."
                )
            if "time_mon" not in res or "time_seas" not in res:
                raise KeyError(
                    f"Results for init month {init_month} must contain 'time_mon' and 'time_seas'. "
                    "Run run() or run_single_init() first."
                )

            skill_mon = compute_skill(
                self.stats,
                res["mon_dd"],
                res["time_mon"],
                obs_mon,
                self.cfg.climy0,
                self.cfg.climy1,
                self.cfg.nlead,
                monthly=True,
            )

            skill_seas = compute_skill(
                self.stats,
                res["seas_dd"],
                res["time_seas"],
                obs_seas,
                self.cfg.climy0,
                self.cfg.climy1,
                seasonal_nlead,
                monthly=False,
            )

            tmp = dict(res)
            tmp["skill_mon"] = skill_mon
            tmp["skill_seas"] = skill_seas
            out[init_month] = tmp

        return out
