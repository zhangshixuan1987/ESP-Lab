"""Reusable monthly gridded ACC/nRMSE cache production."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from esp_lab import data_access_e3sm, data_access_obs, land_input_cache, land_skill, stats
from esp_lab.leadtime_prepared_cache import cache_status
from esp_lab.leadtime_workflow import compute_monthly_skill_lead_range, select_lead_range
from esp_lab.paths import leadtime_acc_dir
from esp_lab.utils import regrid_utils as regrid
from esp_lab.utils.netcdf_utils import atomic_to_netcdf, load_netcdf
from esp_lab.utils.filename_utils import source_init_prefix


MONTHLY_SKILL_VARIABLES = (
    "corr", "pval", "rmse", "msss", "rpc", "sig_obs", "sig_sig", "sig_tot", "s2t",
)


def produce_monthly_skill_cache(
    model_anomaly: xr.DataArray,
    valid_time: xr.DataArray,
    observations: xr.DataArray,
    *,
    climatology_years: tuple[int, int],
    cache_path: str | Path,
    expected_attrs: Mapping[str, object],
    force_compute: bool = False,
    lead_start: int = 1,
    lead_end: int | None = None,
    detrend: bool = True,
    observations_are_anomalies: bool = False,
    write_options: Mapping | None = None,
) -> xr.Dataset:
    """Reuse or create a monthly gridded skill product with ``L=1..24``.

    Callers own input preparation: model data must already be monthly and
    lead-dependent-drift-corrected, while observations remain monthly fields.
    This keeps the atmospheric and land archive conventions in their separate
    drivers while guaranteeing one ACC/nRMSE cache contract.
    """
    path = Path(cache_path)
    attrs = {**expected_attrs, "frequency": "monthly", "cache_kind": "monthly_skill_v1"}
    compatible, _ = cache_status(
        path, expected_attrs=attrs, required_variables=MONTHLY_SKILL_VARIABLES
    )
    if compatible and not force_compute:
        return load_netcdf(path)

    start, end = map(int, climatology_years)
    model, time = select_lead_range(
        model_anomaly, valid_time, lead_start,
        int(model_anomaly.sizes["L"]) if lead_end is None else lead_end,
    )
    # Leads whose target calendar month never occurs in the reference (e.g.
    # C3S SWE has no June-September fields) cannot be verified; keep them in
    # the L=1..24 contract as missing values instead of failing the product.
    reference_months = set(np.unique(observations["time"].dt.month.values).tolist())
    lead_months = time.dt.month.max("Y").values
    keep = [i for i, month in enumerate(lead_months)
            if np.isfinite(month) and int(month) in reference_months]
    if not keep:
        raise ValueError("No monthly lead has a target month present in the reference data.")
    skill = compute_monthly_skill_lead_range(
        model.isel(L=keep), time.isel(L=keep), observations, str(start), str(end),
        lead_start=1, lead_end=len(keep), resamp=0,
        detrend=detrend, is_anomaly=observations_are_anomalies,
    )
    if len(keep) < model.sizes["L"]:
        skipped = [int(v) for i, v in enumerate(model["L"].values) if i not in keep]
        skill = skill.reindex(L=model["L"].values)
        attrs = {**attrs, "unverifiable_leads": ",".join(map(str, skipped))}
    skill = skill.assign_attrs(attrs).compute()
    if not {"L", "lat", "lon"}.issubset(skill["corr"].dims):
        raise ValueError("Monthly skill must contain corr(L, lat, lon).")
    atomic_to_netcdf(skill, path, **dict(write_options or {}))
    return skill


def _skill_attrs(
    *, field: str, source_tag: str, init_month: int,
    initialization_years: tuple[int, int], climatology_years: tuple[int, int],
    reference_product: str, reference_variable: str,
) -> dict[str, object]:
    return {
        "field": field,
        "source_tag": source_tag,
        "initialization_month": int(init_month),
        "initialization_year_start": int(initialization_years[0]),
        "initialization_year_end": int(initialization_years[1]),
        "climatology_year_start": int(climatology_years[0]),
        "climatology_year_end": int(climatology_years[1]),
        "reference_product": reference_product,
        "reference_variable": reference_variable,
        "lead_units": "months",
        "regridding_method": "conservative",
        "target_grid": "5x5deg",
    }


def _monthly_path(
    root: str | Path, source_tag: str, cache_realm: str, field: str,
    init_month: int, year_end: int,
) -> Path:
    directory = leadtime_acc_dir(
        source_tag, "skill_monthly", cache_realm, field, root=Path(root)
    )
    return directory / (
        f"{source_init_prefix(source_tag, init_month)}_{field}_monthly_skill_"
        f"y1980-{year_end}_clim1981-2010_l1-24.nc"
    )


def _cache_is_current(path: Path, attrs: Mapping[str, object]) -> bool:
    expected = {**attrs, "frequency": "monthly", "cache_kind": "monthly_skill_v1"}
    compatible, _ = cache_status(
        path, expected_attrs=expected, required_variables=MONTHLY_SKILL_VARIABLES
    )
    return compatible


def _on_grid(data: xr.DataArray, target: xr.Dataset, *, field: str) -> xr.DataArray:
    if land_input_cache.is_target_grid(data, target):
        return data.copy()
    mapper = regrid.make_regridder(
        data.to_dataset(name=field), target, method="conservative", periodic=True
    )
    return mapper(data).rename(field)


def build_atmospheric_monthly_skill_caches(
    *, field: str, source_tags: Mapping[str, str], cases: Mapping[str, Mapping],
    variable_config: Mapping[str, object], diagnostic_root: str | Path,
    raw_model_root: str | Path, observation_root: str | Path,
    year_end: int, force_compute: bool = False,
    initialization_year_start: int = 1980,
    initialization_months: Sequence[int] = (5, 11),
    climatology_years: tuple[int, int] = (1981, 2010),
    members: Sequence[str] = tuple(f"EN{i:02d}" for i in range(10)),
    nlead: int = 24, cache_realm: str = "atm",
) -> list[Path]:
    """Build independent monthly atmosphere/ocean skill caches.

    Every archive and variable choice is supplied by the calling driver.  The
    function therefore contains reusable processing only and has no notebook
    dependency.  Optional ``variable_config`` keys:

    * ``obs_path_pattern``: read the reference from an explicit file or glob
      (e.g. EN4 yearly files) instead of ``obs_product`` under the obs root.
    * ``pre_regrid_convert``: callable applied to the model field on its
      archive grid before regridding (e.g. OHC700 J -> J m-2).
    """
    cfg = dict(variable_config)
    required = {
        "obs_product", "obs_variable", "obs_year_start", "obs_year_end",
        "model_convert", "obs_convert",
    }
    missing_cfg = required - set(cfg)
    if missing_cfg:
        raise ValueError(f"Monthly configuration for {field} lacks {sorted(missing_cfg)}")
    case_by_tag = {str(value["cache_tag"]): value for value in cases.values()}
    unknown = set(source_tags.values()) - set(case_by_tag)
    if unknown:
        raise ValueError(f"No E3SM case configuration for source tags {sorted(unknown)}")

    years = np.arange(int(initialization_year_start), int(year_end) + 1)
    target = regrid.make_latlon_grid(5.0, 5.0)
    obs_window = dict(
        start_year=str(cfg["obs_year_start"]), end_year=str(cfg["obs_year_end"]),
        chunks={"time": 24, "lat": 90, "lon": 180},
    )
    if cfg.get("obs_path_pattern"):
        obs = data_access_obs.get_monthly_data_from_pattern(
            str(cfg["obs_path_pattern"]), str(cfg["obs_variable"]), **obs_window
        )
    else:
        obs = data_access_obs.get_monthly_data(
            obs_dir=str(observation_root), product=str(cfg["obs_product"]), field=field,
            field_map={field: str(cfg["obs_variable"])}, **obs_window,
        )
    pre_regrid_convert = cfg.get("pre_regrid_convert") or (lambda data: data)
    try:
        obs_name = str(cfg["obs_variable"])
        if obs_name not in obs:
            raise KeyError(f"Observation dataset does not contain {obs_name!r}")
        obs_ready = cfg["obs_convert"](_on_grid(obs[obs_name], target, field=obs_name))
        outputs: list[Path] = []
        for source_tag in source_tags.values():
            case = case_by_tag[source_tag]
            for init_month in map(int, initialization_months):
                attrs = _skill_attrs(
                    field=field, source_tag=source_tag, init_month=init_month,
                    initialization_years=(int(years[0]), int(years[-1])),
                    climatology_years=climatology_years,
                    reference_product=str(cfg["obs_product"]),
                    reference_variable=obs_name,
                )
                path = _monthly_path(
                    diagnostic_root, source_tag, cache_realm, field,
                    init_month, int(year_end),
                )
                if not force_compute and _cache_is_current(path, attrs):
                    print(f"Reusing monthly skill cache: {path}")
                    outputs.append(path)
                    continue
                print(
                    f"Building monthly skill cache: {source_tag}, "
                    f"init={init_month:02d}, field={field}"
                )
                monthly = data_access_e3sm.get_monthly_data(
                    data_dir=str(raw_model_root),
                    case_prefix=str(case["case_prefix"]), members=list(members),
                    init_tags=data_access_e3sm.build_init_tags(years, init_month),
                    field=str(cfg.get("model_variable", field)), nlead=int(nlead),
                    chunks={}, realm=str(cfg.get("model_realm", "atm")),
                    grid="180x360_aave", freq="monthly", ts_split="2yr",
                    engine="netcdf4", require_all_members=True,
                    verify_field_name=True, verify_coverage=True,
                )
                try:
                    model_name = str(cfg.get("model_variable", field))
                    model = pre_regrid_convert(monthly[model_name]).rename(field)
                    model = cfg["model_convert"](_on_grid(model, target, field=field))
                    anomaly, _ = stats.remove_drift(
                        model, monthly["time"], *map(int, climatology_years)
                    )
                    produce_monthly_skill_cache(
                        anomaly, monthly["time"], obs_ready,
                        climatology_years=climatology_years, cache_path=path,
                        expected_attrs=attrs, force_compute=True,
                        lead_start=1, lead_end=int(nlead), detrend=True,
                    ).close()
                    outputs.append(path)
                finally:
                    monthly.close()
        return outputs
    finally:
        obs.close()


def _load_land_reference(
    cfg: Mapping[str, object], field: str,
    depth_range_m: tuple[float, float] | None,
) -> tuple[xr.Dataset, xr.DataArray]:
    paths = land_input_cache.resolve_reference_files(str(cfg["path"]))
    source = (
        xr.open_mfdataset([str(path) for path in paths], combine="by_coords")
        if len(paths) > 1 else xr.open_dataset(paths[0])
    )
    variable = str(cfg["variable"])
    if variable not in source:
        source.close()
        raise KeyError(f"Land reference does not contain {variable!r}")
    data = land_input_cache.standardize_reference_coordinates(source[variable])
    data = (data * float(cfg.get("scale", 1.0)) + float(cfg.get("offset", 0.0))).rename(field)
    if cfg.get("mask_negative_categorical_flags", False):
        data = land_skill.mask_c3s_swe_flags(data)
    data.attrs["units"] = str(cfg["output_units"])
    if field == "H2OSOI":
        represented = cfg.get("represented_depth_range_m")
        if represented is None or depth_range_m is None or not np.allclose(
            represented, depth_range_m, rtol=0.0, atol=1e-6
        ):
            source.close()
            raise ValueError("Reference and requested H2OSOI depth intervals differ")
        data.attrs.update(depth_top_m=depth_range_m[0], depth_bottom_m=depth_range_m[1])
    return source, data


def build_land_monthly_skill_caches(
    *, field: str, source_tags: Mapping[str, str], cases: Mapping[str, Mapping],
    reference_config: Mapping[str, object], diagnostic_root: str | Path,
    raw_model_root: str | Path, year_end: int, force_compute: bool = False,
    initialization_year_start: int = 1980,
    initialization_months: Sequence[int] = (5, 11),
    climatology_years: tuple[int, int] = (1981, 2010),
    members: Sequence[str] = tuple(f"EN{i:02d}" for i in range(10)),
    nlead: int = 24, depth_range_m: tuple[float, float] | None = None,
    cache_realm: str = "lnd",
) -> list[Path]:
    """Build independent monthly land skill caches from raw model/reference data."""
    case_by_tag = {str(value["cache_tag"]): value for value in cases.values()}
    unknown = set(source_tags.values()) - set(case_by_tag)
    if unknown:
        raise ValueError(f"No E3SM case configuration for source tags {sorted(unknown)}")
    years = np.arange(int(initialization_year_start), int(year_end) + 1)
    target = land_input_cache.target_grid(5.0, 5.0)
    source, reference = _load_land_reference(reference_config, field, depth_range_m)
    try:
        reference = _on_grid(reference, target, field=field)
        outputs: list[Path] = []
        for source_tag in source_tags.values():
            case = case_by_tag[source_tag]
            for init_month in map(int, initialization_months):
                attrs = _skill_attrs(
                    field=field, source_tag=source_tag, init_month=init_month,
                    initialization_years=(int(years[0]), int(years[-1])),
                    climatology_years=climatology_years,
                    reference_product=str(reference_config["product"]),
                    reference_variable=str(reference_config["variable"]),
                )
                path = _monthly_path(
                    diagnostic_root, source_tag, cache_realm, field,
                    init_month, int(year_end),
                )
                if not force_compute and _cache_is_current(path, attrs):
                    print(f"Reusing monthly skill cache: {path}")
                    outputs.append(path)
                    continue
                print(
                    f"Building monthly skill cache: {source_tag}, "
                    f"init={init_month:02d}, field={field}"
                )
                monthly = land_skill.load_e3sm_land_monthly(
                    data_dir=str(raw_model_root), case_prefix=str(case["case_prefix"]),
                    members=list(members),
                    init_tags=data_access_e3sm.build_init_tags(years, init_month),
                    field=field, nlead=int(nlead), chunks={},
                )
                try:
                    selection = ({
                        "soil_depth_range_m": depth_range_m,
                        "soil_output": "water_equivalent_mm",
                    } if field == "H2OSOI" else {})
                    model = land_skill.prepare_land_field(monthly, field, **selection)
                    model = _on_grid(model, target, field=field)
                    anomaly, _ = stats.remove_drift(
                        model, monthly["time"], *map(int, climatology_years)
                    )
                    produce_monthly_skill_cache(
                        anomaly, monthly["time"], reference,
                        climatology_years=climatology_years, cache_path=path,
                        expected_attrs=attrs, force_compute=True,
                        lead_start=1, lead_end=int(nlead), detrend=True,
                        observations_are_anomalies=bool(reference_config.get("is_anomaly", False)),
                    ).close()
                    outputs.append(path)
                finally:
                    monthly.close()
        return outputs
    finally:
        source.close()


__all__ = [
    "MONTHLY_SKILL_VARIABLES", "produce_monthly_skill_cache",
    "build_atmospheric_monthly_skill_caches", "build_land_monthly_skill_caches",
]
