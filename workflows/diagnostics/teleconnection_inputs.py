"""On-demand preparation of inputs consumed by teleconnection diagnostics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from esp_lab import data_access_e3sm, data_access_obs, land_input_cache, land_skill, stats
from esp_lab.leadtime_skill_cache import source_fingerprint
from esp_lab.leadtime_workflow import observation_source_identity
from esp_lab.paths import leadtime_acc_dir
from esp_lab.leadtime_prepared_cache import (
    build_prepared_skill_dataset,
    expected_prepared_skill_attrs,
    grid_token,
    prepared_skill_path,
    units_token,
    write_prepared_skill_dataset,
)
from esp_lab.utils import calendar_utils, regrid_utils
from esp_lab.utils.netcdf_utils import atomic_to_netcdf
from esp_lab.utils.unit_conversion import (
    convert_kelvin_to_celsius,
    convert_pa_to_hpa,
    convert_precip_mps_to_mmday,
    no_unit_conversion,
)


ATMOSPHERIC_VARIABLES: dict[str, dict[str, Any]] = {
    "TREFHT": {
        "realm": "atm", "model_field": "TREFHT", "obs_product": "ERA5",
        "obs_field": "tas", "obs_years": (1979, 2019),
        "model_convert": convert_kelvin_to_celsius,
        "obs_convert": convert_kelvin_to_celsius,
    },
    "TS": {
        "realm": "atm", "model_field": "TS", "obs_product": "ERA5",
        "obs_field": "ts", "obs_years": (1979, 2019),
        "model_convert": convert_kelvin_to_celsius,
        "obs_convert": convert_kelvin_to_celsius,
    },
    "PRECT": {
        "realm": "atm", "model_field": "PRECT", "obs_product": "GPCP_v2.3",
        "obs_field": "PRECT", "obs_years": (1979, 2017),
        "model_convert": convert_precip_mps_to_mmday,
        "obs_convert": lambda da: no_unit_conversion(da, units="mm/day"),
    },
    "PSL": {
        "realm": "atm", "model_field": "PSL", "obs_product": "ERA5",
        "obs_field": "psl", "obs_years": (1979, 2019),
        "model_convert": convert_pa_to_hpa,
        "obs_convert": convert_pa_to_hpa,
    },
    "SST": {
        "realm": "ocn", "model_field": "timeMonthly_avg_activeTracers_temperature",
        "obs_product": "HadISST2", "obs_field": "sst", "obs_years": (1979, 2022),
        "model_convert": lambda da: no_unit_conversion(da, units=r"$^\circ$C"),
        "obs_convert": lambda da: no_unit_conversion(da, units=r"$^\circ$C"),
    },
}


LAND_REFERENCES: dict[str, dict[str, Any]] = {
    "H2OSNO": {
        "path": "/global/cfs/cdirs/e3sm/zhan391/data/C3S_SWE/1x1/monthly/swe_*.nc",
        "variable": "swe", "product": "C3S_SWE",
        "documentation": "Copernicus Climate Change Service snow water equivalent",
        "source_revision": "c3s_swe_archive_2026-08-26",
        "scale": 1.0, "offset": 0.0, "output_units": "mm",
        "is_anomaly": False, "already_seasonal": False,
        "mask_negative_categorical_flags": True,
        "require_complete_calendar_months": True,
        "retain_missing_seasons": True,
        "restrict_model_to_reference_months": True,
    },
    "TWS": {
        "path": "/global/cfs/cdirs/e3sm/zhan391/data/C3S_TWSA/1x1/monthly/twsa_*.nc",
        "variable": "twsa", "product": "C3S_TWSA",
        "documentation": "C3S terrestrial water storage anomaly v1.0; doi:10.5880/GFZ.C3S_TWSA_v1.0",
        "source_revision": "c3s_twsa_v1.0_archive_2026-09-09",
        "scale": 1.0, "offset": 0.0, "output_units": "mm",
        "is_anomaly": True, "already_seasonal": False,
        "require_complete_calendar_months": True,
        "retain_missing_seasons": True,
        "restrict_model_to_reference_months": True,
    },
    "H2OSOI": {
        "path": "/global/cfs/cdirs/e3sm/zhan391/data/CPC_SOM/monthly/soilw_*.nc",
        "variable": "soilw", "product": "CPC_Soil_Moisture_V2",
        "documentation": "https://www.cpc.ncep.noaa.gov/soilmst/descrip.htm",
        "source_revision": "cpc_soil_moisture_v2_archive_2026-08-26",
        "scale": 1.0, "offset": 0.0, "output_units": "mm",
        "is_anomaly": False, "already_seasonal": False,
        "vertical_dim": None, "layer_bounds_variable": None,
        "represented_depth_range_m": (0.0, 1.6),
        "restrict_model_to_reference_months": False,
    },
}


def _input_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("inputs", {})


def _years(config: Mapping[str, Any]) -> list[int]:
    settings = _input_settings(config)
    start, end = settings.get("initialization_years", (1980, 2011))
    return list(range(int(start), int(end) + 1))


def _regrid(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config["regrid"]
    return settings


def _target_grid_identity(config: Mapping[str, Any]) -> str:
    settings = _regrid(config)
    return (
        f"latlon_{float(settings['target_dlat'])}x{float(settings['target_dlon'])}_"
        f"periodic-{settings['periodic']}"
    )


def ensure_sst_indices(
    config: Mapping[str, Any], systems: Sequence[str], *,
    include_observation: bool = True, force: bool = False,
) -> None:
    """Run the shared SST-index processor for the selected observations and systems."""
    selection = config["selection"]
    settings = _input_settings(config)
    root = Path(config["paths"]["diag_root"])
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "run_process_sst_index.py"
    years = _years(config)
    climy0, climy1 = selection["climatology_years"]
    common = [
        sys.executable, str(script), "--regions", str(selection["upstream_index"]),
        "--outdir", str(root), "--obs-outdir",
        str(root / "observations" / "sst_index" / "timeseries"),
        "--e3sm-data-dir", str(settings.get(
            "raw_model_root", "/global/cfs/cdirs/e3sm/S2S2D/post_process"
        )),
        "--init-months", *map(str, selection["init_months"]),
        "--year-start", str(years[0]), "--year-end", str(years[-1]),
        "--climy0", str(climy0), "--climy1", str(climy1),
        "--nlead", str(settings.get("monthly_nlead", 24)),
        "--e3sm-nens", str(settings.get("ensemble_member_count", 10)),
        "--workers", str(settings.get("workers", 4)),
    ]
    if selection["upstream_index"] == "ELI":
        eli_grid = str(settings.get("eli_grid", "regridded"))
        if eli_grid not in {"regridded", "native"}:
            raise ValueError("inputs.eli_grid must be 'regridded' or 'native'")
        common.extend(["--eli-grid", eli_grid])
        if settings.get("eli_mesh_file"):
            common.extend(["--mesh-file", str(settings["eli_mesh_file"])])
        if settings.get("eli_raw_model_root"):
            common.extend(["--e3sm-raw-dir", str(settings["eli_raw_model_root"])])
    if force:
        common.append("--force")
    if not settings.get("sst_land_mask", True):
        common.append("--no-sst-land-mask")

    if include_observation:
        subprocess.run([*common, "--sources", "obs"], check=True, cwd=repo_root)
    from workflows.diagnostics.sst_teleconnections import E3SM_CASES

    for system in systems:
        case = E3SM_CASES[system]
        subprocess.run(
            [
                *common, "--sources", "e3sm",
                "--e3sm-case-prefix", case["case_prefix"],
                "--e3sm-cache-tag", case["cache_tag"],
                "--e3sm-display-name", case["display_name"],
            ],
            check=True, cwd=repo_root,
        )


def _atmospheric_expected(
    config: Mapping[str, Any], system: str, init_month: int, variable: str
) -> tuple[Path, Mapping[str, Any]]:
    from workflows.diagnostics.sst_teleconnections import E3SM_CASES

    case = E3SM_CASES[system]
    selection = config["selection"]
    settings = _input_settings(config)
    years = _years(config)
    clim = tuple(map(int, selection["climatology_years"]))
    spec = ATMOSPHERIC_VARIABLES[variable]
    source_identity = source_fingerprint(
        {
            "archive_root": str(settings.get(
                "raw_model_root", "/global/cfs/cdirs/e3sm/S2S2D/post_process"
            )),
            "case_prefix": case["case_prefix"], "realm": spec["realm"],
            "grid": "180x360_aave", "frequency": "monthly",
            "time_split": "2yr", "field": spec["model_field"],
            "years": years, "init_month": int(init_month),
        },
        source_revision=str(case.get("source_revision", "post_process_v1")),
    )
    provenance = dict(
        source_data_identity=source_identity,
        case_prefix=case["case_prefix"], requested_years=years,
        target_grid=_target_grid_identity(config),
        regridding_method=str(_regrid(config)["method"]),
        ensemble_member_count=int(settings.get("ensemble_member_count", 10)),
        lead_count=int(settings.get("monthly_nlead", 24)) // 3,
        unit_conversion_version=(
            "leadtime_skill_units_v2_unmasked_sst"
            if variable == "SST" else "leadtime_skill_units_v1"
        ),
    )
    path = prepared_skill_path(
        case["cache_tag"], spec["realm"], variable, int(init_month), clim,
        root=Path(config["paths"]["diag_root"]), **provenance,
    )
    expected = expected_prepared_skill_attrs(
        source=case["cache_tag"], component=spec["realm"], variable=variable,
        init_month=int(init_month), climatology_years=clim, **provenance,
    )
    return path, expected


def prepare_atmospheric_model(
    config: Mapping[str, Any], system: str, init_month: int, variable: str
) -> Path:
    """Create one analysis-ready atmospheric anomaly cache from raw E3SM output."""
    from workflows.diagnostics.sst_teleconnections import E3SM_CASES

    if variable not in ATMOSPHERIC_VARIABLES:
        raise ValueError(f"No atmospheric input preparer is configured for {variable}")
    case = E3SM_CASES[system]
    selection = config["selection"]
    settings = _input_settings(config)
    spec = ATMOSPHERIC_VARIABLES[variable]
    years = _years(config)
    members = [f"EN{i:02d}" for i in range(int(settings.get("ensemble_member_count", 10)))]
    nlead = int(settings.get("monthly_nlead", 24))
    monthly = data_access_e3sm.get_monthly_data(
        data_dir=str(settings.get("raw_model_root", "/global/cfs/cdirs/e3sm/S2S2D/post_process")),
        case_prefix=case["case_prefix"], members=members,
        init_tags=data_access_e3sm.build_init_tags(years, int(init_month)),
        field=spec["model_field"], nlead=nlead,
        chunks=dict(settings.get("raw_model_chunks", {})), realm=spec["realm"],
        grid="180x360_aave", freq="monthly", ts_split="2yr",
        require_all_members=True, verify_field_name=True, verify_coverage=True,
        engine=str(settings.get("engine", "netcdf4")),
    )
    regridder = None
    try:
        seasonal = calendar_utils.mon_to_seas_dask(monthly)
        dest = regrid_utils.make_latlon_grid(
            dlat=float(_regrid(config)["target_dlat"]),
            dlon=float(_regrid(config)["target_dlon"]),
        )
        regridder = regrid_utils.make_regridder(
            seasonal, dest, method=str(_regrid(config)["method"]),
            periodic=bool(_regrid(config)["periodic"]),
        )
        ready = spec["model_convert"](regridder(seasonal[spec["model_field"]]))
        time = seasonal["time"].load()
        climy0, climy1 = map(int, selection["climatology_years"])
        anomaly, climatology = stats.remove_drift(ready, time, climy0, climy1)
        path, expected = _atmospheric_expected(config, system, init_month, variable)
        prepared = build_prepared_skill_dataset(
            anomaly, climatology, time,
            source=expected["source"], component=expected["component"],
            variable=variable, init_month=int(init_month),
            climatology_years=(climy0, climy1),
            source_data_identity=expected["source_data_identity"],
            case_prefix=expected["case_prefix"], requested_years=years,
            target_grid=expected["target_grid"],
            regridding_method=expected["regridding_method"],
            ensemble_member_count=len(members), lead_count=nlead // 3,
            unit_conversion_version=expected["unit_conversion_version"],
        )
        return write_prepared_skill_dataset(prepared, path)
    finally:
        monthly.close()


def prepare_atmospheric_observation(
    config: Mapping[str, Any], variable: str, *, force: bool = False
) -> Path:
    """Create one seasonal, regridded observational field cache."""
    if variable not in ATMOSPHERIC_VARIABLES:
        raise ValueError(f"No atmospheric observation preparer is configured for {variable}")
    spec = ATMOSPHERIC_VARIABLES[variable]
    settings = _input_settings(config)
    root = Path(config["paths"]["diag_root"])
    if not force:
        from workflows.diagnostics.sst_teleconnections import atmospheric_candidates, select_cache

        _, candidates = atmospheric_candidates("E3SM-FOSIRL", 1, variable, diag_root=root)
        try:
            return select_cache(
                candidates, required_vars=("observation",),
                expected_attrs={
                    "variable": variable,
                    "target_grid": _target_grid_identity(config),
                    "regridding_method": str(_regrid(config)["method"]),
                },
                description=f"prepared {variable} observation",
                allow_ambiguous=True,
            )
        except FileNotFoundError:
            pass
    obs_root = str(settings.get(
        "observation_root", "/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series"
    ))
    obs_start, obs_end = spec["obs_years"]
    obs_path_pattern = f"{obs_root}/{spec['obs_product']}/{spec['obs_field']}_*.nc"
    # Same identity as the 1a notebooks, so either workflow reuses the other's cache.
    identity = observation_source_identity(
        archive_root=obs_root, product=spec["obs_product"], variable=spec["obs_field"],
        years=(obs_start, obs_end), path_pattern=obs_path_pattern, mode="inventory",
        configured_revision=str(settings.get("observation_revision", "obs_archive_v1")),
        snapshot_dir=root / "tmp" / "source_inventory_snapshots",
        paths=data_access_obs.resolve_glob_files(obs_path_pattern),
    )
    expected = {
        "cache_kind": "unmasked_prepared_observations_v1",
        "observation_data_identity": identity, "variable": variable,
        "target_grid": _target_grid_identity(config),
        "regridding_method": str(_regrid(config)["method"]),
        "unit_conversion_version": (
            "leadtime_skill_units_v2_unmasked_sst"
            if variable == "SST" else "leadtime_skill_units_v1"
        ),
    }
    # Same fixed name as the 1a notebooks; provenance is in `expected`, stored
    # in the file's attributes.
    path = leadtime_acc_dir(
        "observations", "prepared_skill", spec["realm"], variable,
        f"{spec['obs_product']}_{variable}_seasonal_"
        f"{grid_token(expected['target_grid'])}_"
        f"{units_token(expected['unit_conversion_version'])}.nc",
        root=root,
    )
    chunks = dict(settings.get("observation_chunks", {"time": 24, "lat": 90, "lon": 180}))
    monthly = data_access_obs.get_monthly_data_from_pattern(
        obs_path_pattern, field=spec["obs_field"],
        start_year=str(obs_start), end_year=str(obs_end), chunks=chunks,
    )
    try:
        seasonal = data_access_obs.mon_to_seas_obs(
            monthly, var=spec["obs_field"], field_map={variable: spec["obs_field"]}
        )
        dest = regrid_utils.make_latlon_grid(
            dlat=float(_regrid(config)["target_dlat"]),
            dlon=float(_regrid(config)["target_dlon"]),
        )
        regridder = regrid_utils.make_regridder(
            monthly, dest, method=str(_regrid(config)["method"]),
            periodic=bool(_regrid(config)["periodic"]),
        )
        observation = spec["obs_convert"](regridder(seasonal)).rename("observation")
        dataset = observation.to_dataset()
        dataset.attrs.update(expected)
        return atomic_to_netcdf(dataset, path)
    finally:
        monthly.close()


def _land_snapshot_dir(root: Path) -> Path:
    tmp_path = root / "tmp" / "source_inventory_snapshots" / "land"
    legacy_path = root / "source_inventory_snapshots" / "land"
    if not tmp_path.exists() and legacy_path.exists():
        return legacy_path
    return tmp_path


def prepare_land_inputs(
    config: Mapping[str, Any], variable: str, pairs: Sequence[tuple[str, int]], *,
    force: bool = False,
) -> None:
    """Create selected land reference and model caches through the shared 1b APIs."""
    from workflows.diagnostics.sst_teleconnections import E3SM_CASES

    if variable not in LAND_REFERENCES:
        raise ValueError(f"No land input preparer is configured for {variable}")
    cfg = LAND_REFERENCES[variable]
    settings = _input_settings(config)
    selection = config["selection"]
    root = Path(config["paths"]["diag_root"])
    snapshot_dir = _land_snapshot_dir(root)
    years = _years(config)
    members = [f"EN{i:02d}" for i in range(int(settings.get("ensemble_member_count", 10)))]
    nlead = int(settings.get("monthly_nlead", 24))
    grid = land_input_cache.target_grid(
        float(_regrid(config)["target_dlat"]), float(_regrid(config)["target_dlon"])
    )
    grid_tag = land_input_cache.target_grid_tag(
        float(_regrid(config)["target_dlat"]), float(_regrid(config)["target_dlon"])
    )
    depth = (0.0, 1.6) if variable == "H2OSOI" else None
    raw_files = land_input_cache.resolve_reference_files(cfg["path"])
    raw_identity = land_input_cache.raw_source_identity(
        paths=raw_files,
        logical_identity={
            "product": cfg["product"], "variable": cfg["variable"],
            "field": variable, "path_pattern": cfg["path"],
        },
        source_revision=cfg["source_revision"], mode="inventory",
        inventory_root=Path(cfg["path"]).parent,
        snapshot_dir=snapshot_dir,
    )
    reference_month_policy = (
        "complete centered 3-month windows; retain explicit missing seasons"
        if cfg.get("require_complete_calendar_months", False)
        else "all seasonal center months"
    )
    common = dict(
        initialization_years=(years[0], years[-1]),
        climatology_years=tuple(selection["climatology_years"]),
        ensemble_members=members, monthly_nlead=nlead,
        target_grid_name=grid_tag, regridding_method=str(_regrid(config)["method"]),
        regridding_periodic=bool(_regrid(config)["periodic"]),
        output_units=cfg["output_units"], reference_is_anomaly=cfg["is_anomaly"],
        depth_range_m=depth, reference_month_policy=reference_month_policy,
    )
    ref_path = land_skill.staged_land_input_path(root, cfg["product"], variable, grid_tag, depth_range_m=depth)
    ref_expected = land_input_cache.expected_attrs(
        field=variable, source_kind="reference", source_name=cfg["product"],
        source_data_identity=raw_identity, **common,
    )
    ref_ok, _ = land_input_cache.prepared_cache_status(
        ref_path, ref_expected, grid=grid
    )
    if force or not ref_ok:
        land_input_cache.prepare_reference_cache(
            paths=raw_files, cfg=cfg, field=variable, depth_range_m=depth,
            grid=grid, periodic=bool(_regrid(config)["periodic"]),
            expected=ref_expected, output_path=ref_path,
        )
    with xr.open_dataset(ref_path) as reference_ds:
        reference = reference_ds[variable].load()

    for system, init_month in pairs:
        case = E3SM_CASES[system]
        raw_model_root = Path(settings.get(
            "raw_model_root", "/global/cfs/cdirs/e3sm/S2S2D/post_process"
        ))
        model_files = land_input_cache.resolve_model_files(
            data_dir=raw_model_root, case_prefix=case["case_prefix"],
            members=members, years=years, init_month=int(init_month),
            field=variable, nlead=nlead,
        )
        model_identity = land_input_cache.raw_source_identity(
            paths=model_files,
            logical_identity={
                "case": system, "case_prefix": case["case_prefix"],
                "field": variable, "init_month": int(init_month),
                "years": years, "members": members, "monthly_nlead": nlead,
            },
            source_revision=str(case.get("source_revision", "post_process_v1")),
            mode="inventory", inventory_root=raw_model_root,
            snapshot_dir=snapshot_dir,
        )
        expected = land_input_cache.expected_attrs(
            field=variable, source_kind="model", source_name=system,
            source_data_identity=model_identity, case_prefix=case["case_prefix"],
            init_month=int(init_month), **common,
        )
        path = land_skill.staged_land_input_path(
            root, case["cache_tag"], variable, grid_tag,
            init_month=int(init_month), depth_range_m=depth,
        )
        model_ok, _ = land_input_cache.prepared_cache_status(path, expected, grid=grid)
        if force or not model_ok:
            land_input_cache.prepare_model_cache(
                data_dir=str(raw_model_root),
                case_prefix=case["case_prefix"], members=members, years=years,
                init_month=int(init_month), field=variable, monthly_nlead=nlead,
                monthly_chunks=dict(settings.get("raw_model_chunks", {})),
                depth_range_m=depth, reference=reference,
                restrict_to_reference_months=cfg.get("restrict_model_to_reference_months", False),
                climatology_years=tuple(selection["climatology_years"]),
                strict_member_completeness=True, grid=grid,
                periodic=bool(_regrid(config)["periodic"]), expected=expected,
                output_path=path,
            )
