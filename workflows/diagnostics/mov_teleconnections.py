"""Reusable modes-of-variability (MOV) teleconnection diagnostic workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from workflows.diagnostics.sst_teleconnections import (
    DEFAULT_DIAG_ROOT,
    DEFAULT_FIGURE_DIR,
    DEFAULT_OUTPUT_DIR,
    DOWNSTREAM_VARIABLES,
    E3SM_CASES,
    MEMBER_DIMS,
    assemble_teleconnection_dataset,
    corr_and_p,
    ensemble_mean,
    file_signature,
    lat_name,
    lead_anomaly,
    lead_signature,
    linear_detrend,
    match_leads,
    monthly_anomaly,
    observed_at_valid_time,
    parse_init_years,
    resolve_downstream_paths,
    select_cache,
    standardize_spatial_grid,
    time_year_month,
    weighted_spatial_metrics,
)

MOV_MODES: dict[str, dict[str, str]] = {
    "NAM": {"kind": "pressure", "description": "Northern Annular Mode"},
    "NAO": {"kind": "pressure", "description": "North Atlantic Oscillation"},
    "SAM": {"kind": "pressure", "description": "Southern Annular Mode"},
    "PNA": {"kind": "pressure", "description": "Pacific-North American pattern"},
    "NPO": {"kind": "pressure", "description": "North Pacific Oscillation"},
    "EA": {"kind": "pressure", "description": "East Atlantic pattern"},
    "SCA": {"kind": "pressure", "description": "Scandinavian pattern"},
    "PSA1": {"kind": "pressure", "description": "Pacific-South American pattern 1"},
    "PSA2": {"kind": "pressure", "description": "Pacific-South American pattern 2"},
    "PDO": {"kind": "temperature", "description": "Pacific Decadal Oscillation"},
    "NPGO": {"kind": "temperature", "description": "North Pacific Gyre Oscillation"},
    "AMO": {"kind": "temperature", "description": "Atlantic Multidecadal Oscillation"},
}

MOV_PRODUCT_TAGS: dict[str, str] = {
    "E3SM-FOSIRL": "JRA55_FOSIRL",
    "E3SM-Reanalysis": "Reanalysis",
    "E3SM-4DEnVarOcn": "4DEnVarOcn",
}
MOV_SOURCE_DIRS: dict[str, str] = dict(MOV_PRODUCT_TAGS)


def load_modes_manifest(diag_root: Path = DEFAULT_DIAG_ROOT) -> tuple[dict[str, Any] | None, Path | None]:
    """Discover and load modes_manifest.json if present."""
    candidates = [
        diag_root / "_manifests" / "modes_manifest.json",
        diag_root / "modes_manifest.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text()), path
            except Exception:
                continue
    return None, None


def _manifest_index_path(manifest: dict[str, Any] | None, product_key: str) -> Path | None:
    if not manifest:
        return None
    entry = manifest.get("products", {}).get(product_key)
    if not entry or not entry.get("index"):
        return None
    candidate = Path(entry["index"])
    return candidate if candidate.is_file() else None


def upstream_mov_paths(
    system: str,
    init_month: int,
    mode: str,
    *,
    diag_root: Path = DEFAULT_DIAG_ROOT,
    manifest: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Locate forecast and reference MOV index netCDF files."""
    mode_token = mode.lower()
    tag = MOV_PRODUCT_TAGS.get(system, system)
    source_dir = MOV_SOURCE_DIRS.get(system, tag)

    forecast = _manifest_index_path(manifest, f"{mode}:{tag}_init{init_month:02d}")
    if forecast is None:
        forecast = (
            diag_root
            / source_dir
            / "modes_variability"
            / "modes"
            / mode_token
            / "indices"
            / f"{tag}_init{init_month:02d}_{mode_token}.nc"
        )

    observed = _manifest_index_path(manifest, f"{mode}:reference")
    if observed is None and manifest:
        entry = manifest.get("products", {}).get("era5")
        if entry and entry.get("index") and Path(entry["index"]).is_file():
            observed = Path(entry["index"])
    if observed is None:
        standard_ref = (
            diag_root
            / "ERA5"
            / "modes_variability"
            / "modes"
            / mode_token
            / "indices"
            / f"era5_{mode_token}_reference.nc"
        )
        if standard_ref.is_file():
            observed = standard_ref
        else:
            candidates = sorted(
                diag_root.glob(f"*/modes_variability/modes/{mode_token}/indices/*_reference.nc")
            )
            candidates += sorted((diag_root / "modes" / mode_token / "indices").glob("*_reference.nc"))
            if len(candidates) == 1:
                observed = candidates[0]
            elif len(candidates) > 1:
                observed = candidates[0]
            else:
                observed = standard_ref
    return forecast, observed


def build_mov_teleconnection_inventory(config: Mapping[str, Any]) -> pd.DataFrame:
    """Scan and verify availability of all required upstream MOV products and downstream fields."""
    diag_root = Path(config["paths"].get("diag_root", DEFAULT_DIAG_ROOT))
    mode = config["selection"]["upstream_mode"].upper().strip()
    target_grid = config["selection"].get("target_grid", "latlon_1.0x1.0_periodic-True")
    allow_ambiguous = config["cache"].get("allow_ambiguous_matches", False)

    manifest, _ = load_modes_manifest(diag_root)
    rows: list[dict[str, Any]] = []

    for system in config["selection"]["systems"]:
        for init_month in config["selection"]["init_months"]:
            idx_fcst, idx_obs = upstream_mov_paths(
                system, init_month, mode, diag_root=diag_root, manifest=manifest
            )
            idx_available = idx_fcst.is_file() and idx_obs.is_file()

            for variable in config["selection"]["downstream_variables"]:
                spec = DOWNSTREAM_VARIABLES[variable]
                # Check if system produces land component
                if spec["family"] == "land" and not E3SM_CASES[system].get("supports_land", True):
                    rows.append({
                        "system": system,
                        "init_month": init_month,
                        "mode": mode,
                        "variable": variable,
                        "index_forecast": str(idx_fcst),
                        "index_observed": str(idx_obs),
                        "field_forecast": None,
                        "field_observed": None,
                        "status": "skipped",
                        "detail": f"{system} does not produce land diagnostics",
                    })
                    continue

                # If upstream MOV index is not available for this system, skip cleanly
                if not idx_available:
                    rows.append({
                        "system": system,
                        "init_month": init_month,
                        "mode": mode,
                        "variable": variable,
                        "index_forecast": str(idx_fcst),
                        "index_observed": str(idx_obs),
                        "field_forecast": None,
                        "field_observed": None,
                        "status": "skipped",
                        "detail": f"{mode} index not computed for {system}",
                    })
                    continue

                try:
                    fld_fcst, fld_obs = resolve_downstream_paths(
                        system,
                        init_month,
                        variable,
                        diag_root=diag_root,
                        target_grid=target_grid,
                        allow_ambiguous=allow_ambiguous,
                    )
                    status = "ready"
                    detail = ""
                except Exception as exc:
                    fld_fcst = fld_obs = None
                    status = "missing"
                    detail = str(exc).splitlines()[0]

                rows.append({
                    "system": system,
                    "init_month": init_month,
                    "mode": mode,
                    "variable": variable,
                    "index_forecast": str(idx_fcst),
                    "index_observed": str(idx_obs),
                    "field_forecast": str(fld_fcst) if fld_fcst else None,
                    "field_observed": str(fld_obs) if fld_obs else None,
                    "status": status,
                    "detail": detail,
                })
    return pd.DataFrame(rows)


def open_mov_inputs(
    system: str,
    init_month: int,
    variable: str,
    config: Mapping[str, Any],
    manifest: dict[str, Any] | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, list[Path]]:
    """Open and extract forecast & observed MOV index and downstream field DataArrays."""
    diag_root = Path(config["paths"].get("diag_root", DEFAULT_DIAG_ROOT))
    mode = config["selection"]["upstream_mode"].upper().strip()
    target_grid = config["selection"].get("target_grid", "latlon_1.0x1.0_periodic-True")
    allow_ambiguous = config["cache"].get("allow_ambiguous_matches", False)

    idx_fcst_path, idx_obs_path = upstream_mov_paths(
        system, init_month, mode, diag_root=diag_root, manifest=manifest
    )
    fld_fcst_path, fld_obs_path = resolve_downstream_paths(
        system, init_month, variable, diag_root=diag_root, target_grid=target_grid, allow_ambiguous=allow_ambiguous
    )

    idx_fcst_ds = xr.open_dataset(idx_fcst_path)
    idx_obs_ds = xr.open_dataset(idx_obs_path)
    fld_fcst_ds = xr.open_dataset(fld_fcst_path)
    fld_obs_ds = xr.open_dataset(fld_obs_path)

    # Standardize MOV index coordinate: prefer mode_index with coord L
    if {"mode_index", "valid_time"} <= set(idx_fcst_ds.data_vars):
        idx_fcst = idx_fcst_ds["mode_index"]
        idx_time = idx_fcst_ds["valid_time"]
    elif {"mode_index_skill", "valid_time_skill"} <= set(idx_fcst_ds.data_vars):
        idx_fcst = idx_fcst_ds["mode_index_skill"].rename({"skill_L": "L"})
        idx_time = idx_fcst_ds["valid_time_skill"].rename({"skill_L": "L"})
    else:
        raise ValueError(f"MOV forecast cache {idx_fcst_path} lacks supported index/time pair")

    if "mode_index" not in idx_obs_ds:
        raise ValueError(f"MOV reference cache {idx_obs_path} lacks mode_index")
    idx_obs = idx_obs_ds["mode_index"]

    fld_fcst = fld_fcst_ds["anomaly"] if "anomaly" in fld_fcst_ds else fld_fcst_ds[variable]
    fld_time = fld_fcst_ds["time"]
    fld_obs = fld_obs_ds["observation"] if "observation" in fld_obs_ds else fld_obs_ds[variable]

    sources = [idx_fcst_path, idx_obs_path, fld_fcst_path, fld_obs_path]
    return idx_fcst, idx_time, idx_obs, fld_fcst, fld_time, fld_obs, sources


def compute_system_mov_teleconnection(
    system: str,
    init_month: int,
    variable: str,
    config: Mapping[str, Any],
    manifest: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Compute lead-dependent teleconnection metrics for one (system, init_month, variable) MOV case."""
    idx_fcst, idx_time, idx_obs, fld_fcst, fld_time, fld_obs, sources = open_mov_inputs(
        system, init_month, variable, config, manifest=manifest
    )
    init_dim = "Y" if "Y" in idx_fcst.dims else "year"

    clim_years = config["selection"]["climatology_years"]
    idx_fcst = lead_anomaly(idx_fcst, idx_time, clim_years)
    fld_fcst = lead_anomaly(fld_fcst, fld_time, clim_years)
    idx_obs = monthly_anomaly(idx_obs, clim_years)
    fld_obs = monthly_anomaly(fld_obs, clim_years)

    lead_pairs = match_leads(idx_time, fld_time, init_dim)
    requested = config["selection"].get("leads")
    if requested is not None:
        requested_set = set(map(int, requested))
        lead_pairs = [(idx_lead, fld_lead) for idx_lead, fld_lead in lead_pairs if idx_lead in requested_set]
    if not lead_pairs:
        raise ValueError(f"No common requested leads for {system}, {variable}, init={init_month}")

    y0, y1 = config["selection"]["verification_years"]
    idx_init_years = parse_init_years(idx_fcst[init_dim].values)
    fld_init_years = parse_init_years(fld_fcst[init_dim].values)

    common_years = sorted(list(set(idx_init_years) & set(fld_init_years) & set(range(y0, y1 + 1))))
    if len(common_years) < config["analysis"].get("minimum_years", 20):
        raise ValueError(
            f"Insufficient common verification years ({len(common_years)}) for {system}, {variable}, init={init_month}"
        )

    idx_pos = [i for i, y in enumerate(idx_init_years) if y in common_years]
    fld_pos = [i for i, y in enumerate(fld_init_years) if y in common_years]

    outputs: list[xr.Dataset] = []
    sample = xr.DataArray(np.arange(len(common_years)), dims="sample", name="sample")

    for index_lead, field_lead in lead_pairs:
        idx_t = idx_time.sel(L=index_lead).isel({init_dim: idx_pos})
        fld_t = fld_time.sel(L=field_lead).isel({init_dim: fld_pos})
        iy, im = time_year_month(idx_t.values)
        fy, fm = time_year_month(fld_t.values)
        if not (np.array_equal(iy, fy) and np.array_equal(im, fm)):
            raise ValueError(
                f"MOV index L={index_lead} and {variable} L={field_lead} target dates differ for common starts"
            )

        xmod = (
            ensemble_mean(idx_fcst.sel(L=index_lead).isel({init_dim: idx_pos}))
            .rename({init_dim: "sample"})
            .assign_coords(sample=sample)
        )
        ymod = (
            ensemble_mean(fld_fcst.sel(L=field_lead).isel({init_dim: fld_pos}))
            .rename({init_dim: "sample"})
            .assign_coords(sample=sample)
        )
        xobs = observed_at_valid_time(idx_obs, idx_t)
        yobs = observed_at_valid_time(fld_obs, fld_t)

        xmod, ymod = xr.align(xmod, ymod, join="inner")
        xobs, yobs = xr.align(xobs, yobs, join="inner")

        if config["analysis"].get("detrend", True):
            xmod, ymod, xobs, yobs = map(linear_detrend, (xmod, ymod, xobs, yobs))

        model_r, model_p, model_n = corr_and_p(xmod, ymod, minimum_years=config["analysis"].get("minimum_years", 20))
        obs_r, obs_p, obs_n = corr_and_p(xobs, yobs, minimum_years=config["analysis"].get("minimum_years", 20))
        model_r, obs_r = xr.align(model_r, obs_r, join="exact")

        metrics = weighted_spatial_metrics(
            model_r,
            obs_r,
            model_p,
            obs_p,
            latitude_bounds=config["analysis"].get("latitude_bounds", (-80.0, 80.0)),
            area_weighted=config["analysis"].get("area_weighted", True),
            sign_agreement_threshold=config["analysis"].get("sign_agreement_threshold", 0.0),
            alpha=config["analysis"].get("alpha", 0.10),
        )

        ds = xr.Dataset({
            "model_correlation": model_r,
            "model_pvalue": model_p,
            "observed_correlation": obs_r,
            "observed_pvalue": obs_p,
            "model_sample_count": model_n,
            "observed_sample_count": obs_n,
            **{name: da for name, da in metrics.data_vars.items()},
            "downstream_lead": xr.DataArray(int(field_lead)),
        }).expand_dims(L=[int(index_lead)])
        outputs.append(ds)

    result = xr.concat(outputs, dim="L").expand_dims(
        init_month=[init_month], system=[system], variable=[variable]
    )
    result.attrs["source_files"] = json.dumps([str(p) for p in sources])
    return result


def compute_mov_provenance_fingerprint(
    config: Mapping[str, Any], source_paths: Sequence[Path | str]
) -> str:
    """Generate a SHA-256 fingerprint hash for MOV teleconnection configuration and inputs."""
    provenance = {
        "schema": "mov_teleconnection_metrics_v1",
        "selection": config["selection"],
        "analysis": config["analysis"],
        "sources": file_signature(source_paths),
    }
    return hashlib.sha256(
        json.dumps(provenance, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def ensure_mov_teleconnection_dataset(
    config: Mapping[str, Any],
    inventory: pd.DataFrame | None = None,
) -> tuple[xr.Dataset, Path, str]:
    """Load or compute cached MOV teleconnection metrics across all configured products."""
    if inventory is None:
        inventory = build_mov_teleconnection_inventory(config)

    missing = inventory.query("status == 'missing'")
    if not missing.empty:
        raise FileNotFoundError(
            "Required upstream products are missing. Please verify preparation workflows:\n"
            + missing.to_string(index=False)
        )

    all_source_paths: list[str] = []
    for row in inventory.itertuples():
        if getattr(row, "status", "") == "ready":
            all_source_paths.extend([
                row.index_forecast,
                row.index_observed,
                row.field_forecast,
                row.field_observed,
            ])

    fingerprint = compute_mov_provenance_fingerprint(config, all_source_paths)
    mode = config["selection"]["upstream_mode"].upper().strip()
    output_dir = Path(config["paths"].get("output_dir", DEFAULT_OUTPUT_DIR))
    out_file = output_dir / f"teleconnection_{mode}_{fingerprint}.nc"
    cache_mode = config["cache"].get("mode", "auto")

    if out_file.is_file() and cache_mode != "rebuild":
        metrics_ds = xr.open_dataset(out_file)
        return metrics_ds, out_file, "loaded"

    if cache_mode == "require":
        raise FileNotFoundError(f"Required teleconnection cache is unavailable: {out_file}")

    manifest, _ = load_modes_manifest(Path(config["paths"].get("diag_root", DEFAULT_DIAG_ROOT)))
    parts: list[xr.Dataset] = []
    for row in inventory.itertuples():
        if getattr(row, "status", "") != "ready":
            continue
        print(f"Computing {row.system} init={row.init_month:02d} {mode} -> {row.variable}")
        part = compute_system_mov_teleconnection(
            row.system, row.init_month, row.variable, config, manifest=manifest
        )
        parts.append(part)

    if not parts:
        raise RuntimeError("No MOV teleconnection datasets were computed from inventory.")

    metrics_ds = assemble_teleconnection_dataset(parts)
    metrics_ds.attrs.update({
        "schema": "mov_teleconnection_metrics_v1",
        "fingerprint": fingerprint,
        "upstream_mode": mode,
        "configuration_json": json.dumps(config, sort_keys=True, default=str),
        "source_inventory_json": json.dumps(file_signature(all_source_paths), sort_keys=True),
        "forecast_definition": "correlation across initialization years using ensemble-mean MOV index and ensemble-mean downstream field",
        "observed_definition": "correlation across matching target dates using observed MOV index and observed downstream anomaly",
        "season_definition": "centered three-month means inherited from upstream caches",
        "pvalue_note": "classical Pearson t test; no field-significance or autocorrelation correction",
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp.nc")
    metrics_ds.to_netcdf(tmp)
    tmp.replace(out_file)
    return metrics_ds, out_file, "computed"

