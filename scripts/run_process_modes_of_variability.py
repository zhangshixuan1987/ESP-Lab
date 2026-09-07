#!/usr/bin/env python3
"""Run restartable PCMDI modes-of-variability preprocessing.

The heavy field preparation is shared by source field and frequency, while
mode-specific conventional EOF and common-basis products are written below
``OUTDIR/modes/MODE/indices``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import dask
import numpy as np
import xarray as xr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflows.modes_of_variability import analysis as modes_analysis


LOG = logging.getLogger(__name__)
PRODUCT_CONFIGURATION_SCHEMA = 6


def smyle_benchmark_fingerprints(
    settings: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Describe the benchmark files that feed a SMYLE field product.

    Including these lightweight file fingerprints in the cache signature
    prevents a corrected upstream benchmark from leaving an older regridded
    field/index cache falsely reusable.
    """
    archive_field = str(
        settings.get(
            "archive_field",
            "TS" if str(settings["field"]) == "SST" else settings["field"],
        )
    )
    frequency = "seas" if settings["frequency"] == "seasonal" else "mon"
    root = Path(args.smyle_benchmark_dir)
    fingerprints: list[dict[str, object]] = []
    for init_month in args.init_months:
        name = modes_analysis.smyle_access.benchmark_filename(
            archive_field,
            init_month,
            nens=args.smyle_nens,
            nlead=args.monthly_nlead,
            freq=frequency,
        )
        candidates = [
            root / "leadtime_acc" / "inputs" / "atm" / archive_field / name,
            root / "leadtime_acc" / "inputs" / archive_field / name,
            root / name,
        ]
        path = next(
            (candidate for candidate in candidates if candidate.exists()),
            candidates[0],
        )
        record: dict[str, object] = {
            "init_month": int(init_month),
            "path": str(path.resolve()),
        }
        try:
            stat = path.stat()
        except FileNotFoundError:
            record["missing"] = True
        else:
            record.update(size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))
        fingerprints.append(record)
    return fingerprints


def configuration_signature(
    source: str,
    settings: dict[str, object],
    args: argparse.Namespace,
    *,
    include_mode: bool,
) -> str:
    """Return a stable signature for deciding whether a cached product is reusable."""
    signature: dict[str, object] = {
        "schema": PRODUCT_CONFIGURATION_SCHEMA,
        "source": source,
        "field": str(settings["field"]),
        "frequency": str(settings["frequency"]),
        "climatology": [args.clim_start, args.clim_end],
        "target_grid": {
            "dlat": args.target_dlat,
            "dlon": args.target_dlon,
            "method": args.regrid_method,
            "periodic": not args.no_periodic,
        },
    }
    if source == "obs":
        signature["years"] = [args.obs_start_year, args.obs_end_year]
        signature["product"] = str(settings["obs_product"])
        signature["variable"] = str(settings["obs_var"])
        signature["obs_dir"] = str(Path(args.obs_dir).resolve())
    else:
        signature["years"] = [args.start_year, args.end_year]
        signature["monthly_nlead"] = args.monthly_nlead
        if source == "e3sm":
            signature.update(
                data_dir=str(Path(args.e3sm_data_dir).resolve()),
                case_prefix=args.e3sm_case_prefix,
                ensemble_size=args.e3sm_nens,
                engine=args.e3sm_engine,
                archive_grid=args.e3sm_grid,
            )
        elif source == "smyle":
            signature.update(
                benchmark_dir=str(Path(args.smyle_benchmark_dir).resolve()),
                ensemble_size=args.smyle_nens,
                benchmark_inputs=smyle_benchmark_fingerprints(settings, args),
            )
        elif source == "nmme":
            signature.update(
                nmme_root=str(Path(args.nmme_root).resolve()),
                nmme_models=list(args.nmme_models),
                nmme_field=str(args.nmme_field),
            )
            if str(settings["field"]) == "SST":
                signature["nmme_sst_mask_version"] = (
                    modes_analysis.nmme_access.NMME_SST_MASK_VERSION
                )
                signature["nmme_sst_land_mask"] = bool(
                    args.nmme_sst_land_mask
                )
                signature["nmme_fixed_dir"] = str(
                    Path(args.nmme_fixed_dir).resolve()
                )
        if source in {"e3sm", "smyle"} and str(settings["field"]) == "SST":
            signature["model_sst_preprocessing_version"] = (
                modes_analysis.sst_utils.SST_PREPROCESSING_VERSION
            )
            signature["model_sst_land_mask"] = bool(args.model_sst_land_mask)
    if include_mode:
        signature["mode"] = str(settings["mode"])
        signature["eof_number"] = int(settings["eof_number"])
        signature["domain_mode"] = str(settings["domain_mode"])
        signature["eof_scaling"] = bool(args.eof_scaling)
        signature["remove_domain_mean"] = bool(args.remove_domain_mean)
        signature["eof_strategy"] = getattr(args, "eof_strategy", "fixed_obs_projection")
        if str(settings["mode"]) in modes_analysis.TEMPERATURE_MODES:
            signature["fixed_basis_projection_mask_version"] = (
                modes_analysis.FIXED_BASIS_PROJECTION_MASK_VERSION
            )
        signature["eof_reference"] = {
            "source": getattr(args, "eof_reference_source", "obs"),
            "years": [
                getattr(args, "eof_reference_start_year", args.obs_start_year),
                getattr(args, "eof_reference_end_year", args.obs_end_year),
            ],
        }
        signature["regression_confidence"] = getattr(args, "regression_confidence", 0.95)
        signature["eof_bootstrap"] = {
            "iterations": args.eof_bootstrap_iterations,
            "seed": args.eof_bootstrap_seed,
            "confidence": args.eof_bootstrap_confidence,
        }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def cached_product_matches(path: Path, attribute: str, expected: str) -> bool:
    """Return True only when a readable cached product has the expected signature."""
    try:
        with xr.open_dataset(path, decode_times=False) as dataset:
            actual = dataset.attrs.get(attribute)
    except Exception as error:
        LOG.warning("Cannot read cached product %s: %s", path, error)
        return False
    def normalized(value: object) -> object:
        if not isinstance(value, str):
            return value
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value
        if isinstance(payload, dict):
            # Dask chunking changes execution cost, not scientific content.
            # Ignore this legacy signature field so products written before
            # chunking was made operational remain reusable.
            payload.pop("nmme_chunks", None)
            return json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return value

    if normalized(actual) != normalized(expected):
        LOG.info("%s has an incompatible or missing %s", path, attribute)
        return False
    return True


def add_mode_aliases(dataset: xr.Dataset, mode: str) -> xr.Dataset:
    result = dataset.copy()
    prefix = mode.lower()
    aliases = {
        "mode_index": f"{prefix}_eof",
        "mode_index_conventional": f"{prefix}_eof_conventional",
        "mode_pattern": f"{prefix}_pattern",
        "mode_variance_fraction": f"{prefix}_variance_fraction",
    }
    for generic_name, alias in aliases.items():
        if generic_name in result and alias not in result:
            result[alias] = result[generic_name]
    return result


def add_skill_lead_subset(dataset: xr.Dataset, mode: str) -> xr.Dataset:
    """Add a seasonal lead subset for ACC/RMSE skill calculations.

    Temperature-mode products keep monthly leads for pattern diagnostics, but
    the lead-dependent skill plots verify only seasonal target months.  Store
    that subset on a separate dimension so consumers do not accidentally use
    the first N monthly leads as seasonal leads.
    """
    required = {"mode_index", "valid_time", "target_month"}
    if not required <= set(dataset):
        return dataset

    seasonal = dataset["target_month"].isin([1, 4, 7, 10])
    skill_leads = dataset["target_month"].L.where(seasonal, drop=True)
    if skill_leads.size == 0:
        return dataset

    result = dataset.copy()
    skill_index = result["mode_index"].sel(L=skill_leads).rename(L="skill_L")
    skill_time = result["valid_time"].sel(L=skill_leads).rename(L="skill_L")
    skill_month = result["target_month"].sel(L=skill_leads).rename(L="skill_L")
    skill_index = skill_index.assign_coords(skill_L=skill_leads.values)
    skill_time = skill_time.assign_coords(skill_L=skill_leads.values)
    skill_month = skill_month.assign_coords(skill_L=skill_leads.values)

    result["mode_index_skill"] = skill_index
    result["valid_time_skill"] = skill_time
    result["target_month_skill"] = skill_month

    prefix = mode.lower()
    alias = f"{prefix}_eof_skill"
    if alias not in result:
        result[alias] = result["mode_index_skill"]

    result["mode_index_skill"].attrs.update(
        long_name="Seasonal target-month subset of mode_index for skill metrics",
        description=(
            "Mode index at target months DJF/MAM/JJA/SON, stored separately "
            "from the full lead axis so ACC/RMSE calculations use seasonal "
            "leads instead of the first monthly leads."
        ),
    )
    return result


def index_dataset(
    field_ds: xr.Dataset,
    source: str,
    settings: dict[str, object],
    station_definition: modes_analysis.StationNaoDefinition,
    args: argparse.Namespace,
    references: dict[int, dict[str, object]] | None = None,
) -> tuple[xr.Dataset, dict[int, dict[str, object]] | None]:
    field = str(settings["field"])
    mode = str(settings["mode"])
    anomalies = field_ds[f"{field}_anom"]
    if source == "obs":
        reference_anomalies = anomalies.sel(
            time=slice(
                str(args.eof_reference_start_year),
                str(args.eof_reference_end_year),
            )
        )
        mode_ds, references = modes_analysis.pcmdi_mode_reference(
            reference_anomalies, settings, args
        )
        if mode == "NAO":
            mode_ds["nao_station"] = modes_analysis.station_nao_obs(
                reference_anomalies, station_definition
            ).load()
        return add_mode_aliases(mode_ds, mode), references

    if references is None:
        raise ValueError(f"Observational PCMDI EOF references are required for {mode}.")
    mode_ds = modes_analysis.pcmdi_mode_model(
        anomalies,
        field_ds["valid_time"],
        references,
        settings,
        args,
    )
    if mode == "NAO":
        mode_ds["nao_station"] = modes_analysis.station_nao_model(
            anomalies, station_definition
        ).load()
    mode_ds["valid_time"] = field_ds["valid_time"]
    mode_ds = add_skill_lead_subset(mode_ds, mode)
    return add_mode_aliases(mode_ds, mode), references


def write_product(
    dataset: xr.Dataset,
    path: Path,
    source: str,
    settings: dict[str, object],
    args: argparse.Namespace,
    station_definition: modes_analysis.StationNaoDefinition,
    product_kind: str,
) -> None:
    dataset = dataset.copy()
    modes_analysis.validate_spatial_coordinates(dataset, f"{source} {product_kind} product")
    mode = str(settings["mode"])
    if product_kind == "index" and "mode_index" in dataset:
        finite_count = int(dataset["mode_index"].notnull().sum().compute())
        if finite_count == 0:
            raise ValueError(
                f"Refusing to write {source} {mode} index product: mode_index "
                "contains no finite values."
            )
    dataset.attrs.update(
        title=f"Processed {mode} mode-of-variability diagnostics",
        source=source.upper(),
        mode=mode,
        eof_number=int(settings["eof_number"]),
        origin_domain=str(settings["domain_mode"]),
        source_field=str(settings["field"]),
        frequency=str(settings["frequency"]),
        climatology=f"{args.clim_start}-{args.clim_end}",
        processing_script="scripts/run_process_modes_of_variability.py",
        eof_method="PCMDI Metrics Package variability_mode",
        eof_strategy=getattr(args, "eof_strategy", "fixed_obs_projection"),
        model_pattern_definition=(
            "regression of model anomalies onto projected observed-EOF PC"
            if source != "obs" and getattr(args, "eof_strategy", "fixed_obs_projection") == "fixed_obs_projection"
            else "PCMDI conventional EOF pattern"
        ),
        common_basis_reference=str(settings["obs_product"]),
        observation_data_period=(
            f"{args.obs_start_year}-{args.obs_end_year}"
        ),
        eof_reference_period=(
            f"{args.eof_reference_start_year}-{args.eof_reference_end_year}"
        ),
        hindcast_initialization_period=(
            "not_applicable"
            if source == "obs"
            else f"{args.start_year}-{args.end_year}"
        ),
        eof_scaling=str(args.eof_scaling),
        remove_domain_mean=str(args.remove_domain_mean),
        sst_ocean_mask_min_valid_fraction=str(
            getattr(args, "sst_ocean_mask_min_valid_fraction", 0.5)
        ),
        sst_ocean_mask_resolution=str(
            getattr(args, "sst_ocean_mask_resolution", "110m")
        ),
        eof_bootstrap_iterations=int(args.eof_bootstrap_iterations),
        eof_bootstrap_seed=int(args.eof_bootstrap_seed),
        eof_bootstrap_confidence=float(args.eof_bootstrap_confidence),
        history=f"created {datetime.now(timezone.utc).isoformat()}",
        spatial_coordinate_convention=(
            "strictly increasing regular latitude/longitude; regional mode "
            "longitudes are continuous in the PCMDI domain convention"
        ),
    )
    if mode == "NAO":
        dataset.attrs["station_nao_definition"] = json.dumps(
            asdict(station_definition), sort_keys=True
        )
    if product_kind == "field":
        dataset.attrs["field_configuration"] = configuration_signature(
            source, settings, args, include_mode=False
        )
    elif product_kind == "index":
        dataset.attrs["index_configuration"] = configuration_signature(
            source, settings, args, include_mode=True
        )
    else:
        raise ValueError(f"Unknown product_kind={product_kind!r}")
    encoding = {
        name: {"zlib": True, "complevel": 1}
        for name, variable in dataset.data_vars.items()
        if np.issubdtype(variable.dtype, np.number)
    }
    modes_analysis.atomic_to_netcdf(dataset, path, encoding=encoding)


def process_one(
    source: str,
    init_month: int | None,
    settings: dict[str, object],
    args: argparse.Namespace,
    destination_grid: xr.Dataset,
    station_definition: modes_analysis.StationNaoDefinition,
    references: dict[int, dict[str, object]] | None = None,
    force_field: bool = False,
) -> tuple[dict[str, str], dict[int, dict[str, object]] | None]:
    mode = str(settings["mode"])
    output_source = (
        str(getattr(args, "e3sm_cache_tag", "e3sm"))
        if source == "e3sm"
        else source
    )
    field_path, index_path = modes_analysis.product_paths(
        Path(args.outdir),
        mode,
        output_source,
        init_month,
        modes_analysis.grid_token(args.target_dlat, args.target_dlon),
        settings,
        args.legacy_nao_layout,
    )
    if source == "obs":
        source_label = str(settings["obs_product"])
    elif source == "e3sm":
        source_label = str(getattr(args, "e3sm_display_name", "E3SM"))
    elif source == "nmme":
        source_label = "NMME"
    else:
        source_label = source.upper()
    label = source_label if init_month is None else f"{source_label} init {init_month:02d}"
    field_signature = configuration_signature(
        source, settings, args, include_mode=False
    )
    index_signature = configuration_signature(
        source, settings, args, include_mode=True
    )

    if args.dry_run:
        LOG.info("[dry-run] %s %s field: %s", mode, label, field_path)
        LOG.info("[dry-run] %s %s index: %s", mode, label, index_path)
        return (
            {"field": str(field_path), "index": str(index_path), "status": "dry_run"},
            references,
        )

    field_reusable = (
        field_path.exists()
        and not force_field
        and cached_product_matches(
            field_path, "field_configuration", field_signature
        )
    )
    if field_reusable:
        LOG.info("%s %s: using field product %s", mode, label, field_path)
        field_ds = xr.open_dataset(field_path)
    else:
        LOG.info("%s %s: processing %s", mode, label, settings["field"])
        if source == "obs":
            field_ds = modes_analysis.obs_field_dataset(settings, args, destination_grid)
        else:
            if init_month is None:
                raise ValueError(f"{source} requires an initialization month.")
            field_ds = modes_analysis.model_field_dataset(
                source, init_month, settings, args, destination_grid
            )
        write_product(
            field_ds,
            field_path,
            source,
            settings,
            args,
            station_definition,
            product_kind="field",
        )
        field_ds = xr.open_dataset(field_path)

    needs_reference_build = source == "obs" and references is None
    index_compatible = (
        index_path.exists()
        and not args.force
        and cached_product_matches(
            index_path, "index_configuration", index_signature
        )
    )
    index_reusable = index_compatible and not needs_reference_build
    if index_reusable:
        LOG.info("%s %s: using mode product %s", mode, label, index_path)
        status = "skipped"
    else:
        indices, references = index_dataset(
            field_ds,
            source,
            settings,
            station_definition,
            args,
            references=references,
        )
        if args.force or not index_compatible:
            write_product(
                indices,
                index_path,
                source,
                settings,
                args,
                station_definition,
                product_kind="index",
            )
            status = "ok"
        else:
            status = "skipped"
    field_ds.close()
    return (
        {"field": str(field_path), "index": str(index_path), "status": status},
        references,
    )


def eof_configs_compatible(eof1: dict, eof2: dict) -> bool:
    if not isinstance(eof1, dict) or not isinstance(eof2, dict):
        return False
    ignore_keys = {"reference_years"}
    keys = set(eof1.keys()) | set(eof2.keys())
    for k in keys - ignore_keys:
        if eof1.get(k) != eof2.get(k):
            return False
    return True


def write_manifest(
    args: argparse.Namespace,
    mode_configs: dict[str, dict[str, object]],
    products: dict[str, dict[str, str]],
    station_definition: modes_analysis.StationNaoDefinition,
) -> Path:
    if args.legacy_nao_layout:
        path = Path(args.outdir) / "nao_manifest.json"
    else:
        path = Path(args.outdir) / "_manifests" / "modes_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PRODUCT_CONFIGURATION_SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(),
        "processing_script": "scripts/run_process_modes_of_variability.py",
        "sources": args.sources,
        "modes": mode_configs,
        "init_months": args.init_months,
        "years": [args.start_year, args.end_year],
        "hindcast_initialization_years": [args.start_year, args.end_year],
        "climatology": [args.clim_start, args.clim_end],
        "monthly_nlead": args.monthly_nlead,
        "target_grid": {
            "dlat": args.target_dlat,
            "dlon": args.target_dlon,
            "method": args.regrid_method,
            "periodic": not args.no_periodic,
        },
        "eof": {
            "package": "pcmdi_metrics",
            "method": "observational EOF reference with model common-basis projection",
            "strategy": getattr(args, "eof_strategy", "fixed_obs_projection"),
            "reference_source": getattr(args, "eof_reference_source", "obs"),
            "reference_years": [
                getattr(args, "eof_reference_start_year", args.obs_start_year),
                getattr(args, "eof_reference_end_year", args.obs_end_year),
            ],
            "eof_scaling": args.eof_scaling,
            "remove_domain_mean": args.remove_domain_mean,
            "bootstrap_iterations": args.eof_bootstrap_iterations,
            "bootstrap_seed": args.eof_bootstrap_seed,
            "bootstrap_confidence": args.eof_bootstrap_confidence,
            "regression_confidence": getattr(args, "regression_confidence", 0.95),
            "sst_ocean_mask": {
                "min_valid_fraction": getattr(
                    args, "sst_ocean_mask_min_valid_fraction", 0.5
                ),
                "natural_earth_resolution": getattr(
                    args, "sst_ocean_mask_resolution", "110m"
                ),
            },
        },
        "products": products,
    }
    if "NAO" in mode_configs:
        payload["nao_definition"] = asdict(station_definition)
    if args.legacy_nao_layout:
        payload["eof"].update(
            mode="NAO",
            common_basis_reference=mode_configs["NAO"]["obs_product"],
        )
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if getattr(args, "merge_manifest", False) and path.exists():
            existing = json.loads(path.read_text())
            compatible_keys = (
                "schema_version",
                "processing_script",
                "years",
                "climatology",
                "monthly_nlead",
                "target_grid",
            )
            if (
                all(existing.get(key) == payload.get(key) for key in compatible_keys)
                and eof_configs_compatible(existing.get("eof", {}), payload.get("eof", {}))
            ):
                payload["sources"] = list(
                    dict.fromkeys([*existing.get("sources", []), *payload["sources"]])
                )
                payload["init_months"] = sorted(
                    set(existing.get("init_months", [])) | set(payload["init_months"])
                )
                payload["modes"] = {
                    **existing.get("modes", {}),
                    **payload["modes"],
                }
                payload["products"] = {
                    **existing.get("products", {}),
                    **payload["products"],
                }
                if "nao_definition" in existing and "nao_definition" not in payload:
                    payload["nao_definition"] = existing["nao_definition"]

        temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def validate_args(args: argparse.Namespace) -> None:
    if args.start_year > args.end_year:
        raise ValueError("--start-year must not exceed --end-year.")
    if args.obs_start_year > args.obs_end_year:
        raise ValueError("--obs-start-year must not exceed --obs-end-year.")
    if args.clim_start > args.clim_end:
        raise ValueError("--clim-start must not exceed --clim-end.")
    if not (args.start_year <= args.clim_start <= args.clim_end <= args.end_year):
        raise ValueError(
            "The climatology period must lie within the model processing years."
        )
    if not (
        args.obs_start_year
        <= args.clim_start
        <= args.clim_end
        <= args.obs_end_year
    ):
        raise ValueError(
            "The climatology period must lie within the observation years."
        )
    eof_reference_start_year = getattr(
        args, "eof_reference_start_year", args.start_year
    )
    eof_reference_end_year = getattr(args, "eof_reference_end_year", args.end_year)
    if eof_reference_start_year > eof_reference_end_year:
        raise ValueError(
            "--eof-reference-start-year must not exceed --eof-reference-end-year."
        )
    if not (
        args.obs_start_year
        <= eof_reference_start_year
        <= eof_reference_end_year
        <= args.obs_end_year
    ):
        raise ValueError(
            "The EOF reference period must lie within the observation years."
        )
    if not set(args.init_months).issubset(range(1, 13)):
        raise ValueError("--init-months values must be between 1 and 12.")
    if args.monthly_nlead < 1:
        raise ValueError("--monthly-nlead must be positive.")
    if args.target_dlat <= 0 or args.target_dlon <= 0:
        raise ValueError("--target-dlat and --target-dlon must be positive.")
    if args.dask_workers < 1:
        raise ValueError("--dask-workers must be positive.")
    if args.e3sm_nens < 1 or args.smyle_nens < 1:
        raise ValueError("Ensemble sizes must be positive.")
    if "nmme" in args.sources:
        if not Path(args.nmme_root).is_dir():
            raise FileNotFoundError(f"--nmme-root does not exist: {args.nmme_root}")
        if not args.nmme_models:
            raise ValueError("--nmme-models is required when --sources includes nmme.")
    if args.eof_bootstrap_iterations < 0:
        raise ValueError("--eof-bootstrap-iterations must be nonnegative.")
    if not 0 < args.eof_bootstrap_confidence < 1:
        raise ValueError("--eof-bootstrap-confidence must lie between 0 and 1.")
    if getattr(args, "eof_strategy", "fixed_obs_projection") not in {"fixed_obs_projection", "conventional"}:
        raise ValueError("--eof-strategy must be 'fixed_obs_projection' or 'conventional'.")
    if getattr(args, "eof_reference_source", "obs") not in {"obs", "era5", "hadisst2"}:
        raise ValueError("--eof-reference-source must be one of obs, era5, hadisst2.")
    if not 0 < getattr(args, "regression_confidence", 0.95) < 1:
        raise ValueError("--regression-confidence must lie between 0 and 1.")
    if not 0 <= getattr(args, "sst_ocean_mask_min_valid_fraction", 0.5) <= 1:
        raise ValueError(
            "--sst-ocean-mask-min-valid-fraction must lie between 0 and 1."
        )
    if getattr(args, "sst_ocean_mask_resolution", "110m") not in {"110m", "50m", "10m"}:
        raise ValueError(
            "--sst-ocean-mask-resolution must be one of 110m, 50m, or 10m."
        )
    if (
        getattr(args, "eof_strategy", "fixed_obs_projection") == "fixed_obs_projection"
        and args.eof_bootstrap_iterations
    ):
        LOG.warning(
            "--eof-bootstrap-iterations applies to observational reference EOFs only "
            "when --eof-strategy=fixed_obs_projection; model EOF bootstrap is skipped."
        )
    if args.legacy_nao_layout and args.modes != ["NAO"]:
        raise ValueError("--legacy-nao-layout only supports --modes NAO.")


def main() -> None:
    args = modes_analysis.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args.sources = list(dict.fromkeys(
        "obs" if source == "era5" else source for source in args.sources
    ))
    args.modes = list(dict.fromkeys(mode.upper() for mode in args.modes))
    validate_args(args)

    destination_grid = modes_analysis.regrid.make_latlon_grid(
        dlat=args.target_dlat, dlon=args.target_dlon
    )
    station_definition = modes_analysis.StationNaoDefinition()
    mode_configs = {mode: modes_analysis.mode_settings(mode) for mode in args.modes}
    for settings in mode_configs.values():
        if settings["field"] == "PSL":
            settings["obs_product"] = args.psl_obs_product
            settings["obs_var"] = args.psl_obs_var
        else:
            settings["obs_product"] = args.ts_obs_product
            settings["obs_var"] = args.ts_obs_var
        settings["obs_years"] = [args.obs_start_year, args.obs_end_year]
        settings["eof_reference_years"] = [
            getattr(args, "eof_reference_start_year", args.obs_start_year),
            getattr(args, "eof_reference_end_year", args.obs_end_year),
        ]
    LOG.info("Modes: %s", ", ".join(args.modes))
    LOG.info("Sources: %s", ", ".join(args.sources))
    LOG.info("Output: %s", args.outdir)
    LOG.info("EOF strategy: %s", getattr(args, "eof_strategy", "fixed_obs_projection"))
    LOG.info(
        "EOF reference period: %s-%s",
        args.eof_reference_start_year,
        args.eof_reference_end_year,
    )

    products: dict[str, dict[str, str]] = {}
    materialized_fields: set[str] = set()
    with dask.config.set(scheduler="threads", num_workers=args.dask_workers):
        for mode, settings in mode_configs.items():
            references: dict[int, dict[str, object]] | None = None
            if "obs" in args.sources or any(
                source in args.sources for source in modes_analysis.MODEL_SOURCES
            ):
                field_path, _ = modes_analysis.product_paths(
                    Path(args.outdir), mode, "obs", None,
                    modes_analysis.grid_token(args.target_dlat, args.target_dlon),
                    settings, args.legacy_nao_layout,
                )
                product, references = process_one(
                    "obs", None, settings, args, destination_grid,
                    station_definition,
                    force_field=args.force and str(field_path) not in materialized_fields,
                )
                materialized_fields.add(str(field_path))
                key = "era5" if args.legacy_nao_layout else f"{mode}:reference"
                products[key] = product

            for source in modes_analysis.MODEL_SOURCES:
                if source not in args.sources:
                    continue
                for init_month in args.init_months:
                    output_source = (
                        str(getattr(args, "e3sm_cache_tag", "e3sm"))
                        if source == "e3sm"
                        else source
                    )
                    field_path, _ = modes_analysis.product_paths(
                        Path(args.outdir), mode, output_source, init_month,
                        modes_analysis.grid_token(args.target_dlat, args.target_dlon),
                        settings, args.legacy_nao_layout,
                    )
                    product, references = process_one(
                        source, init_month, settings, args, destination_grid,
                        station_definition, references=references,
                        force_field=args.force and str(field_path) not in materialized_fields,
                    )
                    materialized_fields.add(str(field_path))
                    key = (
                        f"{output_source}_init{init_month:02d}"
                        if args.legacy_nao_layout
                        else f"{mode}:{output_source}_init{init_month:02d}"
                    )
                    products[key] = product

    if args.dry_run:
        LOG.info("Dry run complete; no manifest was written.")
        return
    manifest = write_manifest(args, mode_configs, products, station_definition)
    LOG.info("Manifest: %s", manifest)


if __name__ == "__main__":
    main()
