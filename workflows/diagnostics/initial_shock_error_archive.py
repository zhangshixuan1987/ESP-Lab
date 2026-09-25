"""Archive-backed RMSE/MAE workflow derived from initial-shock block indices."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import xarray as xr

from esp_lab.diagnostics import initial_shock_error
from esp_lab.diagnostics.initial_shock_error import VERSION, compute_initial_shock_error_index
from esp_lab.utils.code_identity import code_digest
from esp_lab.utils.netcdf_utils import atomic_to_netcdf, load_netcdf
from workflows.diagnostics import initial_shock_archive as block_archive

ARCHIVE_VERSION = "initial_shock_rmse_mae_archive_v3"


def _block_settings(settings):
    """Remove 6b-only thresholds before planning or running shared 6a caches."""
    prepared = copy.deepcopy(settings)
    prepared["metric"].pop("monthly_error_min_samples", None)
    prepared["metric"].pop("seasonal_error_min_samples", None)
    return prepared


def error_code_identity():
    """Return (archive, metric) fingerprints of the RMSE/MAE compute path."""
    return (
        code_digest([compute_archive_plan, _block_settings]),
        code_digest([initial_shock_error.compute_initial_shock_error_index]),
    )


def _cache_valid(path, digest):
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path) as ds:
            required = {
                "rmse", "mae", "normalized_rmse", "normalized_mae",
                "first_member_normalized_rmse", "first_member_normalized_mae",
                "monthly_standardized_error",
                "monthly_first_member_standardized_error",
                "seasonal_normalized_rmse", "seasonal_normalized_mae",
                "paired_sample_count", "valid_metric",
            }
            return ds.attrs.get("identity_sha256") == digest and required <= set(ds.variables)
    except (OSError, ValueError):
        return False


def plan_archive_run(settings, cases, variable):
    """Plan/reuse the block-index cache and a compact RMSE/MAE cache."""
    plan = block_archive.plan_archive_run(_block_settings(settings), cases, variable)
    code_hash, metric_hash = error_code_identity()
    force_compute = settings["cache"].get("force_compute", False)
    for task in plan:
        identity = {
            "source_index_digest": task["digest"], "algorithm": VERSION,
            "archive_algorithm": ARCHIVE_VERSION, "archive_code": code_hash,
            "metric_code": metric_hash,
            "error_settings": {
                "monthly_error_min_samples": settings["metric"].get(
                    "monthly_error_min_samples"
                ),
                "seasonal_error_min_samples": settings["metric"].get(
                    "seasonal_error_min_samples", 3
                ),
            },
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        # Fixed name next to the index cache (<source>_rmse_mae_init<MM>_<y0>_<y1>.nc);
        # the digest is stored in the file and checked by _cache_valid, so a stale
        # cache is rebuilt in place.
        index_name = Path(task["path"]).name.removeprefix(f"{task['source']}_")
        path = Path(task["path"]).with_name(f"{task['source']}_rmse_mae_{index_name}")
        task["error_identity"] = identity
        task["error_digest"] = digest
        task["error_path"] = str(path)
        task["error_cached"] = _cache_valid(path, digest)
        task["error_rebuild"] = (
            force_compute or settings["cache"]["mode"] == "rebuild" or not task["error_cached"]
        )
    if settings["cache"]["mode"] == "require" and any(t["error_rebuild"] for t in plan):
        raise FileNotFoundError("Required RMSE/MAE caches missing: " + ", ".join(
            t["error_path"] for t in plan if t["error_rebuild"]
        ))
    return plan


def compute_archive_plan(plan, settings, variable):
    """Load or prepare global indices, then derive and cache anomaly errors."""
    block_by_month = block_archive.compute_archive_plan(
        plan, _block_settings(settings), variable
    )
    tasks = {(t["month"], t["case"]): t for t in plan}
    output = {}
    for month, comparison in block_by_month.items():
        items = []
        for case in comparison.case.values:
            case_name = str(case)
            task = tasks[(month, case_name)]
            path = Path(task["error_path"])
            if not task["error_rebuild"] and _cache_valid(path, task["error_digest"]):
                result = load_netcdf(path)
                print(f"Reusing {case_name} init {month:02d} RMSE/MAE: {path}")
            else:
                if settings["cache"]["mode"] == "require":
                    raise FileNotFoundError(path)
                source = comparison.sel(case=case, drop=True)
                result = compute_initial_shock_error_index(
                    source,
                    min_samples=settings["metric"].get("monthly_error_min_samples"),
                    seasonal_min_samples=settings["metric"].get(
                        "seasonal_error_min_samples", 3
                    ),
                ).compute()
                result.attrs.update(
                    case=case_name, field=variable["field"], init_month=month,
                    identity_sha256=task["error_digest"],
                    provenance_json=json.dumps(task["error_identity"], sort_keys=True),
                    source_index_identity_sha256=task["digest"],
                    source_index_provenance_json=task["provenance_json"],
                )
                atomic_to_netcdf(result, path)
                print(f"Wrote {path}")
            items.append(result.expand_dims(case=[case_name]))
        output[month] = xr.concat(items, dim="case", join="exact", combine_attrs="drop_conflicts")
    return output
