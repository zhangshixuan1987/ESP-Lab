"""Archive-backed RMSE/MAE workflow derived from initial-shock block indices."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import xarray as xr

from esp_lab.diagnostics import initial_shock_error
from esp_lab.diagnostics.initial_shock_error import VERSION, compute_initial_shock_error_index
from esp_lab.utils.netcdf_utils import atomic_to_netcdf, load_netcdf
from workflows.diagnostics import initial_shock_archive as block_archive

ARCHIVE_VERSION = "initial_shock_rmse_mae_archive_v2"


def _scientific_code_digest():
    """Hash numerical error code without coupling caches to plot styling."""
    return hashlib.sha256(
        inspect.getsource(initial_shock_error.compute_initial_shock_error_index).encode()
    ).hexdigest()


def _cache_valid(path, digest):
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path) as ds:
            required = {"rmse", "mae", "normalized_rmse", "normalized_mae",
                        "observation_climatology_std", "model_anomaly", "observation_anomaly",
                        "error", "paired_sample_count", "valid_metric"}
            return ds.attrs.get("identity_sha256") == digest and required <= set(ds.variables)
    except (OSError, ValueError):
        return False


def plan_archive_run(settings, cases, variable):
    """Plan/reuse the block-index cache and a compact RMSE/MAE cache."""
    plan = block_archive.plan_archive_run(settings, cases, variable)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metric_hash = _scientific_code_digest()
    force_compute = settings["cache"].get("force_compute", False)
    for task in plan:
        identity = {
            "source_index_digest": task["digest"], "algorithm": VERSION,
            "archive_algorithm": ARCHIVE_VERSION, "archive_code": code_hash,
            "metric_code": metric_hash,
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = Path(task["path"]).with_name(f"rmse_mae_init{task['month']:02d}_{digest[:20]}.nc")
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
    block_by_month = block_archive.compute_archive_plan(plan, settings, variable)
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
                    source, min_samples=settings["metric"].get("min_samples")
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
