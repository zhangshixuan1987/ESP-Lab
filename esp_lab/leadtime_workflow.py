"""Execution contracts for the seasonal ACC notebook."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr

from esp_lab import stats
from esp_lab.leadtime_skill_cache import file_inventory_digest, source_fingerprint


def seasonal_cache_encoding(data, requested_chunks):
    """Bound NetCDF chunks by the actual seasonal shape and dimension order.

    Legacy tuples use the notebook's Y, L, M, lat, lon order. Mappings can
    specify chunks by dimension name. Neither form depends on the array's
    physical dimension order, which xarray transformations may change.
    """
    if isinstance(requested_chunks, Mapping):
        chunks = dict(requested_chunks)
    else:
        dimensions = ("Y", "L", "M", "lat", "lon")
        if len(requested_chunks) != len(dimensions):
            raise ValueError("Seasonal encoding chunks must specify Y, L, M, lat, lon")
        chunks = dict(zip(dimensions, requested_chunks))
    if set(chunks) != set(data.dims):
        raise ValueError(
            f"Seasonal encoding dimensions {tuple(chunks)} must match {data.dims}"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
        for value in chunks.values()
    ):
        raise ValueError("Seasonal encoding chunk sizes must be positive integers")
    if any(size == 0 for size in data.sizes.values()):
        raise ValueError("Cannot encode an empty seasonal input")
    return {
        "chunksizes": tuple(min(int(chunks[dim]), data.sizes[dim]) for dim in data.dims),
        "zlib": True,
        "complevel": 1,
    }


def select_lead_range(model, time, lead_start, lead_end):
    """Select an inclusive, one-based range of seasonal positions."""
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
           for v in (lead_start, lead_end)):
        raise ValueError("lead_start and lead_end must be integers")
    if not 1 <= lead_start <= lead_end <= model.sizes["L"]:
        raise ValueError(
            f"Invalid seasonal lead range {lead_start}-{lead_end}; "
            f"available positions are 1-{model.sizes['L']}"
        )
    model, time = xr.align(model, time, join="exact")
    selection = slice(lead_start - 1, lead_end)
    return model.isel(L=selection), time.isel(L=selection)


def compute_skill_lead_range(
    model, time, observations, clim_start, clim_end, lead_start, lead_end, **kwargs
):
    """Compute skill for selected seasons, without multi-year averaging."""
    model, time = select_lead_range(model, time, lead_start, lead_end)
    return stats.compute_skill_seasonal(
        model, time, observations, clim_start, clim_end,
        nleadavg=1, nleads=model.sizes["L"], **kwargs,
    )


def compute_skill_lead_range_batch(
    model, time, observations, clim_start, clim_end, member_indices,
    lead_start, lead_end, **kwargs,
):
    """Compute resampled skill for the same inclusive seasonal range."""
    model, time = select_lead_range(model, time, lead_start, lead_end)
    return stats.compute_skill_seasonal_batch(
        model, time, observations, clim_start, clim_end,
        member_indices_all=member_indices, nleadavg=1,
        nleads=model.sizes["L"], **kwargs,
    )


def resolve_source_revision(
    mode, configured_revision, *, identity, snapshot_dir, paths=(), root=None,
):
    """Record an inventory online, or reuse that exact identity offline.

    Snapshot mode never stats archive paths. It deliberately trusts the last
    recorded inventory for this explicit source request and revision token.
    Revision mode remains an independent, manually maintained identity mode.
    """
    key = source_fingerprint(identity, source_revision=configured_revision)
    if mode == "revision":
        return configured_revision
    if mode not in {"inventory", "snapshot"}:
        raise ValueError("source identity mode must be inventory, snapshot, or revision")
    path = Path(snapshot_dir) / f"{key.split(':')[-1]}.json"
    if mode == "snapshot":
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"No readable source inventory snapshot at {path}; "
                "run once in inventory mode with archive access."
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid source inventory snapshot: {path}")
        revision = record.get("resolved_revision", "")
        if (not isinstance(revision, str) or record.get("source_key") != key
                or not revision.startswith(f"{configured_revision}|inventory:inventory-sha256:")
                or record.get("checksum") != source_fingerprint(
                    {"source_key": key}, source_revision=revision
                )):
            raise ValueError(f"Invalid source inventory snapshot: {path}")
        return revision

    revision = f"{configured_revision}|inventory:{file_inventory_digest(list(paths), root=root)}"
    record = {
        "source_key": key,
        "resolved_revision": revision,
        "checksum": source_fingerprint({"source_key": key}, source_revision=revision),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return revision


def ocean_mask_identity(mask):
    """Identify the actual common domain, including its grid coordinates."""
    mask = mask.transpose("lat", "lon").compute()
    return source_fingerprint(
        {"lat": mask.lat.values.tolist(), "lon": mask.lon.values.tolist(),
         "valid": np.asarray(mask, dtype=bool).tolist()},
        source_revision="common_e3sm_ocean_mask_v1",
    )
