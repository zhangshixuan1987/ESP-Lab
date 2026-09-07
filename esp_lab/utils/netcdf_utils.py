"""Safe, reusable NetCDF cache I/O helpers."""

from __future__ import annotations

import os
import time
import uuid
import warnings
from pathlib import Path
from typing import Mapping

import xarray as xr


def cleanup_netcdf_temp_files(path, *, max_age_hours=24.0):
    """Remove stale temporary files created for one NetCDF destination."""
    path = Path(path)
    if max_age_hours < 0:
        raise ValueError("max_age_hours must be non-negative")
    if not path.parent.exists():
        return []

    cutoff = time.time() - (max_age_hours * 3600.0)
    removed = []
    for temporary in path.parent.glob(f".{path.name}.tmp.*"):
        try:
            if max_age_hours <= 0.0 or temporary.stat().st_mtime <= cutoff:
                temporary.unlink()
                removed.append(temporary)
        except OSError as exc:
            warnings.warn(
                f"could not remove temporary NetCDF file {temporary}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return removed


def atomic_to_netcdf(
    dataset: xr.Dataset,
    path,
    *,
    encoding: Mapping | None = None,
    cleanup_temporary=True,
    temp_file_max_age_hours=24.0,
):
    """Write a dataset atomically and return the destination path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cleanup_temporary:
        cleanup_netcdf_temp_files(path, max_age_hours=temp_file_max_age_hours)

    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        dataset.to_netcdf(temporary, encoding=encoding)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_netcdf(path):
    """Load a small NetCDF dataset into memory without leaving a file open."""
    with xr.open_dataset(path) as dataset:
        return dataset.load()


__all__ = ["atomic_to_netcdf", "cleanup_netcdf_temp_files", "load_netcdf"]
