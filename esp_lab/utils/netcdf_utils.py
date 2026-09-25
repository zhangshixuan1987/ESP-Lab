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


# Datasets up to this size are computed into memory before writing (see below).
MAX_IN_MEMORY_WRITE_BYTES = 4 * 1024**3


def atomic_to_netcdf(
    dataset: xr.Dataset,
    path,
    *,
    encoding: Mapping | None = None,
    cleanup_temporary=True,
    temp_file_max_age_hours=24.0,
    max_in_memory_bytes=MAX_IN_MEMORY_WRITE_BYTES,
):
    """Write a dataset atomically and return the destination path.

    Dask-backed datasets up to ``max_in_memory_bytes`` are computed first (on
    whatever scheduler is active) and then written by the calling process.
    Letting dask.distributed workers write the NetCDF chunks themselves takes
    xarray's HDF5 write lock inside the workers, and that lock has been seen to
    stay held after a task, hanging the write indefinitely.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cleanup_temporary:
        cleanup_netcdf_temp_files(path, max_age_hours=temp_file_max_age_hours)

    if dataset.chunks:
        if dataset.nbytes <= max_in_memory_bytes:
            dataset = dataset.compute()
        else:
            warnings.warn(
                f"writing {dataset.nbytes / 1024**3:.1f} GiB dataset {path.name} lazily; "
                "distributed NetCDF writes can hang (see atomic_to_netcdf)",
                RuntimeWarning,
                stacklevel=2,
            )

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
