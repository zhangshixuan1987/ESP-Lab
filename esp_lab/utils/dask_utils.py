"""
Dask cluster utilities for ESP-Lab.  Canonical public name.

``esp_lab.utils.dask_util`` (without the 's') is the implementation module;
this shim re-exports everything so both names work.
"""
from .dask_util import (
    DaskConfig,
    get_cluster_client,
    close_cluster,
    maybe_persist,
    maybe_load,
)

__all__ = [
    "DaskConfig",
    "get_cluster_client",
    "close_cluster",
    "maybe_persist",
    "maybe_load",
]
