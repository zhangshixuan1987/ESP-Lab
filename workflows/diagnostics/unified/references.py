"""
references.py
=============
Reference assembly for Observation O_X and Free-Running E3SM Historical Climatology M_X.
"""

from __future__ import annotations

from typing import Optional, Sequence
import xarray as xr

from esp_lab.diagnostics.references_io import (
    build_e3sm_historical_climatology,
    build_observation_reference,
)


def assemble_references(
    obs_da: Optional[xr.DataArray] = None,
    historical_ds: Optional[xr.Dataset] = None,
    field: str = "TREFHT",
    frequency: str = "monthly",
    valid_times: Optional[Sequence[object]] = None,
) -> tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    obs_ref = (
        build_observation_reference(obs_da, valid_times)
        if obs_da is not None and valid_times is not None
        else obs_da
    )
    e3sm_clim = build_e3sm_historical_climatology(historical_ds, field=field, frequency=frequency)
    return obs_ref, e3sm_clim
