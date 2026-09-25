"""Validation helpers for lead-time hindcast preprocessing."""

from __future__ import annotations

import cftime
import xarray as xr


def check_remove_drift_sample(
    data,
    verification_time,
    anomaly,
    climatology,
    climatology_start_year,
    climatology_end_year,
    sample=None,
    name="case",
):
    """Validate drift removal at one configurable point without full loading."""
    start = cftime.DatetimeNoLeap(climatology_start_year, 1, 1, 0, 0, 0)
    end = cftime.DatetimeNoLeap(climatology_end_year, 12, 31, 23, 59, 59)
    sample = dict(sample or {})

    print(f"=== {name} ===")
    print("raw dims :", data.dims, data.shape)
    print("anom dims:", anomaly.dims, anomaly.shape)
    print("clim dims:", climatology.dims, climatology.shape)

    # Y and M must remain intact because the independent reconstruction below
    # averages over those dimensions. Sampling them first invalidates the check.
    reduction_dims = {"Y", "M"}
    point = {
        dim: min(int(sample.get(dim, 0)), data.sizes[dim] - 1)
        for dim in sample
        if dim in data.dims and dim not in reduction_dims and data.sizes[dim] > 0
    }
    data_point = data.isel(point)
    anomaly_point = anomaly.isel(
        {dim: index for dim, index in point.items() if dim in anomaly.dims}
    )
    climatology_point = climatology.isel(
        {dim: index for dim, index in point.items() if dim in climatology.dims}
    )
    time_point = verification_time.isel(
        {dim: index for dim, index in point.items() if dim in verification_time.dims}
    )

    masked = data_point.where((time_point >= start) & (time_point <= end))
    manual_climatology = masked.mean("M").mean("Y")
    manual_anomaly = data_point - manual_climatology
    masked_anomaly = anomaly_point.where((time_point >= start) & (time_point <= end))

    xr.testing.assert_allclose(climatology_point.load(), manual_climatology.load())
    xr.testing.assert_allclose(anomaly_point.load(), manual_anomaly.load())
    print(f"sample reconstruction PASS: {point}")

    residual = masked_anomaly.mean(("Y", "M")).load()
    print("sample max abs masked anomaly mean:", float(abs(residual).max(skipna=True)))


__all__ = ["check_remove_drift_sample"]
