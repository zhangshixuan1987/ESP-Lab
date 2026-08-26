import os
import time

import numpy as np
import pytest
import xarray as xr

from esp_lab.utils.netcdf_utils import (
    atomic_to_netcdf,
    cleanup_netcdf_temp_files,
    load_netcdf,
)
from esp_lab.utils.resource_utils import ResourceTracker
from esp_lab.utils.unit_conversion import (
    convert_kelvin_to_celsius,
    convert_pa_to_hpa,
    convert_precip_mps_to_mmday,
    no_unit_conversion,
)


class _Resource:
    def __init__(self, name, closed):
        self.name = name
        self.closed = closed

    def close(self):
        self.closed.append(self.name)


def test_resource_tracker_closes_unique_resources_in_reverse_order():
    closed = []
    first = _Resource("first", closed)
    second = _Resource("second", closed)
    tracker = ResourceTracker(first, second)

    assert tracker.track(first) is first
    assert tracker.track(None) is None
    tracker.close()
    tracker.close()

    assert closed == ["first", "second"]
    with pytest.raises(RuntimeError, match="closed"):
        tracker.track(_Resource("late", closed))


def test_atomic_netcdf_round_trip_and_temp_cleanup(tmp_path):
    destination = tmp_path / "cache.nc"
    stale = tmp_path / ".cache.nc.tmp.stale"
    stale.write_text("incomplete")
    old_time = time.time() - 7200
    os.utime(stale, (old_time, old_time))

    removed = cleanup_netcdf_temp_files(destination, max_age_hours=1)
    assert removed == [stale]

    dataset = xr.Dataset({"value": ("x", [1.0, 2.0])})
    assert atomic_to_netcdf(dataset, destination) == destination
    xr.testing.assert_identical(load_netcdf(destination), dataset)
    assert not list(tmp_path.glob(".cache.nc.tmp.*"))


def test_unit_conversions_preserve_lazy_xarray_behavior():
    kelvin = xr.DataArray([273.15, 274.15], dims="x", attrs={"units": "K"})
    precip = xr.DataArray([1.0e-3], dims="x", attrs={"units": "m/s"})
    pressure = xr.DataArray([100000.0], dims="x", attrs={"units": "Pa"})

    np.testing.assert_allclose(convert_kelvin_to_celsius(kelvin), [0.0, 1.0])
    np.testing.assert_allclose(convert_precip_mps_to_mmday(precip), [86400.0])
    np.testing.assert_allclose(convert_pa_to_hpa(pressure), [1000.0])
    assert convert_kelvin_to_celsius(kelvin).attrs["units"] == r"$^\circ$C"
    assert convert_precip_mps_to_mmday(precip).attrs["units"] == "mm/day"
    assert convert_pa_to_hpa(pressure).attrs["units"] == "hPa"
    assert no_unit_conversion(kelvin, units="standardized").attrs["units"] == "standardized"
