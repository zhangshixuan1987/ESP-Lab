import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr
import cftime


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_process_sst_index.py"
SPEC = importlib.util.spec_from_file_location("run_process_sst_index", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_observed_iod_is_west_anomaly_minus_east_anomaly():
    time = xr.date_range(
        "1980-01-01",
        "2011-12-01",
        freq="MS",
        calendar="noleap",
        use_cftime=True,
    )
    month = np.asarray(time.month)
    year = np.asarray(time.year)

    # Give the two regions different means and seasonal cycles so subtracting
    # absolute SSTs would be detectably different from the expected DMI.
    west = xr.DataArray(
        28.0 + 2.0 * np.sin(2.0 * np.pi * month / 12.0) + 0.1 * (year - 1980),
        dims="time",
        coords={"time": time},
    )
    east = xr.DataArray(
        24.0 + np.cos(2.0 * np.pi * month / 12.0) + 0.04 * (year - 1980),
        dims="time",
        coords={"time": time},
    )

    result = MODULE.derive_indices(
        {"IOD_West": west, "IOD_East": east},
        west["time"],
        1981,
        2010,
        is_model=False,
    )["IOD"]
    expected = MODULE.compute_obs_anom(west, 1981, 2010) - MODULE.compute_obs_anom(
        east,
        1981,
        2010,
    )

    xr.testing.assert_allclose(result, expected)
    clim = result.sel(time=slice("1981-01-01", "2010-12-31"))
    monthly_climatology = clim.groupby("time.month").mean("time")
    assert float(np.abs(monthly_climatology).max()) < 1.0e-12
    assert result.attrs["index_name"] == "DMI"
    assert result.attrs["climatology_start_year"] == 1981
    assert result.attrs["climatology_end_year"] == 2010


def test_derived_model_indices_preserve_missing_final_season():
    years = np.arange(1981, 2011)
    leads = np.arange(1, 9)
    time = xr.DataArray(
        np.asarray(
            [
                [cftime.DatetimeNoLeap(int(year), 1 + 3 * (lead - 1), 15)
                 if 1 + 3 * (lead - 1) <= 12
                 else cftime.DatetimeNoLeap(
                     int(year) + (3 * (lead - 1)) // 12,
                     1 + (3 * (lead - 1)) % 12,
                     15,
                 )
                 for lead in leads]
                for year in years
            ],
            dtype=object,
        ),
        dims=("Y", "L"),
        coords={"Y": years, "L": leads},
    )
    shape = (years.size, leads.size, 2)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape) / 100.0
    base[:, -1, :] = np.nan

    def series(offset):
        return xr.DataArray(
            base + offset,
            dims=("Y", "L", "M"),
            coords={"Y": years, "L": leads, "M": [1, 2]},
        )

    result = MODULE.derive_indices(
        {
            "Nino3.4": series(0.0),
            "Nino12": series(0.5),
            "Nino4": series(-0.5),
            "TropicalMean": series(0.25),
        },
        time,
        1981,
        2010,
        is_model=True,
    )

    for name in ("ONI", "TNI", "RONI"):
        assert result[name].isel(L=-1).isnull().all()
        assert result[name].isel(L=-2).notnull().all()


def test_output_cache_tracks_sst_preprocessing_and_land_mask(tmp_path):
    path = tmp_path / "index.nc"
    dataset = xr.Dataset({"sst": xr.DataArray([1.0], dims="time")})
    MODULE._safe_to_netcdf(dataset, path, sst_land_mask=True)

    assert MODULE._output_is_current(path, SimpleNamespace(sst_land_mask=True))
    assert not MODULE._output_is_current(path, SimpleNamespace(sst_land_mask=False))


def test_smyle_fixed_mask_is_model_level(tmp_path):
    timeseries, mask = MODULE._smyle_output_paths(tmp_path / "CESM-SMYLE")

    assert timeseries == tmp_path / "CESM-SMYLE" / "sst_index" / "timeseries"
    assert mask == tmp_path / "CESM-SMYLE" / "fixed" / "sftlf.CESM-SMYLE.nc"
