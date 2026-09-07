import numpy as np
import xarray as xr

from esp_lab.utils import sst_utils


def _sst(values, units):
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("lat", "lon"),
        coords={"lat": [-1.0, 1.0], "lon": [120.0, 122.0]},
        attrs={"units": units},
        name="TS",
    )


def test_normalize_sst_respects_celsius_units():
    data = _sst([[20.0, 21.0], [22.0, 23.0]], "degC")
    actual = sst_utils.normalize_sst_to_degc(data)
    xr.testing.assert_allclose(actual, data)
    assert actual.attrs["units"] == "degC"


def test_normalize_sst_converts_kelvin_units():
    data = _sst([[293.15, 294.15], [295.15, 296.15]], "K")
    actual = sst_utils.normalize_sst_to_degc(data)
    np.testing.assert_allclose(actual, [[20.0, 21.0], [22.0, 23.0]])
    assert actual.attrs["units"] == "degC"


def test_normalize_sst_infers_missing_units():
    data = _sst([[293.15, 294.15], [295.15, 296.15]], "")
    actual = sst_utils.normalize_sst_to_degc(data)
    np.testing.assert_allclose(actual, [[20.0, 21.0], [22.0, 23.0]])
    assert actual.attrs["sst_unit_inference"].startswith("sample_mean=")


def test_normalize_sst_infers_missing_units_on_chunked_data():
    data = _sst([[293.15, 294.15], [295.15, 296.15]], "").chunk(
        {"lat": 1, "lon": 1}
    )
    actual = sst_utils.normalize_sst_to_degc(data)
    np.testing.assert_allclose(actual.compute(), [[20.0, 21.0], [22.0, 23.0]])


def test_prepare_sst_masks_nonphysical_values_and_land(tmp_path, monkeypatch):
    data = _sst([[293.15, 999.0], [295.15, 296.15]], "K")

    def fake_land_mask(data, *, lon_name, lat_name):
        return xr.DataArray(
            [[0, 0], [1, 0]],
            dims=(lat_name, lon_name),
            coords={lat_name: data[lat_name], lon_name: data[lon_name]},
            name="sftlf",
        ).astype("int8")

    monkeypatch.setattr(sst_utils, "_natural_earth_land_mask", fake_land_mask)
    path = tmp_path / "fixed" / "sftlf.test.nc"
    actual, land = sst_utils.prepare_sst(
        data,
        land_mask_path=path,
        source="test-source",
    )

    assert path.exists()
    assert bool(land.sel(lat=1.0, lon=120.0))
    assert np.isnan(actual.sel(lat=-1.0, lon=122.0))  # nonphysical
    assert np.isnan(actual.sel(lat=1.0, lon=120.0))  # land
    assert float(actual.sel(lat=-1.0, lon=120.0)) == 20.0


def test_cached_land_mask_is_rebuilt_for_a_different_grid(tmp_path, monkeypatch):
    calls = []

    def fake_land_mask(data, *, lon_name, lat_name):
        calls.append(tuple(float(v) for v in data[lon_name].values))
        return xr.zeros_like(
            data.isel({dim: 0 for dim in data.dims if dim not in {lat_name, lon_name}}),
            dtype="int8",
        ).rename("sftlf")

    monkeypatch.setattr(sst_utils, "_natural_earth_land_mask", fake_land_mask)
    path = tmp_path / "sftlf.test.nc"
    first = _sst([[20.0, 21.0], [22.0, 23.0]], "degC")
    second = first.assign_coords(lon=[121.0, 123.0])

    sst_utils.load_or_create_land_mask(first, path, source="test")
    sst_utils.load_or_create_land_mask(first, path, source="test")
    sst_utils.load_or_create_land_mask(second, path, source="test")

    assert calls == [(120.0, 122.0), (121.0, 123.0)]
