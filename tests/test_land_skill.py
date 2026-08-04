import cftime
import numpy as np
import pytest
import xarray as xr

from esp_lab import land_skill


def _land_dataset():
    return xr.Dataset(
        {
            "H2OSNO": (("time", "lat", "lon"), np.ones((2, 2, 2))),
            "TWS": (("time", "lat", "lon"), np.ones((2, 2, 2)) * 2),
            "H2OSOI": (
                ("time", "levgrnd", "lat", "lon"),
                np.arange(24, dtype=float).reshape(2, 3, 2, 2),
            ),
        },
        coords={
            "time": [0, 1],
            "levgrnd": ("levgrnd", [0.007, 0.028, 0.062], {"units": "m"}),
            "lat": [-0.5, 0.5],
            "lon": [0.5, 1.5],
        },
    )


@pytest.mark.parametrize("field", ["H2OSNO", "TWS"])
def test_prepare_land_field_keeps_map_fields(field):
    out = land_skill.prepare_land_field(_land_dataset(), field)

    assert out.name == field
    assert out.dims == ("time", "lat", "lon")
    assert out.attrs["units"] == "mm"


def test_prepare_h2osoi_selects_explicit_surface_layer():
    ds = _land_dataset()
    out = land_skill.prepare_land_field(ds, "H2OSOI", soil_layer=0)

    xr.testing.assert_equal(out, ds.H2OSOI.isel(levgrnd=0).rename("H2OSOI"))
    assert out.attrs["selected_soil_depth_m"] == pytest.approx(0.007)


def test_prepare_h2osoi_selects_nearest_depth():
    out = land_skill.prepare_land_field(_land_dataset(), "H2OSOI", soil_depth_m=0.05)

    assert float(out.levgrnd) == pytest.approx(0.062)
    assert out.attrs["soil_layer_selection"] == "nearest_depth"


def test_prepare_h2osoi_rejects_ambiguous_vertical_selection():
    with pytest.raises(ValueError, match="exactly one"):
        land_skill.prepare_land_field(
            _land_dataset(), "H2OSOI", soil_layer=0, soil_depth_m=0.05
        )


def test_prepare_h2osoi_requires_vertical_definition():
    with pytest.raises(ValueError, match="requires exactly one"):
        land_skill.prepare_land_field(_land_dataset(), "H2OSOI")


def test_depth_weighted_soil_moisture_uses_layer_overlap():
    da = xr.DataArray(
        [1.0, 3.0, 5.0],
        dims="depth",
        coords={"depth": [0.01, 0.035, 0.075]},
        name="H2OSOI",
    )
    out = land_skill.depth_weighted_soil_moisture(
        da,
        (0.0, 0.06),
        vertical_dim="depth",
        layer_bounds_m=[0.0, 0.02, 0.05, 0.1],
    )

    assert float(out) == pytest.approx((1 * 0.02 + 3 * 0.03 + 5 * 0.01) / 0.06)
    assert out.attrs["depth_top_m"] == 0.0
    assert out.attrs["depth_bottom_m"] == 0.06


def test_depth_weighted_soil_moisture_masks_incomplete_local_column():
    da = xr.DataArray(
        [[1.0, 1.0], [3.0, np.nan]],
        dims=("depth", "point"),
        coords={"depth": [0.025, 0.075], "point": [0, 1]},
    )
    out = land_skill.depth_weighted_soil_moisture(
        da,
        (0.0, 0.1),
        vertical_dim="depth",
        layer_bounds_m=[0.0, 0.05, 0.1],
    )

    np.testing.assert_allclose(out.values, [2.0, np.nan])


def test_depth_integrated_soil_water_converts_volumetric_mean_to_mm():
    da = xr.DataArray(
        [0.2, 0.4],
        dims="depth",
        coords={"depth": [0.4, 1.2]},
        name="H2OSOI",
        attrs={"units": "mm3/mm3"},
    )
    out = land_skill.depth_integrated_soil_water_mm(
        da,
        (0.0, 1.6),
        vertical_dim="depth",
        layer_bounds_m=[0.0, 0.8, 1.6],
    )

    assert float(out) == pytest.approx(480.0)
    assert out.attrs["units"] == "mm"
    assert out.attrs["depth_bottom_m"] == 1.6


def test_validate_h2osoi_requires_matching_depth_intervals():
    forecast = xr.DataArray(
        [[1.0]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0]},
        name="H2OSOI",
        attrs={"depth_top_m": 0.0, "depth_bottom_m": 0.1},
    )
    reference = forecast.copy()
    reference.attrs["depth_bottom_m"] = 0.2

    with pytest.raises(ValueError, match="depth intervals differ"):
        land_skill.validate_land_reference_compatibility(forecast, reference)


def test_validate_grid_ignores_coordinate_metadata_differences():
    forecast = xr.DataArray(
        [[1.0]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0]},
        name="TWS",
    )
    reference = forecast.copy()
    forecast.lat.attrs["bounds"] = "lat_bnds"
    reference.lat.attrs["long_name"] = "latitude"

    land_skill.validate_land_reference_compatibility(forecast, reference)


def test_validate_grid_rejects_different_coordinate_values():
    forecast = xr.DataArray(
        [[1.0]],
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0]},
        name="TWS",
    )
    reference = forecast.assign_coords(lat=[0.5])

    with pytest.raises(ValueError, match="lat coordinates"):
        land_skill.validate_land_reference_compatibility(forecast, reference)


def test_compute_land_acc_rejects_unselected_soil_levels():
    forecast = xr.DataArray(
        np.ones((4, 1, 2, 3)),
        dims=("Y", "L", "M", "levgrnd"),
        coords={"Y": range(4), "L": [1], "M": range(2), "levgrnd": range(3)},
    )
    valid_time = xr.DataArray(
        np.ones((4, 1)), dims=("Y", "L"), coords={"Y": range(4), "L": [1]}
    )
    reference = xr.DataArray(np.ones(4), dims="time", coords={"time": range(4)})

    with pytest.raises(ValueError, match="select an H2OSOI layer"):
        land_skill.compute_land_acc_skill(forecast, valid_time, reference, 2000, 2003)


def test_load_land_hindcast_forces_land_realm(monkeypatch):
    called = {}

    def fake_get_monthly_data(**kwargs):
        called.update(kwargs)
        return xr.Dataset()

    monkeypatch.setattr(land_skill.data_access_e3sm, "get_monthly_data", fake_get_monthly_data)
    land_skill.load_e3sm_land_monthly(
        data_dir="/data",
        case_prefix="case",
        members=["EN00"],
        init_tags=["2000050100"],
        field="TWS",
    )

    assert called["realm"] == "lnd"
    assert called["field"] == "TWS"
    assert called["verify_field_name"] is True


def test_seasonal_land_hindcast_preserves_dataset_time_variable(monkeypatch):
    monthly = _land_dataset().rename({"time": "L"}).expand_dims(Y=[2000], M=[0])
    monthly["time"] = xr.DataArray(
        np.arange(2).reshape(1, 2),
        dims=("Y", "L"),
        coords={"Y": monthly.Y, "L": monthly.L},
    )

    def fake_mon_to_seas(ds):
        assert "time" in ds
        return ds

    monkeypatch.setattr(land_skill.calendar_utils, "mon_to_seas_dask", fake_mon_to_seas)
    out = land_skill.seasonal_land_hindcast(monthly, "TWS")

    assert out.name == "TWS"


def test_seasonal_land_hindcast_dataset_returns_valid_time(monkeypatch):
    monthly = _land_dataset().rename({"time": "L"}).expand_dims(Y=[2000], M=[0])
    monthly["time"] = xr.DataArray(
        np.arange(2).reshape(1, 2),
        dims=("Y", "L"),
        coords={"Y": monthly.Y, "L": monthly.L},
    )
    monkeypatch.setattr(land_skill.calendar_utils, "mon_to_seas_dask", lambda ds: ds)

    out = land_skill.seasonal_land_hindcast_dataset(monthly, "H2OSNO")

    assert {"H2OSNO", "time"}.issubset(out)


def test_retain_valid_seasonal_leads_drops_all_missing_endpoint():
    data = xr.DataArray(
        np.ones((2, 3, 2)),
        dims=("Y", "L", "M"),
        coords={"Y": [2000, 2001], "L": [3, 6, 9], "M": [0, 1]},
    )
    data.loc[{"L": 9}] = np.nan
    data.loc[{"Y": 2000, "L": 6, "M": 0}] = np.nan
    valid_time = xr.DataArray(
        np.arange(6).reshape(2, 3),
        dims=("Y", "L"),
        coords={"Y": data.Y, "L": data.L},
    )

    kept, kept_time, dropped = land_skill.retain_valid_seasonal_leads(data, valid_time)

    assert kept.L.values.tolist() == [3, 6]
    assert kept_time.L.values.tolist() == [3, 6]
    assert dropped == [9]


def test_validate_hindcast_evaluation_setup_enforces_members_and_climatology():
    years = [2000, 2001, 2002]
    ylabels = [f"{year}050100" for year in years]
    forecast = xr.DataArray(
        np.ones((3, 1, 2, 1, 1)),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={
            "Y": ylabels, "L": [3], "M": ["EN00", "EN01"],
            "lat": [0.0], "lon": [0.0],
        },
    )
    valid_time = xr.DataArray(
        np.asarray(
            [[cftime.DatetimeNoLeap(year, 7, 15)] for year in years],
            dtype=object,
        ),
        dims=("Y", "L"),
        coords={"Y": ylabels, "L": [3]},
    )

    target_years = land_skill.validate_hindcast_evaluation_setup(
        forecast, valid_time, init_month=5,
        initialization_years=(2000, 2002),
        expected_members=["EN00", "EN01"],
        climatology_years=(2000, 2002),
    )
    assert target_years == {3: years}

    incomplete = forecast.copy()
    incomplete.loc[{"Y": "2001050100", "M": "EN01", "lat": 0.0, "lon": 0.0}] = np.nan
    with pytest.raises(ValueError, match="fewer than the requested ensemble"):
        land_skill.validate_hindcast_evaluation_setup(
            incomplete, valid_time, init_month=5,
            initialization_years=(2000, 2002),
            expected_members=["EN00", "EN01"],
            climatology_years=(2000, 2002),
        )


def test_validate_reference_time_coverage_rejects_missing_target_season():
    years = [2000, 2001, 2002]
    ylabels = [f"{year}050100" for year in years]
    valid_time = xr.DataArray(
        np.asarray(
            [[cftime.DatetimeNoLeap(year, 7, 15)] for year in years],
            dtype=object,
        ),
        dims=("Y", "L"), coords={"Y": ylabels, "L": [3]},
    )
    reference = xr.DataArray(
        np.ones(2), dims="time",
        coords={"time": [
            cftime.DatetimeNoLeap(2000, 7, 1),
            cftime.DatetimeNoLeap(2002, 7, 1),
        ]},
    )

    with pytest.raises(ValueError, match="reference lacks"):
        land_skill.validate_reference_time_coverage(
            reference, {3: years}, valid_time, (2000, 2002)
        )
