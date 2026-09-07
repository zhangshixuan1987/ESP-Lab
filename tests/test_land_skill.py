import json
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from esp_lab import land_skill


LAND_NOTEBOOK = (
    Path(__file__).parents[1]
    / "jupyter"
    / "1b_refactor_lnd_leadtime_acc_skill_map.ipynb"
)


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


def test_mask_c3s_swe_flags_masks_negative_values_but_keeps_zero():
    swe = xr.DataArray(
        [-30.0, -20.0, -10.0, -1.0, 0.0, 12.0],
        dims="point",
        attrs={"units": "mm"},
    )

    out = land_skill.mask_c3s_swe_flags(swe)

    np.testing.assert_allclose(
        out.values, [np.nan, np.nan, np.nan, np.nan, 0.0, 12.0], equal_nan=True
    )
    assert out.attrs["units"] == "mm"
    assert "zero retained" in out.attrs["flag_treatment"]


def test_complete_calendar_seasonal_mean_rejects_sparse_false_season():
    time = pd.to_datetime(
        [
            "1999-12-01", "2000-01-01", "2000-02-01",
            "2000-03-01", "2000-04-01", "2000-05-01",
            "2000-10-01", "2000-11-01", "2000-12-01",
        ]
    )
    monthly = xr.DataArray(np.arange(len(time), dtype=float), dims="time", coords={"time": time})

    out = land_skill.complete_calendar_seasonal_mean(monthly)

    assert out.time.dt.strftime("%Y-%m").values.tolist() == ["2000-01", "2000-04"]
    assert float(out.sel(time="2000-01-01")) == pytest.approx(1.0)
    assert float(out.sel(time="2000-04-01")) == pytest.approx(4.0)


def test_complete_calendar_seasonal_mean_can_retain_missing_seasons():
    time = pd.date_range("1999-12-01", "2000-11-01", freq="MS")
    monthly = xr.DataArray(
        np.arange(len(time), dtype=float), dims="time", coords={"time": time}
    ).where(lambda da: ~da.time.dt.month.isin([6, 7, 8, 9]))

    out = land_skill.complete_calendar_seasonal_mean(monthly, retain_missing=True)

    assert out.time.dt.strftime("%Y-%m").values.tolist() == [
        "2000-01", "2000-04", "2000-07", "2000-10"
    ]
    assert out.sel(time="2000-07-01").isnull()
    assert out.sel(time="2000-10-01").isnull()
    assert out.attrs["missing_season_representation"] == "explicit all-NaN time slices"


def test_complete_calendar_monthly_change_does_not_bridge_gaps():
    monthly = xr.DataArray(
        [10.0, 13.0, 20.0, 25.0],
        dims="time",
        coords={"time": pd.to_datetime([
            "2000-04-01", "2000-05-01", "2000-10-01", "2000-11-01"
        ])},
        attrs={"units": "mm"},
    )

    out = land_skill.complete_calendar_monthly_change(monthly)

    assert float(out.sel(time="2000-05-01")) == pytest.approx(3.0)
    assert np.isnan(out.sel(time="2000-10-01"))
    assert float(out.sel(time="2000-11-01")) == pytest.approx(5.0)
    assert out.attrs["units"] == "mm"
    assert "SWE(t) - SWE(t-1)" in out.attrs["change_definition"]


def test_monthly_land_hindcast_change_uses_later_lead_and_time():
    y = ["2000050100"]
    monthly = xr.Dataset(
        {
            "H2OSNO": (("Y", "L", "M"), [[[2.0], [5.0], [4.0]]]),
            "time": (("Y", "L"), np.asarray([[
                cftime.DatetimeNoLeap(2000, 5, 15),
                cftime.DatetimeNoLeap(2000, 6, 15),
                cftime.DatetimeNoLeap(2000, 7, 15),
            ]], dtype=object)),
        },
        coords={"Y": y, "L": [1, 2, 3], "M": ["EN00"]},
    )

    out = land_skill.monthly_land_hindcast_change_dataset(monthly)

    assert out.L.values.tolist() == [2, 3]
    np.testing.assert_allclose(out.DELTA_H2OSNO.values.ravel(), [3.0, -1.0])
    assert out.time.dt.month.values.tolist() == [[6, 7]]


def test_monthly_land_hindcast_change_rejects_nonconsecutive_time():
    monthly = xr.Dataset(
        {
            "H2OSNO": (("Y", "L", "M"), [[[2.0], [5.0]]]),
            "time": (("Y", "L"), np.asarray([[
                cftime.DatetimeNoLeap(2000, 5, 15),
                cftime.DatetimeNoLeap(2000, 7, 15),
            ]], dtype=object)),
        },
        coords={"Y": ["2000050100"], "L": [1, 2], "M": ["EN00"]},
    )

    with pytest.raises(ValueError, match="consecutive months"):
        land_skill.monthly_land_hindcast_change_dataset(monthly)


def test_retain_reference_supported_leads_uses_verification_month():
    data = xr.DataArray(
        np.ones((2, 4, 1)),
        dims=("Y", "L", "M"),
        coords={"Y": [2000, 2001], "L": [3, 6, 9, 12], "M": [0]},
    )
    valid_time = xr.DataArray(
        np.asarray(
            [
                pd.to_datetime(["2000-07-01", "2000-10-01", "2001-01-01", "2001-04-01"]),
                pd.to_datetime(["2001-07-01", "2001-10-01", "2002-01-01", "2002-04-01"]),
            ]
        ),
        dims=("Y", "L"),
        coords={"Y": data.Y, "L": data.L},
    )
    reference = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0],
        dims="time",
        coords={"time": pd.to_datetime(["2000-01-01", "2000-04-01", "2001-01-01", "2001-04-01"])},
    )

    kept, kept_time, dropped = land_skill.retain_reference_supported_leads(
        data, valid_time, reference
    )

    assert kept.L.values.tolist() == [9, 12]
    assert kept_time.L.values.tolist() == [9, 12]
    assert dropped == [3, 6]


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


def test_staged_land_input_path_preserves_source_identity(tmp_path):
    path = land_skill.staged_land_input_path(
        tmp_path,
        "JRA55_FOSIRL",
        "H2OSOI",
        "1x1deg_cell_centered",
        init_month=5,
        depth_range_m=(0.0, 1.6),
    )

    assert path.parent == (
        tmp_path / "JRA55_FOSIRL" / "leadtime_acc" / "inputs" / "land" / "H2OSOI"
    )
    assert path.name == (
        "JRA55_FOSIRL05_H2OSOI_depth0-1.6m_integrated_mm_"
        "seasonal_1x1deg_cell_centered.nc"
    )


def test_land_cohort_token_changes_when_one_lead_cohort_changes():
    first = {3: [1980, 1981], 6: [1980, 1981]}
    second = {3: [1980, 1981], 6: [1980, 1982]}

    assert land_skill.land_cohort_token(first) != land_skill.land_cohort_token(second)
    assert land_skill.land_cohort_token(first).startswith(
        "y1980-1981_n2perlead_l3-6_nl2"
    )


def test_validate_land_skill_dataset_checks_exact_provenance_and_cohorts():
    cohorts = {3: [1980, 1981], 6: [1981, 1982]}
    expected_attrs = land_skill.expected_land_skill_attrs(
        field="H2OSNO",
        init_month=11,
        climatology_years=(1981, 2010),
        initialization_years=(1980, 2018),
        ensemble_members=("EN00", "EN01"),
        target_years_by_lead=cohorts,
        reference_product="C3S_SWE",
        reference_data_identity="sha256:reference",
        model_data_identity="sha256:model",
        target_grid="1x1deg_cell_centered",
        detrend=True,
        evaluation_protocol="protocol",
    )
    skill = xr.Dataset(
        {
            "corr": ("L", [0.1, 0.2]),
            "pval": ("L", [0.5, 0.4]),
            "sample_count": ("L", [2, 2]),
            "valid_sample_count": ("L", [2, 2]),
            "target_year_start": ("L", [1980, 1981]),
            "target_year_end": ("L", [1981, 1982]),
        },
        coords={"L": [3, 6]},
        attrs=expected_attrs,
    )

    land_skill.validate_land_skill_dataset(skill, expected_attrs, cohorts)
    skill.attrs["model_data_identity"] = "sha256:changed"
    with pytest.raises(ValueError, match="metadata mismatch"):
        land_skill.validate_land_skill_dataset(skill, expected_attrs, cohorts)


def test_land_acc_notebook_uses_explicit_reusable_workflow_contract():
    notebook = json.loads(LAND_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "os.environ" not in source
    assert "def " not in source
    assert 'detrend=RUN["detrend"]' in source
    assert "land_skill.land_skill_cache_status" in source
    assert "land_prepared_skill.prepare_reference_cache" in source
    assert "land_prepared_skill.prepare_model_cache" in source
    assert '"auto", "rebuild", "require"' in source
    assert '"inventory", "snapshot", "revision"' in source
    assert "Snapshot mode requires prepared_land.mode='require'" in source
    assert '"smoke_mode": False' in source
    assert "restart_notebook_cluster" in source
    assert "close_notebook_resources(globals())" in source
    assert '"memory_limit": "4GB"' in source
    assert "atomic_to_netcdf" in source
    assert "/tmp/esp_lab_1b" not in source


def test_all_land_notebook_cells_compile():
    from IPython.core.interactiveshell import InteractiveShell

    notebook = json.loads(LAND_NOTEBOOK.read_text())
    shell = InteractiveShell.instance()
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            transformed = shell.transform_cell("".join(cell.get("source", [])))
            compile(transformed, f"land notebook cell {index}", "exec")


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
