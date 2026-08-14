"""Synthetic validation of start-specific, two-reference drift diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from esp_lab.diagnostics.two_reference_drift import (
    area_weighted_rmse,
    bootstrap_paired_mean_ci,
    classify_drift_regime,
    compare_initializations,
    compute_diagnostics,
    compute_lead_window_mean,
    compute_regime_fraction,
    compute_spatial_drift_summary,
    match_model_climatology_to_valid_time,
    match_observation_to_valid_time,
    regional_subset,
    build_attractor_lookup,
    compute_early_drift_late_error,
    compare_drift_skill_relationship,
    run_pipeline,
)
from workflows.diagnostics.two_reference_drift import (
    apply_unit_transform,
    apply_spread_unit_transform,
    build_monthly_attractor_climatology,
    discover_monthly_hindcast,
    load_drift_references,
    prepare_monthly_hindcast_cache,
    select_initialization_years,
    _regrid_field,
)


def _valid_time() -> xr.DataArray:
    return xr.DataArray(
        np.asarray(
            [
                ["2000-05-15", "2000-06-15", "2000-07-15"],
                ["2001-05-15", "2001-06-15", "2001-07-15"],
            ],
            dtype="datetime64[ns]",
        ),
        dims=("Y", "L"),
        coords={"Y": [2000, 2001], "L": [1, 2, 3]},
        name="valid_time",
    )


def _references():
    valid_time = _valid_time()
    times = pd.date_range("2000-01-01", periods=24, freq="MS") + pd.Timedelta(days=14)
    obs = xr.DataArray(
        np.asarray([timestamp.month for timestamp in times], dtype=float),
        dims="time",
        coords={"time": times},
        attrs={"units": "degC"},
    ).expand_dims(lat=[-30.0, 30.0], lon=[0.0, 90.0])
    climatology = xr.DataArray(
        np.arange(1.0, 13.0),
        dims="month",
        coords={"month": np.arange(1, 13)},
        attrs={"units": "degC"},
    ).expand_dims(lat=[-30.0, 30.0], lon=[0.0, 90.0])
    return valid_time, obs, climatology


def test_reference_matching_uses_exact_valid_year_month_and_calendar_month():
    valid_time, obs, climatology = _references()
    obs_ref = match_observation_to_valid_time(obs, valid_time)
    att_ref = match_model_climatology_to_valid_time(climatology, valid_time)
    assert obs_ref.dims == ("lat", "lon", "Y", "L")
    np.testing.assert_array_equal(obs_ref.isel(lat=0, lon=0), [[5, 6, 7], [5, 6, 7]])
    np.testing.assert_array_equal(att_ref.isel(lat=0, lon=0), [[5, 6, 7], [5, 6, 7]])
    np.testing.assert_array_equal(obs_ref.valid_time, valid_time)
    np.testing.assert_array_equal(att_ref.valid_time, valid_time)


def test_observation_matching_rejects_incomplete_coverage():
    valid_time, obs, _ = _references()
    obs = obs.where(
        ~((obs.time.dt.year == 2001) & (obs.time.dt.month == 7)), drop=True
    )
    with pytest.raises(ValueError, match="missing"):
        match_observation_to_valid_time(obs, valid_time)


def test_full_product_retains_y_averages_only_m_and_has_correct_formulas():
    valid_time, obs, climatology = _references()
    obs_ref = match_observation_to_valid_time(obs, valid_time)
    att_ref = match_model_climatology_to_valid_time(climatology + 2.0, valid_time)
    att_ref.attrs["units"] = "degC"

    # Ensemble mean offsets from obs are [2, 1, 3], so distance changes are [0, -1, +1].
    offsets = xr.DataArray([2.0, 1.0, 3.0], dims="L", coords={"L": [1, 2, 3]})
    mean = obs_ref + offsets
    members = xr.concat([mean - 1.0, mean + 1.0], dim=xr.IndexVariable("M", [0, 1]))
    members.attrs["units"] = "degC"

    product = compute_diagnostics(members, obs_ref, att_ref, distance_tolerance=1.0e-10)
    assert "Y" in product.dims
    assert "M" not in product.dims
    np.testing.assert_allclose(product.e_obs.isel(Y=0, lat=0, lon=0), [2, 1, 3])
    np.testing.assert_allclose(product.e_att.isel(Y=0, lat=0, lon=0), [0, -1, 1])
    np.testing.assert_allclose(product.delta_e_obs.isel(Y=0, lat=0, lon=0), [0, -1, 1])
    np.testing.assert_allclose(product.delta_abs_e_obs.isel(Y=0, lat=0, lon=0), [0, -1, 1])
    np.testing.assert_allclose(product.delta_abs_e_att.isel(Y=0, lat=0, lon=0), [0, 1, 1])
    np.testing.assert_array_equal(product.regime.isel(Y=0, lat=0, lon=0), [0, 2, 4])
    assert product.attrs["baseline_lead"] == 1


def test_units_and_exact_grid_are_required():
    valid_time, obs, climatology = _references()
    obs_ref = match_observation_to_valid_time(obs, valid_time)
    att_ref = match_model_climatology_to_valid_time(climatology, valid_time)
    members = obs_ref.expand_dims(M=[0])
    members.attrs["units"] = "K"
    with pytest.raises(ValueError, match="units"):
        compute_diagnostics(members, obs_ref, att_ref)
    members.attrs["units"] = "degC"
    with pytest.raises(ValueError):
        compute_diagnostics(members, obs_ref.isel(lon=[0]), att_ref)


def test_regime_tolerance_and_nan_policy():
    obs = xr.DataArray([-2.0, -2.0, 2.0, 2.0, 0.01, np.nan], dims="cell")
    att = xr.DataArray([-2.0, 2.0, -2.0, 2.0, 2.0, -2.0], dims="cell")
    regime = classify_drift_regime(obs, att, tolerance=0.1)
    np.testing.assert_array_equal(regime.values[:5], [1, 2, 3, 4, 0])
    assert np.isnan(regime.values[5])


def test_lead_windows_area_weighted_rms_and_regime_fractions():
    field = xr.DataArray(
        np.asarray([[[3.0, 4.0], [0.0, 0.0]], [[5.0, 6.0], [0.0, 0.0]]]),
        dims=("L", "lat", "lon"),
        coords={"L": [1, 2], "lat": [-60.0, 0.0], "lon": [0.0, 90.0]},
        attrs={"units": "degC"},
    )
    window = compute_lead_window_mean(field, [1, 2])
    np.testing.assert_allclose(window, [[4.0, 5.0], [0.0, 0.0]])
    rms = area_weighted_rmse(field)
    expected_l1 = np.sqrt((0.5 * (9 + 16) + 1.0 * 0) / (2 * 0.5 + 2 * 1.0))
    np.testing.assert_allclose(float(rms.sel(L=1)), expected_l1)

    regime = xr.DataArray(
        [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [np.nan, 4.0]]],
        dims=("L", "lat", "lon"),
        coords=field.coords,
    )
    fractions = compute_regime_fraction(regime, leads=[1, 2])
    np.testing.assert_allclose(
        sum(float(fractions[name]) for name in fractions.data_vars), 1.0
    )


def test_region_crosses_dateline_and_comparison_requires_paired_coordinates():
    data = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("lat", "lon"),
        coords={"lat": [-10, 0, 10], "lon": [0, 90, 180, 270]},
    )
    subset = regional_subset(data, lon_bounds=(260, 20), lat_bounds=(-5, 5))
    np.testing.assert_array_equal(subset.lon, [0, 270])
    left = xr.DataArray([2.0, 3.0], dims="Y", coords={"Y": [2000, 2001]})
    right = xr.DataArray([1.0, 1.0], dims="Y", coords={"Y": [2000, 2001]})
    np.testing.assert_allclose(compare_initializations(left, right), [1.0, 2.0])
    with pytest.raises(ValueError):
        compare_initializations(left, right.assign_coords(Y=[2001, 2002]))


def test_attractor_climatology_requires_complete_unique_monthly_coverage():
    times = pd.date_range("2000-01-01", periods=24, freq="MS") + pd.Timedelta(days=14)
    historical = xr.DataArray(
        np.asarray([timestamp.month for timestamp in times], dtype=float),
        dims="time",
        coords={"time": times},
        attrs={"units": "K"},
    )
    climatology = build_monthly_attractor_climatology(
        historical, start_year=2000, end_year=2001
    )
    np.testing.assert_allclose(climatology, np.arange(1.0, 13.0))

    with pytest.raises(ValueError, match="missing 1 monthly record"):
        build_monthly_attractor_climatology(
            historical.isel(time=slice(None, -1)), start_year=2000, end_year=2001
        )
    with pytest.raises(ValueError, match="duplicate monthly records"):
        build_monthly_attractor_climatology(
            xr.concat([historical, historical.isel(time=[0])], dim="time"),
            start_year=2000,
            end_year=2001,
        )


def test_monthly_discovery_rejects_seasonal_cache(tmp_path):
    job = {
        "source": "Reanalysis",
        "component": "atm",
        "variable": "TREFHT",
        "init_month": 5,
    }
    directory = tmp_path / "Reanalysis" / "leadtime_acc" / "inputs" / "atm" / "TREFHT"
    directory.mkdir(parents=True)

    def write_cache(path, leads):
        years = [2000, 2001]
        valid_time = xr.DataArray(
            np.full((len(years), len(leads)), np.datetime64("2000-05-15")),
            dims=("Y", "L"),
            coords={"Y": years, "L": leads},
        )
        xr.Dataset(
            {
                "TREFHT": (("Y", "M", "L"), np.ones((len(years), 1, len(leads)))),
                "time": valid_time,
            },
            coords={"Y": years, "M": [0], "L": leads},
        ).to_netcdf(path)

    seasonal = directory / "Reanalysis_05_TREFHT_mon_seas.nc"
    write_cache(seasonal, [1, 4, 7, 10])
    with pytest.raises(FileNotFoundError, match="Seasonal .* caches are not valid"):
        discover_monthly_hindcast(job, root=tmp_path)

    monthly = directory / "Reanalysis_05_TREFHT_monthly_mon.nc"
    write_cache(monthly, np.arange(1, 25))
    assert discover_monthly_hindcast(job, root=tmp_path) == monthly


def test_daily_attractor_lookup_maps_day_of_year_to_y_l():
    valid_time = xr.DataArray(
        np.asarray([["2001-01-01", "2001-02-01"]], dtype="datetime64[ns]"),
        dims=("Y", "L"), coords={"Y": [2001], "L": [1, 2]},
    )
    climatology = xr.DataArray(
        np.arange(1.0, 366.0), dims="dayofyear", coords={"dayofyear": np.arange(1, 366)}
    )
    matched = build_attractor_lookup(climatology, valid_time, frequency="daily")
    np.testing.assert_array_equal(matched, [[1.0, 32.0]])


def test_regional_regime_frequency_and_start_specific_drift_skill():
    regime = xr.DataArray(
        [[1, 2, 3], [1, 4, 3]], dims=("Y", "L"),
        coords={"Y": [2000, 2001], "L": [1, 2, 3]},
    )
    fractions = compute_regime_fraction(regime, leads=[1, 2, 3], sample_dims=["Y"])
    np.testing.assert_allclose(
        sum(float(fractions[name]) for name in fractions.data_vars), 1.0
    )

    drift = xr.DataArray(
        [[0.0, -1.0, -2.0, -3.0], [0.0, -2.0, -4.0, -6.0]],
        dims=("Y", "L"), coords={"Y": [2000, 2001], "L": [1, 2, 3, 4]},
    )
    error = -drift
    metrics = compute_early_drift_late_error(
        drift, error, early_leads=[1, 2], late_leads=[3, 4]
    )
    assert metrics.early_drift.dims == ("Y",)
    assert float(metrics.correlation) == pytest.approx(-1.0)
    paired = compare_drift_skill_relationship(metrics * 2, metrics)
    assert set(paired.data_vars) == {"delta_drift", "delta_skill", "correlation"}


def test_prepared_reference_loader_inspects_schema_and_validates_hindcast(tmp_path):
    valid_time = _valid_time()
    values = xr.DataArray(
        np.ones((2, 1, 3)), dims=("Y", "M", "L"),
        coords={"Y": valid_time.Y, "M": [0], "L": valid_time.L},
        attrs={"units": "degC"},
    ).assign_coords(valid_time=valid_time)
    ref = xr.DataArray(
        np.ones((2, 3)), dims=("Y", "L"), coords={"Y": valid_time.Y, "L": valid_time.L},
        attrs={"units": "degC"},
    )
    att_ref = ref + 1
    att_ref.attrs["units"] = "degC"
    path = tmp_path / "prepared.nc"
    xr.Dataset({"obs_ref": ref, "att_ref": att_ref}).assign_coords(
        valid_time=valid_time
    ).to_netcdf(path)
    with load_drift_references(
        "Reanalysis", 5, "TREFHT", path=path, hindcast=values
    ) as loaded:
        assert set(loaded.data_vars) == {"X_obs", "X_att"}
        np.testing.assert_array_equal(loaded.valid_time, valid_time)

    with pytest.raises(ValueError, match="lacks the required.*sigma_att"):
        load_drift_references(
            "Reanalysis", 5, "TREFHT", path=path,
            require_attractor_spread=True,
        )


def test_pipeline_pairs_only_common_years_and_returns_explicit_keys():
    coords_a = {"Y": [2000, 2001], "M": [0, 1], "L": [1, 2]}
    coords_b = {"Y": [2001, 2002], "M": [0, 1], "L": [1, 2]}
    a = xr.DataArray(np.ones((2, 2, 2)), dims=("Y", "M", "L"), coords=coords_a,
                     attrs={"units": "K"})
    b = xr.DataArray(np.ones((2, 2, 2)) * 2, dims=("Y", "M", "L"), coords=coords_b,
                     attrs={"units": "K"})
    obs = {
        "Reanalysis": a.mean("M") * 0,
        "JRA55_FOSIRL": b.mean("M") * 0,
    }
    att = {key: value.copy() for key, value in obs.items()}
    for value in (*obs.values(), *att.values()):
        value.attrs["units"] = "K"
    result = run_pipeline(
        {"Reanalysis": a, "JRA55_FOSIRL": b}, obs, att
    )
    assert set(result) == {
        "hindcast_mean", "e_obs", "e_att", "delta_e_obs", "delta_e_att",
        "delta_abs_e_obs", "delta_abs_e_att", "regime", "skill", "paired",
    }
    np.testing.assert_array_equal(result["e_obs"]["Reanalysis"].Y, [2001])
    assert "delta_delta_abs_e_obs" in result["paired"]


def test_skill_and_correlations_are_dimensionless_and_bootstrap_is_reproducible():
    difference = xr.DataArray(
        [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
        dims=("Y", "L"), coords={"Y": [2000, 2001, 2002], "L": [1, 2]},
        attrs={"units": "degC"},
    )
    first = bootstrap_paired_mean_ci(difference, n_bootstrap=100, seed=7)
    second = bootstrap_paired_mean_ci(difference, n_bootstrap=100, seed=7)
    xr.testing.assert_identical(first, second)
    np.testing.assert_allclose(first.estimate, [0.0, 2.0])
    assert bool((first.ci_lower <= first.estimate).all())
    assert bool((first.estimate <= first.ci_upper).all())


def test_missing_monthly_cache_is_materialized_from_project_loader(tmp_path, monkeypatch):
    times = xr.DataArray(
        np.asarray([["2000-05-15", "2000-06-15"]], dtype="datetime64[ns]"),
        dims=("Y", "L"), coords={"Y": ["2000050100"], "L": [1, 2]},
    )
    source = xr.Dataset(
        {
            "TREFHT": (("Y", "L", "M", "lat", "lon"), np.ones((1, 2, 2, 1, 1))),
            "time": times,
        },
        coords={"Y": ["2000050100"], "L": [1, 2], "M": ["EN00", "EN01"],
                "lat": [0.0], "lon": [0.0]},
    )
    source.TREFHT.attrs["units"] = "K"
    monkeypatch.setattr(
        "workflows.diagnostics.two_reference_drift.data_access_e3sm.get_monthly_data",
        lambda **kwargs: source,
    )
    job = {"source": "Reanalysis", "component": "atm", "variable": "TREFHT",
           "init_month": 5}
    path = prepare_monthly_hindcast_cache(
        job, case_prefix="case", init_years=[2000], members=["EN00", "EN01"],
        data_root=tmp_path, output_root=tmp_path, lead_count=2,
    )
    assert path.is_file()
    with xr.open_dataset(path) as cached:
        assert cached.TREFHT.dims == ("Y", "L", "M", "lat", "lon")
        np.testing.assert_array_equal(cached.L, [1, 2])


def test_initialization_year_selection_accepts_years_and_init_tags():
    for coordinates in (
        [1980, 1981, 1982], ["1980050100", "1981050100", "1982050100"]
    ):
        data = xr.DataArray(np.arange(3), dims="Y", coords={"Y": coordinates})
        selected = select_initialization_years(data, (1981, 1982))
        np.testing.assert_array_equal(selected.values, [1, 2])
        np.testing.assert_array_equal(selected.Y.values, coordinates[1:])


def test_regridding_preserves_physical_units(monkeypatch):
    data = xr.DataArray(
        np.ones((2, 2)), dims=("lat", "lon"),
        coords={"lat": [-1.0, 1.0], "lon": [0.0, 180.0]},
        attrs={"units": "degC", "source_product": "ERA5"},
    )
    target = xr.Dataset(coords={"lat": data.lat, "lon": data.lon})

    class FakeRegridder:
        def __call__(self, field):
            result = field.copy()
            result.attrs = {"regrid_method": "conservative"}
            return result

    monkeypatch.setattr(
        "workflows.diagnostics.two_reference_drift.regrid.make_regridder",
        lambda *args, **kwargs: FakeRegridder(),
    )
    result = _regrid_field(data, target, {})
    assert result.attrs["units"] == "degC"
    assert result.attrs["source_product"] == "ERA5"
    assert result.attrs["regrid_method"] == "identity; source and target coordinates match"


def test_spread_unit_conversion_never_applies_temperature_offset():
    spread = xr.DataArray([1.5, 2.0], dims="time", attrs={"units": "K"})
    converted = apply_spread_unit_transform(spread, "atm", "TREFHT")
    np.testing.assert_allclose(converted, spread)
    assert converted.attrs["units"] == "degC"


def test_psl_and_prect_unit_conversions_use_native_metadata():
    psl = xr.DataArray([101325.0], dims="time", attrs={"units": "Pa"})
    model_precip = xr.DataArray([1.0e-5], dims="time", attrs={"units": "m/s"})
    obs_precip = xr.DataArray([2.5], dims="time", attrs={"units": "mm/d"})

    converted_psl = apply_unit_transform(psl, "atm", "PSL")
    converted_model_precip = apply_unit_transform(model_precip, "atm", "PRECT")
    converted_obs_precip = apply_unit_transform(obs_precip, "atm", "PRECT")

    np.testing.assert_allclose(converted_psl, [1013.25])
    np.testing.assert_allclose(converted_model_precip, [864.0])
    np.testing.assert_allclose(converted_obs_precip, obs_precip)
    assert converted_psl.attrs["units"] == "hPa"
    assert converted_model_precip.attrs["units"] == "mm/day"
    assert converted_obs_precip.attrs["units"] == "mm/day"


def test_prect_spread_conversion_uses_scale_without_offset():
    spread = xr.DataArray([2.0e-6], dims="time", attrs={"units": "m/s"})
    converted = apply_spread_unit_transform(spread, "atm", "PRECT")
    np.testing.assert_allclose(converted, [172.8])
    assert converted.attrs["units"] == "mm/day"


def test_h2osoi_is_integrated_over_cpc_column_depth():
    layer = xr.DataArray(
        np.full((15,), 0.25),
        dims="levgrnd",
        coords={
            "levgrnd": 0.025
            * (np.exp(0.5 * (np.arange(1, 16) - 0.5)) - 1.0)
        },
        attrs={"units": "mm3/mm3"},
    )
    integrated = apply_unit_transform(layer, "lnd", "H2OSOI")
    np.testing.assert_allclose(integrated, 400.0)
    assert integrated.attrs["units"] == "mm"
    assert integrated.attrs["depth_top_m"] == 0.0
    assert integrated.attrs["depth_bottom_m"] == 1.6


def test_integrated_cpc_soil_moisture_is_not_converted_again():
    cpc = xr.DataArray([250.0], dims="time", attrs={"units": "mm"})
    converted = apply_unit_transform(cpc, "lnd", "H2OSOI")
    np.testing.assert_allclose(converted, cpc)
    assert converted.attrs["depth_bottom_m"] == 1.6


def test_pipeline_normalizes_attractor_departure_by_historical_spread():
    coords = {"Y": [2000, 2001], "M": [0, 1], "L": [1, 2]}
    values = xr.DataArray(
        np.asarray([[[2.0, 4.0], [2.0, 4.0]], [[4.0, 8.0], [4.0, 8.0]]]),
        dims=("Y", "M", "L"), coords=coords, attrs={"units": "degC"},
    )
    reference = values.mean("M") * 0
    reference.attrs["units"] = "degC"
    spread = xr.ones_like(reference) * 2
    spread.attrs["units"] = "degC"
    result = run_pipeline(
        {"test": values}, {"test": reference}, {"test": reference},
        attractor_spreads={"test": spread},
    )
    np.testing.assert_allclose(result["z_att"]["test"], [[1, 2], [2, 4]])
    assert result["z_att"]["test"].attrs["units"] == "1"


def test_spatial_summary_retains_maps_and_uses_regime_frequencies():
    coords = {
        "Y": [2000, 2001], "M": [0], "L": [1, 2],
        "lat": [-5.0, 5.0], "lon": [190.0, 200.0],
    }
    values = xr.DataArray(
        np.arange(16.0).reshape(2, 1, 2, 2, 2),
        dims=("Y", "M", "L", "lat", "lon"), coords=coords,
        attrs={"units": "degC"},
    )
    reference = xr.zeros_like(values.isel(M=0, drop=True))
    reference.attrs["units"] = "degC"
    product = compute_diagnostics(values, reference, reference)
    product["z_att"] = product.e_att.assign_attrs(units="1")
    summary = compute_spatial_drift_summary(product, {"both": [1, 2]})
    assert summary.delta_abs_e_obs_both.dims == ("lat", "lon")
    fractions = summary[[name for name in summary if name.startswith("fraction_regime_")]]
    np.testing.assert_allclose(fractions.to_array().sum("variable"), 1.0)
