import numpy as np
import xarray as xr

from workflows.leadtime_skill.rmse_comparison import (
    absolute_rmse_from_skill,
    area_weighted_mask_fraction,
    bootstrap_rmse_diff_matched_ensemble_memorysafe,
    build_rmse_significance_dataset,
    compact_year_tag,
    direct_rmse_cache_attrs,
    direct_rmse_cache_path,
    direct_rmse_difference,
    finite_ensemble_rmse_comparison_memorysafe,
    lead_label,
    make_obs_like_model_leads,
    make_obs_like_model_time,
    normalize_direct_rmse_leads,
    require_available_lead,
    safe_model_name,
    valid_area_weighted_fraction,
)


def test_compact_year_tag_uses_range_and_unique_count():
    assert compact_year_tag(range(1980, 2019)) == "1980-2018_ny39"
    assert compact_year_tag([1980, 1981, 1981, 2018]) == "1980-2018_ny3"


def test_shared_direct_rmse_metadata_helpers():
    legacy = xr.Dataset(
        {"rmse": ("L", [1.0, 2.0])},
        coords={"L": [1, 4]},
    )

    normalized = normalize_direct_rmse_leads(legacy)

    np.testing.assert_array_equal(normalized["L"], [3, 6])
    assert require_available_lead(normalized, 3) == 3
    assert lead_label(5, 3, {5: "MAY"}) == "Lead-1 JJA (May init)"
    assert safe_model_name("E3SM / FOSIRL") == "E3SM___FOSIRL"


def test_direct_rmse_difference_and_weighted_fraction():
    coords = {"L": [3], "lat": [0.0, 60.0], "lon": [0.0]}
    left = xr.Dataset(
        {"rmse": (("L", "lat", "lon"), [[[1.0], [3.0]]])},
        coords=coords,
    )
    right = xr.Dataset(
        {"rmse": (("L", "lat", "lon"), [[[2.0], [2.0]]])},
        coords=coords,
    )

    result = direct_rmse_difference(left, right)
    mask = result["rmse_diff"].sel(L=3) < 0

    np.testing.assert_array_equal(result["rmse_diff"], [[[-1.0], [1.0]]])
    np.testing.assert_array_equal(result["rmse_ratio"], [[[0.5], [1.5]]])
    assert np.isclose(area_weighted_mask_fraction(mask), 2.0 / 3.0)


def test_valid_area_fraction_excludes_missing_domain_area():
    mask = xr.DataArray(
        [[True], [False]],
        dims=("lat", "lon"),
        coords={"lat": [0.0, 60.0], "lon": [0.0]},
    )
    valid = xr.DataArray(
        [[True], [False]],
        dims=("lat", "lon"),
        coords=mask.coords,
    )

    assert valid_area_weighted_fraction(mask, valid=valid) == 1.0


def test_absolute_rmse_and_significance_preserve_missing_cells():
    skill = xr.Dataset(
        {
            "rmse": ("point", [0.5, np.nan]),
            "sig_obs": ("point", [4.0, 2.0]),
        }
    )
    absolute = absolute_rmse_from_skill(skill, units="hPa")
    probability = xr.DataArray([1.0, np.nan], dims="point")
    significance = build_rmse_significance_dataset(
        probability,
        alpha=0.1,
        left_model="E3SM",
        right_model="CESM-SMYLE",
        n_iterations=100,
        matched_ensemble_size=10,
        member_selection_checksum="abc",
    )

    np.testing.assert_allclose(absolute.values[0], 2.0)
    assert absolute.attrs["units"] == "hPa"
    assert significance["left_better"].values[0] == 1
    assert np.isnan(significance["left_better"].values[1])
    assert significance.attrs["member_selection_checksum"] == "abc"


def test_direct_rmse_cache_contract_and_name_are_provenance_aware(tmp_path):
    kwargs = dict(
        source="E3SM-FOSIRL",
        source_data_identity="inventory-sha256:model",
        case_prefix="case-prefix",
        variable="PRECT",
        init_month=11,
        verification_years=[1980, 1981],
        observation_product="GPCP",
        observation_variable="PRECT",
        observation_data_identity="inventory-sha256:obs",
        target_grid="latlon_1x1",
        regridding_method="conservative",
        ensemble_member_count=10,
        unit_conversion_version="precip_v1",
    )
    attrs = direct_rmse_cache_attrs(**kwargs)
    path = direct_rmse_cache_path(
        root=tmp_path,
        source=kwargs["source"],
        component="atm",
        variable=kwargs["variable"],
        init_month=kwargs["init_month"],
        verification_years=kwargs["verification_years"],
        expected_attrs=attrs,
    )

    assert attrs["cache_kind"] == "direct_rmse"
    assert attrs["obs_alignment_version"]
    assert "init11_years_1980-1981_ny2" in path.name
    changed = dict(attrs, observation_data_identity="inventory-sha256:new")
    changed_path = direct_rmse_cache_path(
        root=tmp_path,
        source=kwargs["source"],
        component="atm",
        variable=kwargs["variable"],
        init_month=kwargs["init_month"],
        verification_years=kwargs["verification_years"],
        expected_attrs=changed,
    )
    assert changed_path != path


def test_make_obs_like_model_leads_uses_season_verification_month():
    time = np.arange(
        np.datetime64("1980-01"),
        np.datetime64("1982-01"),
        dtype="datetime64[M]",
    )
    obs = xr.DataArray(
        np.arange(time.size, dtype=float)[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [30.0], "lon": [250.0]},
    )
    template = xr.DataArray(
        np.zeros((1, 2, 1, 1, 1)),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={
            "Y": [1980],
            "L": [3, 15],
            "M": [0],
            "lat": [30.0],
            "lon": [250.0],
        },
    )

    result = make_obs_like_model_leads(
        obs_da=obs,
        template_da=template,
        init_month=5,
        years=[1980],
        leads=[3, 15],
    )

    expected = obs.sel(time=[np.datetime64("1980-07"), np.datetime64("1981-07")])
    np.testing.assert_array_equal(result.sel(Y=1980).values[:, 0, 0], expected.values[:, 0, 0])


def test_observation_alignment_preserves_missing_boundary_seasons_as_nan():
    obs_time = np.array([np.datetime64("1980-07")])
    obs = xr.DataArray(
        np.array([[[2.0]]]),
        dims=("time", "lat", "lon"),
        coords={"time": obs_time, "lat": [30.0], "lon": [250.0]},
    )
    template = xr.DataArray(
        np.zeros((2, 1, 1, 1, 1)),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={"Y": [1980, 1981], "L": [3], "M": [0], "lat": [30.0], "lon": [250.0]},
    )
    model_time = xr.DataArray(
        np.array([[np.datetime64("1980-07")], [np.datetime64("1981-07")]]),
        dims=("Y", "L"),
        coords={"Y": [1980, 1981], "L": [3]},
    )

    by_lead = make_obs_like_model_leads(obs, template, 5, [1980, 1981], [3])
    by_time = make_obs_like_model_time(obs, model_time, [1980, 1981])

    for result in (by_lead, by_time):
        assert result.sel(Y=1980, L=3).compute().item() == 2.0
        assert result.sel(Y=1981, L=3).isnull().all().compute().item()


def test_observation_alignment_treats_nat_verification_time_as_missing():
    obs = xr.DataArray(
        np.array([[[2.0]]]),
        dims=("time", "lat", "lon"),
        coords={"time": [np.datetime64("1980-07")], "lat": [30.0], "lon": [250.0]},
    )
    model_time = xr.DataArray(
        np.array([[np.datetime64("NaT", "ns")]]),
        dims=("Y", "L"),
        coords={"Y": [1980], "L": [24]},
    )

    result = make_obs_like_model_time(obs, model_time, [1980])

    assert result.sel(Y=1980, L=24).isnull().all().item()


def test_matched_ensemble_bootstrap_identifies_lower_rmse():
    coords = {
        "Y": [1980, 1981, 1982, 1983],
        "L": [3],
        "lat": [30.0],
        "lon": [250.0],
    }
    left = xr.DataArray(
        np.full((4, 1, 2, 1, 1), 1.0),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={**coords, "M": [0, 1]},
    )
    right = xr.DataArray(
        np.full((4, 1, 4, 1, 1), 2.0),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={**coords, "M": [0, 1, 2, 3]},
    )

    result = bootstrap_rmse_diff_matched_ensemble_memorysafe(
        left,
        right,
        nboot=20,
        seed=7,
        alpha=0.1,
    )

    assert result.attrs["matched_ensemble_size"] == 2
    assert result["rmse_diff"].item() == -1.0
    assert result["prob_left_lower_rmse"].item() == 1.0
    assert result["left_better"].item() == 1
    assert result["right_better"].item() == 0


def test_finite_ensemble_comparison_resamples_only_larger_ensemble():
    coords = {
        "Y": [1980, 1981, 1982, 1983],
        "L": [3],
        "lat": [30.0],
        "lon": [250.0],
    }
    left = xr.DataArray(
        np.full((4, 1, 2, 1, 1), 1.0),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={**coords, "M": [0, 1]},
    )
    right = xr.DataArray(
        np.full((4, 1, 4, 1, 1), 2.0),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={**coords, "M": [0, 1, 2, 3]},
    )

    result = finite_ensemble_rmse_comparison_memorysafe(
        left,
        right,
        n_iterations=20,
        seed=7,
        alpha=0.1,
    )

    assert result.attrs["larger_ensemble_side"] == "right"
    assert result.attrs["right_ensemble_size"] == 4
    assert result.attrs["matched_ensemble_size"] == 2
    assert result["rmse_diff"].item() == -1.0
    assert result["prob_left_lower_rmse"].item() == 1.0


def test_finite_ensemble_accepts_audited_member_selections():
    coords = {"Y": [1980, 1981], "L": [3], "lat": [0.0], "lon": [0.0]}
    left = xr.DataArray(
        np.ones((2, 1, 2, 1, 1)),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={**coords, "M": [0, 1]},
    )
    right = xr.DataArray(
        np.full((2, 1, 4, 1, 1), 2.0),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={**coords, "M": [0, 1, 2, 3]},
    )
    selections = np.array([[0, 1], [2, 3]], dtype=np.int32)

    first = finite_ensemble_rmse_comparison_memorysafe(
        left, right, n_iterations=2, seed=7, member_indices=selections
    )
    second = finite_ensemble_rmse_comparison_memorysafe(
        left, right, n_iterations=2, seed=999, member_indices=selections
    )

    xr.testing.assert_equal(first, second)
    assert first.attrs["member_selection_checksum"]


def test_resampling_preserves_an_all_missing_boundary_lead():
    coords = {
        "Y": [1980, 1981],
        "L": [3, 24],
        "lat": [30.0],
        "lon": [250.0],
    }
    left_values = np.ones((2, 2, 2, 1, 1), dtype=float)
    right_values = np.full((2, 2, 4, 1, 1), 2.0, dtype=float)
    left_values[:, 1] = np.nan
    right_values[:, 1] = np.nan
    left = xr.DataArray(
        left_values,
        dims=("Y", "L", "M", "lat", "lon"),
        coords={**coords, "M": [0, 1]},
    )
    right = xr.DataArray(
        right_values,
        dims=("Y", "L", "M", "lat", "lon"),
        coords={**coords, "M": [0, 1, 2, 3]},
    )

    for result in (
        finite_ensemble_rmse_comparison_memorysafe(
            left, right, n_iterations=4, seed=7, alpha=0.1
        ),
        bootstrap_rmse_diff_matched_ensemble_memorysafe(
            left, right, nboot=4, seed=7, alpha=0.1
        ),
    ):
        assert result["prob_left_lower_rmse"].sel(L=3).item() == 1.0
        assert result["prob_left_lower_rmse"].sel(L=24).isnull().all()
        assert result["left_better"].sel(L=24).isnull().all()
        assert result["right_better"].sel(L=24).isnull().all()
