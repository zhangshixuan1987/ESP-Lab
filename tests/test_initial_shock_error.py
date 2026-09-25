import numpy as np
import pytest
import xarray as xr

from esp_lab.diagnostics.initial_shock_error import (
    compute_initial_shock_error_index,
    plot_error_heatmap,
)


def indices():
    obs = xr.DataArray(
        [[0., 1.], [2., 3.]], dims=("Y", "block"),
        coords={"Y": [1980, 1981], "block": [1, 2]}, attrs={"units": "degC"},
    )
    model = obs * 2 + 100
    model.attrs["units"] = "degC"
    return xr.Dataset({"model_index": model, "observation_index": obs})


def monthly_indices():
    years = [1980, 1981]
    leads = np.arange(1, 25)
    observation = xr.DataArray(
        np.tile(np.arange(24.), (2, 1)), dims=("Y", "lead_month"),
        coords={"Y": years, "lead_month": leads}, attrs={"units": "degC"},
    )
    model = observation + 2.
    first_member = observation + 4.
    model.attrs["units"] = first_member.attrs["units"] = "degC"
    scale = xr.DataArray(
        np.full(12, 2.), dims="month_phase", coords={"month_phase": np.arange(1, 13)},
        attrs={"units": "degC"},
    )
    return xr.Dataset({
        "monthly_model_index": model,
        "monthly_first_member_index": first_member,
        "monthly_observation_index": observation,
        "monthly_observation_climatology_std": scale,
        "monthly_standardized_error": (model - observation) / 2.,
    }, coords={"calendar_month": ("lead_month", np.tile(np.arange(1, 13), 2))})


def test_monthly_shared_climatology_errors_and_seasonal_summaries():
    result = compute_initial_shock_error_index(monthly_indices())
    np.testing.assert_allclose(result.normalized_rmse, 1.)
    np.testing.assert_allclose(result.normalized_mae, 1.)
    np.testing.assert_allclose(result.first_member_normalized_rmse, 2.)
    np.testing.assert_allclose(result.first_member_normalized_mae, 2.)
    np.testing.assert_allclose(result.rmse, 2.)
    np.testing.assert_allclose(result.mae, 2.)
    np.testing.assert_allclose(result.seasonal_normalized_rmse, 1.)
    np.testing.assert_allclose(result.seasonal_normalized_mae, 1.)
    assert result.seasonal_normalized_rmse.dims == ("Y", "lead_year", "season")
    assert (result.paired_sample_count == 24).all()
    assert result.attrs["anomaly_baseline"].startswith("shared reference-observation")


def test_pairing_minimum_and_validation():
    source = monthly_indices()
    source["monthly_standardized_error"].loc[dict(Y=1980, lead_month=1)] = np.nan
    strict = compute_initial_shock_error_index(source)
    assert np.isnan(strict.normalized_rmse.sel(Y=1980))
    relaxed = compute_initial_shock_error_index(source, min_samples=23)
    assert np.isfinite(relaxed.normalized_rmse.sel(Y=1980))
    with pytest.raises(ValueError, match="min_samples"):
        compute_initial_shock_error_index(source, min_samples=25)


def test_normalized_error_heatmap_accepts_explicit_levels():
    import matplotlib.pyplot as plt

    result = compute_initial_shock_error_index(monthly_indices()).expand_dims(case=["case"])
    levels = np.arange(0., 2.01, .2)
    fig, ax = plt.subplots()
    returned = plot_error_heatmap(
        result, variable="normalized_rmse", levels=levels, ax=ax,
        add_colorbar=False, add_invalid_legend=False,
    )
    assert returned is fig
    np.testing.assert_allclose(ax.images[0].norm.boundaries, levels)
    plt.close(fig)
