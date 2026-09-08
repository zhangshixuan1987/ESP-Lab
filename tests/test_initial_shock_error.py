import numpy as np
import pytest
import xarray as xr

from esp_lab.diagnostics.initial_shock_error import (
    compute_initial_shock_error_index, plot_error_heatmap, plot_rmse_mae,
)


def indices():
    obs = xr.DataArray(
        [[0., 1.], [2., 3.]], dims=("Y", "block"),
        coords={"Y": [1980, 1981], "block": [1, 2]}, attrs={"units": "degC"},
    )
    model = obs * 2 + 100
    model.attrs["units"] = "degC"
    return xr.Dataset({"model_index": model, "observation_index": obs})


def test_ncl_independent_full_cohort_anomalies_and_errors():
    result = compute_initial_shock_error_index(indices())
    np.testing.assert_allclose(result.model_climatology, 103.)
    np.testing.assert_allclose(result.observation_climatology, 1.5)
    np.testing.assert_allclose(result.error, [[-1.5, -.5], [.5, 1.5]])
    np.testing.assert_allclose(result.rmse, np.sqrt(1.25))
    np.testing.assert_allclose(result.mae, 1.)
    np.testing.assert_allclose(result.normalized_rmse, np.sqrt(1.25) / np.sqrt(2.))
    np.testing.assert_allclose(result.normalized_mae, 1. / np.sqrt(2.))
    assert (result.paired_sample_count == 2).all()


def test_separate_constant_offsets_do_not_change_metrics():
    source = indices()
    shifted = source.copy()
    shifted["model_index"] = source.model_index + 700
    shifted["observation_index"] = source.observation_index - 20
    xr.testing.assert_allclose(
        compute_initial_shock_error_index(source)[["rmse", "mae"]],
        compute_initial_shock_error_index(shifted)[["rmse", "mae"]],
    )


def test_pairing_minimum_and_validation():
    source = indices()
    source["model_index"].loc[dict(Y=1980, block=1)] = np.nan
    strict = compute_initial_shock_error_index(source)
    assert np.isnan(strict.rmse.sel(Y=1980))
    relaxed = compute_initial_shock_error_index(source, min_samples=1)
    assert np.isfinite(relaxed.rmse.sel(Y=1980))
    with pytest.raises(ValueError, match="min_samples"):
        compute_initial_shock_error_index(source, min_samples=3)


def test_plot_has_two_metric_panels():
    result = compute_initial_shock_error_index(indices()).expand_dims(case=["case"])
    fig = plot_rmse_mae(result)
    assert len(fig.axes) == 4  # two panels and two colorbars
    with pytest.raises(ValueError, match="six increasing"):
        plot_rmse_mae(result, rmse_ranges=(.2, .1), mae_ranges=(.1,) * 6)


def test_normalized_error_heatmap_accepts_explicit_levels():
    import matplotlib.pyplot as plt

    result = compute_initial_shock_error_index(indices()).expand_dims(case=["case"])
    levels = np.arange(0., 2.01, .2)
    fig, ax = plt.subplots()
    returned = plot_error_heatmap(
        result, variable="normalized_rmse", levels=levels, ax=ax,
        add_colorbar=False, add_invalid_legend=False,
    )
    assert returned is fig
    np.testing.assert_allclose(ax.images[0].norm.boundaries, levels)
    plt.close(fig)
