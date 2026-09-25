import numpy as np
import xarray as xr
import xskillscore as xs


def test_pearson_r_and_pvalue():
    rng = np.random.default_rng(0)
    a = xr.DataArray(np.linspace(0, 10, 10) + rng.normal(0, 0.1, 10), dims=['time'])
    b = xr.DataArray(np.linspace(0, 10, 10) + rng.normal(0, 0.1, 10), dims=['time'])

    r = xs.pearson_r(a, b, dim='time')
    pval = xs.pearson_r_p_value(a, b, dim='time')
    pval_eff = xs.pearson_r_eff_p_value(a, b, dim='time')

    assert float(r) > 0.9, f"Expected high correlation, got {float(r)}"
    assert 0.0 <= float(pval) <= 1.0
    # Effective p-value can be NaN when effective dof calculation is degenerate
    assert np.isnan(float(pval_eff)) or (0.0 <= float(pval_eff) <= 1.0)


def test_pearson_statistics_define_small_sample_behavior():
    """Small samples may be undefined, but must not produce invalid probabilities."""
    rng = np.random.default_rng(1)
    for sample_size in (2, 3):
        first = xr.DataArray(rng.normal(size=sample_size), dims="time")
        second = xr.DataArray(rng.normal(size=sample_size), dims="time")

        correlation = float(xs.pearson_r(first, second, dim="time"))
        p_value = float(xs.pearson_r_p_value(first, second, dim="time"))
        effective_p_value = float(xs.pearson_r_eff_p_value(first, second, dim="time"))

        assert np.isnan(correlation) or -1 <= correlation <= 1
        assert np.isnan(p_value) or 0 <= p_value <= 1
        assert np.isnan(effective_p_value) or 0 <= effective_p_value <= 1
