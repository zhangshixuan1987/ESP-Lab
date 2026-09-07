import numpy as np
import xarray as xr
import xskillscore as xs


def test_pearson_r_and_pvalue():
    np.random.seed(0)
    a = xr.DataArray(np.linspace(0, 10, 10) + np.random.normal(0, 0.1, 10), dims=['time'])
    b = xr.DataArray(np.linspace(0, 10, 10) + np.random.normal(0, 0.1, 10), dims=['time'])

    r = xs.pearson_r(a, b, dim='time')
    pval = xs.pearson_r_p_value(a, b, dim='time')
    pval_eff = xs.pearson_r_eff_p_value(a, b, dim='time')

    assert float(r) > 0.9, f"Expected high correlation, got {float(r)}"
    assert 0.0 <= float(pval) <= 1.0
    # Effective p-value can be NaN when effective dof calculation is degenerate
    assert np.isnan(float(pval_eff)) or (0.0 <= float(pval_eff) <= 1.0)
