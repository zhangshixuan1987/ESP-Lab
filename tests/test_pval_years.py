import numpy as np
import xarray as xr
import xskillscore as xs


def test_pearson_r_small_n():
    """Verify xskillscore handles N=2 and N=3 without raising errors."""
    np.random.seed(0)
    for n in (2, 3):
        a = xr.DataArray(np.random.normal(0, 1, n), dims=['time'])
        b = xr.DataArray(np.random.normal(0, 1, n), dims=['time'])
        r = xs.pearson_r(a, b, dim='time')
        pval = xs.pearson_r_p_value(a, b, dim='time')
        pval_eff = xs.pearson_r_eff_p_value(a, b, dim='time')
        assert np.isfinite(float(r)) or np.isnan(float(r))
        assert 0.0 <= float(pval) <= 1.0 or np.isnan(float(pval))
        assert 0.0 <= float(pval_eff) <= 1.0 or np.isnan(float(pval_eff))
