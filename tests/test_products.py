import numpy as np
import pandas as pd
import pytest
import xarray as xr

from esp_lab.diagnostics.attribution_core import (
    across_start_attribution,
    weighted_ic_alignment,
)
from esp_lab.diagnostics.products import (
    PAIRED_SIGN_CONVENTION,
    combine_product_bundles,
    standardize_product_table,
    write_product_bundle,
)
from esp_lab.diagnostics.drift import CaseArray, paired_difference_by_start
from esp_lab.diagnostics.physical_core import compute_land_water_residual


def test_product_bundle_round_trip(tmp_path):
    table = standardize_product_table(
        [{"metric_name": "paired_diff", "value": 1.5, "baseline": "lead 1"}],
        workflow="5c_monthly_spatial",
        configuration_hash="abc123",
    )
    _, manifest = write_product_bundle(
        tmp_path, table, workflow="5c_monthly_spatial", configuration_hash="abc123"
    )
    combined, manifests = combine_product_bundles([manifest])
    assert combined.loc[0, "sign_convention"] == PAIRED_SIGN_CONVENTION
    assert manifests[0]["complete"] is True


def test_product_contract_rejects_reversed_paired_sign():
    with pytest.raises(ValueError, match="Paired products must use"):
        standardize_product_table(
            [{
                "metric_name": "paired_difference", "value": 1.0,
                "sign_convention": "Reanalysis - JRA55_FOSIRL",
            }],
            workflow="bad", configuration_hash="bad",
        )


def test_weighted_ic_alignment_recovers_scaled_pattern():
    ic = xr.DataArray(
        [[-1.0, 0.0], [1.0, 2.0]], dims=("lat", "lon"),
        coords={"lat": [-30.0, 30.0], "lon": [0.0, 1.0]},
    )
    drift = xr.concat([2.0 * ic, -0.5 * ic], dim=pd.Index([1, 2], name="L"))
    weights = xr.DataArray([0.5, 1.0], dims="lat", coords={"lat": ic.lat})
    result = weighted_ic_alignment(
        ic, drift, spatial_dims=("lat", "lon"), weights=weights
    )
    np.testing.assert_allclose(result.r_ic, [1.0, -1.0])
    np.testing.assert_allclose(result.beta_ic, [2.0, -0.5])


def test_across_start_attribution_recovers_regression_by_lead():
    indicator = xr.DataArray([0.0, 1.0, 2.0], dims="Y", coords={"Y": [2000, 2001, 2002]})
    response = xr.DataArray(
        [[1.0, -1.0], [3.0, 2.0], [5.0, 5.0]],
        dims=("Y", "L"), coords={"Y": indicator.Y, "L": [1, 2]},
    )
    result = across_start_attribution(indicator, response)
    np.testing.assert_allclose(result.alpha, [1.0, -1.0])
    np.testing.assert_allclose(result.beta, [2.0, 3.0])
    np.testing.assert_allclose(result.correlation, [1.0, 1.0])


def test_regional_start_product_uses_jra_minus_reanalysis_sign():
    coords = {"Y": [2000, 2001], "M": ["EN00"], "L": [1, 2]}
    reanalysis = CaseArray(
        xr.DataArray(np.zeros((2, 1, 2)), dims=("Y", "M", "L"), coords=coords),
        name="Reanalysis",
    )
    jra = CaseArray(
        xr.DataArray(
            [[[1.0, 3.0]], [[2.0, 5.0]]],
            dims=("Y", "M", "L"), coords=coords,
        ),
        name="JRA55_FOSIRL",
    )
    result = paired_difference_by_start(jra, reanalysis, baseline_lead=1)
    np.testing.assert_allclose(result.sel(L=2), [2.0, 3.0])
    assert result.attrs["sign_convention"] == PAIRED_SIGN_CONVENTION


def test_land_water_residual_definition():
    storage = xr.DataArray([1.0, 2.0], dims="d")
    precipitation = xr.DataArray([5.0, 7.0], dims="d")
    et = xr.DataArray([2.0, 3.0], dims="d")
    runoff = xr.DataArray([1.0, 1.0], dims="d")
    result = compute_land_water_residual(storage, precipitation, et, runoff)
    np.testing.assert_allclose(result, [-1.0, -1.0])
