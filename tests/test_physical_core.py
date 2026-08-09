"""
tests/test_physical_core.py
============================
Offline smoke tests for physical_core.py (pure core).

No filesystem I/O, no real NetCDF files required. All tests use synthetic
numpy and xarray DataArrays.

Test classes
------------
1. TestFluxPartitioning         — compute_evaporative_fraction, compute_bowen_ratio, QC masking
2. TestOceanCoupling            — compute_sst_trefht_contrast
3. TestSoilMoistureIntegration   — integrate_soil_moisture
4. TestSoilMoistureCoupling     — compute_coupling_slope
5. TestPrecipSMLagResponse      — compute_precip_sm_lag_response (lags 0..7)
6. TestApparentEnergyResidual   — compute_apparent_energy_residual
7. TestPhysicalBootstrap        — bootstrap_physical_metric_ci
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from esp_lab.diagnostics.physical_core import (
    bootstrap_physical_metric_ci,
    compute_apparent_energy_residual,
    compute_bowen_ratio,
    compute_coupling_slope,
    compute_evaporative_fraction,
    compute_precip_sm_lag_response,
    compute_sst_trefht_contrast,
    integrate_soil_moisture,
)


# ===========================================================================
# 1. TestFluxPartitioning
# ===========================================================================

class TestFluxPartitioning:
    def test_evaporative_fraction_basic(self):
        lh = xr.DataArray(np.array([[60.0, 80.0], [50.0, 70.0]]), dims=["lat", "lon"])
        sh = xr.DataArray(np.array([[40.0, 20.0], [50.0, 30.0]]), dims=["lat", "lon"])

        ef, masked_frac = compute_evaporative_fraction(lh, sh)
        assert ef.name == "evaporative_fraction"
        # EF = 60/(60+40) = 0.60
        np.testing.assert_allclose(ef.values[0, 0], 0.60)
        # EF = 80/(80+20) = 0.80
        np.testing.assert_allclose(ef.values[0, 1], 0.80)
        assert masked_frac == 0.0

    def test_evaporative_fraction_qc_masking(self):
        # Negative or near-zero net flux masked out
        lh = xr.DataArray(np.array([[-10.0, 50.0]]), dims=["lat", "lon"])
        sh = xr.DataArray(np.array([[5.0, -60.0]]),  dims=["lat", "lon"])

        ef, masked_frac = compute_evaporative_fraction(lh, sh, min_flux_threshold=1.0)
        assert np.isnan(ef.values[0, 0])  # LH < 0
        assert np.isnan(ef.values[0, 1])  # LH + SH = -10 <= 1
        assert masked_frac == 1.0

    def test_bowen_ratio_basic(self):
        lh = xr.DataArray(np.array([[50.0, 100.0]]), dims=["lat", "lon"])
        sh = xr.DataArray(np.array([[25.0, 50.0]]),  dims=["lat", "lon"])

        br, masked_frac = compute_bowen_ratio(lh, sh)
        assert br.name == "bowen_ratio"
        # BR = SH / LH = 25/50 = 0.50
        np.testing.assert_allclose(br.values[0, 0], 0.50)
        np.testing.assert_allclose(br.values[0, 1], 0.50)
        assert masked_frac == 0.0


# ===========================================================================
# 2. TestOceanCoupling
# ===========================================================================

class TestOceanCoupling:
    def test_sst_trefht_contrast(self):
        trefht = xr.DataArray(np.array([25.0, 28.0]), dims=["x"], attrs={"units": "degC"})
        sst    = xr.DataArray(np.array([27.0, 26.0]), dims=["x"])

        dt = compute_sst_trefht_contrast(trefht, sst)
        assert dt.name == "sst_trefht_contrast"
        np.testing.assert_allclose(dt.values, [-2.0, 2.0])


# ===========================================================================
# 3. TestSoilMoistureIntegration
# ===========================================================================

class TestSoilMoistureIntegration:
    def test_integrate_soil_moisture_mean(self):
        h2osoi = xr.DataArray(np.array([0.2, 0.3, 0.4]), dims=["levgrnd"], coords={"levgrnd": [1, 2, 3]})
        sm = integrate_soil_moisture(h2osoi)
        np.testing.assert_allclose(sm.values, 0.3)

    def test_integrate_soil_moisture_weighted(self):
        h2osoi = xr.DataArray(np.array([0.2, 0.3]), dims=["levgrnd"], coords={"levgrnd": [1, 2]})
        weights = [0.1, 0.5]
        sm = integrate_soil_moisture(h2osoi, layer_depths=weights)
        # 0.2*0.1 + 0.3*0.5 = 0.02 + 0.15 = 0.17
        np.testing.assert_allclose(sm.values, 0.17)


# ===========================================================================
# 4. TestSoilMoistureCoupling
# ===========================================================================

class TestSoilMoistureCoupling:
    def test_coupling_slope_perfect_linear(self):
        # Y' = 2 * X'
        x_anom = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0]), dims=["sample"])
        y_anom = xr.DataArray(np.array([2.0, 4.0, 6.0, 8.0]), dims=["sample"])

        slope, corr = compute_coupling_slope(x_anom, y_anom, sample_dim="sample")
        np.testing.assert_allclose(slope.values, 2.0)
        np.testing.assert_allclose(corr.values, 1.0)


# ===========================================================================
# 5. TestPrecipSMLagResponse
# ===========================================================================

class TestPrecipSMLagResponse:
    def test_precip_sm_lag_response(self):
        days = list(range(1, 15))
        rng  = np.random.default_rng(42)

        prect = xr.DataArray(rng.random((5, 14)), dims=["Y", "d"], coords={"Y": range(5), "d": days})
        sm    = xr.DataArray(rng.random((5, 14)), dims=["Y", "d"], coords={"Y": range(5), "d": days})

        beta_lag = compute_precip_sm_lag_response(prect, sm, max_lag=3, day_dim="d")
        assert beta_lag.dims == ("lag",)
        assert beta_lag.sizes["lag"] == 4
        assert np.all(np.isfinite(beta_lag.values))


# ===========================================================================
# 6. TestApparentEnergyResidual
# ===========================================================================

class TestApparentEnergyResidual:
    def test_apparent_energy_residual(self):
        fsns  = xr.DataArray(np.array([200.0]), dims=["x"])
        flns  = xr.DataArray(np.array([50.0]),  dims=["x"])
        lhflx = xr.DataArray(np.array([80.0]),  dims=["x"])
        shflx = xr.DataArray(np.array([30.0]),  dims=["x"])

        # R = 200 - 50 - 80 - 30 = 40.0
        r_app = compute_apparent_energy_residual(fsns, flns, lhflx, shflx)
        np.testing.assert_allclose(r_app.values, [40.0])


# ===========================================================================
# 7. TestPhysicalBootstrap
# ===========================================================================

class TestPhysicalBootstrap:
    def test_bootstrap_physical_metric_ci(self):
        years = [1980, 1981, 1982, 1983, 1984]
        diff_by_year = xr.DataArray(
            np.zeros((5, 2, 2)),
            dims=["Y", "lat", "lon"],
            coords={"Y": years, "lat": [0, 1], "lon": [0, 1]},
        )

        lower, upper = bootstrap_physical_metric_ci(diff_by_year, n_boot=50, seed=42)
        np.testing.assert_allclose(lower.values, 0.0, atol=1e-12)
        np.testing.assert_allclose(upper.values, 0.0, atol=1e-12)
