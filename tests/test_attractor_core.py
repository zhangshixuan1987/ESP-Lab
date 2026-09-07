"""
tests/test_attractor_core.py
=============================
Offline smoke tests for attractor_core.py (pure core).

No filesystem I/O, no real NetCDF files required. All tests use synthetic
numpy and xarray DataArrays.

Test classes
------------
1. TestSpatialRMSD               — spatial_rmsd with/without weights
2. TestAttractorDistances        — compute_attractor_distances d^obs and d^model
3. TestRelativeMovement          — compute_relative_movement δd and d_initial
4. TestMovementClassification    — classify_movement_category, classify_spatial_movement
5. Test2AxisTrajectory           — build_2axis_trajectory points
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from esp_lab.diagnostics.attractor_core import (
    MovementCategory,
    build_2axis_trajectory,
    classify_movement_category,
    classify_spatial_movement,
    compute_attractor_distances,
    compute_relative_movement,
    spatial_rmsd,
)


class TestSpatialRMSD:
    def test_spatial_rmsd_unweighted(self):
        da1 = xr.DataArray(np.array([[2.0, 4.0]]), dims=["lat", "lon"])
        da2 = xr.DataArray(np.array([[0.0, 0.0]]), dims=["lat", "lon"])

        rmsd = spatial_rmsd(da1, da2)
        # sqrt((4 + 16) / 2) = sqrt(10) = 3.162277
        np.testing.assert_allclose(rmsd.values, np.sqrt(10.0))

    def test_spatial_rmsd_identical(self):
        da1 = xr.DataArray(np.array([[5.0, 10.0]]), dims=["lat", "lon"])
        rmsd = spatial_rmsd(da1, da1)
        np.testing.assert_allclose(rmsd.values, 0.0, atol=1e-12)


class TestAttractorDistances:
    def test_compute_attractor_distances(self):
        model = xr.DataArray(np.array([[10.0]]), dims=["lat", "lon"])
        obs   = xr.DataArray(np.array([[8.0]]),  dims=["lat", "lon"])
        clim  = xr.DataArray(np.array([[5.0]]),  dims=["lat", "lon"])

        d_obs, d_model = compute_attractor_distances(model, obs, clim)
        assert d_obs is not None and d_model is not None
        np.testing.assert_allclose(d_obs.values, 2.0)
        np.testing.assert_allclose(d_model.values, 5.0)


class TestRelativeMovement:
    def test_compute_relative_movement(self):
        dists = xr.DataArray(np.array([2.0, 1.5, 3.0]), dims=["L"], coords={"L": [1, 2, 3]})
        delta_d, d_w0 = compute_relative_movement(dists, baseline_coord_val=1, lead_dim="L")

        np.testing.assert_allclose(d_w0.values, 2.0)
        np.testing.assert_allclose(delta_d.sel(L=1).values, 0.0)
        np.testing.assert_allclose(delta_d.sel(L=2).values, -0.5)
        np.testing.assert_allclose(delta_d.sel(L=3).values, 1.0)


class TestMovementClassification:
    def test_classify_movement_category(self):
        # Toward both (do < 0, dm < 0)
        assert classify_movement_category(-0.5, -0.3) == MovementCategory.TOWARD_BOTH

        # Toward obs, away E3SM (do < 0, dm > 0)
        assert classify_movement_category(-0.5, 0.3) == MovementCategory.TOWARD_OBS_AWAY_E3SM

        # Toward E3SM, away obs (do > 0, dm < 0)
        assert classify_movement_category(0.5, -0.3) == MovementCategory.TOWARD_E3SM_AWAY_OBS

        # Away from both (do > 0, dm > 0)
        assert classify_movement_category(0.5, 0.3) == MovementCategory.AWAY_FROM_BOTH

        # Near zero
        assert classify_movement_category(0.01, 0.01, tol_obs=0.05, tol_model=0.05) == MovementCategory.NEAR_ZERO

    def test_classify_spatial_movement(self):
        do_map = xr.DataArray(np.array([[-0.5, 0.5]]), dims=["lat", "lon"])
        dm_map = xr.DataArray(np.array([[-0.3, -0.3]]), dims=["lat", "lon"])

        cat_map = classify_spatial_movement(do_map, dm_map)
        assert cat_map.name == "movement_category_code"
        # cell (0,0): do < 0, dm < 0 -> Code 1 (Toward Both)
        assert int(cat_map.values[0, 0]) == 1
        # cell (0,1): do > 0, dm < 0 -> Code 3 (Toward E3SM, Away Obs)
        assert int(cat_map.values[0, 1]) == 3


class Test2AxisTrajectory:
    def test_build_2axis_trajectory(self):
        dm_ts = xr.DataArray(np.array([0.0, -0.5, 0.2]), dims=["L"], coords={"L": [1, 2, 3]})
        do_ts = xr.DataArray(np.array([0.0, -0.3, -0.1]), dims=["L"], coords={"L": [1, 2, 3]})

        pts = build_2axis_trajectory(dm_ts, do_ts, d_initial_model=1.0, d_initial_obs=2.0, lead_dim="L")
        assert len(pts) == 3
        assert pts[1].lead_name == "2"
        assert pts[1].delta_d_model == -0.5
        assert pts[1].delta_d_obs == -0.3
        assert pts[1].category == MovementCategory.TOWARD_BOTH
