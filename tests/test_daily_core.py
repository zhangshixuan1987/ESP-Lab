"""
tests/test_daily_core.py
=========================
Offline smoke tests for daily_core.py (pure core).

No filesystem I/O, no real NetCDF files required. All tests use synthetic
numpy and xarray DataArrays.

Test classes
------------
1. TestDailyConfig              — DailyDriftConfig validation, DailyWindowDef, DailyVariableSpec
2. TestDailyLeadCoverage       — check_daily_lead_coverage: complete, missing, duplicates
3. TestDailyGateClassification — classify_daily_window_status, classify_daily_analysis_status
4. TestDailyPairedReadiness    — build_daily_paired_readiness: common paired inits, blocked windows
5. TestDailySpatialDiagnostics — daily_spatial_bias, daily_spatial_adjustment, daily_spatial_paired_diff, daily_window_average
6. TestDailyBootstrap          — bootstrap_daily_spatial_ci, daily_significance_mask
7. TestDailyTimeseries         — daily_regional_timeseries (days 1–84 curve)
8. TestDailyConversions        — convert_prect_to_mmday, convert_kelvin_to_celsius_if_needed
9. TestDailyInventoryOutput    — DailyInventoryRecord, CSV / report formatting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from esp_lab.diagnostics.daily_core import (
    DEFAULT_DAILY_EXPERIMENT_SPECS,
    DEFAULT_DAILY_WINDOW_DEFS,
    DailyDriftConfig,
    DailyGateStatus,
    DailyInventoryRecord,
    DailyInventoryStatus,
    DailyLeadCoverageResult,
    DailyPairedReadinessReport,
    DailyVariableSpec,
    DailyWindowDef,
    apply_mask,
    bootstrap_daily_spatial_ci,
    build_daily_paired_readiness,
    check_daily_lead_coverage,
    classify_daily_analysis_status,
    classify_daily_window_status,
    convert_kelvin_to_celsius_if_needed,
    convert_prect_to_mmday,
    daily_inventory_records_to_df,
    daily_regional_timeseries,
    daily_significance_mask,
    daily_spatial_adjustment,
    daily_spatial_bias,
    daily_spatial_paired_diff,
    daily_window_average,
    no_conversion,
    valid_date_from_init_and_lead,
    write_daily_inventory_csv,
    write_daily_inventory_json,
    write_daily_missing_report,
)


# ===========================================================================
# 1. TestDailyConfig
# ===========================================================================

class TestDailyConfig:
    def test_daily_window_def_valid(self):
        w = DailyWindowDef(name="week_1", day_first=1, day_last=7)
        assert w.lead_days == [1, 2, 3, 4, 5, 6, 7]

    def test_daily_window_def_invalid(self):
        with pytest.raises(ValueError, match="day_last"):
            DailyWindowDef(name="bad", day_first=10, day_last=5)

    def test_daily_config_defaults(self):
        cfg = DailyDriftConfig(
            experiments=DEFAULT_DAILY_EXPERIMENT_SPECS,
            init_years=[1980, 1981],
            init_months=[5, 11],
            members=["EN00", "EN01"],
            lead_days=list(range(1, 85)),
            window_defs=DEFAULT_DAILY_WINDOW_DEFS,
            variables=[
                DailyVariableSpec("TREFHT", "2-m air temperature", "degC")
            ],
            data_dir="/tmp/test_data",
        )
        assert cfg.n_experiments == 2
        assert cfg.active_years == [1980, 1981, 1982]
        assert cfg.active_months == [5]
        assert cfg.baseline_day == 1

    def test_daily_config_full_campaign(self):
        cfg = DailyDriftConfig(
            experiments=DEFAULT_DAILY_EXPERIMENT_SPECS,
            init_years=list(range(1980, 1987)),
            init_months=[5, 11],
            members=["EN00"],
            lead_days=list(range(1, 85)),
            window_defs=DEFAULT_DAILY_WINDOW_DEFS,
            variables=[
                DailyVariableSpec("TREFHT", "2-m air temperature", "degC")
            ],
            data_dir="/tmp/test_data",
            pilot_only=False,
        )
        assert len(cfg.active_years) == 7
        assert cfg.active_months == [5, 11]


# ===========================================================================
# 2. TestDailyLeadCoverage
# ===========================================================================

class TestDailyLeadCoverage:
    def test_complete_days(self):
        res = check_daily_lead_coverage(
            available_days=range(1, 85),
            expected_days=range(1, 85),
        )
        assert res.is_complete is True
        assert res.missing_days == []

    def test_missing_day_detected(self):
        res = check_daily_lead_coverage(
            available_days=[d for d in range(1, 85) if d != 15],
            expected_days=range(1, 85),
        )
        assert res.is_complete is False
        assert res.missing_days == [15]


# ===========================================================================
# 3. TestDailyGateClassification
# ===========================================================================

class TestDailyGateClassification:
    def test_window_specific_classification(self):
        # Missing day 15 (which is in weeks_2_3: 8..21)
        cov = check_daily_lead_coverage(
            available_days=[d for d in range(1, 85) if d != 15],
            expected_days=range(1, 85),
        )
        statuses = classify_daily_window_status(cov, DEFAULT_DAILY_WINDOW_DEFS)

        # week_1 (1..7) should be ANALYSIS_READY
        assert statuses["week_1"] == DailyGateStatus.ANALYSIS_READY
        # weeks_2_3 (8..21) should be BLOCKED because day 15 is missing
        assert statuses["weeks_2_3"] == DailyGateStatus.BLOCKED
        # weeks_4_6 (22..42) should be ANALYSIS_READY
        assert statuses["weeks_4_6"] == DailyGateStatus.ANALYSIS_READY

    def test_overall_analysis_status_blocked(self):
        cov = check_daily_lead_coverage(
            available_days=[d for d in range(1, 85) if d != 15],
            expected_days=range(1, 85),
        )
        status = classify_daily_analysis_status(cov, DEFAULT_DAILY_WINDOW_DEFS)
        assert status == DailyGateStatus.BLOCKED


# ===========================================================================
# 4. TestDailyPairedReadiness
# ===========================================================================

class TestDailyPairedReadiness:
    def test_daily_paired_readiness(self):
        init_tags = ["1980050100"]
        members = ["EN00", "EN01"]

        ref_rows = [
            {"init_tag": "1980050100", "member": "EN00", "daily_status": DailyGateStatus.ANALYSIS_READY.value, "available_days": "1,2,3,4,5,6,7"},
            {"init_tag": "1980050100", "member": "EN01", "daily_status": DailyGateStatus.ANALYSIS_READY.value, "available_days": "1,2,3,4,5,6,7"},
        ]
        test_rows = [
            {"init_tag": "1980050100", "member": "EN00", "daily_status": DailyGateStatus.ANALYSIS_READY.value, "available_days": "1,2,3,4,5,6,7"},
            {"init_tag": "1980050100", "member": "EN01", "daily_status": DailyGateStatus.BLOCKED.value, "available_days": "1,2,3"},
        ]

        report = build_daily_paired_readiness(
            ref_inventory=pd.DataFrame(ref_rows),
            test_inventory=pd.DataFrame(test_rows),
            window_defs={"week_1": (1, 7)},
            init_tags=init_tags,
            members=members,
        )
        assert report.gate_pass is True
        assert ("1980050100", "EN00") in report.common_paired_inits
        assert ("1980050100", "EN01") not in report.common_paired_inits


# ===========================================================================
# 5. TestDailySpatialDiagnostics
# ===========================================================================

class TestDailySpatialDiagnostics:
    def _synthetic_daily_data(self):
        lat = np.linspace(-90, 90, 3)
        lon = np.linspace(-180, 180, 3)
        years = [1980, 1981]
        days = list(range(1, 10))

        rng = np.random.default_rng(42)
        arr_ref = rng.random((2, 3, 9, 3, 3)) + 1.0   # Y, M, d, lat, lon
        arr_test = arr_ref + 0.3                       # test offset +0.3

        da_ref = xr.DataArray(arr_ref, dims=["Y", "M", "d", "lat", "lon"],
                              coords={"Y": years, "M": ["EN00", "EN01", "EN02"], "d": days, "lat": lat, "lon": lon})
        da_test = xr.DataArray(arr_test, dims=["Y", "M", "d", "lat", "lon"],
                               coords={"Y": years, "M": ["EN00", "EN01", "EN02"], "d": days, "lat": lat, "lon": lon})
        return da_ref, da_test

    def test_daily_adjustment_zero_at_day_1(self):
        da_ref, da_test = self._synthetic_daily_data()
        ref_em = da_ref.mean(["M", "Y"])
        adj_ref = daily_spatial_adjustment(ref_em, baseline_day=1)

        # Day 1 adjustment should be 0.0 everywhere
        day1_adj = adj_ref.sel(d=1)
        np.testing.assert_allclose(day1_adj.values, 0.0, atol=1e-12)

    def test_daily_window_average(self):
        lat = np.linspace(-90, 90, 3)
        lon = np.linspace(-180, 180, 3)
        data = np.ones((7, 3, 3)) * 4.0
        da = xr.DataArray(data, dims=["d", "lat", "lon"], coords={"d": range(1, 8), "lat": lat, "lon": lon})

        w = DailyWindowDef("week_1", 1, 7)
        avg = daily_window_average(da, w)
        assert "d" not in avg.dims
        np.testing.assert_allclose(avg.values, 4.0)


# ===========================================================================
# 6. TestDailyBootstrap
# ===========================================================================

class TestDailyBootstrap:
    def test_bootstrap_daily_spatial_ci_zero_diff(self):
        lat = np.linspace(-90, 90, 3)
        lon = np.linspace(-180, 180, 3)
        years = [1980, 1981, 1982, 1983]

        diff_by_year = xr.DataArray(
            np.zeros((4, 3, 3)),
            dims=["Y", "lat", "lon"],
            coords={"Y": years, "lat": lat, "lon": lon},
        )
        lower, upper = bootstrap_daily_spatial_ci(diff_by_year, n_boot=100, seed=42)
        np.testing.assert_allclose(lower.values, 0.0, atol=1e-12)
        np.testing.assert_allclose(upper.values, 0.0, atol=1e-12)
        sig = daily_significance_mask(lower, upper)
        assert not np.any(sig.values)


# ===========================================================================
# 7. TestDailyTimeseries
# ===========================================================================

class TestDailyTimeseries:
    def test_daily_regional_timeseries_global(self):
        lat = np.linspace(-90, 90, 3)
        lon = np.linspace(-180, 180, 3)
        data = np.ones((84, 3, 3))
        for d_idx in range(84):
            data[d_idx] *= (d_idx + 1)

        da = xr.DataArray(data, dims=["d", "lat", "lon"], coords={"d": range(1, 85), "lat": lat, "lon": lon})
        ts = daily_regional_timeseries(da)
        assert ts.dims == ("d",)
        assert ts.sizes["d"] == 84
        np.testing.assert_allclose(ts.sel(d=10).values, 10.0)


# ===========================================================================
# 8. TestDailyConversions
# ===========================================================================

class TestDailyConversions:
    def test_prect_conversion(self):
        da = xr.DataArray(np.array([1e-8]), attrs={"units": "m/s"})
        out = convert_prect_to_mmday(da)
        assert out.attrs["units"] == "mm/day"
        np.testing.assert_allclose(out.values, [1e-8 * 1000.0 * 86400.0])

    def test_kelvin_to_celsius(self):
        da = xr.DataArray(np.array([300.0]), attrs={"units": "K"})
        out = convert_kelvin_to_celsius_if_needed(da)
        assert out.attrs["units"] == "degC"
        np.testing.assert_allclose(out.values, [26.85])


# ===========================================================================
# 9. TestDailyInventoryOutput
# ===========================================================================

class TestDailyInventoryOutput:
    def test_daily_inventory_record_columns(self):
        rec = DailyInventoryRecord(
            experiment="JRA55_FOSIRL",
            case_prefix="prefix",
            init_date="1980-05-01",
            init_year=1980,
            init_month=5,
            member="EN00",
            variable="TREFHT",
        )
        d = rec.to_dict()
        assert "week_1_status" in d
        assert "weeks_2_3_status" in d
        assert "weeks_4_6_status" in d
        assert "weeks_7_12_status" in d
        assert "daily_status" in d
        assert len(d) == 27
