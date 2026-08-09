"""
tests/test_monthly_core.py
==========================
Offline smoke tests for monthly_core.py (pure core).

No filesystem I/O, no real NetCDF files required. All tests use synthetic
numpy and xarray DataArrays.

Test classes
------------
1. TestConfiguration          — MonthlyConfig validation, WindowDef, VariableConversionSpec
2. TestLeadCoverage           — check_lead_coverage: complete, missing, duplicates
3. TestGateClassification     — classify_analysis_status (window-aware), classify_archive_status
4. TestPairedReadiness        — build_paired_readiness: common paired inits, blocked logic
5. TestSpatialDiagnostics      — spatial_bias, spatial_adjustment, spatial_paired_diff, window_average
6. TestBootstrap              — bootstrap_spatial_ci, significance_mask
7. TestUnitConversions         — convert_prect_to_mmday, convert_kelvin_to_celsius, no_conversion
8. TestInventoryOutput        — InventoryRecord, dataframe export, JSON / report formatting

Run
---
    conda run -n e3sm_analysis python3 -m pytest tests/test_monthly_core.py -v --override-ini="addopts="
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from esp_lab.diagnostics.monthly_core import (
    DEFAULT_EXPERIMENT_SPECS,
    DEFAULT_WINDOW_DEFS,
    ExperimentSpec,
    GateStatus,
    InventoryRecord,
    InventoryStatus,
    LeadCoverageResult,
    MonthlyConfig,
    PairedReadinessReport,
    VariableConversionSpec,
    WindowDef,
    apply_mask,
    bootstrap_spatial_ci,
    build_paired_readiness,
    check_lead_coverage,
    classify_analysis_status,
    classify_archive_status,
    convert_kelvin_to_celsius,
    convert_prect_to_mmday,
    init_tag_from_year_month,
    inventory_records_to_df,
    no_conversion,
    significance_mask,
    spatial_adjustment,
    spatial_bias,
    spatial_paired_diff,
    valid_year_month_spatial,
    window_average,
    write_inventory_json,
    write_missing_report,
)


# ===========================================================================
# 1. TestConfiguration
# ===========================================================================

class TestConfiguration:
    def test_window_def_valid(self):
        w = WindowDef(name="months_1_3", lead_first=1, lead_last=3)
        assert w.leads == [1, 2, 3]

    def test_window_def_invalid_leads(self):
        with pytest.raises(ValueError, match="lead_last"):
            WindowDef(name="bad", lead_first=4, lead_last=2)

    def test_window_def_invalid_first_lead(self):
        with pytest.raises(ValueError, match="lead_first"):
            WindowDef(name="bad", lead_first=0, lead_last=3)

    def test_variable_conversion_spec_mask(self):
        spec = VariableConversionSpec(
            native_name="PRECT",
            plot_name="Precipitation",
            plot_units="mm/day",
            mask_type="land",
        )
        assert spec.mask_type == "land"

    def test_variable_conversion_spec_invalid_mask(self):
        with pytest.raises(ValueError, match="mask_type"):
            VariableConversionSpec(
                native_name="PRECT",
                plot_name="Precipitation",
                plot_units="mm/day",
                mask_type="invalid_mask",
            )

    def test_monthly_config_defaults(self):
        cfg = MonthlyConfig(
            experiments=DEFAULT_EXPERIMENT_SPECS,
            init_years=[1980, 1981],
            init_months=[5, 11],
            members=["EN00", "EN01"],
            leads=list(range(1, 25)),
            window_defs=DEFAULT_WINDOW_DEFS,
            variables=[
                VariableConversionSpec("PRECT", "Precipitation", "mm/day")
            ],
            data_dir="/tmp/test_data",
        )
        assert cfg.n_experiments == 2
        assert cfg.active_years == [1980, 1981, 1982]  # default pilot_years
        assert cfg.active_months == [5]                 # default pilot_months

    def test_monthly_config_full_campaign(self):
        cfg = MonthlyConfig(
            experiments=DEFAULT_EXPERIMENT_SPECS,
            init_years=[1980, 1981, 1982, 1983],
            init_months=[5, 11],
            members=["EN00"],
            leads=list(range(1, 25)),
            window_defs=DEFAULT_WINDOW_DEFS,
            variables=[
                VariableConversionSpec("PRECT", "Precipitation", "mm/day")
            ],
            data_dir="/tmp/test_data",
            pilot_only=False,
        )
        assert cfg.active_years == [1980, 1981, 1982, 1983]
        assert cfg.active_months == [5, 11]

    def test_invalid_baseline_lead(self):
        with pytest.raises(ValueError, match="baseline_lead"):
            MonthlyConfig(
                experiments=DEFAULT_EXPERIMENT_SPECS,
                init_years=[1980],
                init_months=[5],
                members=["EN00"],
                leads=[1, 2, 3],
                window_defs={"w1": (1, 2)},
                variables=[VariableConversionSpec("P", "P", "mm/day")],
                data_dir="/tmp",
                baseline_lead=10,  # not in leads
            )


# ===========================================================================
# 2. TestLeadCoverage
# ===========================================================================

class TestLeadCoverage:
    def test_complete_leads(self):
        res = check_lead_coverage(
            available_leads=range(1, 25),
            expected_leads=range(1, 25),
        )
        assert res.is_complete is True
        assert res.missing_leads == []
        assert res.duplicate_leads == []

    def test_missing_leads(self):
        res = check_lead_coverage(
            available_leads=[1, 2, 3, 5, 6],
            expected_leads=[1, 2, 3, 4, 5, 6],
        )
        assert res.is_complete is False
        assert res.missing_leads == [4]

    def test_duplicate_leads(self):
        res = check_lead_coverage(
            available_leads=[1, 2, 2, 3, 4],
            expected_leads=[1, 2, 3, 4],
        )
        assert res.is_complete is False
        assert res.has_duplicates is True
        assert res.duplicate_leads == [2]


# ===========================================================================
# 3. TestGateClassification
# ===========================================================================

class TestGateClassification:
    def test_analysis_ready_when_complete(self):
        cov = check_lead_coverage(range(1, 25), range(1, 25))
        status = classify_analysis_status(cov, DEFAULT_WINDOW_DEFS)
        assert status == GateStatus.ANALYSIS_READY

    def test_window_aware_unblocked(self):
        # Missing lead 20, but windows requested are only 1-3, 4-6, 10-12
        cov = check_lead_coverage(
            available_leads=[L for L in range(1, 25) if L != 20],
            expected_leads=range(1, 25),
        )
        subset_windows = {
            "months_1_3": (1, 3),
            "months_4_6": (4, 6),
            "months_10_12": (10, 12),
        }
        status = classify_analysis_status(cov, subset_windows)
        assert status == GateStatus.ANALYSIS_READY

    def test_window_aware_blocked(self):
        # Missing lead 20, and second_year (13-24) IS requested
        cov = check_lead_coverage(
            available_leads=[L for L in range(1, 25) if L != 20],
            expected_leads=range(1, 25),
        )
        status = classify_analysis_status(cov, DEFAULT_WINDOW_DEFS)
        assert status == GateStatus.BLOCKED

    def test_archive_status_complete(self):
        cov = check_lead_coverage(range(1, 25), range(1, 25))
        assert classify_archive_status(cov, nlead=24) == GateStatus.ARCHIVE_COMPLETE

    def test_archive_status_blocked(self):
        cov = check_lead_coverage(range(1, 24), range(1, 25))
        assert classify_archive_status(cov, nlead=24) == GateStatus.BLOCKED


# ===========================================================================
# 4. TestPairedReadiness
# ===========================================================================

class TestPairedReadiness:
    def test_paired_readiness_all_complete(self):
        init_tags = ["1980050100", "1981050100"]
        members = ["EN00", "EN01"]

        rows = []
        for tag in init_tags:
            for m in members:
                rows.append({
                    "init_tag": tag,
                    "member": m,
                    "analysis_status": GateStatus.ANALYSIS_READY.value,
                    "available_leads": "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24",
                })
        ref_df = pd.DataFrame(rows)
        test_df = pd.DataFrame(rows)

        report = build_paired_readiness(
            ref_inventory=ref_df,
            test_inventory=test_df,
            window_defs=DEFAULT_WINDOW_DEFS,
            init_tags=init_tags,
            members=members,
        )
        assert report.gate_pass is True
        assert len(report.common_paired_inits) == 4
        assert len(report.ref_only_inits) == 0

    def test_paired_readiness_missing_one_member(self):
        init_tags = ["1980050100"]
        members = ["EN00", "EN01"]

        ref_rows = [
            {"init_tag": "1980050100", "member": "EN00", "analysis_status": GateStatus.ANALYSIS_READY.value, "available_leads": "1,2,3,4,5,6"},
            {"init_tag": "1980050100", "member": "EN01", "analysis_status": GateStatus.ANALYSIS_READY.value, "available_leads": "1,2,3,4,5,6"},
        ]
        test_rows = [
            {"init_tag": "1980050100", "member": "EN00", "analysis_status": GateStatus.ANALYSIS_READY.value, "available_leads": "1,2,3,4,5,6"},
            {"init_tag": "1980050100", "member": "EN01", "analysis_status": GateStatus.BLOCKED.value, "available_leads": "1,2,3"},
        ]

        report = build_paired_readiness(
            ref_inventory=pd.DataFrame(ref_rows),
            test_inventory=pd.DataFrame(test_rows),
            window_defs={"w1": (1, 6)},
            init_tags=init_tags,
            members=members,
        )
        assert report.gate_pass is True
        assert ("1980050100", "EN00") in report.common_paired_inits
        assert ("1980050100", "EN01") not in report.common_paired_inits


# ===========================================================================
# 5. TestSpatialDiagnostics
# ===========================================================================

class TestSpatialDiagnostics:
    def _synthetic_spatial_data(self):
        lat = np.linspace(-90, 90, 5)
        lon = np.linspace(-180, 180, 5)
        years = [1980, 1981]
        leads = list(range(1, 7))

        rng = np.random.default_rng(42)
        arr_ref = rng.random((2, 3, 6, 5, 5)) + 1.0   # Y, M, L, lat, lon
        arr_test = arr_ref + 0.5                      # test is offset by +0.5

        da_ref = xr.DataArray(arr_ref, dims=["Y", "M", "L", "lat", "lon"],
                              coords={"Y": years, "M": ["EN00", "EN01", "EN02"], "L": leads, "lat": lat, "lon": lon})
        da_test = xr.DataArray(arr_test, dims=["Y", "M", "L", "lat", "lon"],
                               coords={"Y": years, "M": ["EN00", "EN01", "EN02"], "L": leads, "lat": lat, "lon": lon})
        return da_ref, da_test

    def test_spatial_adjustment_zero_at_baseline(self):
        da_ref, da_test = self._synthetic_spatial_data()
        # Model ensemble mean for ref
        ref_em = da_ref.mean(["M", "Y"])  # (L, lat, lon)
        adj_ref = spatial_adjustment(ref_em, baseline_lead=1)

        # Lead 1 adjustment should be identically 0.0 everywhere
        lead1_adj = adj_ref.sel(L=1)
        np.testing.assert_allclose(lead1_adj.values, 0.0, atol=1e-12)

    def test_spatial_paired_diff(self):
        da_ref, da_test = self._synthetic_spatial_data()
        ref_em = da_ref.mean(["M", "Y"])
        test_em = da_test.mean(["M", "Y"])

        adj_ref = spatial_adjustment(ref_em, baseline_lead=1)
        adj_test = spatial_adjustment(test_em, baseline_lead=1)

        paired_diff = spatial_paired_diff(adj_test, adj_ref)
        assert paired_diff.dims == ("L", "lat", "lon")
        # Since test = ref + 0.5 uniformly, (test - test(1)) - (ref - ref(1)) == 0.0
        np.testing.assert_allclose(paired_diff.values, 0.0, atol=1e-12)

    def test_window_average(self):
        lat = np.linspace(-90, 90, 3)
        lon = np.linspace(-180, 180, 3)
        data = np.ones((6, 3, 3))
        # lead 1=1, lead 2=2, lead 3=3 ...
        for l_idx in range(6):
            data[l_idx] *= (l_idx + 1)

        da = xr.DataArray(data, dims=["L", "lat", "lon"], coords={"L": range(1, 7), "lat": lat, "lon": lon})
        w = WindowDef("months_1_3", 1, 3)
        avg = window_average(da, w)

        assert "L" not in avg.dims
        # Average of 1, 2, 3 is 2.0
        np.testing.assert_allclose(avg.values, 2.0)

    def test_window_average_missing_lead_raises(self):
        da = xr.DataArray(np.ones((2, 3, 3)), dims=["L", "lat", "lon"], coords={"L": [1, 2], "lat": [0, 1, 2], "lon": [0, 1, 2]})
        w = WindowDef("months_1_3", 1, 3)
        with pytest.raises(ValueError, match="missing from the field"):
            window_average(da, w, verify_complete=True)


# ===========================================================================
# 6. TestBootstrap
# ===========================================================================

class TestBootstrap:
    def test_bootstrap_spatial_ci_zero_diff(self):
        lat = np.linspace(-90, 90, 3)
        lon = np.linspace(-180, 180, 3)
        years = [1980, 1981, 1982, 1983, 1984]

        # 5 years of zero difference
        diff_by_year = xr.DataArray(
            np.zeros((5, 3, 3)),
            dims=["Y", "lat", "lon"],
            coords={"Y": years, "lat": lat, "lon": lon},
        )
        lower, upper = bootstrap_spatial_ci(diff_by_year, n_boot=100, seed=42)
        np.testing.assert_allclose(lower.values, 0.0, atol=1e-12)
        np.testing.assert_allclose(upper.values, 0.0, atol=1e-12)

        sig = significance_mask(lower, upper)
        assert not np.any(sig.values)

    def test_bootstrap_spatial_ci_significant(self):
        lat = np.linspace(-90, 90, 3)
        lon = np.linspace(-180, 180, 3)
        years = [1980, 1981, 1982, 1983, 1984]

        # 5 years of large positive difference (5.0 + small noise)
        rng = np.random.default_rng(42)
        data = 5.0 + rng.normal(0, 0.1, size=(5, 3, 3))
        diff_by_year = xr.DataArray(
            data,
            dims=["Y", "lat", "lon"],
            coords={"Y": years, "lat": lat, "lon": lon},
        )
        lower, upper = bootstrap_spatial_ci(diff_by_year, n_boot=100, seed=42)
        assert np.all(lower.values > 0.0)
        sig = significance_mask(lower, upper)
        assert np.all(sig.values)


# ===========================================================================
# 7. TestUnitConversions
# ===========================================================================

class TestUnitConversions:
    def test_prect_conversion(self):
        da = xr.DataArray(np.array([1e-8, 2e-8]), attrs={"units": "m/s"})
        out = convert_prect_to_mmday(da)
        assert out.attrs["units"] == "mm/day"
        np.testing.assert_allclose(out.values, da.values * 1000.0 * 86400.0)

    def test_prect_conversion_already_mmday(self):
        da = xr.DataArray(np.array([1.0, 2.0]), attrs={"units": "mm/day"})
        out = convert_prect_to_mmday(da)
        np.testing.assert_allclose(out.values, [1.0, 2.0])

    def test_kelvin_conversion(self):
        da = xr.DataArray(np.array([273.15, 300.0]), attrs={"units": "K"})
        out = convert_kelvin_to_celsius(da)
        assert out.attrs["units"] == "degC"
        np.testing.assert_allclose(out.values, [0.0, 26.85])

    def test_no_conversion(self):
        da = xr.DataArray(np.array([10.0, 20.0]), attrs={"units": "W/m²"})
        out = no_conversion(da)
        np.testing.assert_allclose(out.values, [10.0, 20.0])


# ===========================================================================
# 8. TestInventoryOutput
# ===========================================================================

class TestInventoryOutput:
    def test_inventory_record_to_dict(self):
        rec = InventoryRecord(
            experiment="JRA55_FOSIRL",
            case_prefix="case_prefix",
            init_tag="1980050100",
            member="EN00",
            variable="PRECT",
            analysis_status=GateStatus.ANALYSIS_READY.value,
        )
        d = rec.to_dict()
        assert d["experiment"] == "JRA55_FOSIRL"
        assert d["analysis_status"] == "ANALYSIS_READY"

    def test_inventory_records_to_df(self):
        recs = [
            InventoryRecord("E1", "C1", "1980050100", "EN00", "PRECT"),
            InventoryRecord("E2", "C2", "1980050100", "EN00", "PRECT"),
        ]
        df = inventory_records_to_df(recs)
        assert len(df) == 2
        assert "experiment" in df.columns

    def test_missing_report_formatting(self, tmp_path):
        recs = [
            InventoryRecord("E1", "C1", "1980050100", "EN00", "PRECT", analysis_status="BLOCKED", missing_leads="20"),
        ]
        df = inventory_records_to_df(recs)
        rpt_path = tmp_path / "missing_report.txt"
        write_missing_report(df, rpt_path, config_summary="test config")
        assert rpt_path.exists()
        text = rpt_path.read_text()
        assert "Missing Data Report" in text
        assert "E1 | 1980050100 | EN00 | PRECT | missing_leads=20" in text

    def test_json_output_formatting(self, tmp_path):
        recs = [
            InventoryRecord("E1", "C1", "1980050100", "EN00", "PRECT", analysis_status="ANALYSIS_READY"),
        ]
        df = inventory_records_to_df(recs)
        json_path = tmp_path / "summary.json"
        write_inventory_json(df, json_path, extra_meta={"test_key": "test_val"})
        assert json_path.exists()
        text = json_path.read_text()
        assert "test_val" in text
        assert "ANALYSIS_READY" in text
