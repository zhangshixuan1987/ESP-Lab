"""
tests/test_ic_core.py
=====================
Offline smoke tests for the IC analysis pure core (ic_core.py).

No NERSC access, no real data files, no filesystem writes.  All data is
synthetic.  The I/O bridge (ic_io.py) is not tested here.

Test classes
------------
1. TestAuditResult         — compare_file_records: identical, different, missing,
                             incompatible dimensions
2. TestVariableStats       — ic_variable_stats: known arrays, area-weighted vs
                             unweighted, all-NaN guard, zero-reference guard
3. TestNCSchema            — compare_nc_schema: common vars, extra vars on each
                             side, dimension mismatch, dtype mismatch
4. TestClassifyVariable    — classify_variable: physical, metadata, unknown
5. TestCampaignAggregation — aggregate_campaign_stats: May/Nov split, RMSE mean,
                             sign-agreement fraction, bad group_by raises
6. TestConsistencyChecks   — check_atm_surface_vs_land: aligned mismatch,
                             no-overlap case; check_ocean_ice_consistency:
                             freezing-but-no-ice flagged
7. TestICConfig            — ICConfig construction: valid, invalid date format,
                             empty members, pilot_only vs active_dates,
                             date_season helper

Run
---
    cd /global/homes/z/zhan391/code/ESP-Lab
    conda run -n e3sm_analysis python3 -m pytest tests/test_ic_core.py -v --override-ini="addopts="
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from esp_lab.diagnostics.ic_core import (
    AuditResult,
    ComponentSpec,
    ConsistencyReport,
    DEFAULT_COMPONENTS,
    ExperimentPair,
    FileRecord,
    ICConfig,
    StructureDiff,
    aggregate_campaign_stats,
    check_atm_surface_vs_land,
    check_ocean_ice_consistency,
    classify_variable,
    compare_file_records,
    compare_nc_schema,
    ic_variable_stats,
    write_manifest_json,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_record(
    sha: str = "abc123",
    size: int = 1000,
    dims: dict | None = None,
    path: str = "/tmp/fake.nc",
    component: str = "lnd",
    experiment: str = "BruteForce",
    start_date: str = "1980-05-01-00000",
) -> FileRecord:
    return FileRecord(
        path=path,
        component=component,
        experiment=experiment,
        start_date=start_date,
        size_bytes=size,
        sha256=sha,
        nc_dims=dims or {},
    )


def _make_da(
    values: np.ndarray,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
) -> xr.DataArray:
    if lat is not None and lon is not None:
        return xr.DataArray(
            values,
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
    return xr.DataArray(values.ravel(), dims=["x"])


def _make_ds(var_dict: dict[str, np.ndarray]) -> xr.Dataset:
    return xr.Dataset({k: xr.DataArray(v, dims=["x"]) for k, v in var_dict.items()})


# ===========================================================================
# 1. TestAuditResult
# ===========================================================================


class TestAuditResult:
    def test_identical_checksums(self):
        r = _make_record(sha="deadbeef")
        t = _make_record(sha="deadbeef")
        assert compare_file_records(r, t) == AuditResult.IDENTICAL

    def test_different_checksums(self):
        r = _make_record(sha="aaa")
        t = _make_record(sha="bbb")
        assert compare_file_records(r, t) == AuditResult.DIFFERENT

    def test_missing_ref(self):
        r = _make_record(size=-1, sha="")
        t = _make_record(sha="bbb")
        assert compare_file_records(r, t) == AuditResult.MISSING

    def test_missing_test(self):
        r = _make_record(sha="aaa")
        t = _make_record(size=-1, sha="")
        assert compare_file_records(r, t) == AuditResult.MISSING

    def test_both_missing(self):
        r = _make_record(size=-1)
        t = _make_record(size=-1)
        assert compare_file_records(r, t) == AuditResult.MISSING

    def test_incompatible_dimensions(self):
        r = _make_record(sha="aaa", dims={"nCells": 100, "nVertLevels": 10})
        t = _make_record(sha="bbb", dims={"nCells": 200, "nVertLevels": 10})
        assert compare_file_records(r, t) == AuditResult.INCOMPATIBLE_STRUCTURE

    def test_compatible_dims_different_sha(self):
        r = _make_record(sha="aaa", dims={"nCells": 100})
        t = _make_record(sha="bbb", dims={"nCells": 100})
        assert compare_file_records(r, t) == AuditResult.DIFFERENT

    def test_empty_sha_conservative(self):
        """No checksum computed → conservative DIFFERENT."""
        r = _make_record(sha="")
        t = _make_record(sha="")
        assert compare_file_records(r, t) == AuditResult.DIFFERENT

    def test_disjoint_dim_keys(self):
        """Completely different dimension name sets → INCOMPATIBLE_STRUCTURE."""
        r = _make_record(sha="aaa", dims={"latlon": 64800})
        t = _make_record(sha="bbb", dims={"nCells": 100})
        assert compare_file_records(r, t) == AuditResult.INCOMPATIBLE_STRUCTURE


# ===========================================================================
# 2. TestVariableStats
# ===========================================================================


class TestVariableStats:
    def _lat_lon_arrays(self, n=10):
        lat = np.linspace(-90, 90, n)
        lon = np.linspace(-180, 180, n)
        rng = np.random.default_rng(0)
        ref = _make_da(rng.random((n, n)), lat=lat, lon=lon)
        test = _make_da(rng.random((n, n)) + 0.5, lat=lat, lon=lon)
        return ref, test

    def test_basic_stats_finite(self):
        ref, test = self._lat_lon_arrays()
        stats = ic_variable_stats(ref, test)
        for key in ("rmse", "mad", "mean_diff", "pattern_corr", "n_valid"):
            assert key in stats, f"Missing key: {key}"
        assert stats["n_valid"] > 0
        assert stats["rmse"] >= 0
        assert stats["mad"] >= 0

    def test_rmse_known_value(self):
        """RMSE of a uniform shift of exactly 1.0 should equal 1.0."""
        ref = xr.DataArray(np.zeros(100), dims=["x"])
        test = xr.DataArray(np.ones(100), dims=["x"])
        stats = ic_variable_stats(ref, test)
        assert abs(stats["rmse"] - 1.0) < 1e-9

    def test_complete_scientific_metric_contract(self):
        ref = xr.DataArray(np.array([0.0, 1.0, 2.0, 3.0]), dims=["x"])
        test = xr.DataArray(np.array([0.0, 2.0, 4.0, 6.0]), dims=["x"])
        stats = ic_variable_stats(ref, test, meaningful_threshold=1.5)
        assert stats["max_abs_diff"] == pytest.approx(3.0)
        assert stats["p50_diff"] == pytest.approx(1.5)
        assert stats["frac_exceeding_threshold"] == pytest.approx(0.5)
        assert stats["meaningful_threshold"] == pytest.approx(1.5)
        assert stats["nrmse"] == pytest.approx(stats["rmse"] / stats["reference_std"])
        assert stats["integral_diff"] == pytest.approx(6.0)
        assert stats["integral_pct_diff"] == pytest.approx(100.0)

    def test_negative_threshold_rejected(self):
        ref = xr.DataArray(np.ones(4), dims=["x"])
        with pytest.raises(ValueError, match="non-negative"):
            ic_variable_stats(ref, ref, meaningful_threshold=-1.0)

    def test_identical_returns_zero_rmse(self):
        ref = xr.DataArray(np.linspace(0, 1, 50), dims=["x"])
        test = ref.copy()
        stats = ic_variable_stats(ref, test)
        assert stats["rmse"] < 1e-12

    def test_all_nan_returns_empty(self):
        ref = xr.DataArray(np.full(10, np.nan), dims=["x"])
        test = xr.DataArray(np.full(10, np.nan), dims=["x"])
        stats = ic_variable_stats(ref, test)
        assert stats["n_valid"] == 0
        assert np.isnan(stats["rmse"])

    def test_area_weighted_vs_unweighted(self):
        """With unequal weights, results should differ from unweighted."""
        rng = np.random.default_rng(1)
        n = 20
        ref = xr.DataArray(rng.random(n), dims=["x"])
        test = xr.DataArray(rng.random(n), dims=["x"])
        weights = xr.DataArray(np.abs(rng.random(n)) + 0.1, dims=["x"])
        stats_uw = ic_variable_stats(ref, test)
        stats_w = ic_variable_stats(ref, test, area_weights=weights)
        # Both should be finite
        assert np.isfinite(stats_uw["rmse"])
        assert np.isfinite(stats_w["rmse"])

    def test_frac_differing_all_zero(self):
        ref = xr.DataArray(np.ones(20), dims=["x"])
        test = xr.DataArray(np.ones(20), dims=["x"])
        stats = ic_variable_stats(ref, test)
        assert stats["frac_differing"] < 1e-12

    def test_frac_differing_all_different(self):
        ref = xr.DataArray(np.zeros(20), dims=["x"])
        test = xr.DataArray(np.ones(20), dims=["x"])
        stats = ic_variable_stats(ref, test)
        assert stats["frac_differing"] == pytest.approx(1.0)


# ===========================================================================
# 3. TestNCSchema
# ===========================================================================


class TestNCSchema:
    def test_identical_datasets(self):
        ds = _make_ds({"a": np.ones(10), "b": np.zeros(10)})
        diff = compare_nc_schema(ds, ds)
        assert sorted(diff.common_vars) == ["a", "b"]
        assert diff.only_in_ref == []
        assert diff.only_in_test == []
        assert diff.dim_mismatches == {}
        assert diff.dtype_mismatches == {}

    def test_extra_var_in_ref(self):
        ds_ref = _make_ds({"a": np.ones(10), "b": np.zeros(10), "c": np.ones(10)})
        ds_test = _make_ds({"a": np.ones(10), "b": np.zeros(10)})
        diff = compare_nc_schema(ds_ref, ds_test)
        assert "c" in diff.only_in_ref
        assert diff.only_in_test == []

    def test_extra_var_in_test(self):
        ds_ref = _make_ds({"a": np.ones(10)})
        ds_test = _make_ds({"a": np.ones(10), "new": np.zeros(10)})
        diff = compare_nc_schema(ds_ref, ds_test)
        assert "new" in diff.only_in_test

    def test_dimension_mismatch(self):
        ds_ref = xr.Dataset({"t": xr.DataArray(np.ones(50), dims=["x"])})
        ds_test = xr.Dataset({"t": xr.DataArray(np.ones(100), dims=["x"])})
        diff = compare_nc_schema(ds_ref, ds_test)
        assert "x" in diff.dim_mismatches
        assert diff.dim_mismatches["x"] == (50, 100)

    def test_coordinate_ordering_mismatch(self):
        ds_ref = xr.Dataset(
            {"t": xr.DataArray([1.0, 2.0], dims=["x"])}, coords={"x": [0, 1]}
        )
        ds_test = xr.Dataset(
            {"t": xr.DataArray([2.0, 1.0], dims=["x"])}, coords={"x": [1, 0]}
        )
        diff = compare_nc_schema(ds_ref, ds_test)
        assert diff.coord_mismatches["x"] == "values or ordering differ"

    def test_units_mismatch(self):
        ds_ref = xr.Dataset({"t": xr.DataArray([1.0], dims=["x"], attrs={"units": "K"})})
        ds_test = xr.Dataset({"t": xr.DataArray([1.0], dims=["x"], attrs={"units": "degC"})})
        diff = compare_nc_schema(ds_ref, ds_test)
        assert "t" in diff.variable_attr_mismatches

    def test_dtype_mismatch(self):
        ds_ref = xr.Dataset({"t": xr.DataArray(np.ones(10, dtype=np.float32), dims=["x"])})
        ds_test = xr.Dataset({"t": xr.DataArray(np.ones(10, dtype=np.float64), dims=["x"])})
        diff = compare_nc_schema(ds_ref, ds_test)
        assert "t" in diff.dtype_mismatches

    def test_to_dict_has_expected_keys(self):
        ds = _make_ds({"a": np.ones(5)})
        diff = compare_nc_schema(ds, ds)
        d = diff.to_dict()
        for key in ("n_common", "n_only_ref", "n_only_test", "n_dim_mismatch"):
            assert key in d


# ===========================================================================
# 4. TestClassifyVariable
# ===========================================================================


class TestClassifyVariable:
    def test_ocean_temperature_is_physical(self):
        assert classify_variable("activeTracers_temperature", "ocn") == "physical"

    def test_land_tsoi_is_physical(self):
        assert classify_variable("TSOI", "lnd") == "physical"

    def test_time_is_metadata(self):
        assert classify_variable("time", "lnd") == "metadata"

    def test_nstep_is_metadata(self):
        assert classify_variable("nstep", "ocn") == "metadata"

    def test_counter_is_metadata(self):
        assert classify_variable("restart_counter", "cpl") == "metadata"

    def test_unknown_var(self):
        assert classify_variable("xyz_completely_new_var", "atm") == "unknown"

    def test_override_priority_keywords(self):
        result = classify_variable("xyz", "lnd", priority_keywords=["xyz"])
        assert result == "physical"

    def test_ice_aice_is_physical(self):
        assert classify_variable("aice", "ice") == "physical"


# ===========================================================================
# 5. TestCampaignAggregation
# ===========================================================================


def _make_stat_records(
    n_may: int = 5,
    n_nov: int = 5,
    component: str = "ocn",
    variable: str = "temperature",
    rmse_may: float = 0.5,
    rmse_nov: float = 1.0,
) -> list[dict]:
    rows = []
    for i in range(n_may):
        rows.append({
            "start_date": f"198{i}-05-01-00000",
            "component": component,
            "variable": variable,
            "season": "May",
            "rmse": rmse_may + i * 0.01,
            "mean_diff": 0.1 * (i + 1),
            "mad": 0.2,
            "pattern_corr": 0.9,
            "frac_differing": 0.3,
        })
    for i in range(n_nov):
        rows.append({
            "start_date": f"198{i}-11-01-00000",
            "component": component,
            "variable": variable,
            "season": "November",
            "rmse": rmse_nov + i * 0.01,
            "mean_diff": -0.1 * (i + 1),
            "mad": 0.4,
            "pattern_corr": 0.8,
            "frac_differing": 0.5,
        })
    return rows


class TestCampaignAggregation:
    def test_season_split_produces_two_rows(self):
        records = _make_stat_records()
        agg = aggregate_campaign_stats(records, group_by="season")
        seasons = set(agg["season"].tolist())
        assert "May" in seasons
        assert "November" in seasons

    def test_mean_rmse_closer_to_may(self):
        records = _make_stat_records(rmse_may=0.3, rmse_nov=1.0)
        agg = aggregate_campaign_stats(records, group_by="season,component,variable")
        may_row = agg[agg["season"] == "May"]
        nov_row = agg[agg["season"] == "November"]
        assert float(may_row["mean_rmse"].values[0]) < float(
            nov_row["mean_rmse"].values[0]
        )

    def test_sign_agreement_all_positive(self):
        """All May mean_diffs are positive → sign_agreement_frac = 1.0."""
        records = _make_stat_records(n_nov=0)
        agg = aggregate_campaign_stats(records, group_by="season")
        row = agg[agg["season"] == "May"]
        assert float(row["sign_agreement_frac"].values[0]) == pytest.approx(1.0)

    def test_sign_agreement_all_negative(self):
        """All Nov mean_diffs are negative → sign_agreement_frac = 0.0."""
        records = _make_stat_records(n_may=0)
        agg = aggregate_campaign_stats(records, group_by="season")
        row = agg[agg["season"] == "November"]
        assert float(row["sign_agreement_frac"].values[0]) == pytest.approx(0.0)

    def test_empty_records_returns_empty_df(self):
        agg = aggregate_campaign_stats([])
        assert isinstance(agg, pd.DataFrame)
        assert agg.empty

    def test_invalid_group_by_raises(self):
        records = _make_stat_records()
        with pytest.raises(ValueError, match="invalid group_by"):
            aggregate_campaign_stats(records, group_by="bogus_column")

    def test_missing_required_column_raises(self):
        bad_records = [{"start_date": "1980-05-01-00000", "component": "ocn"}]
        with pytest.raises(ValueError, match="missing columns"):
            aggregate_campaign_stats(bad_records)

    def test_n_starts_correct(self):
        records = _make_stat_records(n_may=4, n_nov=3)
        agg = aggregate_campaign_stats(records, group_by="season")
        may_row = agg[agg["season"] == "May"]
        nov_row = agg[agg["season"] == "November"]
        assert int(may_row["n_starts"].values[0]) == 4
        assert int(nov_row["n_starts"].values[0]) == 3


# ===========================================================================
# 6. TestConsistencyChecks
# ===========================================================================


class TestConsistencyChecks:
    def _make_ts_arrays(self, n=10, offset=0.0):
        lat = np.linspace(-90, 90, n)
        lon = np.linspace(-180, 180, n)
        base = np.full((n, n), 295.0)
        atm = xr.DataArray(base + offset, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        lnd = xr.DataArray(base, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        return atm, lnd

    def test_identical_ts_no_inconsistency(self):
        atm, lnd = self._make_ts_arrays(offset=0.0)
        report = check_atm_surface_vs_land(atm, lnd, threshold_k=2.0)
        assert report.frac_inconsistent == pytest.approx(0.0)
        assert report.n_points > 0

    def test_large_offset_triggers_warning(self):
        atm, lnd = self._make_ts_arrays(offset=10.0)
        report = check_atm_surface_vs_land(atm, lnd, threshold_k=2.0)
        assert report.frac_inconsistent > 0.0
        assert len(report.warnings) > 0

    def test_no_overlap_warns(self):
        """Arrays on disjoint grids produce a no-overlap warning."""
        atm = xr.DataArray(
            np.ones((5, 5)), dims=["lat", "lon"],
            coords={"lat": [10, 20, 30, 40, 50], "lon": [0, 1, 2, 3, 4]},
        )
        lnd = xr.DataArray(
            np.ones((5, 5)), dims=["lat", "lon"],
            coords={"lat": [-50, -40, -30, -20, -10], "lon": [10, 11, 12, 13, 14]},
        )
        report = check_atm_surface_vs_land(atm, lnd)
        assert any("overlap" in w.lower() or "point" in w.lower() for w in report.warnings)

    def test_ocean_ice_no_inconsistency_warm_sst(self):
        """Warm SST everywhere with no ice → no below-freezing cells → 0 inconsistency."""
        sst = xr.DataArray(np.full(100, 285.0), dims=["x"])   # warm
        aice = xr.DataArray(np.zeros(100), dims=["x"])          # no ice
        report = check_ocean_ice_consistency(sst, aice)
        assert report.frac_inconsistent == pytest.approx(0.0)

    def test_ocean_ice_inconsistency_detected(self):
        """Below-freezing SST but no ice → inconsistency flagged."""
        sst = xr.DataArray(np.full(100, 265.0), dims=["x"])   # below freezing
        aice = xr.DataArray(np.zeros(100), dims=["x"])          # no ice
        report = check_ocean_ice_consistency(sst, aice)
        assert report.frac_inconsistent > 0.0
        assert len(report.warnings) > 0

    def test_ocean_ice_consistent_frozen(self):
        """Below-freezing SST WITH ice → consistent."""
        sst = xr.DataArray(np.full(100, 265.0), dims=["x"])
        aice = xr.DataArray(np.full(100, 0.9), dims=["x"])      # lots of ice
        report = check_ocean_ice_consistency(sst, aice, threshold_frac=0.01)
        assert report.frac_inconsistent == pytest.approx(0.0)

    def test_consistency_report_to_dict(self):
        rpt = ConsistencyReport(
            check_name="test",
            n_points=100,
            mean_abs_mismatch=0.5,
            frac_inconsistent=0.1,
        )
        d = rpt.to_dict()
        assert d["check_name"] == "test"
        assert d["n_points"] == 100
        assert d["frac_inconsistent"] == pytest.approx(0.1)


# ===========================================================================
# 7. TestICConfig
# ===========================================================================


class TestICConfig:
    def _valid_pair(self):
        return ExperimentPair(
            ref_label="BruteForce",
            test_label="JRA55-FOSIRL",
            ref_root="/fake/ref",
            test_root="/fake/test",
        )

    def _valid_components(self):
        return [ComponentSpec(name="lnd", file_glob="*.elm.r.*.nc")]

    def test_valid_construction(self):
        cfg = ICConfig(
            experiment_pair=self._valid_pair(),
            start_dates=["1980-05-01-00000", "1980-11-01-00000"],
            seasons=["May", "November"],
            members=["EN00", "EN01"],
            components=self._valid_components(),
            pilot_only=True,
            pilot_date="1980-05-01-00000",
        )
        assert cfg.pilot_only is True
        assert cfg.active_dates == ["1980-05-01-00000"]

    def test_full_campaign_active_dates(self):
        cfg = ICConfig(
            experiment_pair=self._valid_pair(),
            start_dates=["1980-05-01-00000", "1980-11-01-00000"],
            seasons=["May", "November"],
            members=["EN00"],
            components=self._valid_components(),
            pilot_only=False,
        )
        assert set(cfg.active_dates) == {"1980-05-01-00000", "1980-11-01-00000"}

    def test_invalid_date_format_raises(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD-00000"):
            ICConfig(
                experiment_pair=self._valid_pair(),
                start_dates=["1980-05-01"],   # missing time
                seasons=["May"],
                members=["EN00"],
                components=self._valid_components(),
            )

    def test_empty_members_raises(self):
        with pytest.raises(ValueError, match="members"):
            ICConfig(
                experiment_pair=self._valid_pair(),
                start_dates=["1980-05-01-00000"],
                seasons=["May"],
                members=[],
                components=self._valid_components(),
            )

    def test_empty_start_dates_raises(self):
        with pytest.raises(ValueError, match="start_dates"):
            ICConfig(
                experiment_pair=self._valid_pair(),
                start_dates=[],
                seasons=["May"],
                members=["EN00"],
                components=self._valid_components(),
            )

    def test_date_season_may(self):
        cfg = ICConfig(
            experiment_pair=self._valid_pair(),
            start_dates=["1980-05-01-00000"],
            seasons=["May"],
            members=["EN00"],
            components=self._valid_components(),
        )
        assert cfg.date_season("1980-05-01-00000") == "May"
        assert cfg.date_season("1980-11-01-00000") == "November"
        assert cfg.date_season("1980-07-01-00000") == "Month07"

    def test_experiment_pair_empty_label_raises(self):
        with pytest.raises(ValueError):
            ExperimentPair(
                ref_label="",
                test_label="test",
                ref_root="/r",
                test_root="/t",
            )

    def test_component_spec_invalid_name_raises(self):
        with pytest.raises(ValueError, match="not one of"):
            ComponentSpec(name="bogus", file_glob="*.nc")

    def test_default_components_valid(self):
        for cs in DEFAULT_COMPONENTS:
            assert cs.name in ("atm", "lnd", "ocn", "ice", "rof", "cpl")
            assert len(cs.file_glob) > 0
