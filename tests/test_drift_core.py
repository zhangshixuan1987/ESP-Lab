"""
tests/test_drift_core.py
========================
Offline smoke test for the drift-analysis statistical core.

No NERSC access, no real data files, no ESP-Lab I/O.  Two synthetic
hindcast experiments are created with a known imposed drift difference,
and the core functions are exercised end-to-end.

Test classes
------------
1. TestValidYearMonth    — exact calendar rollover, no nearest-neighbour
2. TestAdjustment        — D(τ₀) = 0 at baseline, grows elsewhere, bad arg raises
3. TestBootstrapPairedCI — shape, CI direction matches imposed drift, ≥2-year guard
4. TestWindowSummary     — row count, required columns, n_leads, adj_slope sign
5. TestBuildObsLookup    — raises KeyError on incomplete coverage; passes when complete
6. TestSkillAndPairedDiff — skill keys/dims, paired_difference dims and zero at baseline
7. TestCaseArray         — valid construction, wrong dim order, duplicate Y, non-monotone L,
                           all-NaN data all raise appropriate errors
8. TestCacheAndPipelineContracts — canonical paths and units, partial-year warnings,
                                   and common-sample alignment

Run
---
    cd /global/homes/z/zhan391/code/ESP-Lab
    conda run -n e3sm_analysis python3 -m pytest tests/test_drift_core.py -v --override-ini="addopts="
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from esp_lab.diagnostics.drift import (
    CaseArray,
    DriftConfig,
    ExperimentSpec,
    VariableSpec,
    absolute_bias,
    adjustment,
    bootstrap_paired_ci,
    build_obs_lookup,
    cache_to_case_array,
    compute_skill,
    convert_pa_to_hpa,
    convert_precip_mps_to_mmday,
    lead_to_cal_month,
    no_conversion,
    open_regional_cache,
    paired_difference,
    regional_cache_path,
    run_pipeline,
    valid_year_month,
    window_summary,
)


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _make_obs_series(
    init_years,
    init_month: int,
    leads,
    base_value: float = 27.0,
    seed: int = 0,
) -> xr.DataArray:
    """Build a monthly time-series that covers all required valid months."""
    rng = np.random.default_rng(seed)
    # Span slightly more than the required period
    start_year = min(init_years)
    end_year = max(init_years) + max(leads) // 12 + 2

    times = []
    values = []
    for yr in range(start_year, end_year + 1):
        for mo in range(1, 13):
            # Use cftime-compatible objects via pandas Timestamp fallback
            times.append(
                xr.coding.cftimeindex.CFTimeIndex(
                    [_make_cftime(yr, mo)]
                )[0]
            )
            values.append(base_value + rng.normal(0, 0.5))

    # Build as pandas-backed DatetimeIndex for simplicity in test
    import pandas as pd

    idx = pd.date_range(
        start=f"{start_year}-01-15",
        periods=(end_year - start_year + 1) * 12,
        freq="MS",
    ) + pd.Timedelta(days=14)
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("time",),
        coords={"time": idx},
        name="obs_regional_index",
    )


def _make_cftime(year, month):
    import cftime
    return cftime.DatetimeNoLeap(year, month, 15)


def _make_pandas_obs(init_years, init_month, leads, base_value=27.0, seed=0):
    """Build a pandas-datetime obs series covering all valid months."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    start_year = min(init_years)
    end_year = max(init_years) + max(leads) // 12 + 3

    n_months = (end_year - start_year + 1) * 12
    idx = pd.date_range(
        start=f"{start_year}-01-15", periods=n_months, freq="ME"
    )
    values = base_value + rng.normal(0, 0.5, size=n_months)
    return xr.DataArray(values, dims=("time",), coords={"time": idx})


def _make_case_array(
    init_years,
    members,
    leads,
    bias_offset: float = 0.0,
    drift_per_lead: float = 0.0,
    noise_std: float = 0.1,
    seed: int = 1,
    name: str = "test",
) -> CaseArray:
    """Create a synthetic (Y, M, L) DataArray with a known drift signal."""
    rng = np.random.default_rng(seed)
    n_y = len(init_years)
    n_m = len(members)
    n_l = len(leads)
    # True value = bias_offset + drift_per_lead * (lead - 1) + noise
    data = (
        bias_offset
        + drift_per_lead * (np.asarray(leads, dtype=float) - 1)[np.newaxis, np.newaxis, :]
        + rng.normal(0, noise_std, size=(n_y, n_m, n_l))
    )
    da = xr.DataArray(
        data,
        dims=("Y", "M", "L"),
        coords={
            "Y": list(init_years),
            "M": list(members),
            "L": list(leads),
        },
    )
    return CaseArray(da, name=name)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

INIT_YEARS = list(range(1980, 1991))   # 11 years
MEMBERS = [f"EN{i:02d}" for i in range(5)]
LEADS = list(range(1, 13))             # monthly, 1–12
INIT_MONTH = 11                        # November initialization


@pytest.fixture
def obs_series():
    return _make_pandas_obs(INIT_YEARS, INIT_MONTH, LEADS)


@pytest.fixture
def obs_lookup(obs_series):
    return build_obs_lookup(obs_series, INIT_YEARS, INIT_MONTH, LEADS)


@pytest.fixture
def case_ref(obs_lookup):
    """Reference experiment: small positive bias, slow drift."""
    ca = _make_case_array(
        INIT_YEARS, MEMBERS, LEADS,
        bias_offset=1.0,
        drift_per_lead=0.05,
        name="ref",
    )
    return ca


@pytest.fixture
def case_test(obs_lookup):
    """Test experiment: same bias, faster drift (+0.15/lead extra)."""
    ca = _make_case_array(
        INIT_YEARS, MEMBERS, LEADS,
        bias_offset=1.0,
        drift_per_lead=0.20,
        name="test",
    )
    return ca


# ---------------------------------------------------------------------------
# Test 1 — Calendar rollover
# ---------------------------------------------------------------------------

class TestValidYearMonth:
    def test_no_rollover(self):
        assert valid_year_month(1980, 5, 1) == (1980, 5)

    def test_rollover_nov_lead3(self):
        """Nov init + lead 3  →  Jan of the following year."""
        yr, mo = valid_year_month(1980, 11, 3)
        assert yr == 1981
        assert mo == 1

    def test_rollover_dec_lead12(self):
        yr, mo = valid_year_month(1980, 12, 12)
        assert yr == 1981
        assert mo == 11

    def test_lead_to_cal_month_consistency(self):
        for init_mo in range(1, 13):
            for lead in range(1, 13):
                _, mo = valid_year_month(2000, init_mo, lead)
                assert mo == lead_to_cal_month(init_mo, lead)

    def test_24_months_covers_two_years(self):
        # May init, leads 1–24 should span May->Apr (2 years)
        yrs_months = [valid_year_month(1980, 5, L) for L in range(1, 25)]
        years_seen = {y for y, m in yrs_months}
        assert 1980 in years_seen
        assert 1982 in years_seen


# ---------------------------------------------------------------------------
# Test 2 — Adjustment is zero at baseline lead
# ---------------------------------------------------------------------------

class TestAdjustment:
    def test_zero_at_baseline(self, case_ref, obs_lookup):
        bias = absolute_bias(case_ref, obs_lookup, INIT_MONTH)
        adj = adjustment(bias, baseline_lead=1)
        assert abs(float(adj.sel(L=1))) < 1e-12

    def test_non_zero_elsewhere(self, case_ref, obs_lookup):
        """With imposed drift the adjustment should grow away from baseline."""
        bias = absolute_bias(case_ref, obs_lookup, INIT_MONTH)
        adj = adjustment(bias, baseline_lead=1)
        # At lead 6 the drift_per_lead*5 ≈ 0.25 — clearly non-zero
        assert abs(float(adj.sel(L=6))) > 0.01

    def test_invalid_baseline_raises(self, case_ref, obs_lookup):
        bias = absolute_bias(case_ref, obs_lookup, INIT_MONTH)
        with pytest.raises(ValueError, match="baseline_lead"):
            adjustment(bias, baseline_lead=99)


# ---------------------------------------------------------------------------
# Test 3 — Bootstrap CI captures imposed drift direction
# ---------------------------------------------------------------------------

class TestBootstrapPairedCI:
    def test_ci_shape(self, case_test, case_ref, obs_lookup):
        lo, hi = bootstrap_paired_ci(
            case_test, case_ref, obs_lookup, obs_lookup,
            INIT_MONTH, n_boot=200, seed=42,
        )
        assert lo.dims == ("L",)
        assert hi.dims == ("L",)
        assert len(lo) == len(case_test.leads)

    def test_ci_captures_imposed_drift(self, case_test, case_ref, obs_lookup):
        """Test experiment drifts faster → ΔD > 0 at late leads."""
        lo, hi = bootstrap_paired_ci(
            case_test, case_ref, obs_lookup, obs_lookup,
            INIT_MONTH, n_boot=500, seed=0,
        )
        # At lead 12 the imposed ΔD ≈ (0.20 - 0.05) * 11 = 1.65
        # The 95 % CI lower bound should be well above zero
        assert float(lo.sel(L=12)) > 0.0, (
            "Bootstrap CI lower bound should be positive at lead 12 "
            "given the imposed positive drift difference."
        )

    def test_requires_at_least_two_common_years(self):
        """Raises ValueError when experiments share fewer than 2 years."""
        ca1 = _make_case_array([1980], MEMBERS, LEADS, name="a")
        ca2 = _make_case_array([1981], MEMBERS, LEADS, name="b")
        obs = _make_pandas_obs(list(range(1980, 1990)), INIT_MONTH, LEADS)
        lk1 = build_obs_lookup(obs, [1980], INIT_MONTH, LEADS)
        lk2 = build_obs_lookup(obs, [1981], INIT_MONTH, LEADS)
        with pytest.raises(ValueError, match="common init years"):
            bootstrap_paired_ci(ca1, ca2, lk1, lk2, INIT_MONTH, n_boot=10)

    def test_uses_requested_baseline(self, case_test, case_ref, obs_lookup):
        lo, hi = bootstrap_paired_ci(
            case_test, case_ref, obs_lookup, obs_lookup,
            INIT_MONTH, n_boot=100, seed=0, baseline_lead=3,
        )
        assert abs(float(lo.sel(L=3))) < 1e-12
        assert abs(float(hi.sel(L=3))) < 1e-12


# ---------------------------------------------------------------------------
# Test 4 — window_summary shape and columns
# ---------------------------------------------------------------------------

WINDOW_DEFS = {
    "early_adj":    (1, 3),
    "first_season": (1, 6),
    "first_year":   (1, 12),
}


class TestWindowSummary:
    def test_expected_rows(self, case_ref, obs_lookup):
        bias = absolute_bias(case_ref, obs_lookup, INIT_MONTH)
        adj = adjustment(bias, baseline_lead=1)
        skill = compute_skill(case_ref, obs_lookup, INIT_MONTH)
        df = window_summary(
            bias, adj, skill, WINDOW_DEFS,
            baseline_lead=1,
            experiment="ref",
            init_month=INIT_MONTH,
        )
        assert len(df) == len(WINDOW_DEFS)

    def test_expected_columns(self, case_ref, obs_lookup):
        bias = absolute_bias(case_ref, obs_lookup, INIT_MONTH)
        adj = adjustment(bias, baseline_lead=1)
        skill = compute_skill(case_ref, obs_lookup, INIT_MONTH)
        df = window_summary(
            bias, adj, skill, WINDOW_DEFS,
            baseline_lead=1,
            experiment="ref",
            init_month=INIT_MONTH,
        )
        required = {
            "window", "lead_first", "lead_last", "n_leads",
            "mean_bias", "mean_abs_adj", "adj_slope",
            "mean_rmse", "mean_acc", "mean_spread",
            "experiment", "init_month", "baseline_lead",
        }
        assert required <= set(df.columns), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_early_adj_window_n_leads(self, case_ref, obs_lookup):
        bias = absolute_bias(case_ref, obs_lookup, INIT_MONTH)
        adj = adjustment(bias, baseline_lead=1)
        skill = compute_skill(case_ref, obs_lookup, INIT_MONTH)
        df = window_summary(
            bias, adj, skill, WINDOW_DEFS,
            baseline_lead=1,
            experiment="ref",
            init_month=INIT_MONTH,
        )
        row = df[df["window"] == "early_adj"].iloc[0]
        assert row["n_leads"] == 3   # leads 1, 2, 3

    def test_adj_slope_sign(self, case_ref, obs_lookup):
        """With positive drift_per_lead, first-year slope should be positive."""
        bias = absolute_bias(case_ref, obs_lookup, INIT_MONTH)
        adj = adjustment(bias, baseline_lead=1)
        skill = compute_skill(case_ref, obs_lookup, INIT_MONTH)
        df = window_summary(
            bias, adj, skill, WINDOW_DEFS,
            baseline_lead=1,
            experiment="ref",
            init_month=INIT_MONTH,
        )
        row = df[df["window"] == "first_year"].iloc[0]
        assert row["adj_slope"] > 0, "Positive drift should give positive adj_slope."


# ---------------------------------------------------------------------------
# Test 5 — Missing obs coverage raises informative error
# ---------------------------------------------------------------------------

class TestBuildObsLookup:
    def test_missing_coverage_raises(self):
        """build_obs_lookup must raise KeyError when obs months are missing."""
        import pandas as pd
        # obs series that only covers 1980 — missing all years after
        obs = _make_pandas_obs([1980], INIT_MONTH, LEADS)
        with pytest.raises(KeyError, match="Observation coverage is incomplete"):
            build_obs_lookup(obs, init_years=[1980, 1985], init_month=INIT_MONTH, leads=LEADS)

    def test_complete_coverage_ok(self, obs_series):
        lk = build_obs_lookup(obs_series, INIT_YEARS, INIT_MONTH, LEADS)
        assert len(lk) >= len(INIT_YEARS) * len(LEADS)

    def test_keys_are_year_month_tuples(self, obs_lookup):
        for key in obs_lookup:
            assert isinstance(key, tuple) and len(key) == 2
            y, m = key
            assert 1 <= m <= 12


# ---------------------------------------------------------------------------
# Test 6 — Compute skill and paired_difference compatibility
# ---------------------------------------------------------------------------

class TestSkillAndPairedDiff:
    def test_skill_keys(self, case_ref, obs_lookup):
        skill = compute_skill(case_ref, obs_lookup, INIT_MONTH)
        assert set(skill.keys()) == {"rmse", "acc", "spread", "spread_rmse_ratio"}

    def test_skill_dims(self, case_ref, obs_lookup):
        skill = compute_skill(case_ref, obs_lookup, INIT_MONTH)
        for k, v in skill.items():
            assert v.dims == ("L",), f"{k} should have dim (L,)"
            assert v.sizes["L"] == len(LEADS)

    def test_paired_difference_dims(self, case_test, case_ref, obs_lookup):
        b_ref = absolute_bias(case_ref, obs_lookup, INIT_MONTH)
        b_test = absolute_bias(case_test, obs_lookup, INIT_MONTH)
        adj_ref = adjustment(b_ref, baseline_lead=1)
        adj_test = adjustment(b_test, baseline_lead=1)
        pd_arr = paired_difference(adj_test, adj_ref)
        assert pd_arr.dims == ("L",)
        assert pd_arr.name == "paired_diff"

    def test_paired_difference_at_baseline_near_zero(self, case_test, case_ref, obs_lookup):
        """Both adjustments are zero at baseline → ΔD(τ₀) = 0."""
        b_ref = absolute_bias(case_ref, obs_lookup, INIT_MONTH)
        b_test = absolute_bias(case_test, obs_lookup, INIT_MONTH)
        adj_ref = adjustment(b_ref, baseline_lead=1)
        adj_test = adjustment(b_test, baseline_lead=1)
        pd_arr = paired_difference(adj_test, adj_ref)
        assert abs(float(pd_arr.sel(L=1))) < 1e-12


# ---------------------------------------------------------------------------
# Test 7 — CaseArray validation
# ---------------------------------------------------------------------------

class TestCaseArray:
    def _simple(self, dims=("Y", "M", "L"), yvals=None, mvals=None, lvals=None):
        y = yvals or [1980, 1981, 1982]
        m = mvals or ["E0", "E1"]
        l = lvals or [1, 2, 3]
        data = np.ones((len(y), len(m), len(l)))
        return xr.DataArray(
            data, dims=dims,
            coords={"Y": y, "M": m, "L": l},
        )

    def test_valid(self):
        ca = CaseArray(self._simple())
        assert ca.n_years == 3

    def test_wrong_dim_order_raises(self):
        da = self._simple(dims=("L", "M", "Y")).transpose("L", "M", "Y")
        with pytest.raises(ValueError, match="expected dims"):
            CaseArray(da)

    def test_duplicate_years_raises(self):
        da = self._simple(yvals=[1980, 1980, 1981])
        with pytest.raises(ValueError, match="duplicate"):
            CaseArray(da)

    def test_non_monotone_leads_raises(self):
        da = self._simple(lvals=[3, 1, 2])
        # Need to construct with correct dims after manual reorder
        data = np.ones((3, 2, 3))
        da2 = xr.DataArray(
            data, dims=("Y", "M", "L"),
            coords={"Y": [1980, 1981, 1982], "M": ["E0", "E1"], "L": [3, 1, 2]},
        )
        with pytest.raises(ValueError, match="monotonically"):
            CaseArray(da2)

    def test_all_nan_raises(self):
        data = np.full((3, 2, 3), np.nan)
        da = xr.DataArray(
            data, dims=("Y", "M", "L"),
            coords={"Y": [1980, 1981, 1982], "M": ["E0", "E1"], "L": [1, 2, 3]},
        )
        with pytest.raises(ValueError, match="no finite"):
            CaseArray(da)


# ---------------------------------------------------------------------------
# Test 8 — cache paths, units, and comparative sample alignment
# ---------------------------------------------------------------------------

class TestCacheAndPipelineContracts:
    def _config(self, years):
        return DriftConfig(
            init_years=list(years),
            leads=list(LEADS),
            members=list(MEMBERS),
            init_months=[INIT_MONTH],
            baseline_lead=1,
            n_bootstrap=50,
        )

    def test_cache_path_preserves_canonical_case(self, tmp_path):
        path = regional_cache_path(
            tmp_path, "JRA55_FOSIRL", 5, "mon", "PRECT", "Nino3.4"
        )
        assert path == (
            tmp_path / "JRA55_FOSIRL" / "leadtime_drift"
            / "atm"
            / "JRA55_FOSIRL_init05_PRECT_Nino3_4_mon.nc"
        )

    def test_precip_converter_is_idempotent(self):
        da = xr.DataArray([2.5], dims=("x",), attrs={"units": "mm/day"})
        converted = convert_precip_mps_to_mmday(da)
        assert float(converted.item()) == 2.5
        assert converted.attrs["units"] == "mm/day"

    def test_pressure_converter_is_idempotent(self):
        da = xr.DataArray([1012.0], dims=("x",), attrs={"units": "hPa"})
        converted = convert_pa_to_hpa(da)
        assert float(converted.item()) == 1012.0
        assert converted.attrs["units"] == "hPa"

    def test_open_cache_does_not_reconvert_canonical_units(self, tmp_path):
        path = tmp_path / "prect.nc"
        da = xr.DataArray(
            np.full((2, 2, 3), 1.25),
            dims=("Y", "M", "L"),
            coords={"Y": [1980, 1981], "M": ["EN00", "EN01"], "L": [1, 2, 3]},
            attrs={"units": "mm/day"},
            name="regional_index",
        )
        da.to_netcdf(path)
        spec = VariableSpec(
            native_field="PRECT", plot_name="precipitation", plot_units="mm/day",
            obs_product=None, obs_var=None,
            model_convert=convert_precip_mps_to_mmday,
            obs_convert=no_conversion,
        )
        opened = open_regional_cache(path, spec)
        assert float(opened.mean()) == 1.25
        opened.close()
        with pytest.raises(ValueError, match="metadata does not match"):
            open_regional_cache(
                path, spec, expected_attrs={"experiment": "wrong_case"}
            )

    def test_cache_warns_about_missing_requested_years(self):
        cfg = self._config([1980, 1981, 1982])
        ca = _make_case_array(
            [1980, 1981], MEMBERS, LEADS, name="partial"
        )
        with pytest.warns(RuntimeWarning, match="missing requested initialization years"):
            result = cache_to_case_array(ca.data, "partial", cfg)
        assert list(result.init_years) == [1980, 1981]

    def test_pipeline_aligns_all_metrics_to_common_years(self):
        ref_years = [1980, 1981, 1982, 1983]
        test_years = [1980, 1981, 1982, 1983, 1984, 1985, 1986]
        cfg = self._config(test_years)
        ref = _make_case_array(ref_years, MEMBERS, LEADS, name="ref")
        test = _make_case_array(test_years, MEMBERS, LEADS, name="test")
        obs = _make_pandas_obs(test_years, INIT_MONTH, LEADS)
        specs = {
            "ref": ExperimentSpec("ref", "ref_case"),
            "test": ExperimentSpec("test", "test_case"),
        }
        var_spec = VariableSpec("X", "X", "1", "obs", "x")

        result = run_pipeline(
            specs,
            var_spec,
            cfg,
            {"ref": {INIT_MONTH: ref}, "test": {INIT_MONTH: test}},
            obs,
            write_outputs=False,
        )

        assert result["sample_years"][INIT_MONTH] == ref_years
        assert all(
            table["n_years"].eq(len(ref_years)).all()
            for table in result["window_tables"]
        )
        lookup = build_obs_lookup(obs, ref_years, INIT_MONTH, LEADS)
        expected_test_bias = absolute_bias(
            CaseArray(test.data.sel(Y=ref_years), name="test"),
            lookup,
            INIT_MONTH,
        )
        xr.testing.assert_allclose(
            result["bias"]["test"][INIT_MONTH], expected_test_bias
        )
