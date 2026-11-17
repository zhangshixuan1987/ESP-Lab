import cftime
import numpy as np
import pytest
import xarray as xr

from esp_lab import sample_size_skill as sss
from esp_lab import stats


def _synthetic(n_years=12, n_members=5, seed=0):
    rng = np.random.default_rng(seed)
    years = np.arange(1980, 1980 + n_years)
    leads = [1, 2]
    months = {1: 7, 2: 10}
    signal = rng.normal(size=(n_years, len(leads), 2, 3)) + 0.05 * (years - 1980)[:, None, None, None]
    model = signal[:, :, None] + rng.normal(scale=0.8, size=(n_years, len(leads), n_members, 2, 3))
    mod = xr.DataArray(
        model, dims=("Y", "L", "M", "lat", "lon"),
        coords={"Y": [f"{y}050100" for y in years], "L": leads, "M": range(n_members),
                "lat": [-10.0, 10.0], "lon": [0.0, 120.0, 240.0]},
    )
    valid_time = xr.DataArray(
        np.asarray([[cftime.DatetimeNoLeap(y, months[l], 15) for l in leads] for y in years], dtype=object),
        dims=("Y", "L"), coords={"Y": mod.Y, "L": leads},
    )
    obs_times = [cftime.DatetimeNoLeap(y, m, 15) for y in years for m in (7, 10)]
    obs_values = np.stack([signal[i, j] + rng.normal(scale=0.5, size=(2, 3))
                           for i in range(n_years) for j in range(len(leads))])
    obs = xr.DataArray(obs_values, dims=("time", "lat", "lon"),
                       coords={"time": obs_times, "lat": mod.lat, "lon": mod.lon})
    return mod, valid_time, obs


@pytest.mark.parametrize("detrend", [True, False])
def test_full_sample_scores_match_compute_skill_seasonal(detrend):
    mod, valid_time, obs = _synthetic()
    reference = stats.compute_skill_seasonal(
        mod, valid_time, obs, nleads=2, detrend=detrend, is_anomaly=True
    ).compute()
    obs_at = sss.observations_at_valid_time(obs, valid_time)
    scores = sss.year_subset_scores(
        mod.mean("M"), obs_at, np.ones((1, mod.sizes["Y"])), detrend=detrend
    ).isel(draw=0)

    np.testing.assert_allclose(scores.acc.values, reference.corr.values, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(scores.nrmse.values, reference.rmse.values, rtol=1e-10, atol=1e-12)


def test_year_subsets_score_each_subset_like_a_direct_call():
    mod, valid_time, obs = _synthetic()
    obs_at = sss.observations_at_valid_time(obs, valid_time)
    weights = sss.random_subsets(mod.sizes["Y"], 6, 4, np.random.default_rng(1))
    scores = sss.year_subset_scores(mod.mean("M"), obs_at, weights, detrend=True, batch=3)
    for draw in range(len(weights)):
        keep = np.flatnonzero(weights[draw])
        direct = stats.compute_skill_seasonal(
            mod.isel(Y=keep), valid_time.isel(Y=keep), obs, nleads=2, detrend=True, is_anomaly=True
        ).compute()
        np.testing.assert_allclose(scores.acc.isel(draw=draw).values, direct.corr.values, atol=1e-9)
    assert int(scores.n.max()) == 6


def test_member_subsets_with_all_members_equal_full_ensemble():
    mod, valid_time, obs = _synthetic()
    obs_at = sss.observations_at_valid_time(obs, valid_time)
    full = sss.year_subset_scores(mod.mean("M"), obs_at, np.ones((1, mod.sizes["Y"])), detrend=True)
    members = sss.member_subset_scores(
        mod.transpose("Y", "M", ...), obs_at, np.ones((1, mod.sizes["M"])), detrend=True
    )
    np.testing.assert_allclose(members.acc.values, full.acc.values, rtol=1e-12)
    one_member = sss.member_subset_scores(
        mod.transpose("Y", "M", ...), obs_at, np.eye(mod.sizes["M"]), detrend=True
    )
    assert one_member.sizes["draw"] == mod.sizes["M"]


def test_climatology_from_all_years_has_no_bias_and_keeps_acc():
    mod, valid_time, obs = _synthetic()
    obs_at = sss.observations_at_valid_time(obs, valid_time)
    full = sss.climatology_subset_scores(
        mod.transpose("Y", "M", ...), obs_at, np.ones((1, mod.sizes["Y"]))
    ).isel(draw=0)
    np.testing.assert_allclose(full.bias.values, 0.0, atol=1e-12)
    np.testing.assert_allclose(full.clim_error.values, 0.0, atol=1e-12)
    raw = sss.year_subset_scores(mod.mean("M"), obs_at, np.ones((1, mod.sizes["Y"])), detrend=False)
    # A constant shift per lead and point leaves ACC unchanged.
    np.testing.assert_allclose(full.acc.values, raw.acc.isel(draw=0).values, rtol=1e-10)



def test_cross_validated_climatology_is_leave_one_out_at_n_minus_one():
    mod, valid_time, obs = _synthetic()
    obs_at = sss.observations_at_valid_time(obs, valid_time)
    n = mod.sizes["Y"]
    weights = sss.cross_validated_subsets(n, n - 1, 50, np.random.default_rng(3))
    assert weights.shape == (1, n, n) and np.all(np.diagonal(weights[0]) == 0)
    scores = sss.climatology_subset_scores(mod.transpose("Y", "M", ...), obs_at, weights).isel(draw=0)

    fmean, obsv = mod.mean("M").values, obs_at.values
    total_f, total_o = fmean.sum(0), obsv.sum(0)
    fa = fmean - (total_f - fmean) / (n - 1)       # each year minus the mean of the others
    oa = obsv - (total_o - obsv) / (n - 1)
    rmse = np.sqrt(((fa - oa) ** 2).mean(0))
    np.testing.assert_allclose(scores.nrmse.values, rmse / oa.std(0), rtol=1e-10)
    np.testing.assert_allclose(scores.rmse.values, rmse, rtol=1e-10)
    one = sss.climatology_subset_scores(
        mod.transpose("Y", "M", ...), obs_at, sss.cross_validated_subsets(n, 1, 200, np.random.default_rng(5))
    )
    # A one-year climatology inflates the cross-validated RMSE (not the nRMSE) by ~sqrt(2).
    assert float((one.rmse.median("draw") / scores.rmse).mean()) > 1.2
    assert int(scores.n.max()) == n

    few = sss.cross_validated_subsets(n, 3, 7, np.random.default_rng(4))
    assert few.shape == (7, n, n) and (few.sum(2) == 3).all()


def test_random_subsets_enumerates_small_spaces():
    rng = np.random.default_rng(0)
    assert sss.random_subsets(5, 5, 100, rng).shape == (1, 5)
    assert sss.random_subsets(5, 2, 100, rng).shape == (10, 5)
    big = sss.random_subsets(30, 10, 50, rng)
    assert big.shape == (50, 30) and (big.sum(1) == 10).all()


def test_acc_critical_value_and_member_extrapolation():
    assert sss.acc_critical_value(10, 0.1, detrend=True) == pytest.approx(0.582, abs=2e-3)
    assert sss.acc_critical_value(10, 0.1, detrend=False) == pytest.approx(0.549, abs=2e-3)
    members = [1, 2, 3, 5, 8, 10]
    curve = sss.member_curve(0.6, 4.0, members)
    acc_inf, ratio = sss.fit_member_curve(members, curve)
    assert acc_inf == pytest.approx(0.6) and ratio == pytest.approx(4.0)
    assert sss.member_curve(acc_inf, ratio, [np.inf])[0] == pytest.approx(0.6)
    assert np.isnan(sss.fit_member_curve([1, 2], [-0.1, -0.2])[0])
    # A steep curve that the free fit would extrapolate past ACC = 1 is capped at 1.
    steep = [0.3, 0.42, 0.6, 0.7]
    assert np.polyfit(1 / np.array([1, 2, 5, 10]), 1 / np.array(steep) ** 2, 1)[1] < 1
    acc_inf, ratio = sss.fit_member_curve([1, 2, 5, 10], steep)
    assert acc_inf == 1.0 and ratio > 0
    assert np.all(sss.member_curve(acc_inf, ratio, [1, 10, np.inf]) <= 1.0)


def test_smallest_stable_size():
    sizes = [3, 5, 8, 12]
    assert sss.smallest_stable_size(sizes, [1.3, 1.08, 1.03, 1.0], [1.6, 1.15, 1.08, 1.0], 1.0, 0.05) == 8
    assert sss.smallest_stable_size(sizes, [1.3, 1.3, 1.3, 1.3], [2, 2, 2, 2], 1.0, 0.05) is None


def test_missing_values_are_excluded_pairwise():
    mod, valid_time, obs = _synthetic()
    obs_at = sss.observations_at_valid_time(obs, valid_time)
    gappy = obs_at.copy()
    gappy[2, 0, 0, 0] = np.nan
    scores = sss.year_subset_scores(mod.mean("M"), gappy, np.ones((1, mod.sizes["Y"])), detrend=True)
    keep = [i for i in range(mod.sizes["Y"]) if i != 2]
    direct = sss.year_subset_scores(
        mod.mean("M").isel(Y=keep), obs_at.isel(Y=keep), np.ones((1, len(keep))), detrend=True
    )
    assert float(scores.n.isel(draw=0, L=0, lat=0, lon=0)) == len(keep)
    np.testing.assert_allclose(
        scores.acc.isel(draw=0, L=0, lat=0, lon=0), direct.acc.isel(draw=0, L=0, lat=0, lon=0), atol=1e-10
    )
