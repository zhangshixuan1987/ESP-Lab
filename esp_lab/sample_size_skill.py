"""Hindcast skill as a function of sample size.

Three sampling questions share one vectorized scorer:

* start years: score random subsets of N initialization years;
* ensemble members: score ensemble means of random subsets of M members;
* lead-time climatology: build the drift (and paired observed) climatology
  from K start years and score the resulting anomalies.

Scores match :func:`esp_lab.stats.compute_skill_seasonal` for one lead:
ACC is the Pearson correlation of the ensemble-mean and observed anomalies
over the selected years, nRMSE is their RMSE divided by the observed standard
deviation (population form), and ``detrend`` removes a linear trend in
initialization year from both before scoring. Many subsets are scored at once
from weighted moments, so thousands of resamples stay cheap.
"""

from __future__ import annotations

from math import comb
from typing import Callable, Optional, Sequence

import numpy as np
import xarray as xr
from scipy import stats as sps

Reducer = Optional[Callable[[xr.DataArray], xr.DataArray]]
_VAR_RTOL = 1e-10   # variance below this fraction of the second moment counts as constant


def init_years(values: Sequence) -> np.ndarray:
    """Initialization years from Y labels such as 1980, '1980' or '1980050100'."""
    return np.asarray([int(str(value)[:4]) for value in np.asarray(values)], dtype=int)


def observations_at_valid_time(obs: xr.DataArray, valid_time: xr.DataArray) -> xr.DataArray:
    """Observed value at each hindcast verification season.

    ``obs`` is (time, ...) and ``valid_time`` is (Y, L); the result is
    (Y, L, ...) with NaN where the observations do not cover a season.
    """
    obs = obs.load()
    keys = list(zip(obs.time.dt.year.values.tolist(), obs.time.dt.month.values.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("observations contain duplicate year/month timestamps")
    lookup = {key: index for index, key in enumerate(keys)}
    years = valid_time.dt.year.values
    months = valid_time.dt.month.values
    index = np.vectorize(lambda y, m: lookup.get((int(y), int(m)), -1))(years, months)
    data = np.take(obs.values, np.where(index < 0, 0, index), axis=0).astype("float64")
    data[index < 0] = np.nan
    space = obs.dims[1:]
    return xr.DataArray(
        data, dims=("Y", "L", *space),
        coords={"Y": valid_time.Y, "L": valid_time.L, **{dim: obs[dim] for dim in space}},
    )


def random_subsets(n_total: int, size: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """0/1 weights (draw, n_total) selecting ``size`` items per draw.

    All combinations are used when there are no more than ``n_draws`` of them
    (e.g. size == n_total gives one draw); otherwise ``n_draws`` random subsets.
    """
    if not 1 <= size <= n_total:
        raise ValueError(f"subset size {size} must lie in 1..{n_total}")
    if comb(n_total, size) <= n_draws:
        from itertools import combinations

        picks = np.asarray(list(combinations(range(n_total), size)))
    else:
        picks = np.argsort(rng.random((n_draws, n_total)), axis=1)[:, :size]
    weights = np.zeros((len(picks), n_total))
    np.put_along_axis(weights, picks, 1.0, axis=1)
    return weights


def _wsum(w, x):
    """Weighted sum over years: w (D, Y) with x (1 or D, Y, *S) -> (D, *S)."""
    if x.shape[0] == 1:
        return np.tensordot(w, x[0], axes=(1, 0))   # one BLAS product for every draw
    return np.einsum("dy,dy...->d...", w, x)


def _scores(f, o, t, w, detrend):
    """ACC, nRMSE, RMSE, bias and sample count from weighted moments.

    f, o: (1 or D, Y, *S); t: (Y,); w: (D, Y). Returns arrays (D, *S).
    """
    extra = f.ndim - 2
    valid = (np.isfinite(f) & np.isfinite(o)).astype("float64")
    f0 = np.where(valid > 0, f, 0.0)
    o0 = np.where(valid > 0, o, 0.0)
    tt = np.asarray(t, dtype="float64").reshape((1, -1) + (1,) * extra)
    n = _wsum(w, valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        def mean(x):
            return _wsum(w, x) / n

        mf, mo = mean(f0), mean(o0)
        cff = mean(f0 * f0) - mf ** 2
        coo = mean(o0 * o0) - mo ** 2
        cfo = mean(f0 * o0) - mf * mo
        if detrend:
            tv = tt * valid   # year only where the pair is valid
            mt = mean(tv)
            ctt = mean(tv * tt) - mt ** 2
            cft = mean(f0 * tt) - mf * mt
            cot = mean(o0 * tt) - mo * mt
            cff = cff - cft ** 2 / ctt
            coo = coo - cot ** 2 / ctt
            cfo = cfo - cft * cot / ctt
            bias = np.zeros_like(mf)
        else:
            bias = mf - mo
        cff = np.maximum(cff, 0.0)
        coo = np.maximum(coo, 0.0)
        mse = np.maximum(cff + coo - 2.0 * cfo, 0.0) + bias ** 2
        rmse = np.sqrt(mse)
        # A (near-)constant series, e.g. a snow-free cell where H2OSNO is always 0,
        # has no anomalies: ACC and nRMSE are undefined there, not infinite.
        varies_o = coo > _VAR_RTOL * mean(o0 * o0)
        varies_f = cff > _VAR_RTOL * mean(f0 * f0)
        acc = np.where(varies_o & varies_f, cfo / np.sqrt(cff * coo), np.nan)
        nrmse = np.where(varies_o, rmse / np.sqrt(coo), np.nan)
    enough = n >= 3
    return {name: np.where(enough, value, np.nan)
            for name, value in (("acc", acc), ("nrmse", nrmse), ("rmse", rmse), ("bias", bias))} | {"n": n}


def _batched(n_draws, batch):
    for start in range(0, n_draws, batch):
        yield slice(start, min(start + batch, n_draws))


def _wrap(arrays, template, reduce, extra_coords=None):
    dims = ("draw", *template.dims[1:])
    coords = {dim: template[dim] for dim in template.dims[1:] if dim in template.coords}
    ds = xr.Dataset({name: xr.DataArray(value, dims=dims, coords=coords) for name, value in arrays.items()})
    if extra_coords:
        ds = ds.assign_coords(extra_coords)
    return ds.map(reduce) if reduce is not None else ds


def year_subset_scores(
    fcst_mean: xr.DataArray, obs: xr.DataArray, weights: np.ndarray, *,
    detrend: bool, reduce: Reducer = None, batch: int = 256,
) -> xr.Dataset:
    """Scores of ``fcst_mean`` (Y, ...) against ``obs`` (Y, ...) on year subsets.

    ``weights`` is (draw, Y), e.g. from :func:`random_subsets`. ``reduce`` is
    applied to each (draw, ...) batch, e.g. an area mean, to bound memory.
    """
    fcst_mean, obs = xr.align(fcst_mean, obs, join="exact")
    t = init_years(fcst_mean.Y.values)
    f = fcst_mean.values[None].astype("float64")
    o = obs.values[None].astype("float64")
    parts = []
    for chunk in _batched(len(weights), batch):
        parts.append(_wrap(_scores(f, o, t, weights[chunk], detrend), fcst_mean, reduce))
    return xr.concat(parts, "draw")


def member_subset_scores(
    fcst_members: xr.DataArray, obs: xr.DataArray, member_weights: np.ndarray, *,
    detrend: bool, reduce: Reducer = None, batch: int = 16,
) -> xr.Dataset:
    """Scores over all years of ensemble means of member subsets.

    ``fcst_members`` is (Y, M, ...); ``member_weights`` is (draw, M) with 0/1 entries.
    """
    fcst_members = fcst_members.transpose("Y", "M", ...)
    template = fcst_members.isel(M=0, drop=True)
    template, obs = xr.align(template, obs, join="exact")
    t = init_years(template.Y.values)
    members = fcst_members.values.astype("float64")
    o = obs.values[None].astype("float64")
    all_years = np.ones((1, members.shape[0]))
    parts = []
    for chunk in _batched(len(member_weights), batch):
        mw = member_weights[chunk]
        mw = mw / mw.sum(axis=1, keepdims=True)
        f = np.einsum("dm,ym...->dy...", mw, members)
        scores = _scores(f, o, t, np.repeat(all_years, len(mw), axis=0), detrend)
        parts.append(_wrap(scores, template, reduce))
    return xr.concat(parts, "draw")


def cross_validated_subsets(n_years: int, size: int, n_draws: int,
                            rng: np.random.Generator) -> np.ndarray:
    """0/1 weights (draw, target year, year): for every target year, ``size``
    climatology years drawn from the other ``n_years - 1`` years."""
    if not 1 <= size <= n_years - 1:
        raise ValueError(f"cross-validated climatology size {size} must lie in 1..{n_years - 1}")
    if comb(n_years - 1, size) == 1:
        n_draws = 1   # leave-one-out: every draw would be identical
    noise = rng.random((n_draws, n_years, n_years))
    diagonal = np.arange(n_years)
    noise[:, diagonal, diagonal] = np.inf   # never use the target year itself
    picks = np.argsort(noise, axis=2)[..., :size]
    weights = np.zeros((n_draws, n_years, n_years))
    np.put_along_axis(weights, picks, 1.0, axis=2)
    return weights


def contiguous_subsets(n_years: int, size: int, n_draws: int,
                       rng: np.random.Generator) -> np.ndarray:
    """0/1 weights (draw, target year, year): for every target year, a block of
    ``size + 1`` consecutive years that contains it, minus the target itself.

    This is how a climatology period is used in practice: consecutive years
    around (or, at the record edges, before or after) the evaluated year, never
    the year itself. The block position is drawn at random per draw and target;
    ``size = n_years - 1`` is leave-one-out on the full record.
    """
    if not 1 <= size <= n_years - 1:
        raise ValueError(f"contiguous climatology size {size} must lie in 1..{n_years - 1}")
    if size == n_years - 1:
        n_draws = 1   # the only block is the whole record
    target = np.arange(n_years)
    lo = np.maximum(0, target - size)                 # earliest block start containing the target
    hi = np.minimum(target, n_years - size - 1)       # latest block start inside the record
    start = lo[None] + np.floor(rng.random((n_draws, n_years)) * (hi - lo + 1)[None]).astype(int)
    year = np.arange(n_years)
    inside = (year[None, None] >= start[..., None]) & (year[None, None] <= start[..., None] + size)
    weights = inside.astype("float64")
    weights[:, target, target] = 0.0
    return weights


def climatology_subset_scores(
    fcst_members: xr.DataArray, obs: xr.DataArray, weights: np.ndarray, *,
    reduce: Reducer = None, batch: int = 8,
) -> xr.Dataset:
    """Scores when the lead-time climatology comes from a subset of start years.

    ``weights`` (draw, Y) builds one climatology from the selected years and
    removes it from every year (in-sample, the workflow convention); weights
    (draw, target Y, Y), e.g. from :func:`cross_validated_subsets`, build a
    separate climatology for each target year. The model drift climatology
    (selected years, all members) and the paired observed climatology (same
    verification seasons) are removed, and every year is scored without
    detrending, since a linear detrend would remove the climatology again.
    Use ``rmse`` to compare cross-validated climatologies: their noise enters
    the forecast and observed anomalies alike, so it cancels in ``nrmse``.
    ``clim_error`` is the model climatology minus its all-year value in units
    of the interannual standard deviation of the ensemble mean (RMS over
    target years).
    """
    fcst_members = fcst_members.transpose("Y", "M", ...)
    fmean_da = fcst_members.mean("M")
    fmean_da, obs = xr.align(fmean_da, obs, join="exact")
    t = init_years(fmean_da.Y.values)
    fmean = fmean_da.values.astype("float64")
    o = obs.values.astype("float64")
    n_years, space = fmean.shape[0], fmean.shape[1:]
    valid = np.isfinite(fmean) & np.isfinite(o)
    f2 = np.where(valid, fmean, 0.0).reshape(n_years, -1)
    o2 = np.where(valid, o, 0.0).reshape(n_years, -1)
    v2 = valid.reshape(n_years, -1).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        full_clim = f2.sum(0) / v2.sum(0)
        spread = np.nanstd(np.where(v2 > 0, f2, np.nan), axis=0)
    parts = []
    for chunk in _batched(len(weights), batch):
        w = weights[chunk]
        if w.ndim == 2:
            w = np.repeat(w[:, None, :], n_years, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            den = np.einsum("dyj,js->dys", w, v2)
            clim_f = np.einsum("dyj,js->dys", w, f2) / den
            clim_o = np.einsum("dyj,js->dys", w, o2) / den
            error = (clim_f - full_clim) / spread
            clim_error = np.sqrt(np.nanmean(error ** 2, axis=1))
        fa = (np.where(v2 > 0, f2, np.nan)[None] - clim_f).reshape((len(w), n_years, *space))
        oa = (np.where(v2 > 0, o2, np.nan)[None] - clim_o).reshape((len(w), n_years, *space))
        scores = _scores(fa, oa, t, np.ones((len(w), n_years)), detrend=False)
        scores["clim_error"] = clim_error.reshape((len(w), *space))
        parts.append(_wrap(scores, fmean_da, reduce))
    return xr.concat(parts, "draw")


def fit_member_curve(members: Sequence[int], acc: Sequence[float]) -> tuple[float, float]:
    """Fit ACC(M) = acc_inf / sqrt(1 + ratio / M) to an ACC-vs-ensemble-size curve.

    ``ratio`` is the noise-to-signal variance ratio. The model is linear in 1/M
    for 1/ACC**2, so it is fit by least squares on the points with ACC > 0.
    A correlation cannot exceed 1: when the free fit gives acc_inf > 1, acc_inf
    is capped at 1 and ``ratio`` refit with the intercept fixed at 1.
    Returns (nan, nan) when the curve gives no usable fit.
    """
    m = np.asarray(members, dtype=float)
    r = np.asarray(acc, dtype=float)
    keep = np.isfinite(r) & (r > 0)
    if keep.sum() < 2:
        return float("nan"), float("nan")
    x, y = 1.0 / m[keep], 1.0 / r[keep] ** 2
    slope, intercept = np.polyfit(x, y, 1)
    if intercept <= 0:
        return float("nan"), float("nan")
    if intercept < 1:
        return 1.0, float(max(np.sum(x * (y - 1.0)) / np.sum(x * x), 0.0))
    return float(1.0 / np.sqrt(intercept)), float(slope / intercept)


def member_curve(acc_inf: float, ratio: float, members) -> np.ndarray:
    """ACC for ensemble sizes ``members`` (np.inf allowed) from :func:`fit_member_curve`."""
    m = np.asarray(members, dtype=float)
    return acc_inf / np.sqrt(1.0 + ratio / m)


def fit_member_nrmse_curve(members: Sequence[int], nrmse: Sequence[float]) -> tuple[float, float]:
    """Fit nRMSE(M)**2 = nrmse_inf**2 + noise / M to an nRMSE-vs-ensemble-size curve.

    For an M-member mean the forecast variance is signal + noise / M while its
    covariance with the observations does not depend on M, so the squared nRMSE
    is linear in 1/M; ``noise`` is the member-noise variance over the observed
    variance. Least squares on the finite points; a negative intercept is set
    to 0 and ``noise`` refit through the origin. Returns (nan, nan) without a
    usable curve.
    """
    m = np.asarray(members, dtype=float)
    e = np.asarray(nrmse, dtype=float)
    keep = np.isfinite(e)
    if keep.sum() < 2:
        return float("nan"), float("nan")
    x, y = 1.0 / m[keep], e[keep] ** 2
    slope, intercept = np.polyfit(x, y, 1)
    if intercept < 0:
        return 0.0, float(np.sum(x * y) / np.sum(x * x))
    return float(np.sqrt(intercept)), float(slope)


def member_nrmse_curve(nrmse_inf: float, noise: float, members) -> np.ndarray:
    """nRMSE for ensemble sizes ``members`` (np.inf allowed) from :func:`fit_member_nrmse_curve`."""
    m = np.asarray(members, dtype=float)
    return np.sqrt(np.maximum(nrmse_inf ** 2 + noise / m, 0.0))


def acc_critical_value(n_years, alpha: float = 0.1, *, detrend: bool = True,
                       two_sided: bool = True) -> np.ndarray:
    """Smallest ACC significant at ``alpha`` for ``n_years`` independent samples.

    Student t with n - 2 degrees of freedom (one fewer after detrending).
    """
    n = np.asarray(n_years, dtype=float)
    dof = n - 2 - (1 if detrend else 0)
    tcrit = sps.t.ppf(1 - alpha / (2 if two_sided else 1), dof)
    return tcrit / np.sqrt(dof + tcrit ** 2)


def smallest_stable_size(sizes: Sequence[int], center, upper, reference: float,
                         tolerance: float, lower=None) -> Optional[int]:
    """Smallest size from which ``center`` and ``upper`` stay within tolerance of reference.

    ``center``/``upper`` are per-size values (e.g. median and 95th percentile
    nRMSE); a size qualifies when it and every larger size satisfy
    center <= reference * (1 + tolerance) and upper <= reference * (1 + 2 * tolerance).
    With ``lower`` (e.g. the 5th percentile) the test is two-sided: also
    center >= reference * (1 - tolerance) and lower >= reference * (1 - 2 * tolerance),
    so a score that moves *below* the reference (a changed target, not a better
    estimate) does not count as stable.
    """
    lower = [None] * len(sizes) if lower is None else list(lower)
    ok = [
        (c <= reference * (1 + tolerance)) and (u <= reference * (1 + 2 * tolerance))
        and (lo is None or (c >= reference * (1 - tolerance) and lo >= reference * (1 - 2 * tolerance)))
        for c, u, lo in zip(center, upper, lower)
    ]
    for i, size in enumerate(sizes):
        if all(ok[i:]):
            return int(size)
    return None
