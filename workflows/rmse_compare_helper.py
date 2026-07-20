"""Reusable helpers for direct-RMSE hindcast comparisons.

These utilities are intentionally notebook-friendly: they keep common
alignment, lead handling, comparison, and bootstrap logic in one importable
place while leaving experiment-specific metadata, loops, and plotting choices
in notebooks.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr


DIRECT_RMSE_LEADS = (3, 6, 9, 12, 15, 18, 21, 24)
LEGACY_DIRECT_RMSE_LEADS = (1, 4, 7, 10, 13, 16, 19, 22)
SEASON_NAMES = ("DJF", "MAM", "JJA", "SON")
OBS_ALIGNMENT_VERSION = "exact_year_month_nan_boundary_v2"


def safe_model_name(model):
    """Return a model name suitable for cache and figure filenames."""
    return str(model).replace(" ", "_").replace("/", "_")


def compact_year_tag(year_values):
    """Return a short, deterministic tag such as ``1980-2018_ny39``."""
    years = np.unique(np.asarray(year_values, dtype=int).reshape(-1))
    if years.size == 0:
        raise ValueError("year_values must contain at least one year")
    return f"{int(years.min())}-{int(years.max())}_ny{years.size}"


def normalize_direct_rmse_leads(
    obj,
    native_leads=DIRECT_RMSE_LEADS,
    legacy_leads=LEGACY_DIRECT_RMSE_LEADS,
):
    """Migrate cached 1,4,... lead labels to native 3,6,... labels."""
    if "L" not in obj.coords:
        return obj

    leads = tuple(np.asarray(obj["L"].values, dtype=int))
    legacy_subset = tuple(legacy_leads[:len(leads)])
    if leads == legacy_subset:
        return obj.assign_coords(
            L=np.asarray(native_leads[:len(leads)], dtype=int)
        )
    return obj


def seasonal_label(init_month, lead, season_names=SEASON_NAMES):
    """Return the verification season for a native seasonal lead."""
    center_month = ((int(init_month) + int(lead) - 2) % 12) + 1
    return season_names[(center_month % 12) // 3]


def lead_label(init_month, lead, init_month_names, season_names=SEASON_NAMES):
    """Format a native seasonal lead using the original Lead-1 convention."""
    init_label = init_month_names.get(init_month, f"{init_month:02d}").title()
    season = seasonal_label(init_month, lead, season_names=season_names)
    return f"Lead-{int(lead) - 2} {season} ({init_label} init)"


def require_available_lead(obj, target_lead):
    """Validate and return an exact native lead coordinate."""
    available = np.asarray(obj["L"].values, dtype=int)
    if target_lead not in available:
        raise ValueError(
            f"Requested lead {target_lead} is unavailable; "
            f"native leads are {available.tolist()}."
        )
    return int(target_lead)


def direct_rmse_difference(left, right):
    """Return aligned left-minus-right RMSE and RMSE ratio fields."""
    left_rmse = left["rmse"] if isinstance(left, xr.Dataset) else left
    right_rmse = right["rmse"] if isinstance(right, xr.Dataset) else right
    left_rmse, right_rmse = xr.align(left_rmse, right_rmse, join="inner")

    rmse_diff = (left_rmse - right_rmse).astype("float32")
    rmse_diff.name = "rmse_diff"
    rmse_ratio = (left_rmse / right_rmse).astype("float32")
    rmse_ratio.name = "rmse_ratio"
    return xr.Dataset({"rmse_diff": rmse_diff, "rmse_ratio": rmse_ratio})


def area_weighted_mask_fraction(mask):
    """Return the cosine-latitude-weighted fraction where a mask is true."""
    weights = xr.DataArray(
        np.cos(np.deg2rad(mask["lat"])),
        dims="lat",
        coords={"lat": mask["lat"]},
    )
    valid = mask.notnull()
    numerator = xr.where(mask.fillna(False), weights, 0).sum()
    denominator = xr.where(valid, weights, 0).sum()
    return float(numerator / denominator) if float(denominator) > 0 else np.nan


def convert_kelvin_to_celsius(da):
    """Convert K to degC while preserving lazy evaluation."""
    da = da - 273.15
    da.attrs["units"] = r"$^\circ$C"
    return da


def convert_precip_mps_to_mmday(da):
    """Convert precipitation from m/s to mm/day while preserving lazy evaluation."""
    da = da * (1000.0 * 86400.0)
    da.attrs["units"] = "mm/day"
    return da


def convert_pa_to_hpa(da):
    """Convert pressure from Pa to hPa while preserving lazy evaluation."""
    da = da * 1.0e-2
    da.attrs["units"] = "hPa"
    return da


def no_unit_conversion(da, units=None):
    """Return a lazy copy and optionally standardize the unit attribute."""
    da = da * 1.0
    if units is not None:
        da.attrs["units"] = units
    return da


def normalize_y_to_year(obj, require_y=False):
    """Normalize a Y coordinate containing YYYY or YYYYMMDDHH labels to years."""
    if "Y" not in obj.coords:
        if require_y:
            raise KeyError(
                f"No Y coordinate found. dims={obj.dims}, coords={list(obj.coords)}"
            )
        return obj

    yvals = [int(y) for y in obj["Y"].values]
    years = [int(str(y)[:4]) for y in yvals]
    return obj.assign_coords(Y=("Y", years))


def drop_non_dim_coords(da, keep=("Y", "L", "M", "lat", "lon")):
    """Drop auxiliary coordinates that are not needed for direct RMSE arithmetic."""
    drop_names = [
        name for name in da.coords
        if name not in da.dims and name not in keep
    ]
    if drop_names:
        da = da.reset_coords(drop_names, drop=True)
    return da


def get_ensemble_mean(da):
    """Return ensemble mean if M exists; otherwise return unchanged."""
    if "M" in da.dims:
        return da.mean("M", skipna=True)
    return da


def _empty_obs_slice(obs_da):
    """Return one spatial observation slice filled with NaN."""
    if obs_da.sizes.get("time", 0) == 0:
        raise ValueError("Cannot construct a missing observation from an empty time series")
    return xr.full_like(obs_da.isel(time=0, drop=True), np.nan, dtype=float)


def _select_obs_year_month(obs_da, target_year, target_month, allow_missing=False):
    """Select exactly one observed season by year/month."""
    target = obs_da.sel(
        time=(
            (obs_da["time"].dt.year == int(target_year))
            & (obs_da["time"].dt.month == int(target_month))
        )
    )
    count = target.sizes["time"]
    if count == 0 and allow_missing:
        return _empty_obs_slice(obs_da)
    if count != 1:
        raise ValueError(
            f"Expected exactly one obs time for target="
            f"{int(target_year):04d}-{int(target_month):02d}; got {count}"
        )
    return target.isel(time=0, drop=True)


def _select_obs_at_time(obs_da, target_time, allow_missing=False):
    """Select an exact observed seasonal mean for a model verification time."""
    if hasattr(target_time, "year") and hasattr(target_time, "month"):
        target_year = int(target_time.year)
        target_month = int(target_time.month)
    else:
        target_month_value = np.datetime64(target_time, "M")
        if np.isnat(target_month_value):
            if allow_missing:
                return _empty_obs_slice(obs_da)
            raise ValueError("Model verification time is NaT")
        target_text = np.datetime_as_string(target_month_value, unit="M")
        target_year, target_month = (int(part) for part in target_text.split("-"))
    return _select_obs_year_month(
        obs_da,
        target_year,
        target_month,
        allow_missing=allow_missing,
    )


def make_obs_like_model_time(obs_da, model_time, common_years, obs_chunks=None):
    """Convert continuous observed seasons into Y,L observations using model time."""
    model_time = normalize_y_to_year(model_time)
    model_time = model_time.sel(Y=common_years)

    y_labels = [int(y) for y in model_time["Y"].values]
    l_labels = list(model_time["L"].values)

    obs_by_y = []
    for y in y_labels:
        obs_by_l = []
        for lead in l_labels:
            target_time = model_time.sel(Y=y, L=lead).item()
            obs_sel = _select_obs_at_time(obs_da, target_time, allow_missing=True)
            obs_by_l.append(obs_sel)

        obs_by_y.append(xr.concat(obs_by_l, dim=xr.IndexVariable("L", l_labels)))

    obs_like = xr.concat(obs_by_y, dim=xr.IndexVariable("Y", y_labels))
    if obs_chunks is not None:
        obs_like = obs_like.chunk(obs_chunks)
    return obs_like


def make_obs_like_model_leads(obs_da, template_da, init_month, years, leads):
    """Build Y,L observations, retaining unavailable boundary seasons as NaN."""
    if "time" not in obs_da.coords:
        raise KeyError(
            f"obs_da must have a time coordinate. dims={obs_da.dims}, coords={list(obs_da.coords)}"
        )

    years = [int(y) for y in years]
    leads = [int(l) for l in leads]

    pieces_by_lead = []
    for lead in leads:
        pieces_by_year = []
        for year in years:
            # Native seasonal leads are the final month number in each
            # three-month window: L=3 verifies at init_month + 2 months.
            total_month = int(init_month) + int(lead) - 1
            target_year = year + (total_month - 1) // 12
            target_month = ((total_month - 1) % 12) + 1

            target = _select_obs_year_month(
                obs_da,
                target_year,
                target_month,
                allow_missing=True,
            )
            pieces_by_year.append(target.expand_dims(Y=[year]))

        pieces_by_lead.append(xr.concat(pieces_by_year, dim="Y").expand_dims(L=[lead]))

    obs_init = xr.concat(pieces_by_lead, dim="L").transpose("Y", "L", "lat", "lon")
    template_align = template_da.isel(M=0, drop=True) if "M" in template_da.dims else template_da
    _, obs_init = xr.align(template_align, obs_init, join="inner")
    return obs_init.chunk({"Y": min(4, len(years)), "L": -1, "lat": 90, "lon": 180})


def prepare_model_for_direct_rmse(model_da, common_years):
    """Normalize Y, subset years, compute ensemble mean, and drop aux coords."""
    model_da = normalize_y_to_year(model_da)
    model_da = model_da.sel(Y=common_years)
    return drop_non_dim_coords(get_ensemble_mean(model_da))


def compute_direct_rmse(
    model_da,
    obs_da,
    model_time,
    common_years,
    field=None,
    units=None,
    lead_coord=None,
):
    """Compute direct RMSE, bias, MAE, and valid-year count over available years."""
    model_em = prepare_model_for_direct_rmse(model_da, common_years)
    obs_like = make_obs_like_model_time(
        obs_da=obs_da,
        model_time=model_time,
        common_years=common_years,
        obs_chunks={"Y": -1, "L": -1, "lat": 90, "lon": 180},
    )
    obs_like = drop_non_dim_coords(obs_like)

    model_em, obs_like = xr.align(model_em, obs_like, join="inner")

    if lead_coord is not None:
        lead_coord = list(lead_coord)
        if len(lead_coord) != model_em.sizes["L"]:
            raise ValueError(
                "lead_coord length must match the aligned L dimension: "
                f"got {len(lead_coord)} labels for {model_em.sizes['L']} leads"
            )
        model_em = model_em.assign_coords(L=("L", lead_coord))
        obs_like = obs_like.assign_coords(L=("L", lead_coord))

    diff = model_em - obs_like
    rmse = np.sqrt((diff ** 2).mean("Y", skipna=True))
    bias = diff.mean("Y", skipna=True)
    mae = np.abs(diff).mean("Y", skipna=True)
    n_years = diff.notnull().sum("Y")

    ds_out = xr.Dataset(
        {
            "rmse": rmse.astype("float32"),
            "bias": bias.astype("float32"),
            "mae": mae.astype("float32"),
            "n_years": n_years.astype("int16"),
        }
    )
    ds_out["rmse"].name = "rmse"
    ds_out["bias"].name = "bias"
    ds_out["mae"].name = "mae"
    ds_out["n_years"].name = "n_years"
    ds_out.attrs["description"] = (
        "Direct model-vs-observation RMSE for limited initialization years. "
        "Model is ensemble mean before differencing. No correlation/p-value/MSSS/RPC is computed."
    )
    if field is not None:
        ds_out.attrs["field"] = field
    if units is not None:
        ds_out.attrs["units"] = units
    return ds_out


def finite_fraction(da, name):
    """Print and return finite fraction for a DataArray."""
    frac = da.notnull().mean().compute().item()
    print(f"{name}: finite fraction = {frac:.4f}")
    print(f"{name}: dims={da.dims}, shape={da.shape}")
    if "Y" in da.coords:
        print(f"{name}: Y={list(da['Y'].values)}")
    if "L" in da.coords:
        print(f"{name}: L={list(da['L'].values)}")
    return frac


def print_direct_rmse_input_check(label, model_da, obs_da, model_time, common_years):
    """Print a diagnostic check before direct-RMSE calculation."""
    model_em = prepare_model_for_direct_rmse(model_da, common_years)
    obs_like = make_obs_like_model_time(
        obs_da=obs_da,
        model_time=model_time,
        common_years=common_years,
        obs_chunks={"Y": -1, "L": -1, "lat": 90, "lon": 180},
    )
    obs_like = drop_non_dim_coords(obs_like)

    model_em, obs_like = xr.align(model_em, obs_like, join="inner")
    diff = model_em - obs_like

    print("\n" + "=" * 80)
    print(label)
    print("common_years:", common_years)
    finite_fraction(model_em, "model ensemble mean")
    finite_fraction(obs_like, "matched obs")
    finite_fraction(diff, "model - obs")


def prepare_member_error(model_da, obs_da, common_years):
    """Return member-level error = model - obs with dimensions Y,L,M,lat,lon."""
    model_da = normalize_y_to_year(model_da, require_y=True)
    obs_da = normalize_y_to_year(obs_da, require_y=True)
    common_years = [int(y) for y in common_years]

    model_da = model_da.sel(Y=common_years)
    obs_da = obs_da.sel(Y=common_years)
    model_da, obs_da = xr.align(model_da, obs_da, join="inner")

    if "M" not in model_da.dims:
        raise ValueError(
            f"Expected model_da to have ensemble dimension M. Got dims={model_da.dims}"
        )
    return model_da - obs_da


def _nanmean_sq(arr):
    """Mean of squared errors over year/member axes, preserving lat/lon."""
    return _nanmean_preserve_missing(arr * arr, axis=(0, 1), dtype=np.float64)


def _nanmean_preserve_missing(arr, axis=None, dtype=None):
    """Compute nanmean while allowing intentionally all-missing slices."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(arr, axis=axis, dtype=dtype)


def _finite_comparison_probability(left, right, axis=0):
    """Return P(left < right) using only finite paired comparisons."""
    left, right = np.broadcast_arrays(left, right)
    valid = np.isfinite(left) & np.isfinite(right)
    valid_count = valid.sum(axis=axis)
    win_count = ((left < right) & valid).sum(axis=axis)
    probability = np.full(np.shape(valid_count), np.nan, dtype="float32")
    np.divide(
        win_count,
        valid_count,
        out=probability,
        where=valid_count > 0,
    )
    return probability


def bootstrap_rmse_diff_member_level_memorysafe(
    e3sm_err,
    smyle_err,
    nboot=100,
    seed=42,
    alpha=0.05,
):
    """Memory-safe RMSE-difference bootstrap using year/member resampling."""
    rng = np.random.default_rng(seed)
    e3sm_err, smyle_err = xr.align(
        e3sm_err, smyle_err, join="inner", exclude={"M"}
    )

    if set(e3sm_err["L"].values.tolist()) != set(smyle_err["L"].values.tolist()):
        common_leads = np.intersect1d(e3sm_err["L"].values, smyle_err["L"].values)
        e3sm_err = e3sm_err.sel(L=common_leads)
        smyle_err = smyle_err.sel(L=common_leads)

    leads = [int(l) for l in e3sm_err["L"].values]
    lat = e3sm_err["lat"].values
    lon = e3sm_err["lon"].values

    out_rmse_diff = []
    out_rmse_e3sm = []
    out_rmse_smyle = []
    out_prob = []
    out_p = []
    out_sig_better = []
    out_sig_worse = []

    for lead in leads:
        print(f"    bootstrap lead L={lead}")

        e = e3sm_err.sel(L=lead).transpose("Y", "M", "lat", "lon").astype("float32").load().values
        s = smyle_err.sel(L=lead).transpose("Y", "M", "lat", "lon").astype("float32").load().values

        n_year = e.shape[0]
        nmem_e3sm = e.shape[1]
        nmem_smyle = s.shape[1]

        rmse_e_obs = np.sqrt(_nanmean_sq(e)).astype("float32")
        rmse_s_obs = np.sqrt(_nanmean_sq(s)).astype("float32")
        rmse_diff_obs = (rmse_e_obs - rmse_s_obs).astype("float32")
        bootstrap_diffs = []

        for _ in range(nboot):
            y_index = rng.integers(0, n_year, size=n_year)
            m_index_e3sm = rng.integers(0, nmem_e3sm, size=nmem_e3sm)
            m_index_smyle = rng.integers(0, nmem_smyle, size=nmem_smyle)
            e_sample = e[np.ix_(y_index, m_index_e3sm)]
            s_sample = s[np.ix_(y_index, m_index_smyle)]
            boot_diff = np.sqrt(_nanmean_sq(e_sample)) - np.sqrt(_nanmean_sq(s_sample))
            bootstrap_diffs.append(boot_diff)

        bootstrap_diffs = np.asarray(bootstrap_diffs)
        prob = _finite_comparison_probability(
            bootstrap_diffs,
            np.zeros_like(bootstrap_diffs),
        )
        p_two = (2.0 * np.minimum(prob, 1.0 - prob)).clip(0, 1).astype("float32")
        valid_prob = np.isfinite(prob)
        sig_better = np.where(valid_prob, prob > (1.0 - alpha / 2.0), np.nan).astype("float32")
        sig_worse = np.where(valid_prob, prob < (alpha / 2.0), np.nan).astype("float32")

        coords = {"lat": lat, "lon": lon}
        out_rmse_e3sm.append(xr.DataArray(rmse_e_obs, dims=("lat", "lon"), coords=coords).expand_dims(L=[lead]))
        out_rmse_smyle.append(xr.DataArray(rmse_s_obs, dims=("lat", "lon"), coords=coords).expand_dims(L=[lead]))
        out_rmse_diff.append(xr.DataArray(rmse_diff_obs, dims=("lat", "lon"), coords=coords).expand_dims(L=[lead]))
        out_prob.append(xr.DataArray(prob, dims=("lat", "lon"), coords=coords).expand_dims(L=[lead]))
        out_p.append(xr.DataArray(p_two, dims=("lat", "lon"), coords=coords).expand_dims(L=[lead]))
        out_sig_better.append(xr.DataArray(sig_better, dims=("lat", "lon"), coords=coords).expand_dims(L=[lead]))
        out_sig_worse.append(xr.DataArray(sig_worse, dims=("lat", "lon"), coords=coords).expand_dims(L=[lead]))

        del e, s, bootstrap_diffs

    ds_out = xr.Dataset(
        {
            "rmse_diff": xr.concat(out_rmse_diff, dim="L").astype("float32"),
            "rmse_e3sm": xr.concat(out_rmse_e3sm, dim="L").astype("float32"),
            "rmse_smyle": xr.concat(out_rmse_smyle, dim="L").astype("float32"),
            "prob_e3sm_lower_rmse": xr.concat(out_prob, dim="L").astype("float32"),
            "p_two_sided_bootstrap": xr.concat(out_p, dim="L").astype("float32"),
            "significant_e3sm_better": xr.concat(out_sig_better, dim="L").astype("float32"),
            "significant_e3sm_worse": xr.concat(out_sig_worse, dim="L").astype("float32"),
        }
    )
    ds_out["rmse_diff"].attrs["description"] = "E3SM RMSE minus CESM-SMYLE RMSE; negative means E3SM lower RMSE"
    ds_out["prob_e3sm_lower_rmse"].attrs["description"] = "Bootstrap fraction with E3SM RMSE lower than CESM-SMYLE"
    ds_out.attrs["bootstrap_method"] = "Memory-safe year/member bootstrap; probability-based significance; exploratory for 3-4 years"
    ds_out.attrs["nboot"] = int(nboot)
    ds_out.attrs["alpha"] = float(alpha)
    return ds_out


def bootstrap_rmse_diff_matched_ensemble_memorysafe(
    left_err,
    right_err,
    nboot=100,
    seed=42,
    alpha=0.1,
):
    """Bootstrap ensemble-mean RMSE differences using matched ensemble sizes.

    Years are resampled as paired verification cases. For each bootstrap
    replicate, both ensembles are sampled without replacement to the smaller
    member count, averaged over members, and then scored over years.
    """
    rng = np.random.default_rng(seed)
    left_err, right_err = xr.align(
        left_err, right_err, join="inner", exclude={"M"}
    )

    common_leads = np.intersect1d(left_err["L"].values, right_err["L"].values)
    left_err = left_err.sel(L=common_leads)
    right_err = right_err.sel(L=common_leads)

    leads = [int(lead) for lead in common_leads]
    lat = left_err["lat"].values
    lon = left_err["lon"].values

    out_diff = []
    out_prob = []
    out_p = []
    out_left_better = []
    out_right_better = []

    for lead in leads:
        print(f"    matched-ensemble bootstrap lead L={lead}")

        left = (
            left_err.sel(L=lead)
            .transpose("Y", "M", "lat", "lon")
            .astype("float32")
            .load()
            .values
        )
        right = (
            right_err.sel(L=lead)
            .transpose("Y", "M", "lat", "lon")
            .astype("float32")
            .load()
            .values
        )

        n_year = left.shape[0]
        matched_nmem = min(left.shape[1], right.shape[1])
        left_obs = np.sqrt(_nanmean_preserve_missing(_nanmean_preserve_missing(left, axis=1) ** 2, axis=0))
        right_obs = np.sqrt(_nanmean_preserve_missing(_nanmean_preserve_missing(right, axis=1) ** 2, axis=0))
        diff_obs = (left_obs - right_obs).astype("float32")
        bootstrap_diffs = []

        for _ in range(nboot):
            year_index = rng.integers(0, n_year, size=n_year)
            left_members = rng.choice(left.shape[1], matched_nmem, replace=False)
            right_members = rng.choice(right.shape[1], matched_nmem, replace=False)

            left_mean = left[np.ix_(year_index, left_members)].mean(axis=1)
            right_mean = right[np.ix_(year_index, right_members)].mean(axis=1)
            boot_diff = (
                np.sqrt(_nanmean_preserve_missing(left_mean ** 2, axis=0))
                - np.sqrt(_nanmean_preserve_missing(right_mean ** 2, axis=0))
            )
            bootstrap_diffs.append(boot_diff)

        bootstrap_diffs = np.asarray(bootstrap_diffs)
        prob = _finite_comparison_probability(
            bootstrap_diffs,
            np.zeros_like(bootstrap_diffs),
        )
        p_two = (2.0 * np.minimum(prob, 1.0 - prob)).clip(0, 1).astype("float32")
        valid_prob = np.isfinite(prob)
        left_better = np.where(valid_prob, prob >= 1.0 - alpha, np.nan).astype("float32")
        right_better = np.where(valid_prob, prob <= alpha, np.nan).astype("float32")
        coords = {"lat": lat, "lon": lon}

        def as_lead(data):
            return xr.DataArray(
                data, dims=("lat", "lon"), coords=coords
            ).expand_dims(L=[lead])

        out_diff.append(as_lead(diff_obs))
        out_prob.append(as_lead(prob))
        out_p.append(as_lead(p_two))
        out_left_better.append(as_lead(left_better))
        out_right_better.append(as_lead(right_better))

        del left, right, bootstrap_diffs

    ds_out = xr.Dataset(
        {
            "rmse_diff": xr.concat(out_diff, dim="L").astype("float32"),
            "prob_left_lower_rmse": xr.concat(out_prob, dim="L").astype("float32"),
            "p_two_sided_bootstrap": xr.concat(out_p, dim="L").astype("float32"),
            "left_better": xr.concat(out_left_better, dim="L").astype("float32"),
            "right_better": xr.concat(out_right_better, dim="L").astype("float32"),
        }
    )
    ds_out.attrs.update(
        bootstrap_method=(
            "Paired-year bootstrap with matched member subsampling; "
            "ensemble mean before RMSE"
        ),
        nboot=int(nboot),
        alpha=float(alpha),
        matched_ensemble_size=int(matched_nmem),
        interpretation="Exploratory robustness evidence for 3-4 verification years",
    )
    return ds_out


def finite_ensemble_rmse_comparison_memorysafe(
    left_err,
    right_err,
    n_iterations=100,
    seed=42,
    alpha=0.1,
):
    """Compare fixed smaller-ensemble RMSE with resampled larger-ensemble RMSE.

    This follows the finite-ensemble method used by the ACC workflow: hold the
    smaller ensemble fixed and repeatedly sample the larger ensemble down to
    the smaller member count without replacement. Years are not resampled.
    """
    rng = np.random.default_rng(seed)
    left_err, right_err = xr.align(
        left_err, right_err, join="inner", exclude={"M"}
    )

    left_nmem = left_err.sizes["M"]
    right_nmem = right_err.sizes["M"]
    if left_nmem == right_nmem:
        raise ValueError(
            "Finite-ensemble resampling requires unequal ensemble sizes; "
            f"both inputs have {left_nmem} members."
        )

    larger_side = "left" if left_nmem > right_nmem else "right"
    matched_nmem = min(left_nmem, right_nmem)
    common_leads = np.intersect1d(left_err["L"].values, right_err["L"].values)
    left_err = left_err.sel(L=common_leads)
    right_err = right_err.sel(L=common_leads)

    out_diff = []
    out_prob = []
    out_p = []
    out_left_better = []
    out_right_better = []

    for lead in [int(value) for value in common_leads]:
        print(f"    finite-ensemble comparison lead L={lead}")
        left = (
            left_err.sel(L=lead)
            .transpose("Y", "M", "lat", "lon")
            .astype("float32")
            .load()
            .values
        )
        right = (
            right_err.sel(L=lead)
            .transpose("Y", "M", "lat", "lon")
            .astype("float32")
            .load()
            .values
        )

        fixed = left if larger_side == "right" else right
        larger = right if larger_side == "right" else left
        fixed_rmse = np.sqrt(
            _nanmean_preserve_missing(
                _nanmean_preserve_missing(fixed, axis=1) ** 2,
                axis=0,
            )
        )

        sampled_rmse = []
        for _ in range(n_iterations):
            members = rng.choice(larger.shape[1], matched_nmem, replace=False)
            sampled_mean = larger[:, members].mean(axis=1)
            sampled_rmse.append(
                np.sqrt(_nanmean_preserve_missing(sampled_mean ** 2, axis=0))
            )
        sampled_rmse = np.asarray(sampled_rmse)

        if larger_side == "right":
            left_rmse = fixed_rmse
            right_rmse = sampled_rmse
            prob_left_lower = _finite_comparison_probability(left_rmse, right_rmse)
            diff = left_rmse - _nanmean_preserve_missing(right_rmse, axis=0)
        else:
            left_rmse = sampled_rmse
            right_rmse = fixed_rmse
            prob_left_lower = _finite_comparison_probability(left_rmse, right_rmse)
            diff = _nanmean_preserve_missing(left_rmse, axis=0) - right_rmse

        p_two = (2.0 * np.minimum(prob_left_lower, 1.0 - prob_left_lower)).clip(0, 1)
        coords = {"lat": left_err["lat"].values, "lon": left_err["lon"].values}

        def as_lead(data):
            return xr.DataArray(
                np.asarray(data), dims=("lat", "lon"), coords=coords
            ).expand_dims(L=[lead])

        out_diff.append(as_lead(diff))
        out_prob.append(as_lead(prob_left_lower))
        out_p.append(as_lead(p_two))
        valid_prob = np.isfinite(prob_left_lower)
        out_left_better.append(as_lead(np.where(valid_prob, prob_left_lower >= 1.0 - alpha, np.nan)))
        out_right_better.append(as_lead(np.where(valid_prob, prob_left_lower <= alpha, np.nan)))

    ds_out = xr.Dataset(
        {
            "rmse_diff": xr.concat(out_diff, dim="L").astype("float32"),
            "prob_left_lower_rmse": xr.concat(out_prob, dim="L").astype("float32"),
            "p_two_sided_resampling": xr.concat(out_p, dim="L").astype("float32"),
            "left_better": xr.concat(out_left_better, dim="L").astype("float32"),
            "right_better": xr.concat(out_right_better, dim="L").astype("float32"),
        }
    )
    ds_out.attrs.update(
        resampling_method=(
            "Fixed smaller ensemble; larger ensemble sampled without replacement "
            "to matched size; years held fixed"
        ),
        larger_ensemble_side=larger_side,
        left_ensemble_size=int(left_nmem),
        right_ensemble_size=int(right_nmem),
        matched_ensemble_size=int(matched_nmem),
        n_iterations=int(n_iterations),
        alpha=float(alpha),
        interpretation=(
            "Finite-ensemble sensitivity following the ACC comparison method; "
            "limited verification years remain a major uncertainty"
        ),
    )
    return ds_out
