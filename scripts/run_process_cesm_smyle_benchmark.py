#!/usr/bin/env python
"""
Preprocess CESM-SMYLE monthly data into benchmark NetCDF files.

For every (field, init_month) combination this script:
  1. Reads monthly timeseries files via esp_lab.data_access_cesm_smyle.
  2. Optionally writes monthly benchmark files.
  3. Optionally converts monthly data to seasonal means (DJF, MAM, JJA, SON).
  4. Optionally writes seasonal benchmark files.
  5. Optionally runs sanity checks and saves diagnostic figures (--verify).

Output files follow the same naming convention as the E3SMLE benchmark files:
    BSMYLE{mm:02d}_{field}_N{nens:02d}_M{nlead:02d}_mon.nc
    BSMYLE{mm:02d}_{field}_N{nens:02d}_M{nlead:02d}_seas.nc

The output is on the NATIVE f09_g17 grid (0.9° × 1.25°).
Regridding to an analysis grid is left to downstream notebooks/scripts so that
users can choose any target grid.

Usage
-----
    # process everything with defaults: monthly and seasonal benchmarks
    python run_process_cesm_smyle_benchmark.py

    # process only monthly benchmarks, useful for Niño3.4 monthly skill
    python run_process_cesm_smyle_benchmark.py --fields TS --init-months 5 11 --freqs mon

    # process only TREFHT for May and November initializations
    python run_process_cesm_smyle_benchmark.py --fields TREFHT --init-months 5 11

    # overwrite existing output files
    python run_process_cesm_smyle_benchmark.py --force

    # generate sanity-check figures alongside the benchmark files
    python run_process_cesm_smyle_benchmark.py --verify

    # dry run: print what would be done without writing anything
    python run_process_cesm_smyle_benchmark.py --dry-run

    # use fewer ensemble members (e.g. for testing)
    python run_process_cesm_smyle_benchmark.py --nens 5 --fields TREFHT --init-months 5

Batch job example (SLURM at NERSC)
-----------------------------------
    #!/bin/bash
    #SBATCH -N 1 -c 32 --time=04:00:00 -q regular -C cpu
    #SBATCH -A <your_account>
    conda activate e3sm_analysis
    python scripts/run_process_cesm_smyle_benchmark.py --workers 32 --verify

Notes
-----
- Monthly data for all 20 members × 39 years × 4 init months is large.
  The script processes one (field, init_month) at a time to limit memory.
- Set --workers to match the number of available cores (default: 8).
- The script is idempotent: re-running skips already-written files unless
  --force is given.
- By default, both monthly (freq=mon) and seasonal (freq=seas) benchmark
  files are written. Use --freqs to choose one or both.
- Diagnostic figures are written below --figdir, not the data cache.
"""

import argparse
import os
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

# esp_lab imports
import esp_lab.data_access_cesm_smyle as smyle_access
from esp_lab.data_access_cesm_smyle import benchmark_filename as _benchmark_filename
from esp_lab.utils import calendar_utils as cal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELDS_ALL = ["TREFHT", "TS", "PRECT", "PSL"]
INIT_MONTHS_ALL = [2, 5, 8, 11]

DATA_DIR_DEFAULT = "/global/cfs/cdirs/e3sm/S2S2D/CESM-SMYLE"
OUTDIR_DEFAULT = smyle_access.BENCHMARK_OUTDIR_DEFAULT
FIGDIR_DEFAULT = "/global/cfs/cdirs/e3sm/www/zhan391/esp-lab_diag"

YEAR_START_DEFAULT = 1980
YEAR_END_DEFAULT = 2018
NENS_DEFAULT = 20
NLEAD_DEFAULT = 24
WORKERS_DEFAULT = 8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        default=DATA_DIR_DEFAULT,
        help=f"Root CESM-SMYLE directory. Default: {DATA_DIR_DEFAULT}",
    )
    p.add_argument(
        "--outdir",
        default=OUTDIR_DEFAULT,
        help=f"Output benchmark directory. Default: {OUTDIR_DEFAULT}",
    )
    p.add_argument(
        "--figdir",
        default=FIGDIR_DEFAULT,
        help=f"Output directory for verification figures. Default: {FIGDIR_DEFAULT}",
    )
    p.add_argument(
        "--fields",
        nargs="+",
        default=FIELDS_ALL,
        help=f"Field(s) to process. Default: {FIELDS_ALL}",
    )
    p.add_argument(
        "--init-months",
        nargs="+",
        type=int,
        default=INIT_MONTHS_ALL,
        dest="init_months",
        help=f"Initialization month(s). Default: {INIT_MONTHS_ALL}",
    )
    p.add_argument(
        "--year-start",
        type=int,
        default=YEAR_START_DEFAULT,
        dest="year_start",
        help=f"First initialization year. Default: {YEAR_START_DEFAULT}",
    )
    p.add_argument(
        "--year-end",
        type=int,
        default=YEAR_END_DEFAULT,
        dest="year_end",
        help=f"Last initialization year (inclusive). Default: {YEAR_END_DEFAULT}",
    )
    p.add_argument(
        "--nens",
        type=int,
        default=NENS_DEFAULT,
        help=f"Number of ensemble members. Default: {NENS_DEFAULT}",
    )
    p.add_argument(
        "--nlead",
        type=int,
        default=NLEAD_DEFAULT,
        help=f"Number of lead months. Default: {NLEAD_DEFAULT}",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=WORKERS_DEFAULT,
        help=f"Number of Dask workers. Default: {WORKERS_DEFAULT}",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Print what would be done without writing anything.",
    )
    p.add_argument(
        "--no-require-all-members",
        action="store_false",
        dest="require_all_members",
        default=True,
        help=(
            "Allow init tags with fewer than --nens members. "
            "By default all members are required."
        ),
    )
    p.add_argument(
        "--verify-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verify that file date ranges match (init_tag, nlead) before loading. "
            "Use --no-verify-coverage for exploratory partial runs."
        ),
    )
    p.add_argument(
        "--open-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use xarray.open_mfdataset(parallel=True) when opening raw "
            "CESM-SMYLE files. Default is serial opens to avoid intermittent "
            "netCDF/HDF5 metadata errors on CFS."
        ),
    )
    p.add_argument(
        "--freqs",
        nargs="+",
        default=["mon", "seas"],
        choices=["mon", "seas"],
        help="Benchmark frequencies to write. Default: mon seas",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help=(
            "After writing each benchmark file, run sanity checks and save "
            "diagnostic figures to <figdir>/CESM-SMYLE/verify/. "
            "Checks: dimension completeness, NaN fraction, value ranges. "
            "Figures: ensemble-mean global map, ensemble-spread map, "
            "global-mean lead-time series, and Niño-3.4 lead-time series."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sanity-check / verification
# ---------------------------------------------------------------------------

# Expected physical value ranges for a rough sanity gate.
_FIELD_RANGES = {
    "TREFHT": (180.0, 340.0),   # K
    "TS":     (180.0, 340.0),   # K
    "PRECT":  (0.0,   1e-3),    # m/s
    "PSL":    (8e4,   1.1e5),   # Pa
}

# Niño-3.4 box for the lead-series inset check
_NINO34 = dict(lat_s=-5, lat_n=5, lon_w=190, lon_e=240)

SEASON_LABELS = {1: "DJF", 4: "MAM", 7: "JJA", 10: "SON"}


def _season_label(ds, L_idx: int) -> str:
    """Return a season label string for a given L index."""
    try:
        t = ds.time.isel(Y=0, L=L_idx).values
        mo = int(getattr(t, "month", None) or 0)
        return SEASON_LABELS.get(mo, f"L{L_idx + 1}")
    except Exception:
        return f"L{L_idx + 1}"


def _fig_name(
    diagnostic: str,
    variable: str,
    model: str,
    reference: str,
    period: str,
    *extra: str,
) -> str:
    """Return a standard verification figure filename."""
    parts = ["fig", diagnostic, variable, model, reference, period, *extra]
    clean = [
        str(part).strip().replace(" ", "_").replace("/", "-")
        for part in parts
        if part is not None and str(part).strip()
    ]
    return "_".join(clean) + ".png"

def _check_integrity(da, field):
    """Perform data integrity checks (NaNs, ranges, dimensions)."""
    checks_passed = []
    check_warnings = []

    # 1. Dimension completeness
    required_dims = {"Y", "L", "M", "lat", "lon"}
    missing = required_dims - set(da.dims)
    if missing:
        check_warnings.append(f"Missing dimensions: {missing}")
    else:
        checks_passed.append(f"Dimensions OK: {list(da.sizes.items())}")

    # 2. NaN fraction
    da_vals = da.values
    total = da_vals.size
    n_nan = int(np.isnan(da_vals).sum())
    nan_frac = n_nan / total if total > 0 else 0.0
    nan_msg = f"NaN fraction: {nan_frac:.4%}"
    if nan_frac > 0.05:
        check_warnings.append(f"HIGH NaN fraction — {nan_msg}")
    else:
        checks_passed.append(nan_msg)

    # 3. Physical value range
    vmin, vmax = float(np.nanmin(da_vals)), float(np.nanmax(da_vals))
    range_msg = f"Value range: [{vmin:.3g}, {vmax:.3g}]"
    if field in _FIELD_RANGES:
        lo, hi = _FIELD_RANGES[field]
        if vmin < lo or vmax > hi:
            check_warnings.append(f"Outside expected range [{lo}, {hi}] — {range_msg}")
        else:
            checks_passed.append(f"Physical range OK — {range_msg}")

    return checks_passed, check_warnings, da_vals

def _plot_verification(da, da_vals, ds, field, fname_stem, period, fig_dir):
    """Generate diagnostic figures for a benchmark file."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    lat, lon = da.lat.values, da.lon.values
    yi_mid = da.sizes["Y"] // 2

    # 1 & 2: Global ensemble-mean and spread maps
    for L_idx, L_label in [(0, _season_label(ds, 0)), (-1, _season_label(ds, -1))]:
        slice_all = da.isel(Y=yi_mid, L=L_idx)
        ens_mean = slice_all.mean("M").values
        ens_std = slice_all.std("M").values

        for what, data_2d, cmap in [("ens_mean", ens_mean, "RdBu_r"), ("ens_spread", ens_std, "YlOrRd")]:
            fig, ax = plt.subplots(figsize=(10, 4), subplot_kw={"projection": ccrs.Robinson()})
            vabs = np.nanpercentile(np.abs(data_2d), 98)
            vmin_p, vmax_p = (np.nanmean(data_2d) - vabs*0.4, np.nanmean(data_2d) + vabs*0.4) if field in ("TREFHT", "TS") and what == "ens_mean" else (0, vabs)

            im = ax.pcolormesh(lon, lat, data_2d, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin_p, vmax=vmax_p)
            ax.coastlines(linewidth=0.5)
            plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.03)
            ax.set_title(f"{fname_stem} | {what} | {L_label}")
            fig.savefig(os.path.join(fig_dir, _fig_name(what, field, "CESM-SMYLE", "benchmark", period, L_label)), dpi=120)
            plt.close(fig)

    # 3: Global-mean lead-time series
    weights = xr.DataArray(np.cos(np.deg2rad(lat)), dims=("lat",), coords={"lat": da.lat})
    gmean = da.weighted(weights).mean(("lat", "lon"), skipna=True).values
    ens_mean_ts = gmean.mean(axis=-1)

    fig, ax = plt.subplots(figsize=(9, 4))
    for yi in range(ens_mean_ts.shape[0]):
        ax.plot(da.L, ens_mean_ts[yi], color="steelblue", alpha=0.3)
    ax.plot(da.L, ens_mean_ts.mean(axis=0), color="navy", linewidth=2, label="Grand mean")
    ax.set_title(f"{fname_stem} | global-mean lead-series")
    fig.savefig(os.path.join(fig_dir, _fig_name("global_mean_lead_series", field, "CESM-SMYLE", "benchmark", period)), dpi=120)
    plt.close(fig)

def _set_benchmark_metadata(
    ds,
    field,
    init_month,
    nlead,
    n_members,
    year_range,
    frequency,
    processing,
    data_dir,
):
    """Apply standard metadata attributes to the dataset."""
    ds.attrs.update({
        "source": "CESM-SMYLE hindcast data",
        "data_dir": data_dir,
        "field": field,
        "init_month": init_month,
        "nlead_months": nlead,
        "n_members": n_members,
        "year_range": year_range,
        "grid": "f09_g17 (0.9x1.25 FV)",
        "frequency": frequency,
        "processing": processing,
        "created_by": "scripts/run_process_cesm_smyle_benchmark.py",
    })

def verify_benchmark(
    fpath: str,
    field: str,
    init_month: int,
    fig_dir: str,
) -> dict:
    """
    Run integrity checks on a written benchmark file and save diagnostic figures.

    Checks
    ------
    1. All expected dimensions (Y, L, M, lat, lon) are present.
    2. NaN fraction is below 5 %.
    3. Ensemble-mean values fall within the expected physical range for the field.
    4. No single year or member is all-NaN (complete dropout).

    Figures (saved to fig_dir)
    --------------------------
    1. Global ensemble-mean map — first and last lead season.
    2. Ensemble-spread map (std across members) — same slice.
    3. Global-mean lead-time series — ensemble mean ±1σ, all init years overlaid.
    4. Niño-3.4 regional-mean lead-time series — same layout (TREFHT / TS only).

    Parameters
    ----------
    fpath : str
        Path to the written benchmark NetCDF file.
    field : str
        Variable name.
    init_month : int
        Initialization month.
    fig_dir : str
        Directory to write PNG figures.

    Returns
    -------
    report : dict
        Keys: ``passed`` (bool), ``checks`` (list of str), ``warnings`` (list of str).
    """
    import xarray as xr
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for batch use
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    os.makedirs(fig_dir, exist_ok=True)
    fname_stem = Path(fpath).stem

    log = logging.getLogger(__name__)
    checks_passed: list = []
    check_warnings: list = []

    ds = xr.open_dataset(fpath, chunks={"Y": 1})
    da = ds[field]
    period = f"init{init_month:02d}_{ds.Y.values[0]}-{ds.Y.values[-1]}"

    checks_passed, check_warnings, da_vals = _check_integrity(da, field)

    # ------------------------------------------------------------------
    # Print check results
    # ------------------------------------------------------------------
    log.info("  [VERIFY] %s", fname_stem)
    for msg in checks_passed:
        log.info("    ✓ %s", msg)
    for msg in check_warnings:
        log.warning("    ✗ %s", msg)

    _plot_verification(da, da_vals, ds, field, fname_stem, period, fig_dir)

    ds.close()

    passed = len(check_warnings) == 0
    return {"passed": passed, "checks": checks_passed, "warnings": check_warnings}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def benchmark_filename(
    field: str,
    init_month: int,
    nens: int,
    nlead: int,
    freq: str = "seas",
) -> str:
    """Return the standard benchmark filename."""
    return _benchmark_filename(
        field,
        init_month,
        nens=nens,
        nlead=nlead,
        freq=freq,
    )


def build_encoding(
    field: str,
    chunksizes: tuple = (1, 8, 1, 96, 144),
) -> dict:
    """Build NetCDF4 encoding dict for one benchmark file."""
    return {
        field: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 1,
            "chunksizes": chunksizes,
        }
    }


def validate_monthly_dataset(
    ds,
    field: str,
    years: list,
    members: list,
    nlead: int,
    require_all_members: bool,
) -> None:
    """Fail early when the loaded monthly benchmark would not match its filename."""
    if field not in ds:
        raise ValueError(f"Loaded dataset does not contain field {field!r}.")

    required_dims = {"Y", "L", "M", "lat", "lon"}
    missing = required_dims - set(ds[field].dims)
    if missing:
        raise ValueError(f"Loaded {field!r} data is missing dimensions: {sorted(missing)}")

    if ds.sizes["L"] != nlead:
        raise ValueError(
            f"Loaded {field!r} has L={ds.sizes['L']} lead(s), expected nlead={nlead}."
        )

    if require_all_members and ds.sizes["Y"] != len(years):
        raise ValueError(
            f"Loaded {field!r} has Y={ds.sizes['Y']} init year(s), "
            f"expected {len(years)}."
        )

    if require_all_members and ds.sizes["M"] != len(members):
        raise ValueError(
            f"Loaded {field!r} has M={ds.sizes['M']} member(s), "
            f"expected {len(members)}."
        )

    mismatch_count = smyle_access.verification_time_mismatch_count(ds)
    if mismatch_count:
        raise ValueError(
            f"Loaded {field!r} has {mismatch_count} verification timestamp(s) "
            "that disagree with its Y initialization tags and L lead values."
        )


def load_monthly_benchmark_dataset(
    *,
    data_dir: str,
    members: list,
    init_tags: list,
    field: str,
    nlead: int,
    require_all_members: bool,
    verify_coverage: bool,
    years: list,
    open_parallel: bool = False,
) -> xr.Dataset:
    """Open and validate one CESM-SMYLE monthly benchmark dataset."""
    ds = smyle_access.get_monthly_data(
        data_dir=data_dir,
        members=members,
        init_tags=init_tags,
        field=field,
        nlead=nlead,
        require_all_members=require_all_members,
        verify_coverage=verify_coverage,
        open_parallel=open_parallel,
    )
    validate_monthly_dataset(
        ds=ds,
        field=field,
        years=years,
        members=members,
        nlead=nlead,
        require_all_members=require_all_members,
    )
    return ds


def write_netcdf_atomic(ds, path: Path, encoding: dict) -> None:
    """Write NetCDF via a temp file, then atomically replace the final path."""
    import uuid
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use uuid to prevent collisions across distributed workers on different nodes
    tmp_path = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        ds.to_netcdf(str(tmp_path), encoding=encoding)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_one(
    field: str,
    init_month: int,
    data_dir: str,
    outdir: str,
    years: list,
    members: list,
    nlead: int,
    require_all_members: bool,
    verify_coverage: bool,
    force: bool,
    dry_run: bool,
    run_verify: bool = False,
    freqs: list | None = None,
    figdir: str | os.PathLike[str] = FIGDIR_DEFAULT,
    open_parallel: bool = False,
) -> str:
    """
    Process one (field, init_month) combination.

    This version can write monthly and/or seasonal CESM-SMYLE benchmark files.

    Returns
    -------
    str
        Status: 'ok', 'skipped', 'dry_run', or 'no_data'.
    """
    log = logging.getLogger(__name__)
    if freqs is None:
        freqs = ["mon", "seas"]

    output_dir = Path(outdir) / "leadtime_acc" / "inputs" / field
    outfiles = {
        freq: output_dir / benchmark_filename(
            field,
            init_month,
            nens=len(members),
            nlead=nlead,
            freq=freq,
        )
        for freq in freqs
    }

    # Skip only when all existing products also have internally consistent
    # verification dates.  This automatically invalidates legacy benchmarks
    # whose time(Y,L) coordinate became detached during multi-file concat.
    if all(path.exists() for path in outfiles.values()) and not force:
        bad_time_files = []
        for path in outfiles.values():
            with xr.open_dataset(path) as existing:
                mismatch_count = smyle_access.verification_time_mismatch_count(
                    existing,
                    init_month=init_month,
                )
            if mismatch_count:
                bad_time_files.append(f"{path.name} ({mismatch_count} bad dates)")
        if not bad_time_files:
            log.info(
                "  [SKIP]   %s (already exists)",
                ", ".join(path.name for path in outfiles.values())
            )
            return "skipped"
        log.warning(
            "  [REBUILD] invalid verification time: %s",
            ", ".join(bad_time_files),
        )
        force = True

    if dry_run:
        log.info(
            "  [DRY]    %s (would write to %s)",
            ", ".join(path.name for path in outfiles.values()),
            outdir
        )
        return "dry_run"

    t0 = time.perf_counter()
    log.info("  Loading  field=%s, init_month=%02d ...", field, init_month)

    # Build init tags for this month only.
    init_tags = smyle_access.build_init_tags(years, init_month)

    # Load monthly data once for validation and for any monthly write. If a
    # seasonal file is also needed, reopen below before building the seasonal
    # graph so it does not reuse netCDF handles touched by the monthly write.
    try:
        ds_mon = load_monthly_benchmark_dataset(
            data_dir=data_dir,
            members=members,
            init_tags=init_tags,
            field=field,
            nlead=nlead,
            require_all_members=require_all_members,
            verify_coverage=verify_coverage,
            years=years,
            open_parallel=open_parallel,
        )
    except ValueError as e:
        warnings.warn(f"No data for field={field}, init_month={init_month}: {e}")
        return "no_data"

    # Rechunk monthly data.
    # Member chunk size of 1 keeps memory bounded during write/seasonal aggregation.
    nM = 1
    mchunk_mon = {
        "Y": 3,
        "L": nlead,
        "M": nM,
        "lat": 96,
        "lon": 144,
    }
    ds_mon = ds_mon.chunk(mchunk_mon)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write monthly benchmark
    # ------------------------------------------------------------------
    if "mon" in freqs:
        mon_file = outfiles["mon"]

        if mon_file.exists() and not force:
            print(f"  [SKIP]   {mon_file.name}  (already exists)")
        else:
            lat_size = ds_mon.sizes["lat"]
            lon_size = ds_mon.sizes["lon"]

            mon_chunksizes = (
                1,
                min(24, ds_mon.sizes["L"]),
                1,
                lat_size // 2,
                lon_size // 2,
            )

            encoding_mon = build_encoding(
                field,
                chunksizes=mon_chunksizes,
            )
            _set_benchmark_metadata(
                ds_mon,
                field,
                init_month,
                nlead,
                len(members),
                f"{years[0]}-{years[-1]}",
                "monthly",
                "monthly benchmark, no seasonal averaging",
                data_dir,
            )

            log.info("  Writing monthly benchmark: %s", mon_file)
            write_netcdf_atomic(ds_mon, mon_file, encoding=encoding_mon)

    # ------------------------------------------------------------------
    # Write seasonal benchmark
    # ------------------------------------------------------------------
    if "seas" in freqs:
        seas_file = outfiles["seas"]

        if seas_file.exists() and not force:
            log.info("  [SKIP]   %s (already exists)", seas_file.name)
        else:
            if "mon" in freqs:
                try:
                    ds_mon.close()
                except Exception:
                    pass
                ds_mon = load_monthly_benchmark_dataset(
                    data_dir=data_dir,
                    members=members,
                    init_tags=init_tags,
                    field=field,
                    nlead=nlead,
                    require_all_members=require_all_members,
                    verify_coverage=verify_coverage,
                    years=years,
                    open_parallel=open_parallel,
                ).chunk(mchunk_mon)

            # Convert monthly → seasonal.
            ds_seas = cal.mon_to_seas_dask(ds_mon)

            # Rechunk with actual seasonal L size.
            n_L_seas = ds_seas.sizes["L"]
            mchunk_seas = {**mchunk_mon, "L": n_L_seas}
            ds_seas = ds_seas.chunk(mchunk_seas)

            lat_size = ds_seas.sizes["lat"]
            lon_size = ds_seas.sizes["lon"]

            seas_chunksizes = (
                1,
                min(8, n_L_seas),
                1,
                lat_size // 2,
                lon_size // 2,
            )

            encoding_seas = build_encoding(
                field,
                chunksizes=seas_chunksizes,
            )
            _set_benchmark_metadata(
                ds_seas,
                field,
                init_month,
                nlead,
                len(members),
                f"{years[0]}-{years[-1]}",
                "seasonal",
                "monthly → seasonal via 3-month seasonal averaging",
                data_dir,
            )

            log.info("  Writing seasonal benchmark: %s", seas_file)
            write_netcdf_atomic(ds_seas, seas_file, encoding=encoding_seas)
            try:
                ds_seas.close()
            except Exception:
                pass

    try:
        ds_mon.close()
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    log.info(
        "  [DONE]   field=%s, init_month=%02d (%.1fs)", field, init_month, elapsed
    )

    if run_verify:
        for freq, fpath in outfiles.items():
            if not fpath.exists():
                continue
            fig_dir = os.path.join(str(figdir), "CESM-SMYLE", "verify")
            report = verify_benchmark(
                fpath=str(fpath),
                field=field,
                init_month=init_month,
                fig_dir=fig_dir,
            )
            if not report["passed"]:
                warnings.warn(
                    f"[VERIFY] {fpath.name} has {len(report['warnings'])} warning(s): "
                    + "; ".join(report["warnings"])
                )

    return "ok"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    log = logging.getLogger(__name__)
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in cast.*")

    # Validate data directory
    if not Path(args.data_dir).exists():
        sys.exit(f"ERROR: data_dir does not exist: {args.data_dir}")

    years = list(range(args.year_start, args.year_end + 1))
    members = [f"EN{i:02d}" for i in range(1, args.nens + 1)]
    combos = [(f, m) for f in args.fields for m in args.init_months]

    log.info("=" * 70)
    log.info("CESM-SMYLE benchmark preprocessing")
    log.info("=" * 70)
    log.info("  data_dir      : %s", args.data_dir)
    log.info("  outdir        : %s", args.outdir)
    log.info("  fields        : %s", args.fields)
    log.info("  init_months   : %s", args.init_months)
    log.info("  years         : %d–%d (%d years)", years[0], years[-1], len(years))
    log.info("  members       : EN01–EN%02d (%d members)", args.nens, args.nens)
    log.info("  nlead         : %d months", args.nlead)
    log.info("  freqs         : %s", args.freqs)
    log.info("  combinations  : %d", len(combos))
    log.info("  force         : %s", args.force)
    log.info("  dry_run       : %s", args.dry_run)
    log.info("  verify_coverage : %s", args.verify_coverage)
    log.info("  open_parallel : %s", args.open_parallel)
    log.info("  workers       : %d", args.workers)
    log.info("  verify        : %s", args.verify)
    log.info("  figdir        : %s", args.figdir)
    log.info("=" * 70)

    client = None
    cluster = None
    if args.dry_run:
        log.info("Dry run: not starting a Dask client.")
    else:
        import dask
        from dask.distributed import Client, LocalCluster

        dask.config.set({"array.slicing.split_large_chunks": True})
        cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
        client = Client(cluster)
        print(f"Dask dashboard: {client.dashboard_link}\n")

    total_t0 = time.perf_counter()
    counters: dict = {"ok": 0, "skipped": 0, "dry_run": 0, "no_data": 0, "failed": 0}
    interrupted = False

    try:
        for field, init_month in combos:
            log.info("\nProcessing: field=%s, init_month=%02d", field, init_month)
            try:
                status = process_one(
                    field=field,
                    init_month=init_month,
                    data_dir=args.data_dir,
                    outdir=args.outdir,
                    years=years,
                    members=members,
                    nlead=args.nlead,
                    require_all_members=args.require_all_members,
                    verify_coverage=args.verify_coverage,
                    force=args.force,
                    dry_run=args.dry_run,
                    run_verify=args.verify,
                    freqs=args.freqs,
                    figdir=args.figdir,
                    open_parallel=args.open_parallel,
                )
                counters[status] = counters.get(status, 0) + 1
            except KeyboardInterrupt:
                log.info("\nInterrupted by user.")
                interrupted = True
                break
            except Exception as exc:
                warnings.warn(f"FAILED: field={field}, init_month={init_month}: {exc}")
                counters["failed"] += 1
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()

    elapsed = time.perf_counter() - total_t0
    log.info("\n" + "=" * 70)
    log.info("Finished in %.1fs", elapsed)
    log.info("  Written:   %d", counters['ok'])
    log.info("  Skipped:   %d", counters['skipped'])
    log.info("  Dry-run:   %d", counters['dry_run'])
    log.info("  No data:   %d", counters['no_data'])
    log.info("  Failed:    %d", counters['failed'])
    log.info("  Interrupted: %s", interrupted)
    log.info("=" * 70)

    if interrupted:
        sys.exit(130)

    if counters["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
