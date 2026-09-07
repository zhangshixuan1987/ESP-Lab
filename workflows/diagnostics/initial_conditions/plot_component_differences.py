#!/usr/bin/env python3
"""
04_plot_component_differences.py
=================================
Step 4: Generate spatial and vertical diagnostic plots for priority IC
        variables using the ΔX NetCDF files from Step 3.

Plots produced per (date, component, variable)
----------------------------------------------
* Four-panel native-grid fingerprint (signed, relative, standardised, threshold mask)
* Area-weighted zonal-mean difference profile
* Vertical RMSE profile  (for 3-D fields: ocean layers, soil levels)
* Global histogram and regional empirical CDFs

Plots produced per (date, component)
-------------------------------------
* Multi-variable RMSE bar chart (top-N by RMSE)

Usage
-----
    python 04_plot_component_differences.py
    python 04_plot_component_differences.py --full-campaign
    python 04_plot_component_differences.py --date 1980-05-01-00000
    python 04_plot_component_differences.py --components ocn lnd --top-n 10

Outputs
-------
    output/maps/<date>_<component>_<var>_fingerprint.png
    output/maps/<date>_<component>_<var>_distribution.png
    output/profiles/<date>_<component>_<var>_zonal_mean.png
    output/profiles/<date>_<component>_<var>_vertical_rmse.png
    output/maps/<date>_<component>_rmse_bar.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .config import build_ic_config, load_config

# ---------------------------------------------------------------------------
# Figure style (mirrors 5a_refactor style)
# ---------------------------------------------------------------------------
FONTZ = 11
FS = {
    "title":     int(round(FONTZ * 1.0)),
    "label":     int(round(FONTZ * 0.9)),
    "tick":      int(round(FONTZ * 0.85)),
    "colorbar":  int(round(FONTZ * 0.85)),
    "suptitle":  int(round(FONTZ * 1.1)),
}
FIG_DPI = 150


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _map_plot(diff_da: xr.DataArray, title: str, outpath: Path) -> None:
    """Simple filled-contour map of a 2-D difference field."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=FIG_DPI)
    try:
        # Try pcolormesh with lat/lon if available
        if "lat" in diff_da.dims and "lon" in diff_da.dims:
            vmax = float(np.nanpercentile(np.abs(diff_da.values), 98))
            vmax = vmax if vmax > 0 else 1.0
            pcm = ax.pcolormesh(
                diff_da["lon"].values,
                diff_da["lat"].values,
                diff_da.values,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
            )
            ax.set_xlabel("Longitude", fontsize=FS["label"])
            ax.set_ylabel("Latitude", fontsize=FS["label"])
        else:
            # Unstructured mesh: 1-D array → histogram fallback
            vals = diff_da.values.ravel()
            ax.hist(vals[np.isfinite(vals)], bins=80, color="steelblue", alpha=0.8)
            ax.axvline(0, color="k", lw=1, ls="--")
            ax.set_xlabel("ΔX", fontsize=FS["label"])
            ax.set_ylabel("Count", fontsize=FS["label"])
            pcm = None
    except Exception as exc:
        warnings.warn(f"_map_plot fallback for {outpath.name}: {exc}", stacklevel=2)
        vals = diff_da.values.ravel()
        ax.hist(vals[np.isfinite(vals)], bins=80, color="steelblue", alpha=0.8)
        pcm = None

    if pcm is not None:
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.03)
        cbar.ax.tick_params(labelsize=FS["colorbar"])

    ax.set_title(title, fontsize=FS["title"])
    ax.tick_params(labelsize=FS["tick"])
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _plot_native_field(ax, da: xr.DataArray, title: str, *, categorical: bool = False):
    """Plot a structured or MPAS-native field without regridding."""
    plot_da = da
    depth_dim = _detect_depth_dim(plot_da)
    if depth_dim:
        plot_da = plot_da.isel({depth_dim: 0})
    for dim in list(plot_da.dims):
        if dim not in ("lat", "lon", "nCells", "x", "y"):
            plot_da = plot_da.isel({dim: 0})

    cmap = "Greys" if categorical else "RdBu_r"
    kwargs = {"vmin": 0, "vmax": 1} if categorical else {}
    if not categorical:
        vmax = float(np.nanpercentile(np.abs(plot_da.values), 98))
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        kwargs = {"vmin": -vmax, "vmax": vmax}

    if "lat" in plot_da.dims and "lon" in plot_da.dims:
        artist = ax.pcolormesh(
            plot_da["lon"], plot_da["lat"], plot_da,
            shading="auto", cmap=cmap, **kwargs,
        )
    elif "nCells" in plot_da.dims and {"latCell", "lonCell"} <= set(plot_da.coords):
        lat = np.asarray(plot_da["latCell"])
        lon = np.asarray(plot_da["lonCell"])
        if np.nanmax(np.abs(lat)) <= np.pi + 0.1:
            lat = np.rad2deg(lat)
        if np.nanmax(np.abs(lon)) <= 2 * np.pi + 0.1:
            lon = np.rad2deg(lon)
        artist = ax.scatter(lon, lat, c=plot_da.values, s=1, cmap=cmap, rasterized=True, **kwargs)
    else:
        values = np.asarray(plot_da).ravel()
        artist = ax.scatter(np.arange(values.size), values, s=2, c=values, cmap=cmap, **kwargs)
        ax.set_xlabel("native cell index")
    ax.set_title(title, fontsize=FS["title"])
    return artist


def _fingerprint_plot(ds: xr.Dataset, title: str, outpath: Path) -> None:
    """Four-panel native-grid physical fingerprint."""
    panels = [
        ("signed_difference", "signed difference", False),
        ("relative_difference", "relative difference", False),
        ("standardized_difference", "standardized difference", False),
        ("threshold_exceedance", "threshold exceedance", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=FIG_DPI, constrained_layout=True)
    for ax, (name, label, categorical) in zip(axes.flat, panels):
        if name not in ds:
            ax.set_axis_off()
            continue
        artist = _plot_native_field(ax, ds[name], label, categorical=categorical)
        fig.colorbar(artist, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(title, fontsize=FS["suptitle"])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _distribution_plot(diff_da: xr.DataArray, title: str, outpath: Path) -> None:
    plot_da = diff_da
    depth_dim = _detect_depth_dim(plot_da)
    if depth_dim:
        plot_da = plot_da.isel({depth_dim: 0})
    for dim in list(plot_da.dims):
        if dim not in ("lat", "lon", "nCells", "x", "y"):
            plot_da = plot_da.isel({dim: 0})
    values = np.asarray(plot_da).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=FIG_DPI, constrained_layout=True)
    axes[0].hist(values, bins=80, color="steelblue", alpha=0.85)
    axes[0].axvline(0, color="k", lw=1, ls="--")
    axes[0].set(xlabel="ΔX", ylabel="cell count", title="Histogram")
    region_values = {"Global": values}
    lat = None
    if "lat" in plot_da.coords:
        lat = plot_da["lat"].broadcast_like(plot_da).values.ravel()
    elif "latCell" in plot_da.coords:
        lat = np.asarray(plot_da["latCell"])
        if np.nanmax(np.abs(lat)) <= np.pi + 0.1:
            lat = np.rad2deg(lat)
    if lat is not None and lat.size == np.asarray(plot_da).size:
        raw_values = np.asarray(plot_da).ravel()
        bands = {
            "Arctic (≥60°N)": lat >= 60,
            "Tropics (20°S–20°N)": np.abs(lat) <= 20,
            "Antarctic (≤60°S)": lat <= -60,
        }
        for label, mask in bands.items():
            regional = raw_values[mask & np.isfinite(raw_values)]
            if regional.size:
                region_values[label] = regional
    for label, regional in region_values.items():
        ordered = np.sort(regional)
        axes[1].plot(ordered, np.arange(1, ordered.size + 1) / ordered.size, label=label)
    axes[1].axvline(0, color="k", lw=1, ls="--")
    axes[1].set(xlabel="ΔX", ylabel="empirical probability", title="Empirical CDF")
    axes[1].legend(fontsize=max(6, FS["tick"] - 1))
    fig.suptitle(title, fontsize=FS["suptitle"])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _zonal_mean_plot(
    diff_da: xr.DataArray, var: str, title: str, outpath: Path
) -> None:
    """Area-weighted zonal-mean difference profile."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=FIG_DPI)
    try:
        if "lat" in diff_da.dims:
            zm = diff_da.mean(dim="lon", skipna=True) if "lon" in diff_da.dims else diff_da
            ax.plot(zm.values, zm["lat"].values if "lat" in zm.coords else range(len(zm)),
                    color="steelblue", lw=1.5)
            ax.axvline(0, color="k", lw=1, ls="--")
            ax.set_xlabel(f"Zonal mean ΔX  ({var})", fontsize=FS["label"])
            ax.set_ylabel("Latitude", fontsize=FS["label"])
        else:
            ax.text(0.5, 0.5, "Zonal mean N/A\n(unstructured grid)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=FS["label"])
    except Exception as exc:
        warnings.warn(f"_zonal_mean_plot for {outpath.name}: {exc}", stacklevel=2)

    ax.set_title(title, fontsize=FS["title"])
    ax.tick_params(labelsize=FS["tick"])
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _vertical_rmse_plot(
    diff_da: xr.DataArray, depth_dim: str, title: str, outpath: Path
) -> None:
    """Vertical profiles of layer-wise weighted mean difference and RMSE."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=FIG_DPI, sharey=True)
    try:
        reduce_dims = [d for d in diff_da.dims if d != depth_dim]
        if "areaCell" in diff_da.coords and "nCells" in reduce_dims:
            weights = diff_da["areaCell"]
            mean_profile = diff_da.weighted(weights).mean(reduce_dims, skipna=True)
            rmse_profile = np.sqrt((diff_da ** 2).weighted(weights).mean(reduce_dims, skipna=True))
        else:
            mean_profile = diff_da.mean(dim=reduce_dims, skipna=True)
            rmse_profile = np.sqrt((diff_da ** 2).mean(dim=reduce_dims, skipna=True))
        depth = rmse_profile[depth_dim].values if depth_dim in rmse_profile.coords \
            else np.arange(len(rmse_profile))
        axes[0].plot(mean_profile.values, depth, color="firebrick", lw=1.5, marker="o", markersize=3)
        axes[0].axvline(0, color="k", lw=1, ls="--")
        axes[0].set_xlabel("weighted mean ΔX", fontsize=FS["label"])
        axes[1].plot(rmse_profile.values, depth, color="steelblue", lw=1.5, marker="o", markersize=3)
        axes[1].set_xlabel("RMSE", fontsize=FS["label"])
        axes[0].invert_yaxis()
        axes[0].set_ylabel(depth_dim, fontsize=FS["label"])
    except Exception as exc:
        warnings.warn(f"_vertical_rmse_plot for {outpath.name}: {exc}", stacklevel=2)

    fig.suptitle(title, fontsize=FS["title"])
    for ax in axes:
        ax.tick_params(labelsize=FS["tick"])
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _rmse_bar_chart(
    stat_df: pd.DataFrame, date: str, comp: str, top_n: int, outpath: Path
) -> None:
    """Horizontal bar chart of top-N variables by RMSE."""
    rank_metric = "nrmse" if "nrmse" in stat_df else "rmse"
    top = stat_df.nlargest(top_n, rank_metric)
    fig, ax = plt.subplots(figsize=(7, max(3, len(top) * 0.4)), dpi=FIG_DPI)
    ax.barh(top["variable"], top[rank_metric], color="steelblue", alpha=0.85)
    ax.set_xlabel(rank_metric.upper(), fontsize=FS["label"])
    ax.set_title(
        f"{comp.upper()}  IC {rank_metric.upper()}  |  {date}", fontsize=FS["title"]
    )
    ax.tick_params(labelsize=FS["tick"])
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Depth dimension detection
# ---------------------------------------------------------------------------

_DEPTH_DIMS = {"nVertLevels", "nVertLevelsP1", "levgrnd", "levsoi",
               "lev", "depth", "z_l", "z_i", "nLayers"}


def _detect_depth_dim(da: xr.DataArray) -> str | None:
    for d in da.dims:
        if d in _DEPTH_DIMS or "lev" in d.lower() or "depth" in d.lower():
            return d
    return None


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    config_path: Path,
    pilot_only: bool | None = None,
    single_date: str | None = None,
    filter_components: list[str] | None = None,
    top_n: int = 15,
    figure_outdir: str | Path | None = None,
    verbose: bool = True,
) -> None:
    cfg = load_config(config_path)
    ic_cfg = build_ic_config(cfg, pilot_only=pilot_only)

    out_root = Path(config_path.parent) / ic_cfg.output_root
    subdirs = cfg.get("output", {}).get("subdirs", {})
    vs_dir = out_root / subdirs.get("variable_statistics", "variable_statistics")
    figure_root = Path(
        figure_outdir or cfg.get("output", {}).get("figure_outdir", out_root)
    )
    maps_dir = figure_root / subdirs.get("maps", "maps")
    profiles_dir = figure_root / subdirs.get("profiles", "profiles")
    maps_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    active_dates = [single_date] if single_date else ic_cfg.active_dates
    fingerprint_keywords = cfg.get("diagnostics", {}).get("fingerprint_keywords", {})

    if verbose:
        print("=" * 70)
        print("IC Analysis — Step 4: Plot Component Differences")
        print("=" * 70)

    for date in active_dates:
        for comp_spec in ic_cfg.components:
            comp = comp_spec.name
            if filter_components and comp not in filter_components:
                continue

            legacy_diff = vs_dir / f"{date}_{comp}_diff.nc"
            diff_files = sorted(vs_dir.glob(f"{date}_*_{comp}_*_diff.nc"))
            if legacy_diff.exists():
                diff_files.insert(0, legacy_diff)

            if not diff_files:
                if verbose:
                    print(f"  [SKIP] {date} {comp} — diff file not found")
                continue

            if verbose:
                print(f"  [{date}] {comp}")

            for diff_nc in diff_files:
                try:
                    ds_diff = xr.open_dataset(diff_nc, chunks="auto")
                except Exception as exc:
                    warnings.warn(f"  Cannot open {diff_nc}: {exc}", stacklevel=2)
                    continue

                member = str(ds_diff.attrs.get("member", "nomember"))
                var = str(ds_diff.attrs.get("source_variable", next(iter(ds_diff.data_vars))))
                keywords = [str(k).lower() for k in fingerprint_keywords.get(comp, [])]
                if keywords and not any(k in var.lower() for k in keywords):
                    ds_diff.close()
                    continue
                diff_da = ds_diff.get("signed_difference", ds_diff[next(iter(ds_diff.data_vars))])
                base = f"{date}_{member}_{comp}_{var}"
                title_base = f"IC fingerprint ({comp.upper()} · {var} · {member}) | {date}"
                _fingerprint_plot(ds_diff, title_base, maps_dir / f"{base}_fingerprint.png")
                _distribution_plot(
                    diff_da,
                    f"ΔX distribution | {comp.upper()} {var} {member} | {date}",
                    maps_dir / f"{base}_distribution.png",
                )
                if "lat" in diff_da.dims or diff_da.ndim >= 2:
                    _zonal_mean_plot(
                        diff_da, var,
                        title=f"Zonal mean ΔX | {comp.upper()} {var} {member} | {date}",
                        outpath=profiles_dir / f"{base}_zonal_mean.png",
                    )
                depth_dim = _detect_depth_dim(diff_da)
                if depth_dim and diff_da.ndim >= 2:
                    _vertical_rmse_plot(
                        diff_da, depth_dim,
                        title=f"Vertical RMSE | {comp.upper()} {var} {member} | {date}",
                        outpath=profiles_dir / f"{base}_vertical_rmse.png",
                    )
                ds_diff.close()

            stats_files = sorted(vs_dir.glob(f"{date}_*_{comp}_stats.csv"))
            legacy_stats = vs_dir / f"{date}_{comp}_stats.csv"
            if legacy_stats.exists():
                stats_files.insert(0, legacy_stats)
            if stats_files:
                try:
                    stat_df = pd.concat(
                        [pd.read_csv(path) for path in stats_files], ignore_index=True
                    )
                    if not stat_df.empty and "rmse" in stat_df.columns:
                        _rmse_bar_chart(
                            stat_df, date, comp, top_n,
                            outpath=maps_dir / f"{date}_{comp}_rmse_bar.png",
                        )
                except Exception as exc:
                    warnings.warn(f"  RMSE bar chart: {exc}", stacklevel=2)

    if verbose:
        print(f"\n  Maps   → {maps_dir}")
        print(f"  Profiles → {profiles_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IC Analysis Step 4: Generate spatial and vertical diagnostic plots."
    )
    parser.add_argument("--config", default=str(_SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--full-campaign", action="store_true")
    parser.add_argument("--date", default=None)
    parser.add_argument("--components", nargs="+", default=None)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--figure-outdir", default=None)
    args = parser.parse_args()
    run(
        config_path=Path(args.config),
        pilot_only=False if args.full_campaign else None,
        single_date=args.date,
        filter_components=args.components,
        top_n=args.top_n,
        figure_outdir=args.figure_outdir,
    )


if __name__ == "__main__":
    main()
