"""
plotting.py
===========
Step 9: Generate all diagnostic figures.

Figure suite per (variable, init_month, window)
------------------------------------------------
1.  Bias map         — model mean minus obs at lead 1  (if obs available)
2.  Adjustment maps  — D_ref, D_test  (window average)
3.  Paired diff map  — ΔD = D_test − D_ref  with significance stippling
4.  Zonal means      — latitude profile of ΔD
5.  RMSE bar chart   — top-N variables by window-mean ΔD
6.  Regional table   — compact global + Niño-3.4 bias statistics

Figures are saved under the configured figure output directory. Diagnostic
summary tables remain under ``{output_root}/summary_tables``.

Design notes
------------
- Cartopy is used for map projections when available; falls back to simple
  pcolormesh without projection if cartopy is not installed.
- Regional Niño-3.4 (5°S–5°N, 190–240°E) is computed inside the figure
  routine but kept separate from the spatial domain logic.
- Figure style mirrors the 5a_refactor aesthetic (clean, publication-ready).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.monthly_core import MonthlyConfig
from .config import DEFAULT_FIGURE_OUTDIR


# ── Try cartopy; fall back gracefully ────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

# ── Figure style ─────────────────────────────────────────────────────────────
FONTZ = 11
DPI   = 150
CMAP_BIAS = "RdBu_r"
CMAP_ADJ  = "RdBu_r"
CMAP_DIFF = "RdBu_r"

# Niño-3.4 region (for summary statistics only)
NINO34 = dict(lat_min=-5, lat_max=5, lon_min=190, lon_max=240)


# ===========================================================================
# Helper: symmetric color scale
# ===========================================================================

def _sym_vmax(da: xr.DataArray, pctile: float = 98.0) -> float:
    vals = da.values.ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1.0
    v = float(np.nanpercentile(np.abs(vals), pctile))
    return max(v, 1e-9)


# ===========================================================================
# Helper: map axes
# ===========================================================================

def _map_axes(fig, pos=111) -> plt.Axes:
    """Return a map Axes — cartopy Robinson if available, else plain."""
    if _HAS_CARTOPY:
        ax = fig.add_subplot(pos, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.4")
    else:
        ax = fig.add_subplot(pos)
    return ax


def _pcolormesh_map(ax, lon, lat, data, vmax, cmap, transform=None):
    """Wrapper around pcolormesh for cartopy and plain axes."""
    kw = dict(cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    if _HAS_CARTOPY and transform is not None:
        return ax.pcolormesh(lon, lat, data, transform=transform, **kw)
    return ax.pcolormesh(lon, lat, data, **kw)


def _add_coastlines(ax):
    if not _HAS_CARTOPY:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")


# ===========================================================================
# 1. Bias map (lead-1, absolute)
# ===========================================================================

def plot_bias_map(
    bias_da: xr.DataArray,
    title: str,
    outpath: Path,
    units: str = "",
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=DPI,
                           subplot_kw={"projection": ccrs.Robinson()} if _HAS_CARTOPY else {})
    vmax = _sym_vmax(bias_da)
    try:
        lat  = bias_da[lat_name].values
        lon  = bias_da[lon_name].values
        data = bias_da.values
        pcm  = _pcolormesh_map(
            ax, lon, lat, data, vmax, CMAP_BIAS,
            transform=ccrs.PlateCarree() if _HAS_CARTOPY else None,
        )
        _add_coastlines(ax)
    except Exception as exc:
        ax.text(0.5, 0.5, f"Cannot render map:\n{exc}",
                ha="center", va="center", transform=ax.transAxes)
        pcm = None

    if pcm is not None:
        cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.04)
        cbar.set_label(units, fontsize=FONTZ - 1)
    ax.set_title(title, fontsize=FONTZ)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 2. Adjustment / paired-diff map (generic 2-panel)
# ===========================================================================

def plot_two_panel_maps(
    da_left: xr.DataArray,
    da_right: xr.DataArray,
    title_left: str,
    title_right: str,
    suptitle: str,
    outpath: Path,
    units: str = "",
    significance_mask: Optional[xr.DataArray] = None,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> None:
    """Two-panel side-by-side comparison map."""
    proj = {"projection": ccrs.Robinson()} if _HAS_CARTOPY else {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=DPI, subplot_kw=proj)
    fig.suptitle(suptitle, fontsize=FONTZ + 1, y=1.01)

    vmax = max(_sym_vmax(da_left), _sym_vmax(da_right))

    for ax, da, title in zip(axes, [da_left, da_right], [title_left, title_right]):
        try:
            lat  = da[lat_name].values
            lon  = da[lon_name].values
            pcm  = _pcolormesh_map(
                ax, lon, lat, da.values, vmax, CMAP_ADJ,
                transform=ccrs.PlateCarree() if _HAS_CARTOPY else None,
            )
            _add_coastlines(ax)
        except Exception:
            pcm = None
        ax.set_title(title, fontsize=FONTZ)

    # Colorbar below both panels
    if pcm is not None:
        cbar_ax = fig.add_axes([0.2, -0.04, 0.6, 0.025])
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(units, fontsize=FONTZ - 1)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 3. Paired difference map with stippling
# ===========================================================================

def plot_paired_diff_map(
    paired_diff: xr.DataArray,
    title: str,
    outpath: Path,
    units: str = "",
    significance: Optional[xr.DataArray] = None,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> None:
    proj = {"projection": ccrs.Robinson()} if _HAS_CARTOPY else {}
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=DPI, subplot_kw=proj)
    vmax = _sym_vmax(paired_diff)

    try:
        lat  = paired_diff[lat_name].values
        lon  = paired_diff[lon_name].values
        pcm  = _pcolormesh_map(
            ax, lon, lat, paired_diff.values, vmax, CMAP_DIFF,
            transform=ccrs.PlateCarree() if _HAS_CARTOPY else None,
        )
        _add_coastlines(ax)

        # Stipple significant regions
        if significance is not None:
            sig_vals = significance.values
            sig_lat  = lat[np.any(sig_vals, axis=1)] if lat.ndim == 1 else lat
            sig_lon  = lon[np.any(sig_vals, axis=0)] if lon.ndim == 1 else lon
            try:
                lon2d, lat2d = np.meshgrid(lon, lat)
                mask = significance.values.astype(bool)
                stip_kw = {"s": 0.3, "c": "k", "alpha": 0.4, "linewidths": 0}
                if _HAS_CARTOPY:
                    ax.scatter(lon2d[mask], lat2d[mask], transform=ccrs.PlateCarree(), **stip_kw)
                else:
                    ax.scatter(lon2d[mask], lat2d[mask], **stip_kw)
            except Exception:
                pass

    except Exception as exc:
        ax.text(0.5, 0.5, f"Cannot render:\n{exc}", ha="center", va="center",
                transform=ax.transAxes)
        pcm = None

    if pcm is not None:
        cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.04)
        cbar.set_label(units, fontsize=FONTZ - 1)
    ax.set_title(title, fontsize=FONTZ)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 3b. Paired difference grid: lead windows x initialization months
# ===========================================================================

def plot_paired_diff_grid(
    results: Dict[int, Dict[str, Any]],
    boot_results: Dict[int, Dict[str, Any]],
    window_defs: Dict[str, Tuple[int, int]],
    title: str,
    outpath: Path,
    units: str = "",
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> None:
    """Plot all paired differences in one lead-window-by-init-month figure.

    May and November are ordered from left to right when present. Each row is
    one configured lead window and all panels share a symmetric color scale.
    Missing month/window combinations are shown as unavailable panels.
    """
    preferred_months = [5, 11]
    init_months = [m for m in preferred_months if m in results]
    init_months.extend(sorted(m for m in results if m not in init_months))
    window_names = [
        name
        for name in window_defs
        if any(
            isinstance(results.get(month, {}).get(name), dict)
            and results[month][name].get("paired_diff") is not None
            for month in init_months
        )
    ]
    if not init_months or not window_names:
        warnings.warn("paired-difference grid skipped: no plottable panels", stacklevel=2)
        return

    panel_fields = [
        results[month][name]["paired_diff"]
        for name in window_names
        for month in init_months
        if isinstance(results.get(month, {}).get(name), dict)
        and results[month][name].get("paired_diff") is not None
    ]
    vmax = max(_sym_vmax(field) for field in panel_fields)
    nrows, ncols = len(window_names), len(init_months)
    # A rectangular projection allows reliable latitude/longitude labels on
    # the outer panels. Centering at 180° also avoids splitting the Pacific.
    proj = (
        {"projection": ccrs.PlateCarree(central_longitude=180)}
        if _HAS_CARTOPY else {}
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 3.6 * nrows),
        dpi=DPI,
        subplot_kw=proj,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=FONTZ + 1)
    pcm = None

    for col, init_month in enumerate(init_months):
        month_label = {5: "May", 11: "November"}.get(
            init_month, f"Month {init_month:02d}"
        )
        axes[0, col].set_title(f"{month_label} initialization", fontsize=FONTZ)

        for row, window_name in enumerate(window_names):
            ax = axes[row, col]
            window_result = results.get(init_month, {}).get(window_name)
            paired = (
                window_result.get("paired_diff")
                if isinstance(window_result, dict)
                else None
            )
            lead_first, lead_last = window_defs[window_name]
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    f"Lead months {lead_first}–{lead_last}",
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=FONTZ,
                    transform=ax.transAxes,
                )

            if paired is None:
                ax.set_axis_off()
                ax.text(
                    0.5, 0.5, "Unavailable", ha="center", va="center",
                    transform=ax.transAxes,
                )
                continue

            try:
                lat = paired[lat_name].values
                lon = paired[lon_name].values
                pcm = _pcolormesh_map(
                    ax,
                    lon,
                    lat,
                    paired.values,
                    vmax,
                    CMAP_DIFF,
                    transform=ccrs.PlateCarree() if _HAS_CARTOPY else None,
                )
                if _HAS_CARTOPY:
                    ax.set_global()
                    ax.coastlines(linewidth=0.55, color="0.25")
                    gridliner = ax.gridlines(
                        crs=ccrs.PlateCarree(),
                        draw_labels=True,
                        linewidth=0.35,
                        color="0.45",
                        alpha=0.55,
                        linestyle="--",
                        x_inline=False,
                        y_inline=False,
                    )
                    gridliner.top_labels = False
                    gridliner.right_labels = False
                    gridliner.bottom_labels = row == nrows - 1
                    gridliner.left_labels = col == 0
                    gridliner.xlocator = mticker.FixedLocator(
                        [-120, -60, 0, 60, 120, 180]
                    )
                    gridliner.ylocator = mticker.FixedLocator(
                        [-60, -30, 0, 30, 60]
                    )
                    gridliner.xformatter = LONGITUDE_FORMATTER
                    gridliner.yformatter = LATITUDE_FORMATTER
                    gridliner.xlabel_style = {"size": FONTZ - 2}
                    gridliner.ylabel_style = {"size": FONTZ - 2}
                else:
                    _add_coastlines(ax)

                significance = (
                    boot_results.get(init_month, {})
                    .get(window_name, {})
                    .get("significant")
                )
                if significance is not None:
                    lon2d, lat2d = np.meshgrid(lon, lat)
                    mask = significance.values.astype(bool)
                    stip_kw = {
                        "s": 0.3, "c": "k", "alpha": 0.4, "linewidths": 0
                    }
                    if _HAS_CARTOPY:
                        ax.scatter(
                            lon2d[mask], lat2d[mask],
                            transform=ccrs.PlateCarree(), **stip_kw,
                        )
                    else:
                        ax.scatter(lon2d[mask], lat2d[mask], **stip_kw)
            except Exception as exc:
                ax.text(
                    0.5, 0.5, f"Cannot render:\n{exc}",
                    ha="center", va="center", transform=ax.transAxes,
                )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12,
                        hspace=0.16, wspace=0.08)
    if pcm is not None:
        cbar_ax = fig.add_axes([0.30, 0.035, 0.40, 0.015])
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(units, fontsize=FONTZ - 1)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 4. Zonal mean
# ===========================================================================

def plot_zonal_mean(
    paired_diff: xr.DataArray,
    title: str,
    outpath: Path,
    units: str = "",
    lat_name: str = "lat",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 6), dpi=DPI)
    try:
        zm = paired_diff.mean([d for d in paired_diff.dims if d != lat_name], skipna=True)
        lat_vals = zm[lat_name].values if lat_name in zm.coords else np.arange(len(zm))
        ax.plot(zm.values, lat_vals, color="steelblue", lw=1.8)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(f"Zonal mean ΔD  ({units})", fontsize=FONTZ - 1)
        ax.set_ylabel("Latitude (°N)", fontsize=FONTZ - 1)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    except Exception as exc:
        ax.text(0.5, 0.5, f"Zonal mean N/A:\n{exc}", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title(title, fontsize=FONTZ)
    ax.tick_params(labelsize=FONTZ - 2)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 5. Regional statistics table
# ===========================================================================

def build_summary_table(
    results: Dict[int, Dict[str, Any]],
    config: MonthlyConfig,
    variable: str,
) -> pd.DataFrame:
    """Compact table of global and Niño-3.4 ΔD statistics."""
    rows = []
    for init_month, month_results in results.items():
        season = {5: "May", 11: "November"}.get(init_month, f"M{init_month:02d}")
        for win_name, win_dict in month_results.items():
            if not isinstance(win_dict, dict):
                continue
            pd_da = win_dict.get("paired_diff")
            if pd_da is None:
                continue
            try:
                vals = pd_da.values.ravel()
                vals = vals[np.isfinite(vals)]
                global_mean = float(np.nanmean(vals))
                global_rmse = float(np.sqrt(np.nanmean(vals ** 2)))

                # Niño-3.4 subset
                nino_mean = np.nan
                try:
                    lat_name = "lat" if "lat" in pd_da.dims else pd_da.dims[-2]
                    lon_name = "lon" if "lon" in pd_da.dims else pd_da.dims[-1]
                    nino = pd_da.sel(
                        **{lat_name: slice(NINO34["lat_min"], NINO34["lat_max"]),
                           lon_name: slice(NINO34["lon_min"], NINO34["lon_max"])}
                    )
                    nino_mean = float(nino.mean(skipna=True))
                except Exception:
                    pass

                rows.append({
                    "variable":    variable,
                    "season":      season,
                    "window":      win_name,
                    "global_mean_dD": round(global_mean, 4),
                    "global_rmse_dD": round(global_rmse, 4),
                    "nino34_mean_dD": round(nino_mean, 4),
                })
            except Exception as exc:
                warnings.warn(f"summary_table: {win_name}: {exc}", stacklevel=2)

    return pd.DataFrame(rows)


# ===========================================================================
# Top-level figure generation
# ===========================================================================

def run_plotting(
    results: Dict[int, Dict[str, Any]],
    boot_results: Dict[int, Dict[str, Any]],
    config: MonthlyConfig,
    variable: str = "PRECT",
    figure_outdir: str | Path = DEFAULT_FIGURE_OUTDIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate the full figure suite for one variable.

    Parameters
    ----------
    results:
        Output of ``diagnostics.run_diagnostics()``.
    boot_results:
        Output of ``bootstrap.run_bootstrap()``.
    config:
        MonthlyConfig.
    variable:
        Variable native name.
    figure_outdir:
        Destination for figures. Defaults to the shared ESP-Lab web directory.

    Returns
    -------
    Summary statistics DataFrame.
    """
    try:
        var_spec = config.get_variable(variable)
        units = var_spec.plot_units
    except KeyError:
        units = ""

    experiments = list(config.experiments.keys())
    ref_label, test_label = experiments[0], experiments[1]

    out_root = Path(figure_outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for init_month, month_results in results.items():
        season_str = {5: "May", 11: "November"}.get(init_month, f"month{init_month:02d}")
        season_token = season_str.lower()

        if verbose:
            print("=" * 70)
            print(f"Step 9: Figures — {season_str}  ({variable})")
            print("=" * 70)

        # Lead-1 bias map
        for key, label in [
            ("model_ref_lead1",  f"Lead-1 mean  {ref_label}"),
            ("model_test_lead1", f"Lead-1 mean  {test_label}"),
        ]:
            da = month_results.get(key)
            if da is not None:
                plot_bias_map(
                    da,
                    title=f"{label}  |  {season_str}  lead=1  {variable}",
                    outpath=(
                        out_root
                        / f"fig_{variable.lower()}_monthly_drift_"
                          f"{season_token}_lead1_{key}.png"
                    ),
                    units=units,
                )

        for win_name, win_dict in month_results.items():
            if not isinstance(win_dict, dict):
                continue

            adj_ref  = win_dict.get("adj_ref")
            adj_test = win_dict.get("adj_test")
            paired   = win_dict.get("paired_diff")
            if paired is None:
                continue

            # Figure 1: Adjustment maps (2-panel)
            if adj_ref is not None and adj_test is not None:
                plot_two_panel_maps(
                    adj_ref, adj_test,
                    title_left=f"D  {ref_label}",
                    title_right=f"D  {test_label}",
                    suptitle=f"IC drift adjustment  |  {variable}  {season_str}  {win_name}",
                    outpath=(
                        out_root
                        / f"fig_{variable.lower()}_monthly_drift_"
                          f"{season_token}_{win_name}_adjustment_comparison.png"
                    ),
                    units=units,
                )

            # Figure 2: Zonal mean
            plot_zonal_mean(
                paired,
                title=f"Zonal mean ΔD  |  {variable}  {season_str}  {win_name}",
                outpath=(
                    out_root
                    / f"fig_{variable.lower()}_monthly_drift_"
                      f"{season_token}_{win_name}_zonal_mean.png"
                ),
                units=units,
            )

            if verbose:
                print(f"  {season_str}/{win_name}  → {out_root}")

            # Summary stats
            try:
                weighted_mean, weighted_rmse = _spatial_summary(paired)
                summary_rows.append({
                    "variable": variable,
                    "season":   season_str,
                    "window":   win_name,
                    "global_mean_dD": round(weighted_mean, 4),
                    "global_rmse_dD": round(weighted_rmse, 4),
                })
            except Exception:
                pass

    # One paired-difference figure for every lead window and initialization.
    paired_grid_path = (
        out_root / f"fig_{variable.lower()}_monthly_drift_paired_diff_grid.png"
    )
    plot_paired_diff_grid(
        results=results,
        boot_results=boot_results,
        window_defs=config.window_defs,
        title=f"ΔD = D({test_label}) − D({ref_label})  |  {variable}",
        outpath=paired_grid_path,
        units=units,
    )
    if verbose:
        print(f"\n  Paired-difference grid → {paired_grid_path}")

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_dir = Path(config.output_root) / "summary_tables"
        summary_dir.mkdir(parents=True, exist_ok=True)
        tbl_path = summary_dir / f"{variable.lower()}_monthly_spatial_drift_summary.csv"
        summary_df.to_csv(tbl_path, index=False)
        if verbose:
            print(f"\n  Summary table → {tbl_path}")
            print(summary_df.to_string(index=False))

    return summary_df
def _spatial_summary(field: xr.DataArray) -> tuple[float, float]:
    """Return latitude-area-weighted mean and RMS for rectilinear grids."""
    spatial_dims = [d for d in ("lat", "lon") if d in field.dims]
    if "lat" in field.dims:
        weights = np.cos(np.deg2rad(field["lat"])).clip(min=0)
        mean = field.weighted(weights).mean(spatial_dims, skipna=True)
        rms = np.sqrt((field ** 2).weighted(weights).mean(spatial_dims, skipna=True))
    else:
        dims = spatial_dims or list(field.dims)
        mean = field.mean(dims, skipna=True)
        rms = np.sqrt((field ** 2).mean(dims, skipna=True))
    return float(mean), float(rms)
