"""
plotting.py
===========
Step 11: Produce daily diagnostic products.

Figure suite per (variable, init_month, window)
------------------------------------------------
1.  Day-1 bias / mean map
2.  Adjustment maps (two-panel: D_ref, D_test for weekly window)
3.  Paired diff map (ΔD = D_test − D_ref with significance stippling)
4.  Zonal mean profile (latitude vs ΔD)
5.  Daily regional time series plot (days 1–84 evolution curve)
6.  Compact regional statistics table
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

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

from esp_lab.diagnostics.daily_core import DailyDriftConfig
from .config import DEFAULT_FIGURE_OUTDIR

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False

FONTZ = 11
DPI   = 150
CMAP_ADJ  = "RdBu_r"
CMAP_DIFF = "RdBu_r"


def _sym_vmax(da: xr.DataArray, pctile: float = 98.0) -> float:
    vals = da.values.ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1.0
    v = float(np.nanpercentile(np.abs(vals), pctile))
    return max(v, 1e-9)


def _pcolormesh_map(ax, lon, lat, data, vmax, cmap, transform=None):
    kw = dict(cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    if _HAS_CARTOPY and transform is not None:
        return ax.pcolormesh(lon, lat, data, transform=transform, **kw)
    return ax.pcolormesh(lon, lat, data, **kw)


def plot_daily_timeseries(
    ref_ts: xr.DataArray,
    test_ts: xr.DataArray,
    ref_label: str,
    test_label: str,
    title: str,
    outpath: Path,
    units: str = "",
) -> None:
    """Plot days 1–84 regional evolution curve."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=DPI)
    try:
        days_ref = ref_ts.d.values if "d" in ref_ts.coords else range(1, len(ref_ts) + 1)
        days_test = test_ts.d.values if "d" in test_ts.coords else range(1, len(test_ts) + 1)

        ax.plot(days_ref, ref_ts.values, label=f"D({ref_label})", color="crimson", lw=1.8)
        ax.plot(days_test, test_ts.values, label=f"D({test_label})", color="navy", lw=1.8)
        ax.axhline(0, color="k", lw=0.8, ls="--")

        ax.set_xlabel("Forecast day", fontsize=FONTZ - 1)
        ax.set_ylabel(f"Adjustment  ({units})", fontsize=FONTZ - 1)
        ax.legend(fontsize=FONTZ - 2, loc="upper right")
    except Exception as exc:
        ax.text(0.5, 0.5, f"Time series error:\n{exc}", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title, fontsize=FONTZ)
    ax.tick_params(labelsize=FONTZ - 2)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_two_panel_maps(
    da_left: xr.DataArray,
    da_right: xr.DataArray,
    title_left: str,
    title_right: str,
    suptitle: str,
    outpath: Path,
    units: str = "",
) -> None:
    proj = {"projection": ccrs.Robinson()} if _HAS_CARTOPY else {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=DPI, subplot_kw=proj)
    fig.suptitle(suptitle, fontsize=FONTZ + 1, y=1.01)

    vmax = max(_sym_vmax(da_left), _sym_vmax(da_right))

    for ax, da, title in zip(axes, [da_left, da_right], [title_left, title_right]):
        try:
            lat  = da["lat"].values if "lat" in da.coords else range(da.shape[-2])
            lon  = da["lon"].values if "lon" in da.coords else range(da.shape[-1])
            pcm  = _pcolormesh_map(
                ax, lon, lat, da.values, vmax, CMAP_ADJ,
                transform=ccrs.PlateCarree() if _HAS_CARTOPY else None,
            )
            if _HAS_CARTOPY:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.4")
        except Exception:
            pcm = None
        ax.set_title(title, fontsize=FONTZ)

    if pcm is not None:
        cbar_ax = fig.add_axes([0.2, -0.04, 0.6, 0.025])
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(units, fontsize=FONTZ - 1)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_paired_diff_map(
    paired_diff: xr.DataArray,
    title: str,
    outpath: Path,
    units: str = "",
    significance: Optional[xr.DataArray] = None,
) -> None:
    proj = {"projection": ccrs.Robinson()} if _HAS_CARTOPY else {}
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=DPI, subplot_kw=proj)
    vmax = _sym_vmax(paired_diff)

    try:
        lat  = paired_diff["lat"].values if "lat" in paired_diff.coords else range(paired_diff.shape[-2])
        lon  = paired_diff["lon"].values if "lon" in paired_diff.coords else range(paired_diff.shape[-1])
        pcm  = _pcolormesh_map(
            ax, lon, lat, paired_diff.values, vmax, CMAP_DIFF,
            transform=ccrs.PlateCarree() if _HAS_CARTOPY else None,
        )
        if _HAS_CARTOPY:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.4")

        if significance is not None:
            mask = significance.values.astype(bool)
            lon2d, lat2d = np.meshgrid(lon, lat)
            stip_kw = {"s": 0.3, "c": "k", "alpha": 0.4, "linewidths": 0}
            if _HAS_CARTOPY:
                ax.scatter(lon2d[mask], lat2d[mask], transform=ccrs.PlateCarree(), **stip_kw)
            else:
                ax.scatter(lon2d[mask], lat2d[mask], **stip_kw)
    except Exception as exc:
        ax.text(0.5, 0.5, f"Render error: {exc}", ha="center", va="center", transform=ax.transAxes)
        pcm = None

    if pcm is not None:
        cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.04)
        cbar.set_label(units, fontsize=FONTZ - 1)
    ax.set_title(title, fontsize=FONTZ)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def run_plotting(
    results: Dict[int, Dict[str, Any]],
    boot_results: Dict[int, Dict[str, Any]],
    config: DailyDriftConfig,
    variable: str = "TREFHT",
    figure_outdir: str | Path = DEFAULT_FIGURE_OUTDIR,
    verbose: bool = True,
) -> pd.DataFrame:
    try:
        var_spec = config.get_variable(variable)
        units = var_spec.plot_units
    except KeyError:
        units = ""

    experiments = list(config.experiments.keys())
    ref_label, test_label = experiments[0], experiments[1]

    out_root = Path(figure_outdir)
    summary_rows = []

    for init_month, month_results in results.items():
        season_str = {5: "May", 11: "November"}.get(init_month, f"month{init_month:02d}")
        fig_dir = out_root / season_str

        if verbose:
            print("=" * 70)
            print(f"Daily Step 11: Figures — {season_str}  ({variable})")
            print("=" * 70)

        ref_ts = month_results.get("ref_ts_d")
        test_ts = month_results.get("test_ts_d")
        if ref_ts is not None and test_ts is not None:
            plot_daily_timeseries(
                ref_ts, test_ts,
                ref_label=ref_label, test_label=test_label,
                title=f"Days 1–84 adjustment evolution  |  {variable}  {season_str}",
                outpath=fig_dir / f"daily_timeseries_{variable}_{season_str}.png",
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

            win_dir = fig_dir / win_name
            sig = boot_results.get(init_month, {}).get(win_name, {}).get("significant")

            # Figure 1: Adjustment maps
            if adj_ref is not None and adj_test is not None:
                plot_two_panel_maps(
                    adj_ref, adj_test,
                    title_left=f"D  {ref_label}",
                    title_right=f"D  {test_label}",
                    suptitle=f"Daily IC drift adjustment  |  {variable}  {season_str}  {win_name}",
                    outpath=win_dir / f"adj_comparison_{variable}_{season_str}_{win_name}.png",
                    units=units,
                )

            # Figure 2: Paired diff map with stippling
            plot_paired_diff_map(
                paired,
                title=f"ΔD = D({test_label}) − D({ref_label})  |  {variable}  {season_str}  {win_name}",
                outpath=win_dir / f"paired_diff_{variable}_{season_str}_{win_name}.png",
                units=units,
                significance=sig,
            )

            if verbose:
                print(f"  {season_str}/{win_name}  → {win_dir}")

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

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_dir = Path(config.output_root) / "summary_tables"
        summary_dir.mkdir(parents=True, exist_ok=True)
        tbl_path = summary_dir / f"daily_summary_{variable}.csv"
        summary_df.to_csv(tbl_path, index=False)
        if verbose:
            print(f"\n  Daily summary table → {tbl_path}")

    return summary_df
def _spatial_summary(field: xr.DataArray) -> tuple[float, float]:
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
