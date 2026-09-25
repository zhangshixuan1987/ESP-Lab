#!/usr/bin/env python3
"""Batch execution of Modes of Variability (MOV) teleconnection diagnostics.

Computes teleconnection NetCDF datasets and generates the 3 diagnostic figures:
1. Multi-panel teleconnection correlation maps
2. Teleconnection fidelity summary (leads vs. metrics)
3. Taylor diagram synthesis

Usage:
    python scripts/run_batch_mov_teleconnections.py
    python scripts/run_batch_mov_teleconnections.py --modes NPO NAO --variables TS PRECT
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
import xarray as xr

# Silence proj/gdal warnings if standard paths are present
proj_candidate = "/global/homes/z/zhan391/.conda/envs/e3sm_analysis/share/proj"
if os.path.isdir(proj_candidate) and "PROJ_DATA" not in os.environ:
    os.environ["PROJ_DATA"] = proj_candidate
gdal_candidate = "/global/homes/z/zhan391/.conda/envs/e3sm_analysis/share/gdal"
if os.path.isdir(gdal_candidate) and "GDAL_DATA" not in os.environ:
    os.environ["GDAL_DATA"] = gdal_candidate

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from esp_lab.leadtime_plot_utils import seasonal_label
from esp_lab import env_paths
from esp_lab.utils import colormap_utils as mycolors
from workflows.diagnostics import mov_teleconnections as mov_telecon

ALL_MODES = [
    "NAM", "NAO", "SAM", "PNA", "NPO", "EA",
    "SCA", "PSA1", "PSA2", "PDO", "NPGO", "AMO",
]
ALL_VARIABLES = [
    "TREFHT", "TS", "PRECT", "PSL", "SST", "H2OSNO", "H2OSOI",
]

DEFAULT_DIAG_ROOT = str(env_paths.s2d_diag_root())
DEFAULT_OUTPUT_DIR = str(env_paths.s2d_diag_root() / "multimodel" / "leadtime_telec")
DEFAULT_FIGURE_DIR = str(env_paths.figure_root() / "teleconnections")


def initialization_label(month: int) -> str:
    return pd.Timestamp(2000, int(month), 1).strftime("%b").upper()


def display_lead(lead: int) -> int:
    """Convert stored centered-season L coordinate to displayed lead months."""
    return int(lead) - 2


def draw_map(
    ax,
    da,
    *,
    pvalue=None,
    correlation_limits=(-1.0, 1.0),
    correlation_interval=0.1,
    correlation_cutoff=0.5,
    cmap_name="blue2red_acc",
    show_significant=True,
    alpha_sig=0.10,
    sig_hatch="...",
    coastline_color="0.25",
    coastline_width=0.5,
    border_color="0.45",
    border_width=0.2,
    gridline_color="0.65",
    gridline_width=0.25,
    gridline_alpha=0.4,
    gridline_style=":",
):
    lat = "lat" if "lat" in da.coords else "latitude"
    lon = "lon" if "lon" in da.coords else "longitude"
    lower, upper = correlation_limits
    interval = correlation_interval
    levels = np.arange(lower, upper + 0.5 * interval, interval)
    cmap = (
        mycolors.blue2red_acc_cmap(levels, correlation_cutoff)
        if cmap_name == "blue2red_acc" else plt.get_cmap(cmap_name)
    )
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    kwargs = dict(
        x=lon, y=lat, ax=ax, cmap=cmap, norm=norm,
        add_colorbar=False, transform=ccrs.PlateCarree() if HAVE_CARTOPY else None
    )
    image = da.plot(**{k: v for k, v in kwargs.items() if v is not None})

    if show_significant and pvalue is not None:
        sig = (pvalue.values <= alpha_sig).astype(float)
        if np.any(sig > 0):
            ax.contourf(
                da[lon].values, da[lat].values, sig,
                levels=[0.5, 1.5],
                colors="none",
                hatches=[sig_hatch],
                transform=ccrs.PlateCarree() if HAVE_CARTOPY else None,
            )

    if HAVE_CARTOPY:
        ax.set_global()
        ax.coastlines(linewidth=coastline_width, color=coastline_color)
        ax.add_feature(cfeature.BORDERS, linewidth=border_width, edgecolor=border_color)
        ax.gridlines(linewidth=gridline_width, color=gridline_color, alpha=gridline_alpha, linestyle=gridline_style)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    return image


def plot_correlation_maps(
    metrics_ds: xr.Dataset,
    config: Mapping[str, Any],
    mode: str,
    variable: str,
    out_file: Path,
) -> None:
    fontz = 18
    fs_suptitle = fontz * 1.0
    fs_group_header = fontz * 0.95
    fs_col_title = fontz * 0.9
    fs_cbar_label = fontz * 0.95
    fs_cbar_ticks = fontz * 0.9
    fs_axis_ticks = fontz * 0.9
    fs_badge = fontz * 0.9
    fs_unavailable = fontz * 0.9

    badge_loc = "lower left"
    badge_pad = 0.18
    badge_borderpad = 0.25
    badge_facecolor = (1.0, 1.0, 1.0, 0.85)
    badge_edgecolor = "0.4"
    badge_linewidth = 0.5
    badge_boxstyle = "round,pad=0.15"

    col_width = 3.3
    row_height = 1.75
    width_pad = 0.8
    height_pad = 2.2
    grid_left = 0.035
    grid_right = 0.985
    grid_bottom = 0.08
    grid_top = 0.90
    grid_hspace = 0.07
    grid_wspace = 0.03
    gap_ratio = 0.22

    lon_ticks = [-120, 0, 120]
    lat_ticks = [-60, -30, 0, 30, 60]
    coastline_color = "0.25"
    coastline_width = 0.5
    border_color = "0.45"
    border_width = 0.2
    gridline_color = "0.65"
    gridline_width = 0.25
    gridline_alpha = 0.4
    gridline_style = ":"

    group_header_y = 0.938
    group_line_y = 0.925
    group_line_color = "0.4"
    group_line_width = 1.2
    suptitle_y = 0.985

    colorbar_rect = [0.18, 0.024, 0.64, 0.016]
    colorbar_ticks = config["figures"].get("colorbar_ticks", (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9))
    correlation_limits = config["figures"].get("correlation_limits", (-1.0, 1.0))
    correlation_interval = config["figures"].get("correlation_interval", 0.1)
    correlation_cutoff = config["figures"].get("correlation_cutoff", 0.5)
    cmap_name = config["figures"].get("cmap", "blue2red_acc")

    show_significant = config["figures"].get("show_significant_only", True)
    alpha_sig = config["analysis"].get("alpha", 0.10)

    available_leads = [int(lead) for lead in metrics_ds.L.values]
    configured_leads = config["figures"].get("map_leads", None)
    map_leads = (
        available_leads if configured_leads is None
        else [int(lead) for lead in configured_leads if int(lead) in available_leads]
    )
    plot_systems = [
        s for s in config["selection"]["systems"]
        if s in set(metrics_ds.system.values.astype(str))
    ]
    plot_months = [
        int(m) for m in config["selection"]["init_months"]
        if int(m) in set(map(int, metrics_ds.init_month.values))
    ]
    figure_dpi = config["figures"].get("dpi", 300)
    index_display = mode

    variable_leads = [
        lead for lead in map_leads
        if not metrics_ds.model_correlation.sel(variable=variable, L=lead).isnull().all()
    ]
    if not variable_leads:
        print(f"Skipped correlation maps for {variable}: no valid leads")
        return

    nrows = len(variable_leads)
    nsys = len(plot_systems)
    group_width = nsys + 1
    width_ratios = []
    for month_index in range(len(plot_months)):
        width_ratios.extend([1.0] * group_width)
        if month_index < len(plot_months) - 1:
            width_ratios.append(gap_ratio)
    total_cols = len(width_ratios)
    projection = ccrs.PlateCarree() if HAVE_CARTOPY else None

    fig = plt.figure(figsize=(col_width * len(plot_months) * group_width + width_pad, row_height * nrows + height_pad))
    grid = fig.add_gridspec(
        nrows, total_cols, width_ratios=width_ratios,
        left=grid_left, right=grid_right, bottom=grid_bottom, top=grid_top,
        hspace=grid_hspace, wspace=grid_wspace,
    )
    axes = {}
    image = None

    for month_index, init_month in enumerate(plot_months):
        column_offset = month_index * (group_width + 1)
        for row, lead in enumerate(variable_leads):
            ax = fig.add_subplot(grid[row, column_offset], projection=projection)
            axes[(init_month, "Reference", lead)] = ax
            reference = reference_pvalue = None
            for system in plot_systems:
                candidate = metrics_ds.sel(
                    variable=variable, system=system, init_month=init_month, L=lead
                )
                if not candidate.observed_correlation.isnull().all():
                    reference = candidate.observed_correlation
                    reference_pvalue = candidate.observed_pvalue
                    break
            if reference is None:
                if HAVE_CARTOPY:
                    ax.set_global()
                    ax.coastlines(linewidth=coastline_width, color=coastline_color)
                ax.text(
                    0.5, 0.5, "Unavailable", transform=ax.transAxes,
                    ha="center", va="center", color="0.45", fontsize=fs_unavailable,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                drawn = draw_map(
                    ax, reference, pvalue=reference_pvalue,
                    correlation_limits=correlation_limits,
                    correlation_interval=correlation_interval,
                    correlation_cutoff=correlation_cutoff,
                    cmap_name=cmap_name,
                    show_significant=show_significant,
                    alpha_sig=alpha_sig,
                )
                image = drawn if image is None else image

            if HAVE_CARTOPY:
                ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                ax.tick_params(
                    labelsize=fs_axis_ticks, length=2, pad=1,
                    labelbottom=(row == nrows - 1),
                    labelleft=True,
                )
            if row == 0:
                ax.set_title("Observed Reference", fontsize=fs_col_title, fontweight="bold", pad=6)
            season_str = seasonal_label(init_month, display_lead(lead))
            badge_text = f"L{display_lead(lead)}: {season_str}"
            at = AnchoredText(
                badge_text, loc=badge_loc,
                prop=dict(size=fs_badge, weight="bold", family="sans-serif"),
                frameon=True, pad=badge_pad, borderpad=badge_borderpad,
            )
            at.patch.set(
                facecolor=badge_facecolor,
                edgecolor=badge_edgecolor,
                linewidth=badge_linewidth,
                boxstyle=badge_boxstyle,
            )
            ax.add_artist(at)

        for system_index, system in enumerate(plot_systems):
            column = column_offset + 1 + system_index
            for row, lead in enumerate(variable_leads):
                ax = fig.add_subplot(grid[row, column], projection=projection)
                axes[(init_month, system, lead)] = ax
                subset = metrics_ds.sel(
                    variable=variable, system=system, init_month=init_month, L=lead
                )
                if subset.model_correlation.isnull().all():
                    if HAVE_CARTOPY:
                        ax.set_global()
                        ax.coastlines(linewidth=coastline_width, color=coastline_color)
                    ax.text(
                        0.5, 0.5, "Unavailable", transform=ax.transAxes,
                        ha="center", va="center", color="0.45", fontsize=fs_unavailable,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    drawn = draw_map(
                        ax, subset.model_correlation, pvalue=subset.model_pvalue,
                        correlation_limits=correlation_limits,
                        correlation_interval=correlation_interval,
                        correlation_cutoff=correlation_cutoff,
                        cmap_name=cmap_name,
                        show_significant=show_significant,
                        alpha_sig=alpha_sig,
                    )
                    image = drawn if image is None else image
                if HAVE_CARTOPY:
                    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
                    ax.yaxis.set_major_formatter(LatitudeFormatter())
                    ax.tick_params(
                        labelsize=fs_axis_ticks, length=2, pad=1,
                        labelbottom=(row == nrows - 1),
                        labelleft=False,
                    )
                if row == 0:
                    disp_name = mov_telecon.E3SM_CASES.get(system, {}).get("display_name", system)
                    ax.set_title(disp_name, fontsize=fs_col_title, fontweight="bold", pad=6)

    for month_index, init_month in enumerate(plot_months):
        left_ax = axes[(init_month, "Reference", variable_leads[0])]
        right_ax = axes[(init_month, plot_systems[-1], variable_leads[0])]
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        center_x = (left_pos.x0 + right_pos.x1) / 2
        month_name = initialization_label(init_month)
        fig.text(
            center_x, group_header_y,
            f"{month_name.upper()} INITIALIZATION",
            ha="center", va="center",
            fontsize=fs_group_header, fontweight="bold",
            color="0.15",
        )
        fig.add_artist(plt.Line2D(
            [left_pos.x0, right_pos.x1], [group_line_y, group_line_y],
            transform=fig.transFigure,
            color=group_line_color, linewidth=group_line_width,
        ))

    if image is None:
        plt.close(fig)
        print(f"Skipped {variable}: no image drawn")
        return

    significance_note = f"stippled where p ≤ {alpha_sig:.2f}" if show_significant else ""
    detrend_note = "linear detrend" if config["analysis"]["detrend"] else "no detrend"
    notes = ", ".join(filter(None, [detrend_note, significance_note]))
    title_suffix = f"\n({notes})" if notes else ""
    fig.suptitle(
        f"{index_display}–{variable} Teleconnection Correlation Patterns{title_suffix}",
        y=suptitle_y, fontsize=fs_suptitle, fontweight="bold",
    )

    colorbar_ax = fig.add_axes(colorbar_rect)
    colorbar = fig.colorbar(
        image, cax=colorbar_ax, orientation="horizontal",
        ticks=colorbar_ticks,
    )
    colorbar.set_label(f"Teleconnection Correlation $r$ ({index_display} vs. {variable})", fontsize=fs_cbar_label, fontweight="bold")
    colorbar.ax.tick_params(labelsize=fs_cbar_ticks, length=3)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_fidelity_summary(
    metrics_ds: xr.Dataset,
    config: Mapping[str, Any],
    mode: str,
    variable: str,
    out_file: Path,
) -> None:
    fontz = 14
    fs_suptitle = fontz * 1.0
    fs_col_title = fontz * 0.95
    fs_axis_label = fontz * 0.9
    fs_axis_ticks = fontz * 0.85
    fs_legend = fontz * 0.85

    figsize = (8.5, 10.5)
    layout_rect = (0.02, 0.055, 0.98, 0.96)
    suptitle_y = 0.985
    col_title_pad = 10
    lead_pad = 0.8

    series_linewidth = 1.5
    series_markersize = 6
    grid_color = "0.88"
    grid_linewidth = 0.6
    grid_linestyle = "-"
    grid_alpha = 0.7

    ref_amplitude = 1.0
    ref_amplitude_style = dict(color="0.55", linestyle="--", linewidth=0.9, alpha=0.7)
    ref_correlation = 0.0
    ref_correlation_style = dict(color="0.65", linestyle=":", linewidth=0.8, alpha=0.6)
    ref_sign_agreement = 0.5
    ref_sign_agreement_style = dict(color="0.65", linestyle=":", linewidth=0.8, alpha=0.6)

    legend_loc = "lower center"
    legend_bbox = (0.5, 0.015)
    legend_framealpha = 0.9
    legend_edgecolor = "0.8"

    system_styles = {
        "E3SM-4DEnVarOcn": {"short_name": "4DEnVarOcn", "color": "tab:purple", "marker": "o"},
        "E3SM-FOSIRL": {"short_name": "FOSIRL", "color": "black", "marker": "s"},
        "E3SM-Reanalysis": {"short_name": "Reanalysis", "color": "tab:blue", "marker": "D"},
        "BruteForce": {"short_name": "BruteForce", "color": "tab:green", "marker": "^"},
        "E3SM-BruteForce": {"short_name": "BruteForce", "color": "tab:green", "marker": "^"},
    }

    summary_metrics = ["pattern_correlation", "centered_rmse", "amplitude_ratio", "sign_agreement_fraction"]
    ylabels = ["Pattern correlation", "Centered RMSE", "Amplitude ratio", "Sign agreement"]
    plot_systems = [
        s for s in config["selection"]["systems"]
        if s in set(metrics_ds.system.values.astype(str))
    ]
    plot_months = [
        int(m) for m in config["selection"]["init_months"]
        if int(m) in set(map(int, metrics_ds.init_month.values))
    ]
    figure_dpi = config["figures"].get("dpi", 300)
    index_display = mode

    valid_leads = [
        int(lead) - 2 for lead in metrics_ds.L.values
        if np.any(np.isfinite(metrics_ds.pattern_correlation.sel(variable=variable, L=lead)))
    ]

    nrows = len(summary_metrics)
    ncols = len(plot_months)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for col, init_month in enumerate(plot_months):
        month_label = pd.Timestamp(2000, int(init_month), 1).strftime("%b")
        col_title = f"{month_label} Initialization"

        for row, (metric, ylabel) in enumerate(zip(summary_metrics, ylabels)):
            ax = axes[row, col]
            if metric == "amplitude_ratio":
                ax.axhline(ref_amplitude, **ref_amplitude_style, zorder=1)
            elif metric == "pattern_correlation":
                ax.axhline(ref_correlation, **ref_correlation_style, zorder=1)
            elif metric == "sign_agreement_fraction":
                ax.axhline(ref_sign_agreement, **ref_sign_agreement_style, zorder=1)

            for system in metrics_ds.system.values:
                sys_str = str(system)
                if sys_str not in mov_telecon.E3SM_CASES:
                    continue
                series = metrics_ds[metric].sel(
                    variable=variable, system=sys_str, init_month=init_month
                )
                if series.isnull().all():
                    continue

                style = METHOD_STYLES.get(sys_str, {
                    "color": mov_telecon.E3SM_CASES[sys_str]["color"],
                    "marker": "o",
                    "short_name": mov_telecon.E3SM_CASES[sys_str]["display_name"],
                })

                leads_display = series.L.astype(int) - 2
                ax.plot(
                    leads_display, series,
                    marker=style["marker"],
                    markersize=series_markersize,
                    linewidth=series_linewidth,
                    color=style["color"],
                    label=mov_telecon.E3SM_CASES[sys_str]["display_name"],
                    zorder=3,
                )

            ax.grid(True, linestyle=grid_linestyle, color=grid_color, linewidth=grid_linewidth, alpha=grid_alpha)
            ax.tick_params(labelsize=fs_axis_ticks)

            if row == 0:
                ax.set_title(col_title, fontsize=fs_col_title, fontweight="bold", pad=col_title_pad)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=fs_axis_label, fontweight="bold")
            if row == nrows - 1:
                ax.set_xlabel("Lead time (months)", fontsize=fs_axis_label)
                if valid_leads:
                    ax.set_xticks(valid_leads)
                    ax.set_xlim(valid_leads[0] - lead_pad, valid_leads[-1] + lead_pad)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc=legend_loc,
            ncol=len(handles),
            fontsize=fs_legend,
            frameon=True,
            framealpha=legend_framealpha,
            edgecolor=legend_edgecolor,
            bbox_to_anchor=legend_bbox,
        )

    fig.suptitle(
        f"{index_display}–{variable} Teleconnection Fidelity Summary",
        fontsize=fs_suptitle,
        fontweight="bold",
        y=suptitle_y,
    )
    plt.tight_layout(rect=layout_rect)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_taylor_diagram(
    metrics_ds: xr.Dataset,
    config: Mapping[str, Any],
    mode: str,
    variable: str,
    out_file: Path,
) -> None:
    fontz = 14
    fs_suptitle = fontz * 1.0
    fs_panel_title = fontz * 0.95
    fs_axis_label = fontz * 0.9
    fs_corr_arc = fontz * 0.85
    fs_ticks = fontz * 0.85
    fs_legend = fontz * 0.9
    fs_lead_annot = fontz * 0.8
    fs_crmse_clabel = fontz * 0.8

    r_max = 1.6
    r_ticks = [0.5, 1.0, 1.5]
    crmse_levels = [0.25, 0.50, 0.75, 1.00, 1.25]
    corr_label_offset = 1.13

    grid_color = "0.85"
    grid_linewidth = 0.6
    grid_linestyle = "-"
    crmse_contour_color = "0.70"
    crmse_contour_style = ":"
    crmse_contour_width = 0.7
    crmse_contour_alpha = 0.7
    crmse_clabel_color = "0.45"

    ref_circle_color = "0.35"
    ref_circle_style = "--"
    ref_circle_width = 0.9
    ref_circle_alpha = 0.6
    ref_marker = "*"
    ref_marker_size = 11
    ref_marker_color = "gold"
    ref_marker_edgecolor = "black"
    ref_marker_edgewidth = 0.9

    connect_leads = True
    traj_linewidth = 1.1
    traj_line_alpha = 0.35
    marker_size_range = (5.0, 10.5)
    marker_alpha_range = (0.45, 0.95)

    annot_boxstyle = "round,pad=0.12"
    annot_facecolor = "white"
    annot_alpha = 0.75
    annot_xytext = (0, 6)

    panel_figsize_width_per_month = 7.5
    panel_figsize_height = 7.0
    panel_title_pad = 24
    suptitle_y = 0.98
    adjust_bottom = 0.14
    adjust_top = 0.86
    adjust_wspace = 0.32

    legend_loc = "lower center"
    legend_bbox = (0.5, 0.01)
    legend_framealpha = 0.9
    legend_edgecolor = "0.8"

    METHOD_STYLES = {
        "E3SM-FOSIRL": {"short_name": "FOSIRL", "color": "black", "marker": "s"},
        "E3SM-Reanalysis": {"short_name": "Reanalysis", "color": "tab:blue", "marker": "D"},
        "E3SM-4DEnVarOcn": {"short_name": "4DEnVarOcn", "color": "tab:purple", "marker": "o"},
        "BruteForce": {"short_name": "BruteForce", "color": "tab:green", "marker": "^"},
        "E3SM-BruteForce": {"short_name": "BruteForce", "color": "tab:green", "marker": "^"},
    }

    figure_dpi = config["figures"].get("dpi", 300)
    plot_months = [
        int(m) for m in config["selection"]["init_months"]
        if int(m) in set(map(int, metrics_ds.init_month.values))
    ]
    index_display = mode

    min_corr = float(metrics_ds.pattern_correlation.sel(variable=variable).min(skipna=True))
    thetamax = 180 if min_corr < 0 else 90

    if thetamax == 90:
        corr_ticks = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        th_ref = np.linspace(0, np.pi / 2, 100)
        corr_label_angle = np.deg2rad(45)
        corr_label_rotation = -45
    else:
        corr_ticks = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
        th_ref = np.linspace(0, np.pi, 100)
        corr_label_angle = np.deg2rad(90)
        corr_label_rotation = 0

    fig = plt.figure(figsize=(panel_figsize_width_per_month * len(plot_months), panel_figsize_height))
    legend_handles = []
    legend_labels = []

    for idx, init_month in enumerate(plot_months):
        ax = fig.add_subplot(1, len(plot_months), idx + 1, projection="polar")
        ax.set_thetamin(0)
        ax.set_thetamax(thetamax)
        ax.set_ylim(0, r_max)

        ax.grid(True, color=grid_color, linewidth=grid_linewidth, linestyle=grid_linestyle)

        theta_ticks = np.arccos(corr_ticks)
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels([f"{c:.2g}" for c in corr_ticks], fontsize=fs_ticks, color="0.25")

        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f"{t:.1f}" for t in r_ticks], fontsize=fs_ticks, color="0.25")

        ax.set_xlabel(
            r"Normalized Standard Deviation ($\sigma_m / \sigma_o$)",
            fontsize=fs_axis_label,
            labelpad=12,
            color="0.2",
        )

        ax.text(
            corr_label_angle,
            r_max * corr_label_offset,
            "Pattern Correlation",
            fontsize=fs_corr_arc,
            ha="center",
            va="bottom",
            rotation=corr_label_rotation,
            color="0.3",
        )

        rs = np.linspace(0, r_max, 150)
        thetas = np.linspace(0, np.deg2rad(thetamax), 150)
        R_grid, TH_grid = np.meshgrid(rs, thetas)
        X_grid = R_grid * np.cos(TH_grid)
        Y_grid = R_grid * np.sin(TH_grid)
        E_grid = np.sqrt((X_grid - 1.0) ** 2 + Y_grid ** 2)
        cs = ax.contour(
            TH_grid,
            R_grid,
            E_grid,
            levels=crmse_levels,
            colors=crmse_contour_color,
            linestyles=crmse_contour_style,
            linewidths=crmse_contour_width,
            alpha=crmse_contour_alpha,
        )
        ax.clabel(cs, inline=True, fontsize=fs_crmse_clabel, fmt="%.2f", colors=crmse_clabel_color)

        ax.plot(
            th_ref,
            np.ones_like(th_ref),
            color=ref_circle_color,
            linestyle=ref_circle_style,
            linewidth=ref_circle_width,
            alpha=ref_circle_alpha,
        )

        ref_line, = ax.plot(
            0,
            1.0,
            marker=ref_marker,
            markersize=ref_marker_size,
            color=ref_marker_color,
            markeredgecolor=ref_marker_edgecolor,
            markeredgewidth=ref_marker_edgewidth,
            linestyle="none",
            zorder=6,
        )
        if idx == 0:
            legend_handles.append(ref_line)
            legend_labels.append("Reference (Obs)")

        for system in metrics_ds.system.values:
            sys_str = str(system)
            if sys_str not in METHOD_STYLES:
                continue
            style = METHOD_STYLES[sys_str]
            sub = metrics_ds.sel(variable=variable, system=sys_str, init_month=init_month)
            r_vals = sub.amplitude_ratio.values
            p_vals = sub.pattern_correlation.values
            leads = sub.L.values

            valid = (
                np.isfinite(r_vals)
                & np.isfinite(p_vals)
                & (p_vals >= (-1.0 if thetamax == 180 else 0.0))
                & (p_vals <= 1.0)
            )
            if not np.any(valid):
                continue

            thetas_mod = np.arccos(p_vals[valid])
            rs_mod = r_vals[valid]
            leads_valid = leads[valid]
            n_pts = len(leads_valid)

            color = style["color"]
            marker_shape = style["marker"]
            short_name = style["short_name"]

            if connect_leads:
                ax.plot(
                    thetas_mod,
                    rs_mod,
                    color=color,
                    linestyle="-",
                    linewidth=traj_linewidth,
                    alpha=traj_line_alpha,
                    zorder=4,
                )

            s_min, s_max = marker_size_range
            a_min, a_max = marker_alpha_range
            sizes = np.linspace(s_min, s_max, n_pts)
            alphas = np.linspace(a_min, a_max, n_pts)

            for k, (th, r, l, sz, al) in enumerate(zip(thetas_mod, rs_mod, leads_valid, sizes, alphas)):
                lead_label = display_lead(l)
                ax.plot(
                    th,
                    r,
                    marker=marker_shape,
                    markersize=sz,
                    color=color,
                    alpha=al,
                    markeredgecolor="black" if (k == 0 or k == n_pts - 1) else color,
                    markeredgewidth=0.9 if (k == 0 or k == n_pts - 1) else 0.4,
                    zorder=5,
                )
                if k == 0 or k == n_pts - 1:
                    ax.annotate(
                        f"L{lead_label}",
                        xy=(th, r),
                        xytext=annot_xytext,
                        textcoords="offset points",
                        fontsize=fs_lead_annot,
                        ha="center",
                        va="bottom",
                        color=color,
                        fontweight="bold",
                        bbox=dict(boxstyle=annot_boxstyle, facecolor=annot_facecolor, edgecolor="none", alpha=annot_alpha),
                        zorder=7,
                    )

            if idx == 0:
                h_line, = ax.plot(
                    [], [],
                    marker=marker_shape,
                    markersize=8,
                    color=color,
                    linestyle="-",
                    linewidth=traj_linewidth,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                )
                legend_handles.append(h_line)
                legend_labels.append(short_name)

        ax.set_title(
            f"{initialization_label(init_month)} initialization",
            fontsize=fs_panel_title,
            fontweight="bold",
            pad=panel_title_pad,
        )

    fig.suptitle(
        f"{index_display}–{variable} Teleconnection Fidelity",
        fontsize=fs_suptitle,
        fontweight="bold",
        y=suptitle_y,
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc=legend_loc,
        ncol=len(legend_handles),
        fontsize=fs_legend,
        frameon=True,
        framealpha=legend_framealpha,
        edgecolor=legend_edgecolor,
        bbox_to_anchor=legend_bbox,
    )

    plt.subplots_adjust(bottom=adjust_bottom, top=adjust_top, wspace=adjust_wspace)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def _pair_config(mode: str, var: str, args: argparse.Namespace) -> dict[str, Any]:
    """Teleconnection configuration for one (mode, variable) pair."""
    return {
        "paths": {
            "diag_root": str(args.diag_root),
            "output_dir": str(args.output_dir),
            "figure_dir": str(args.figure_dir),
        },
        "inputs": {
            "mode": "auto",
            "initialization_years": (1980, int(args.year_end)),
            "year_end": args.year_end,
            "raw_model_root": "/global/cfs/cdirs/e3sm/S2S2D/post_process",
            "observation_root": "/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series",
            "ensemble_member_count": 10,
            "monthly_nlead": 24,
            "workers": args.workers,
        },
        "selection": {
            "upstream_mode": mode,
            "downstream_variable": var,
            "systems": args.selected_systems,
            "init_months": [5, 11],
            "leads": "all",
            "year_end": args.year_end,
            "verification_years": (1981, int(args.year_end)),
            "climatology_years": (1981, 2010),
        },
        "regrid": {
            "target_dlat": 5.0,
            "target_dlon": 5.0,
            "method": "conservative",
            "periodic": True,
        },
        "analysis": {
            "detrend": True,
            "alpha": 0.10,
            "minimum_years": 20,
            "latitude_bounds": (-80.0, 80.0),
            "area_weighted": True,
            "sign_agreement_threshold": 0.0,
        },
        "cache": {
            "mode": "rebuild" if args.force else "auto",
            "allow_ambiguous_matches": True,
        },
        "figures": {
            "map_leads": None,
            "dpi": 300,
            "cmap": "blue2red_acc",
            "correlation_limits": (-1.0, 1.0),
            "correlation_interval": 0.1,
            "correlation_cutoff": 0.5,
            "colorbar_ticks": (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9),
            "show_significant_only": True,
        },
    }


def process_pair(mode: str, var: str, args: argparse.Namespace) -> bool:
    figure_dir = Path(args.figure_dir)

    nc_file = mov_telecon.mov_teleconnection_cache_path(_pair_config(mode, var, args))
    fig1 = figure_dir / f"teleconnection_{mode}_{var}_correlation_reference_comparison.png"
    fig2 = figure_dir / f"teleconnection_{mode}_{var}_summary.png"
    fig3 = figure_dir / f"teleconnection_{mode}_{var}_taylor_diagram.png"

    all_exist = nc_file.exists() and fig1.exists() and fig2.exists() and fig3.exists()
    if all_exist and not args.force:
        print(f"[{mode} - {var}] All products exist. Skipping.")
        return True

    print(f"\n{'='*70}\nProcessing [{mode} - {var}] (nc: {nc_file.exists()}, fig1: {fig1.exists()}, fig2: {fig2.exists()}, fig3: {fig3.exists()})\n{'='*70}")
    t0 = time.time()

    config = _pair_config(mode, var, args)

    try:
        # Step 1: Ensure NetCDF dataset exists or compute it
        if nc_file.exists() and not args.force:
            print(f"Loading existing dataset: {nc_file}")
            metrics_ds = xr.open_dataset(nc_file)
        else:
            print(f"Computing teleconnection dataset...")
            inventory = mov_telecon.ensure_upstream_products(config)
            metrics_ds, out_path, status = mov_telecon.ensure_mov_teleconnection_dataset(config, inventory=inventory)
            print(f"Dataset ready ({status}): {out_path}")

        # Step 2: Render missing figures
        if not fig1.exists() or args.force:
            print(f"Rendering Figure 1 (Correlation Reference Comparison)...")
            t_fig = time.time()
            plot_correlation_maps(metrics_ds, config, mode, var, fig1)
            print(f"Figure 1 saved ({time.time() - t_fig:.1f}s): {fig1}")
        else:
            print(f"Figure 1 already exists: {fig1}")

        if not fig2.exists() or args.force:
            print(f"Rendering Figure 2 (Fidelity Summary)...")
            t_fig = time.time()
            plot_fidelity_summary(metrics_ds, config, mode, var, fig2)
            print(f"Figure 2 saved ({time.time() - t_fig:.1f}s): {fig2}")
        else:
            print(f"Figure 2 already exists: {fig2}")

        if not fig3.exists() or args.force:
            print(f"Rendering Figure 3 (Taylor Diagram Synthesis)...")
            t_fig = time.time()
            plot_taylor_diagram(metrics_ds, config, mode, var, fig3)
            print(f"Figure 3 saved ({time.time() - t_fig:.1f}s): {fig3}")
        else:
            print(f"Figure 3 already exists: {fig3}")

        metrics_ds.close()
        del metrics_ds
        plt.close("all")
        gc.collect()

        print(f"Completed [{mode} - {var}] in {time.time() - t0:.1f}s")
        return True

    except Exception as e:
        print(f"ERROR processing [{mode} - {var}]: {e}")
        import traceback
        traceback.print_exc()
        plt.close("all")
        gc.collect()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch MOV teleconnections")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES, help="Modes of variability")
    parser.add_argument("--variables", nargs="+", default=ALL_VARIABLES, help="Downstream variables")
    parser.add_argument(
        "--systems",
        "--system",
        dest="systems",
        nargs="+",
        default=["E3SM-FOSIRL", "E3SM-Reanalysis", "E3SM-4DEnVarOcn"],
        help="Systems/experiments to process (e.g. BruteForce, 4DEnVarOcn, JRA55_FOSIRL, Reanalysis)",
    )
    parser.add_argument("--diag-root", default=DEFAULT_DIAG_ROOT, help="Diagnostic root directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="NetCDF output directory")
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR, help="Figure output directory")
    parser.add_argument("--year-end", type=int, default=2011, help="Verification end year")
    parser.add_argument("--workers", type=int, default=4, help="Worker count")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing files")
    args = parser.parse_args()

    system_alias_map = {
        "4DEnVarOcn": "E3SM-4DEnVarOcn",
        "E3SM-4DEnVarOcn": "E3SM-4DEnVarOcn",
        "JRA55_FOSIRL": "E3SM-FOSIRL",
        "FOSIRL": "E3SM-FOSIRL",
        "E3SM-FOSIRL": "E3SM-FOSIRL",
        "Reanalysis": "E3SM-Reanalysis",
        "BruteForce": "E3SM-Reanalysis",
        "E3SM-BruteForce": "E3SM-Reanalysis",
    }
    args.selected_systems = [system_alias_map.get(s, s) for s in args.systems]

    # If output_dir was left as default and a single experiment was requested, route to that experiment's dir
    if args.output_dir == DEFAULT_OUTPUT_DIR and len(args.selected_systems) == 1:
        single_sys = args.selected_systems[0]
        sys_dir_map = {
            "E3SM-4DEnVarOcn": "4DEnVarOcn",
            "E3SM-FOSIRL": "JRA55_FOSIRL",
            "E3SM-Reanalysis": "Reanalysis",
            "E3SM-BruteForce": "Reanalysis",
            "BruteForce": "Reanalysis",
        }
        exp_name = sys_dir_map.get(single_sys, single_sys)
        args.output_dir = str(Path(args.diag_root) / exp_name / "leadtime_telec")

    # Create directories if they do not exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    pairs = [(m, v) for m in args.modes for v in args.variables]
    total_pairs = len(pairs)
    print(f"Starting batch MOV teleconnections run at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Total pairs to evaluate: {total_pairs}")
    print(f"Modes ({len(args.modes)}): {args.modes}")
    print(f"Variables ({len(args.variables)}): {args.variables}")
    print(f"NetCDF Dir: {args.output_dir}")
    print(f"Figure Dir: {args.figure_dir}")
    print("-" * 70)

    start_time = time.time()
    success_count = 0
    fail_count = 0
    skipped_count = 0

    for idx, (mode, var) in enumerate(pairs, start=1):
        nc_file = mov_telecon.mov_teleconnection_cache_path(_pair_config(mode, var, args))
        fig1 = Path(args.figure_dir) / f"teleconnection_{mode}_{var}_correlation_reference_comparison.png"
        fig2 = Path(args.figure_dir) / f"teleconnection_{mode}_{var}_summary.png"
        fig3 = Path(args.figure_dir) / f"teleconnection_{mode}_{var}_taylor_diagram.png"
        
        is_done = nc_file.exists() and fig1.exists() and fig2.exists() and fig3.exists()
        if is_done and not args.force:
            skipped_count += 1
            print(f"[{idx}/{total_pairs}] Mode: {mode:<4} Var: {var:<7} -> ALREADY COMPLETED (Skipping)")
            continue

        print(f"\n[{idx}/{total_pairs}] Mode: {mode:<4} Var: {var:<7} -> PROCESSING")
        ok = process_pair(mode, var, args)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Batch execution finished at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Total: {total_pairs}, Skipped (pre-existing): {skipped_count}, Newly completed: {success_count}, Failed: {fail_count}")
    print(f"Total time elapsed: {elapsed / 60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()

