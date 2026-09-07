"""Reusable labeling helpers for lead-time skill figures."""

import numpy as np
import xarray as xr


SEASON_NAMES = ("DJF", "MAM", "JJA", "SON")


def seasonal_label(init_month, lead, *, season_names=SEASON_NAMES):
    """Return the three-month season centered on a monthly lead."""
    if not 1 <= int(init_month) <= 12:
        raise ValueError("init_month must be between 1 and 12")
    center_month = ((int(init_month) + int(lead)) % 12) + 1
    return season_names[(center_month % 12) // 3]


def style_global_map_axis(ax, *, row, column, nrows, label_size, style):
    """Apply consistent ticks, labels, and gridlines to a global map panel."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    ax.set_xticks(style.get("lon_ticks", [-120, -60, 0, 60, 120]), crs=ccrs.PlateCarree())
    ax.set_yticks(style.get("lat_ticks", [-60, -30, 0, 30, 60]), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(
        LongitudeFormatter(zero_direction_label=True)
        if row == nrows - 1 else plt.NullFormatter()
    )
    ax.yaxis.set_major_formatter(
        LatitudeFormatter() if column == 0 else plt.NullFormatter()
    )
    ax.tick_params(
        labelsize=label_size,
        length=style["tick_length"],
        width=style["tick_width"],
        top=False,
        right=False,
        labelbottom=(row == nrows - 1),
        labelleft=(column == 0),
    )
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=style["grid_linewidth"],
        color=style["grid_color"],
        alpha=style["grid_alpha"],
        linestyle="-",
        draw_labels=False,
    )


def add_lead_badge(ax, text, *, font_size, location="lower left", style=None):
    """Add a consistent framed lead/season label to a map panel."""
    from matplotlib.offsetbox import AnchoredText

    style = dict(style or {})
    badge = AnchoredText(
        text,
        loc=location,
        prop={
            "size": font_size,
            "weight": style.get("font_weight", "bold"),
            "family": style.get("font_family", "monospace"),
        },
        frameon=True,
        pad=style.get("pad", 0.2),
        borderpad=style.get("borderpad", 0.3),
    )
    badge.patch.set(
        facecolor=style.get("facecolor", (1, 1, 1, 0.78)),
        edgecolor=style.get("edgecolor", "0.25"),
        linewidth=style.get("linewidth", 0.4),
        boxstyle=style.get("boxstyle", "round,pad=0.1"),
    )
    ax.add_artist(badge)
    return badge


def add_missing_map_panel(
    figure,
    *,
    nrows,
    ncols,
    subplot,
    title,
    message,
    font_size,
):
    """Create a map-shaped placeholder for an unavailable seasonal panel."""
    import cartopy.crs as ccrs

    ax = figure.add_subplot(nrows, ncols, subplot, projection=ccrs.PlateCarree())
    ax.set_aspect("auto")
    ax.set_global()
    ax.coastlines(linewidth=0.6, color="0.45")
    ax.set_facecolor("0.94")
    ax.set_title(title, fontsize=font_size, fontweight="bold")
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=font_size * 0.8,
        color="0.35",
        fontweight="bold",
    )
    return ax


def add_acc_comparison_markers(
    ax,
    base_skill,
    superiority_fraction,
    lead_index,
    *,
    area,
    latitude,
    longitude_2d,
    latitude_2d,
    style,
    font_size,
    font_weight,
    label_bbox,
):
    """Plot ACC comparison markers and annotate valid-area percentages."""
    values = xr.where(
        ~base_skill.corr.isel(L=lead_index).isnull(),
        superiority_fraction.isel(L=lead_index),
        np.nan,
    )
    valid_panel = values.notnull() & (abs(latitude) < style["latlim"])
    valid_area = area.where(valid_panel).sum()

    smyle_better = values > (1 - style["sig_level"])
    e3sm_better = values < style["sig_level"]
    smyle_fraction = float(area.where(smyle_better & valid_panel, 0).sum() / valid_area)
    e3sm_fraction = float(area.where(e3sm_better & valid_panel, 0).sum() / valid_area)

    stride = style["marker_stride"]
    display_lon = longitude_2d[::stride, ::stride]
    display_lat = latitude_2d[::stride, ::stride]
    latitude_mask = abs(display_lat) < style["latlim"]

    display_mask = np.asarray(smyle_better)[::stride, ::stride] & latitude_mask
    ax.scatter(
        display_lon[display_mask], display_lat[display_mask],
        facecolor="none", edgecolor=style["marker_color"],
        s=style["open_marker_size"], linewidth=style["marker_linewidth"], zorder=10,
    )
    display_mask = np.asarray(e3sm_better)[::stride, ::stride] & latitude_mask
    ax.scatter(
        display_lon[display_mask], display_lat[display_mask],
        facecolor=style["marker_color"], edgecolor=style["marker_color"],
        s=style["filled_marker_size"], linewidth=0, zorder=10,
    )
    ax.text(
        0.98, 0.05,
        f"({smyle_fraction * 100:3.1f}%/{e3sm_fraction * 100:3.1f}%)",
        fontsize=font_size, fontweight=font_weight, bbox=label_bbox,
        zorder=10, transform=ax.transAxes, ha="right", va="bottom",
    )
    return smyle_fraction, e3sm_fraction


__all__ = [
    "SEASON_NAMES",
    "add_acc_comparison_markers",
    "add_lead_badge",
    "add_missing_map_panel",
    "seasonal_label",
    "style_global_map_axis",
]
