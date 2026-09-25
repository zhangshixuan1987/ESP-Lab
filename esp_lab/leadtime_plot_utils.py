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


def add_pointwise_significance_markers(
    ax,
    skill,
    lead_index,
    *,
    longitude_2d,
    latitude_2d,
    significance_level,
    style,
    font_size,
    font_weight="bold",
    label_bbox=None,
    label_position="lower right",
):
    """Stipple significant ACC cells and label their valid-area percentage.

    ``label_position`` is ``"lower right"`` or ``"lower left"`` (use the latter
    when the panel's lead label already occupies the lower-right corner).

    The percentage uses cosine-latitude weights and includes only cells with
    the complete sample cohort recorded by the skill dataset.
    """
    panel = skill.isel(L=lead_index)
    valid = panel.corr.notnull()
    if {"valid_sample_count", "sample_count"}.issubset(panel):
        valid = valid & (panel.valid_sample_count == panel.sample_count)

    latitude = panel.lat
    latitude_limit = float(style.get("latlim", 80))
    within_latitude = abs(latitude) < latitude_limit
    valid_panel = valid & within_latitude
    significant = valid_panel & (panel.pval < float(significance_level))

    area_weight = np.cos(np.deg2rad(latitude)).clip(min=0)
    valid_area = area_weight.where(valid_panel).sum()
    significant_area = area_weight.where(significant).sum()
    denominator = float(valid_area)
    significant_fraction = (
        float(significant_area) / denominator if denominator > 0 else np.nan
    )

    stride = int(style.get("marker_stride", 1))
    if stride < 1:
        raise ValueError("marker_stride must be at least 1")
    display_lon = np.asarray(longitude_2d)[::stride, ::stride]
    display_lat = np.asarray(latitude_2d)[::stride, ::stride]
    display_mask = np.asarray(significant)[::stride, ::stride]
    ax.scatter(
        display_lon[display_mask], display_lat[display_mask],
        facecolor=style.get("marker_color", "black"),
        edgecolor=style.get("marker_color", "black"),
        s=style.get("marker_size", 5), linewidth=0,
        zorder=10,
    )
    if style.get("annotate_percentage", True):
        label = "(n/a)" if not np.isfinite(significant_fraction) else (
            f"({significant_fraction * 100:3.1f}% sig.)"
        )
        x, ha = {"lower right": (0.98, "right"), "lower left": (0.02, "left")}[label_position]
        ax.text(
            x, 0.05, label,
            fontsize=font_size, fontweight=font_weight,
            bbox=dict(label_bbox or {}), zorder=10,
            transform=ax.transAxes, ha=ha, va="bottom",
        )
    return significant_fraction


__all__ = [
    "SEASON_NAMES",
    "add_acc_comparison_markers",
    "add_pointwise_significance_markers",
    "add_lead_badge",
    "add_missing_map_panel",
    "seasonal_label",
    "style_global_map_axis",
]


# CONUS zoom used by the lead-time RMSE notebooks (1b_atm layout): each season
# is verified in forecast year 1 (upper block) and forecast year 2 (lower block).
CONUS_EXTENT = (-125.0, -66.0, 24.0, 50.0)
CONUS_SEASON_LEADS = (
    {"season": "JJA", "init_month": 5, "year_leads": (3, 15)},
    {"season": "SON", "init_month": 5, "year_leads": (6, 18)},
    {"season": "DJF", "init_month": 11, "year_leads": (3, 15)},
    {"season": "MAM", "init_month": 11, "year_leads": (6, 18)},
)
_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def subset_conus(data, extent=CONUS_EXTENT):
    """Return ``data`` cut to the CONUS box, with longitudes in -180..180."""
    lon0, lon1, lat0, lat1 = extent
    if float(data.lon.max()) > 180:
        data = data.assign_coords(lon=((data.lon + 180) % 360) - 180).sortby("lon")
    data = data.sel(lon=slice(lon0, lon1))
    lat_slice = slice(lat0, lat1)
    if data.lat.size > 1 and float(data.lat[0]) > float(data.lat[-1]):
        lat_slice = slice(lat1, lat0)
    return data.sel(lat=lat_slice)


def conus_season_specs(init_months):
    """Return the CONUS season rows available for the configured start months."""
    return [spec for spec in CONUS_SEASON_LEADS if spec["init_month"] in set(init_months)]


def plot_conus_rmse_panels(
    fields_by_model,
    *,
    model_names,
    init_months,
    levels,
    colorbar_label,
    title,
    cmap="YlOrRd",
    extend="max",
    tick_decimals=1,
    column_width=3.9,
    missing_message="No data",
):
    """Draw the CONUS RMSE layout of ``1b_atm_leadtime_rmse_skill_map``.

    ``fields_by_model[model][init_month]`` is a (L, lat, lon) map with seasonal
    lead coordinates (3 = first season). Columns are models; rows are the
    seasons of ``CONUS_SEASON_LEADS`` for forecast years 1 and 2. A lead the
    data lacks, or one with no finite values, is drawn as a grey panel showing
    ``missing_message``. Returns the figure.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    from matplotlib.colors import BoundaryNorm
    from matplotlib.offsetbox import AnchoredText

    specs = conus_season_specs(init_months)
    if not specs:
        raise ValueError(f"No CONUS season rows are defined for init_months={list(init_months)}")
    models = list(fields_by_model)
    levels = np.asarray(levels, dtype=float)
    color_map = plt.get_cmap(cmap)
    norm = BoundaryNorm(levels, color_map.N, extend=extend)
    font = 14
    projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(max(column_width * len(models), 8.0), 19.5))
    fig.suptitle(title, fontsize=font * 1.3, fontweight="bold", y=0.985)
    grid = fig.add_gridspec(
        nrows=9, ncols=len(models), height_ratios=[1, 1, 1, 1, 0.35, 1, 1, 1, 1],
        top=0.915, bottom=0.055, left=0.065, right=0.975, hspace=0.22, wspace=0.09,
    )
    banner = dict(boxstyle="round,pad=0.28", facecolor="#f0f4f8", edgecolor="#b0c4de", alpha=0.92)
    for y, text in ((0.945, "Forecast Year 1 (Leads 1–4 seasons)"), (0.490, "Forecast Year 2 (Leads 13–16 seasons)")):
        fig.text(0.52, y, text, ha="center", va="center", fontsize=font * 1.15, fontweight="bold", bbox=banner)

    mappable = None
    for year_index in range(2):
        for row, spec in enumerate(specs):
            grid_row = row + (0 if year_index == 0 else 5)
            lead = spec["year_leads"][year_index]
            month = spec["init_month"]
            badge_text = f"Lead-{lead - 2} {spec['season']} ({_MONTH_NAMES[month - 1]} init)"
            for column, model in enumerate(models):
                ax = fig.add_subplot(grid[grid_row, column], projection=projection)
                ax.set_aspect("auto")
                if row == 0:
                    ax.set_title(model_names.get(model, model), fontsize=font, fontweight="bold", pad=5)
                data = fields_by_model[model].get(month)
                panel = (
                    subset_conus(data.sel(L=lead))
                    if data is not None and lead in data.L.values else None
                )
                if panel is not None and bool(np.isfinite(panel).any()):
                    mappable = ax.pcolormesh(
                        panel.lon, panel.lat, panel, shading="nearest", cmap=color_map,
                        norm=norm, rasterized=True, transform=projection,
                    )
                else:
                    ax.set_facecolor("0.94")
                    ax.text(0.5, 0.5, missing_message, transform=ax.transAxes, ha="center",
                            va="center", fontsize=font * 0.8, color="0.35", fontweight="bold")
                ax.set_extent(CONUS_EXTENT, crs=projection)
                for feature, kwargs in (
                    (cfeature.COASTLINE, {"linewidth": 0.6}),
                    (cfeature.BORDERS, {"linewidth": 0.4}),
                    (cfeature.STATES, {"linewidth": 0.25, "edgecolor": "0.35"}),
                ):
                    try:
                        ax.add_feature(feature, **kwargs)
                    except Exception as err:  # offline Natural Earth data
                        print(f"Skipping Cartopy feature {feature}: {err}")
                ax.set_xticks([-120, -105, -90, -75], crs=projection)
                ax.set_yticks([25, 35, 45], crs=projection)
                ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                ax.tick_params(
                    labelsize=font - 3.5, length=2.5, width=0.5, top=False, right=False,
                    labelbottom=grid_row == 8, labelleft=column == 0,
                )
                badge = AnchoredText(
                    badge_text, loc="lower left", frameon=True, pad=0.18, borderpad=0.25,
                    prop=dict(size=font - 4.5, weight="bold", family="sans-serif"),
                )
                badge.patch.set(boxstyle="round,pad=0.2", facecolor="white", edgecolor="lightgray", alpha=0.85)
                ax.add_artist(badge)

    if mappable is not None:
        colorbar = fig.colorbar(
            mappable, cax=fig.add_axes([0.25, 0.020, 0.5, 0.012]),
            orientation="horizontal", extend=extend,
        )
        colorbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{tick_decimals}f"))
        colorbar.set_label(colorbar_label, fontsize=font, fontweight="bold")
        colorbar.ax.tick_params(labelsize=font - 2.5)
    return fig
