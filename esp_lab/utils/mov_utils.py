"""Helpers for extra-tropical modes-of-variability diagnostics.

This single-file utility module groups map/projection, plotting, metric, and
small workflow helpers used by the MOV analysis notebook.
"""

import numpy as np
import matplotlib.path as mpath
import matplotlib.ticker as mticker

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
except Exception as error:  # pragma: no cover - depends on optional cartopy stack
    ccrs = None
    cfeature = None
    LATITUDE_FORMATTER = None
    LONGITUDE_FORMATTER = None
    CARTOPY_IMPORT_ERROR = error
else:
    CARTOPY_IMPORT_ERROR = None


DEFAULT_PROJECTION_BY_MODE = {
    "NAM": "north_polar",
    "SAM": "south_polar",
    "NAO": "atlantic",
    "EA": "atlantic",
    "SCA": "atlantic",
    "PNA": "north_pacific",
    "PSA1": "south_polar",
    "PSA2": "south_polar",
    "NPO": "north_pacific",
    "PDO": "north_pacific",
    "NPGO": "north_pacific",
    # AMO spans the tropical through subpolar Atlantic.  An Albers map with a
    # rectangular geographic extent clips the northwest corner of this tall
    # domain, which can hide physically important EOF lobes near 40N, 70W.
    "AMO": "platecarree",
}


def has_cartopy():
    return ccrs is not None

def make_projection(
    mode,
    projection_by_mode=None,
    albers_projection_by_mode=None,
    use_cartopy=True,
):
    """Return a Cartopy projection and symbolic projection name for a mode.

    Parameters
    ----------
    mode : str
        Mode name, e.g., "NPO", "NAO", "SCA".
    projection_by_mode : dict, optional
        Mapping from mode name to projection family.
    albers_projection_by_mode : dict, optional
        Mapping from mode name to projection keyword arguments.  Historical
        name retained for notebook compatibility.
    use_cartopy : bool
        If False, return plain matplotlib axes.
    """
    if ccrs is None or not use_cartopy:
        return None, None

    mode = mode.upper().strip()
    projection_by_mode = projection_by_mode or DEFAULT_PROJECTION_BY_MODE
    albers_projection_by_mode = albers_projection_by_mode or {}

    projection_name = projection_by_mode.get(mode, "platecarree")

    if projection_name in {"north_pacific", "atlantic", "south_pacific"}:
        default_albers = {
            "north_pacific": {
                "central_longitude": 180,
                "central_latitude": 40,
                "standard_parallels": (20, 60),
            },
            "atlantic": {
                "central_longitude": -30,
                "central_latitude": 50,
                "standard_parallels": (30, 70),
            },
            "south_pacific": {
                "central_longitude": -120,
                "central_latitude": -50,
                "standard_parallels": (-65, -30),
            },
        }

        params = {
            **default_albers[projection_name],
            **albers_projection_by_mode.get(mode, {}),
        }

        return ccrs.AlbersEqualArea(**params), projection_name

    if projection_name == "north_polar":
        return (
            ccrs.NorthPolarStereo(**albers_projection_by_mode.get(mode, {})),
            projection_name,
        )

    if projection_name == "south_polar":
        return (
            ccrs.SouthPolarStereo(**albers_projection_by_mode.get(mode, {})),
            projection_name,
        )

    if projection_name == "pacific":
        return ccrs.PlateCarree(central_longitude=180), projection_name

    if projection_name == "pacific_global":
        return ccrs.Robinson(central_longitude=180), projection_name

    if projection_name == "atlantic_global":
        return ccrs.Robinson(central_longitude=0), projection_name

    return ccrs.PlateCarree(), projection_name

def data_crs():
    return ccrs.PlateCarree() if ccrs is not None else None


def mode_extent(mode, pattern):
    """Return lon/lat extent for the mode using the pattern coordinates."""
    lon_min = float(pattern.lon.min())
    lon_max = float(pattern.lon.max())
    lat_min = float(pattern.lat.min())
    lat_max = float(pattern.lat.max())
    if mode == "NAM":
        return [-180, 180, max(20, lat_min), 90]
    if mode in {"SAM", "PSA1", "PSA2"}:
        return [-180, 180, -90, min(-20, lat_max)]
    if mode == "PNA":
        return [120, 240, 15, min(85, lat_max)]
    if mode in {"NPO", "PDO", "NPGO"}:
        return [120, 240, 15, 75]
    return [lon_min, lon_max, lat_min, lat_max]


def tick_values(projection_name, settings):
    """Return longitude/latitude ticks with robust fallback.

    New preferred keys:
        longitude_ticks, latitude_ticks

    Legacy projection-specific keys are still supported for compatibility.
    """
    if projection_name == "atlantic":
        if (
            "atlantic_longitude_ticks" in settings
            and "atlantic_latitude_ticks" in settings
        ):
            return (
                settings["atlantic_longitude_ticks"],
                settings["atlantic_latitude_ticks"],
            )

    if projection_name == "north_pacific":
        if (
            "north_pacific_longitude_ticks" in settings
            and "north_pacific_latitude_ticks" in settings
        ):
            return (
                settings["north_pacific_longitude_ticks"],
                settings["north_pacific_latitude_ticks"],
            )

    if projection_name in {"pacific_global", "atlantic_global"}:
        if (
            "global_longitude_ticks" in settings
            and "global_latitude_ticks" in settings
        ):
            return (
                settings["global_longitude_ticks"],
                settings["global_latitude_ticks"],
            )

    if "longitude_ticks" in settings and "latitude_ticks" in settings:
        return settings["longitude_ticks"], settings["latitude_ticks"]

    return (
        settings.get("longitude_ticks", np.arange(-180, 181, 60)),
        settings.get("latitude_ticks", np.arange(-90, 91, 20)),
    )

def set_extent(ax, projection_name, extent, data_projection):
    if projection_name in {"pacific_global", "atlantic_global"}:
        ax.set_global()
        return
    ax.set_extent(extent, crs=data_projection)


def add_map_features(ax, projection_name, settings):
    """Add mode-aware land/coast features."""
    if cfeature is None:
        ax.coastlines(linewidth=settings["coastline_linewidth"])
        return

    if projection_name in {"pacific_global", "atlantic_global"}:
        ax.add_feature(
            cfeature.LAND,
            facecolor=settings["global_land_facecolor"],
            edgecolor=settings["global_land_edgecolor"],
            linewidth=settings["global_land_linewidth"],
            zorder=settings["global_land_zorder"],
        )
        ax.coastlines(
            linewidth=settings["global_coastline_linewidth"],
            color=settings["global_coastline_color"],
            zorder=settings["global_coastline_zorder"],
        )
        return

    ax.coastlines(linewidth=settings["coastline_linewidth"])


def format_longitude_label(lon):
    lon = float(lon)
    if np.isclose(abs(lon), 180):
        return "180°"
    if np.isclose(lon, 0):
        return "0°"
    suffix = "E" if lon > 0 else "W"
    return f"{abs(lon):g}°{suffix}"


def format_latitude_label(lat):
    lat = float(lat)
    if np.isclose(lat, 0):
        return "0°"
    suffix = "N" if lat > 0 else "S"
    return f"{abs(lat):g}°{suffix}"


def add_polar_longitude_labels(
    ax,
    mode,
    extent,
    data_projection,
    *,
    longitude_ticks,
    label_offset,
    fontsize,
    color="0.25",
    skip_longitudes=None,
    label_position="data_edge",
    axes_radius=0.56,
    central_longitude=0.0,
):
    """Draw longitude labels around polar panels."""
    if mode not in {"NAM", "SAM", "PSA1", "PSA2"}:
        return
    skip_longitudes = [] if skip_longitudes is None else skip_longitudes

    if label_position == "outside_axes":
        for lon in longitude_ticks:
            if any(np.isclose(lon, skip_lon) for skip_lon in skip_longitudes):
                continue
            # South polar maps put 180 deg near the bottom and 0 deg near
            # the top.  North polar maps use the opposite visual orientation.
            if mode == "NAM":
                angle = np.deg2rad(float(lon) - float(central_longitude) - 90.0)
            else:
                angle = np.deg2rad(90.0 - (float(lon) - float(central_longitude)))
            ax.text(
                0.5 + float(axes_radius) * np.cos(angle),
                0.5 + float(axes_radius) * np.sin(angle),
                format_longitude_label(lon),
                transform=data_projection,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                clip_on=False,
                zorder=22,
            )
        return

    if mode == "NAM":
        label_lat = extent[2] + label_offset
        va = "top"
    else:
        label_lat = extent[3] - label_offset
        va = "bottom"

    for lon in longitude_ticks:
        if any(np.isclose(lon, skip_lon) for skip_lon in skip_longitudes):
            continue
        ax.text(
            float(lon),
            label_lat,
            format_longitude_label(lon),
            transform=data_projection,
            ha="center",
            va=va,
            fontsize=fontsize,
            color=color,
            clip_on=False,
        )


def add_polar_latitude_labels(
    ax,
    mode,
    *,
    latitude_ticks,
    label_longitude,
    data_projection,
    fontsize,
    color="0.25",
    axes_angle_degrees=None,
    label_position="radial",
    side_x=1.035,
):
    """Draw latitude-ring labels inside polar panels."""
    if mode not in {"NAM", "SAM", "PSA1", "PSA2"}:
        return

    if label_position == "side":
        labels = [format_latitude_label(lat) for lat in latitude_ticks]
        if not labels:
            return
        x_position = float(side_x)
        y_positions = np.linspace(0.35, 0.65, len(labels))
        for y, label in zip(y_positions, labels):
            ax.text(
                x_position,
                y,
                label,
                transform=data_projection,
                ha="left",
                va="center",
                fontsize=fontsize,
                color=color,
                clip_on=False,
                zorder=21,
            )
        return

    if axes_angle_degrees is not None:
        angle = np.deg2rad(float(axes_angle_degrees))
        if mode == "NAM":
            max_radius_lat = 20.0
            radial = (90.0 - np.asarray(latitude_ticks, dtype=float)) / (
                90.0 - max_radius_lat
            )
        else:
            max_radius_lat = -20.0
            radial = (np.asarray(latitude_ticks, dtype=float) + 90.0) / (
                max_radius_lat + 90.0
            )

        for lat, radius_fraction in zip(latitude_ticks, radial):
            if not np.isfinite(radius_fraction):
                continue
            radius = 0.5 * float(np.clip(radius_fraction, 0.0, 1.0))
            ax.text(
                0.5 + radius * np.cos(angle),
                0.5 + radius * np.sin(angle),
                format_latitude_label(lat),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                clip_on=False,
                zorder=21,
            )
        return

    for lat in latitude_ticks:
        if mode == "NAM" and lat >= 89:
            continue
        if mode != "NAM" and lat <= -89:
            continue
        ax.text(
            float(label_longitude),
            float(lat),
            format_latitude_label(lat),
            transform=data_projection,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            clip_on=True,
            zorder=21,
        )


def apply_polar_circular_boundary(ax, projection_name, *, enabled=True):
    if projection_name not in {"north_polar", "south_polar"} or not enabled:
        return
    theta = np.linspace(0, 2 * np.pi, 181)
    vertices = np.column_stack([np.sin(theta), np.cos(theta)]) * 0.5 + 0.5
    ax.set_boundary(mpath.Path(vertices), transform=ax.transAxes)


def _as_label_list(values, labels=None):
    """Return string labels matching the given tick values."""
    if labels is not None:
        return [str(label) for label in labels]
    return [str(value) for value in values]


def _draw_manual_lonlat_labels(
    ax,
    *,
    row,
    col,
    section_last_rows,
    extent,
    data_projection,
    settings,
    fontsize,
):
    """Draw box-parallel manual lon/lat labels for projected regional maps.

    These labels are intentionally placed in axes coordinates, not map/data
    coordinates. This keeps them horizontal and parallel to the subplot box.
    """

    show_left = (
        settings.get("draw_left_labels", True)
        and (col == 0 or not settings.get("label_left_column_only", True))
    )
    show_bottom = (
        settings.get("draw_bottom_labels", True)
        and (
            row in section_last_rows
            or not settings.get("label_bottom_row_only", True)
        )
    )

    longitude_ticks = np.asarray(settings.get("longitude_ticks", []), dtype=float)
    latitude_ticks = np.asarray(settings.get("latitude_ticks", []), dtype=float)

    longitude_labels = _as_label_list(
        longitude_ticks, settings.get("longitude_ticklabels")
    )
    latitude_labels = _as_label_list(
        latitude_ticks, settings.get("latitude_ticklabels")
    )

    label_color = settings.get("manual_label_color", "0.15")
    label_fontsize = settings.get("manual_label_fontsize", fontsize)

    if show_left and latitude_labels:
        y_positions = settings.get(
            "manual_lat_label_y_positions",
            np.linspace(0.35, 0.65, len(latitude_labels)),
        )
        x_position = settings.get("manual_lat_label_x", -0.035)

        for y, label in zip(y_positions, latitude_labels):
            ax.text(
                x_position,
                y,
                label,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=label_fontsize,
                color=label_color,
                rotation=0,
                clip_on=False,
                zorder=20,
            )

    if show_bottom and longitude_labels:
        x_positions = settings.get(
            "manual_lon_label_x_positions",
            np.linspace(0.25, 0.75, len(longitude_labels)),
        )
        y_position = settings.get("manual_lon_label_y", -0.065)

        for x, label in zip(x_positions, longitude_labels):
            ax.text(
                x,
                y_position,
                label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=label_fontsize,
                color=label_color,
                rotation=0,
                clip_on=False,
                zorder=20,
            )

def configure_gridlines(
    ax,
    *,
    mode,
    projection_name,
    row,
    col,
    section_last_rows,
    extent,
    data_projection,
    settings,
    fontsize,
):
    """Configure mode-aware gridlines and coordinate labels for a panel.

    Label strategies:
      - cartopy: use Cartopy Gridliner labels.
      - pacific: use Cartopy gridlines, but draw lon/lat labels manually.
      - polar: use manual polar longitude labels.
    """
    is_polar = projection_name in {"north_polar", "south_polar"}
    label_strategy = settings.get("label_strategy", "cartopy")
    longitude_ticks, latitude_ticks = tick_values(projection_name, settings)

    draw_cartopy_labels = (
        settings.get("map_draw_labels", True)
        and label_strategy == "cartopy"
        and not is_polar
    )

    gridliner = ax.gridlines(
        draw_labels=draw_cartopy_labels,
        linewidth=settings["grid_linewidth"],
        alpha=settings["grid_alpha"],
        rotate_labels=settings.get("grid_rotate_labels", False),
        linestyle=settings.get("grid_linestyle", "--"),
        color=settings.get("grid_color", "0.55"),
        x_inline=False,
        y_inline=False,
    )

    gridliner.xlocator = mticker.FixedLocator(longitude_ticks)
    gridliner.ylocator = mticker.FixedLocator(latitude_ticks)
    gridliner.rotate_labels = settings.get("grid_rotate_labels", False)

    if not settings.get("map_draw_labels", True):
        return gridliner

    if is_polar or label_strategy == "polar":
        if row in section_last_rows or settings.get(
            "polar_longitude_labels_each_panel", False
        ):
            add_polar_longitude_labels(
                ax,
                mode,
                extent,
                data_projection,
                longitude_ticks=settings.get("polar_longitude_ticks", longitude_ticks),
                label_offset=settings.get("polar_longitude_label_offset", 2.5),
                fontsize=fontsize,
                color=settings.get("polar_label_color", "0.25"),
                skip_longitudes=settings.get("polar_longitude_skip", []),
                label_position=settings.get("polar_longitude_label_position", "data_edge"),
                axes_radius=settings.get("polar_longitude_label_radius", 0.56),
                central_longitude=settings.get("polar_central_longitude", 0.0),
            )
        if settings.get("draw_polar_latitude_labels", True) and (
            col == 0
            or not settings.get("label_left_column_only", True)
            or settings.get("polar_latitude_labels_each_panel", False)
        ):
            add_polar_latitude_labels(
                ax,
                mode,
                latitude_ticks=settings.get("polar_latitude_ticks", latitude_ticks),
                label_longitude=settings.get("polar_latitude_label_longitude", 0),
                data_projection=data_projection,
                fontsize=fontsize,
                color=settings.get("polar_label_color", "0.25"),
                axes_angle_degrees=settings.get("polar_latitude_label_angle"),
                label_position=settings.get("polar_latitude_label_position", "radial"),
                side_x=settings.get("polar_latitude_label_x", 1.035),
            )
        return gridliner

    # Default Cartopy label path.
    gridliner.top_labels = settings.get("draw_top_labels", False)
    gridliner.right_labels = settings.get("draw_right_labels", False)
    gridliner.left_labels = (
        settings.get("draw_left_labels", True)
        and (col == 0 or not settings.get("label_left_column_only", True))
    )
    gridliner.bottom_labels = (
        settings.get("draw_bottom_labels", True)
        and (row in section_last_rows or not settings.get("label_bottom_row_only", True))
    )
    gridliner.xlabel_style = {"size": fontsize}
    gridliner.ylabel_style = {"size": fontsize}

    if LONGITUDE_FORMATTER is not None:
        gridliner.xformatter = LONGITUDE_FORMATTER
    if LATITUDE_FORMATTER is not None:
        gridliner.yformatter = LATITUDE_FORMATTER

    return gridliner

# -----------------------------------------------------------------------------
# General notebook/workflow helpers
# -----------------------------------------------------------------------------
def safe_token(value):
    """Return a filesystem-safe token for figure/file names."""
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def figure_filename(*parts, ext="png"):
    """Build a consistent figure filename from descriptive tokens."""
    clean = ["fig"] + [safe_token(part) for part in parts if str(part).strip()]
    return "_".join(clean) + f".{ext.lstrip('.').lower()}"


def save_figure(fig, figpath, *, mode, metric, title="", caption="", dpi=150, **extra):
    """Save a figure and upsert its entry in the figures.json manifest.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    figpath : pathlib.Path or str
        Full path where the PNG will be written.
    mode : str
        Mode name (e.g. ``"NAO"``).
    metric : str
        Metric / figure-type key (e.g. ``"skill"`` or ``"eof_patterns"``).
    title : str, optional
        Short human-readable title for the webpage.
    caption : str, optional
        Longer caption carrying runtime details (years, baseline, reference …).
    dpi : int, optional
        Resolution for the saved PNG (default 150).
    **extra
        Any additional key-value pairs to store in the manifest entry.
    """
    import json
    import datetime
    from pathlib import Path

    figpath = Path(figpath)
    fig.savefig(figpath, dpi=dpi, bbox_inches="tight")

    manifest_path = figpath.parent / "figures.json"

    # Load existing manifest or start fresh
    if manifest_path.exists():
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    else:
        manifest = {"figures": []}

    # Upsert: replace existing entry for same file, or append
    filename = figpath.name
    entry = {
        "file": filename,
        "mode": mode,
        "metric": metric,
        "title": title,
        "caption": caption,
        **extra,
    }
    figures = manifest.get("figures", [])
    idx = next((i for i, e in enumerate(figures) if e.get("file") == filename), None)
    if idx is not None:
        figures[idx] = entry
    else:
        figures.append(entry)

    manifest["figures"] = figures
    manifest["updated"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    try:
        from esp_lab.diagnostics.web import generate_diagnostics_webpage
        generate_diagnostics_webpage(figpath.parent)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not automatically update webpage: {e}")


# -----------------------------------------------------------------------------
# Data/plot helpers for MOV EOF-pattern panels
# -----------------------------------------------------------------------------

def add_cyclic_longitude_for_plot(data_array):
    """Append the first longitude column to the end for global contour plots.

    The function only adds a cyclic column when the longitude coordinate appears
    to cover nearly the full globe. Regional fields are returned unchanged.
    """
    import xarray as xr

    lon = data_array.lon
    if lon.size < 2:
        return data_array
    lon_span = float(lon.max() - lon.min())
    lon_step = float(lon.diff("lon").median())
    if lon_span + lon_step < 350:
        return data_array
    cyclic = xr.concat([data_array, data_array.isel(lon=0)], dim="lon")
    cyclic_lon = np.concatenate([lon.values, [float(lon.values[-1]) + lon_step]])
    return cyclic.assign_coords(lon=cyclic_lon)


def mask_pattern_for_plot(data_array, mask=None):
    """Return a spatially masked pattern for plotting.

    The mask may be a 2-D lat/lon array or may include already-selected
    non-spatial dimensions. Grid cells where the mask is false are set to NaN.
    """
    if mask is None:
        return data_array

    import xarray as xr

    pattern, plot_mask = xr.align(data_array, mask.astype(bool), join="exact")
    return pattern.where(plot_mask)


def spatial_pattern_metrics(model_pattern, reference_pattern):
    """Return cosine-latitude-weighted RMSE and pattern correlation."""
    import xarray as xr

    model_pattern, reference_pattern = xr.align(
        model_pattern, reference_pattern, join="exact"
    )
    valid = np.isfinite(model_pattern) & np.isfinite(reference_pattern)
    weights = np.cos(np.deg2rad(reference_pattern.lat)).broadcast_like(
        reference_pattern
    ).where(valid)
    weight_sum = weights.sum(("lat", "lon"))
    difference = model_pattern - reference_pattern
    rmse = np.sqrt((weights * difference**2).sum(("lat", "lon")) / weight_sum)
    model_mean = (weights * model_pattern).sum(("lat", "lon")) / weight_sum
    reference_mean = (weights * reference_pattern).sum(("lat", "lon")) / weight_sum
    model_anomaly = model_pattern - model_mean
    reference_anomaly = reference_pattern - reference_mean
    covariance = (weights * model_anomaly * reference_anomaly).sum(("lat", "lon"))
    model_variance = (weights * model_anomaly**2).sum(("lat", "lon"))
    reference_variance = (weights * reference_anomaly**2).sum(("lat", "lon"))
    pcc = covariance / np.sqrt(model_variance * reference_variance)
    return float(rmse), float(pcc)


def bootstrap_interval_excludes_zero(dataset, selector):
    """Return a boolean significance mask when bootstrap CI excludes zero."""
    import xarray as xr

    required = {"mode_pattern_bootstrap_lower", "mode_pattern_bootstrap_upper"}
    if not required <= set(dataset.data_vars):
        return None
    lower = dataset["mode_pattern_bootstrap_lower"].sel(selector)
    upper = dataset["mode_pattern_bootstrap_upper"].sel(selector)
    lower, upper = xr.align(lower, upper, join="exact")
    return np.isfinite(lower) & np.isfinite(upper) & ((lower > 0) | (upper < 0))


def pattern_significance(
    dataset,
    selector,
    *,
    allow_regression,
    eof_bootstrap_confidence=np.nan,
):
    """Return significance mask and a human-readable significance label."""
    if allow_regression and "mode_regression_significant" in dataset:
        confidence = float(dataset.attrs.get("regression_confidence", np.nan))
        label = "projected-regression significance"
        if np.isfinite(confidence):
            label = f"{confidence:.0%} {label}"
        return dataset["mode_regression_significant"].sel(selector).astype(bool), label

    significant = bootstrap_interval_excludes_zero(dataset, selector)
    if significant is None:
        return None, None
    confidence = float(dataset.attrs.get("eof_bootstrap_confidence", eof_bootstrap_confidence))
    label = "bootstrap CI excludes zero"
    if np.isfinite(confidence):
        label = f"{confidence:.0%} {label}"
    return significant, label


def target_year_from_display_lead(display_lead):
    """Convert a displayed monthly lead into a one-indexed target year."""
    return 1 + (int(display_lead) - 1) // 12
