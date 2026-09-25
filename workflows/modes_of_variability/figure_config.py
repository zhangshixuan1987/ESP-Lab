"""Central figure configuration for modes-of-variability notebooks.

Shared visual defaults live in the preset dictionaries below.  The only table
that is organized by scientific mode is :data:`MODE_FIGURE_PROFILES`; each
entry selects shared presets and contains only genuine mode-specific changes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import numpy as np


PRESSURE_LEVELS = np.arange(-3.0, 3.01, 0.5)
TEMPERATURE_LEVELS = np.arange(-1.0, 1.01, 0.2)

PACIFIC_LONGITUDE_TICKS = np.array([120, 150, 180, -150, -120])
PACIFIC_LONGITUDE_LABELS = ["120°E", "150°E", "180°", "150°W", "120°W"]
NORTH_PACIFIC_LATITUDE_TICKS = np.array([10, 20, 30, 40, 50, 60, 70])
NORTH_PACIFIC_LATITUDE_LABELS = [
    "10°N", "20°N", "30°N", "40°N", "50°N", "60°N", "70°N",
]
ATLANTIC_LONGITUDE_TICKS = np.array([-90, -60, -30, 0, 30])
ATLANTIC_LATITUDE_TICKS = np.array([20, 30, 40, 50, 60, 70, 80])
GLOBAL_LONGITUDE_TICKS = np.arange(-180, 181, 60)
GLOBAL_LATITUDE_TICKS = np.arange(-60, 61, 30)


# Shared by every regional EOF-pattern figure before a profile is applied.
SHARED_EOF_PATTERN_SETTINGS = {
    "figsize_width": 14.5,
    "figsize_row_height": 2.75,
    "figsize_min_height": 4.0,
    "pattern_years": (1, 2),
    "pattern_seasons": ("DJF", "MAM", "JJA", "SON"),
    "preferred_init_by_season": {"DJF": 11, "MAM": 11, "JJA": 5, "SON": 5},
    "suptitle_scale": 1.0,
    "column_header_scale": 0.95,
    "panel_title_scale": 0.72,
    "panel_title_pad": 3,
    "panel_title_linespacing": 1.10,
    "year_section_title_scale": 1.0,
    "rmse_decimals": 2,
    "pcc_decimals": 2,
    "metric_box_x": 0.03,
    "metric_box_y": 0.045,
    "metric_box_font_scale": 0.62,
    "metric_box_alpha": 0.5,
    "metric_box_pad": 0.16,
    "metric_box_linewidth": 0.45,
    "metric_box_zorder": 10,
    "axis_label_scale": 0.90,
    "tick_label_scale": 0.58,
    "colorbar_label_scale": 0.78,
    "colorbar_tick_scale": 0.66,
    "colorbar_shrink": 0.80,
    "colorbar_pad": 0.035,
    "colorbar_fraction": 0.030,
    "stipple_stride": 2,
    "stipple_size": 9,
    "stipple_color": "black",
    "stipple_alpha": 0.50,
    "stipple_zorder": 4,
    "levels": PRESSURE_LEVELS,
    "draw_contour_lines": True,
    "contour_line_levels": PRESSURE_LEVELS,
    "contour_line_color": "0.20",
    "contour_linewidth": 0.45,
    "contour_line_alpha": 0.70,
    "grid_alpha": 0.45,
    "grid_linewidth": 0.8,
    "grid_linestyle": "--",
    "grid_color": "0.55",
    "grid_rotate_labels": False,
    "label_strategy": "cartopy",
    "map_draw_labels": True,
    "map_label_scale": 0.55,
    "draw_left_labels": True,
    "draw_bottom_labels": True,
    "draw_top_labels": False,
    "draw_right_labels": False,
    "label_left_column_only": True,
    "label_bottom_row_only": True,
    "longitude_ticks": np.arange(-180, 181, 30),
    "latitude_ticks": np.arange(-90, 91, 20),
    "longitude_ticklabels": None,
    "latitude_ticklabels": None,
    "manual_lon_label_offset": 1.5,
    "manual_lat_label_offset": 1.5,
    "polar_circular_boundary": True,
    "polar_draw_regular_labels": False,
    "polar_longitude_label_offset": 2.5,
    "polar_longitude_ticks": np.arange(-180, 181, 60),
    "top_section_gap_ratio": 0.015,
    "year_section_header_ratio": 0.58,
    "year_section_gap_ratio": 0.16,
    "subplot_left": 0.070,
    "subplot_right": 0.965,
    "subplot_bottom": 0.145,
    "subplot_top": 0.890,
    "subplot_wspace": 0.080,
    "subplot_hspace": 0.300,
    "use_cartopy": True,
    "coastline_linewidth": 0.45,
    "global_land_facecolor": "0.18",
    "global_land_edgecolor": "0.05",
    "global_land_linewidth": 0.25,
    "global_land_zorder": 4,
    "global_coastline_color": "0.05",
    "global_coastline_linewidth": 0.35,
    "global_coastline_zorder": 5,
}


# Reusable geometry/typography presets.  Mode profiles select these by name.
LAYOUT_PRESETS = {
    "polar": {
        "figsize_width": 12.0,
        "figsize_row_height": 3.8,
        "subplot_left": 0.055,
        "subplot_right": 0.845,
        "subplot_bottom": 0.06,
        "subplot_top": 0.88,
        "subplot_wspace": 0.70,
        "subplot_hspace": 0.22,
        "year_section_header_ratio": 0.20,
        "map_label_scale": 0.70,
        "metric_box_font_scale": 0.58,
        "polar_metric_box_x": 0.90,
        "polar_metric_box_y": 0.020,
        "polar_metric_box_ha": "center",
        "polar_metric_box_va": "center",
        "polar_metric_two_lines": True,
        "colorbar_shrink": 0.85,
        "colorbar_pad": 0.060,
        "colorbar_fraction": 0.024,
        "suptitle_scale": 1.0,
        "column_header_scale": 0.85,
        "year_title_y_ratio": 0.76,
        "column_title_y_ratio": 0.12,
        "panel_title_y": 1.03,
        "column_header_shift_by_col": {1: -0.035, 2: -0.06, 3: -0.075, 4: -0.11},
        "column_header_y_shift": 0.16,
        "column_header_y_shift_by_col": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
    },
    "atlantic": {
        "figsize_width": 10.0,
        "figsize_row_height": 2.2,
        "subplot_left": 0.08,
        "subplot_right": 0.94,
        "subplot_bottom": 0.06,
        "subplot_top": 0.90,
        "subplot_wspace": 0.05,
        "subplot_hspace": 0.40,
        "year_section_title_scale": 0.90,
        "year_section_header_ratio": 0.55,
        "panel_title_scale": 0.70,
        "map_label_scale": 0.75,
        "suptitle_scale": 1.0,
        "column_header_scale": 0.85,
        "panel_title_pad": 8,
        "panel_title_y": 1.02,
        "colorbar_pad": 0.015,
    },
    "north_pacific": {
        "figsize_width": 15.5,
        "figsize_row_height": 2.2,
        "subplot_left": 0.08,
        "subplot_right": 0.94,
        "subplot_bottom": 0.06,
        "subplot_top": 0.90,
        "subplot_wspace": 0.05,
        "subplot_hspace": 0.40,
        "year_section_title_scale": 0.90,
        "year_section_header_ratio": 0.55,
        "panel_title_scale": 0.70,
        "map_label_scale": 0.75,
        "suptitle_scale": 1.0,
        "column_header_scale": 0.85,
        "panel_title_pad": 8,
        "panel_title_y": 1.02,
        "colorbar_pad": 0.015,
    },
    "north_pacific_temperature": {
        "figsize_width": 12.0,
        "figsize_row_height": 2.5,
        "subplot_left": 0.08,
        "subplot_right": 0.94,
        "subplot_bottom": 0.06,
        "subplot_top": 0.90,
        "subplot_wspace": 0.14,
        "subplot_hspace": 0.30,
        "year_section_title_scale": 0.90,
        "year_section_header_ratio": 0.55,
        "panel_title_scale": 0.70,
        "map_label_scale": 0.75,
        "suptitle_scale": 1.0,
        "column_header_scale": 0.85,
        "panel_title_pad": 0,
        "colorbar_pad": 0.015,
    },
    "pacific_temperature": {
        "figsize_width": 14.0,
        "figsize_row_height": 3.0,
        "subplot_left": 0.08,
        "subplot_right": 0.94,
        "subplot_bottom": 0.06,
        "subplot_top": 0.90,
        "subplot_wspace": 0.04,
        "subplot_hspace": 0.30,
        "year_section_title_scale": 0.90,
        "year_section_header_ratio": 0.55,
        "panel_title_scale": 0.70,
        "map_label_scale": 0.75,
        "suptitle_scale": 1.0,
        "column_header_scale": 0.85,
        "panel_title_pad": 8,
        "panel_title_y": 1.02,
        "colorbar_pad": 0.015,
    },
}

COMPACT_REGIONAL_EOF = {
    "year_section_title_scale": 0.70,
    "panel_title_scale": 0.44,
    "metric_box_font_scale": 0.48,
    "map_label_scale": 0.58,
    "colorbar_label_scale": 0.60,
    "colorbar_tick_scale": 0.52,
    "suptitle_scale": 0.82,
    "column_header_scale": 0.60,
}

ATLANTIC_EOF = {
    **COMPACT_REGIONAL_EOF,
    "subplot_hspace": 0.18,
    "year_section_header_ratio": 0.42,
    "panel_title_pad": 5,
}

MAP_PRESETS = {
    "north_polar": {
        "label_strategy": "polar",
        "longitude_ticks": np.arange(-180, 181, 60),
        "latitude_ticks": np.array([25, 55, 85]),
        "polar_longitude_ticks": np.array([-120, -60, 60, 120]),
        "polar_latitude_ticks": np.array([25, 55, 85]),
    },
    "south_polar": {
        "label_strategy": "polar",
        "longitude_ticks": np.arange(-180, 181, 60),
        "latitude_ticks": np.array([-85, -55, -25]),
        "polar_longitude_ticks": np.array([-120, -60, 60, 120]),
        "polar_latitude_ticks": np.array([-85, -55, -25]),
    },
    "atlantic": {
        "label_strategy": "cartopy",
        "longitude_ticks": ATLANTIC_LONGITUDE_TICKS,
        "latitude_ticks": ATLANTIC_LATITUDE_TICKS,
    },
    "north_pacific": {
        "label_strategy": "cartopy",
        "longitude_ticks": PACIFIC_LONGITUDE_TICKS,
        "longitude_ticklabels": PACIFIC_LONGITUDE_LABELS,
        "latitude_ticks": NORTH_PACIFIC_LATITUDE_TICKS,
        "latitude_ticklabels": NORTH_PACIFIC_LATITUDE_LABELS,
    },
}

POLAR_LABELS = {
    "polar_longitude_skip": [],
    "polar_longitude_label_position": "outside_axes",
    "polar_longitude_label_radius": 0.625,
    "polar_central_longitude": 0,
    "polar_latitude_label_position": "radial",
    "polar_latitude_label_angle": -20,
    "polar_longitude_labels_each_panel": True,
    "polar_latitude_labels_each_panel": True,
    "draw_polar_latitude_labels": True,
    "draw_left_labels": False,
    "draw_bottom_labels": False,
    "polar_draw_regular_labels": False,
    "polar_longitude_label_offset": 2.5,
    "polar_label_color": "0.25",
    "map_label_scale": 0.70,
    "grid_rotate_labels": False,
}
for preset_name in ("north_polar", "south_polar"):
    MAP_PRESETS[preset_name] = {**MAP_PRESETS[preset_name], **POLAR_LABELS}


# This is the only mode-indexed configuration.  Everything else is shared.
MODE_FIGURE_PROFILES: dict[str, dict[str, object]] = {
    "NAM": {"layout": "polar", "map": "north_polar"},
    "SAM": {"layout": "polar", "map": "south_polar"},
    "PSA1": {
        "layout": "polar", "map": "south_polar",
        "layout_overrides": {
            "subplot_wspace": 0.50,
            "subplot_hspace": 0.15,
        },
        "eof_overrides": {
            "polar_longitude_labels_each_panel": True,
            "polar_latitude_labels_each_panel": True,
        },
        "projection": {"central_longitude": 0},
    },
    "PSA2": {
        "layout": "polar", "map": "south_polar",
        "layout_overrides": {
            "subplot_wspace": 0.50,
            "subplot_hspace": 0.15,
        },
        "eof_overrides": {
            "polar_longitude_labels_each_panel": True,
            "polar_latitude_labels_each_panel": True,
        },
        "projection": {"central_longitude": 0},
    },
    "NAO": {
        "layout": "atlantic", "map": "atlantic", "eof_style": "atlantic_compact",
        "layout_overrides": {
            "figsize_width": 15.5,
            "figsize_row_height": 2.0,
            "subplot_wspace": 0.08,
        },
        "projection": {"central_longitude": -30, "central_latitude": 50,
                       "standard_parallels": (30, 70)},
    },
    "EA": {
        "layout": "atlantic", "map": "atlantic", "eof_style": "atlantic_compact",
        "layout_overrides": {
            "figsize_width": 15.5,
            "figsize_row_height": 2.0,
            "subplot_wspace": 0.08,
        },
        "projection": {"central_longitude": -20, "central_latitude": 50,
                       "standard_parallels": (30, 70)},
    },
    "SCA": {
        "layout": "atlantic", "map": "atlantic", "eof_style": "atlantic_compact",
        "layout_overrides": {
            "figsize_width": 15.5,
            "figsize_row_height": 2.0,
            "subplot_wspace": 0.08,
        },
        "projection": {"central_longitude": -20, "central_latitude": 50,
                       "standard_parallels": (35, 70)},
    },
    "AMO": {
        "layout": "atlantic", "map": "atlantic", "eof_style": "atlantic_compact",
        "layout_overrides": {"figsize_width": 11.0, "figsize_row_height": 2.0,
                             "subplot_wspace": 0.08, "subplot_hspace": 0.18,
                             "panel_title_y": 1.0, "map_label_scale": 0.80,
                             "panel_title_pad": 4},
        "eof_overrides": {"panel_title_scale": 0.48,
                          "column_header_x_shift_by_col": {3: -0.012},
                          "transform_first": True},
        "teleconnection_overrides": {"transform_first": True},
        "levels": "temperature",
    },
    "PNA": {
        "layout": "north_pacific", "map": "north_pacific", "eof_style": "compact",
        "projection": {"central_longitude": 180, "central_latitude": 40,
                       "standard_parallels": (20, 60)},
        "eof_overrides": {"panel_title_scale": 0.48, "panel_title_y": 1.04,
                          "panel_title_pad": 3, "panel_title_two_lines": False,
                          "metric_box_two_lines": True},
    },
    "NPO": {
        "layout": "north_pacific", "map": "north_pacific", "eof_style": "compact",
        "projection": {"central_longitude": 180, "central_latitude": 40,
                       "standard_parallels": (20, 60)},
        "layout_overrides": {"figsize_row_height": 2.0, "subplot_wspace": 0.08,
                             "subplot_hspace": 0.22},
        "eof_overrides": {"panel_title_scale": 0.48, "panel_title_y": 1.04,
                          "panel_title_pad": 3, "panel_title_two_lines": False,
                          "metric_box_two_lines": True},
    },
    "NPGO": {
        "layout": "north_pacific_temperature", "map": "north_pacific",
        "eof_style": "compact", "levels": "temperature",
        "layout_overrides": {
            "figsize_width": 15.5,
            "figsize_row_height": 2.0,
            "subplot_wspace": 0.08,
            "subplot_hspace": 0.22,
        },
        "projection": {"central_longitude": 180, "central_latitude": 45,
                       "standard_parallels": (20, 60)},
        "eof_overrides": {"panel_title_scale": 0.50, "panel_title_y": 1.04,
                          "panel_title_pad": 3,
                          "column_header_x_shift_by_col": {3: -0.012}},
    },
    "PDO": {
        "layout": "pacific_temperature", "map": "north_pacific",
        "eof_style": "compact", "levels": "temperature",
        "layout_overrides": {
            "figsize_width": 15.5,
            "figsize_row_height": 2.0,
            "subplot_wspace": 0.08,
            "subplot_hspace": 0.22,
        },
    },
}


SHARED_TELECONNECTION_OVERRIDES = {
    "figsize_width": 16.0,
    "figsize_column_width": 2.90,
    "figsize_colorbar_width": 1.40,
    "figsize_row_height": 1.95,
    "subplot_left": 0.045,
    "subplot_right": 0.925,
    "subplot_bottom": 0.035,
    "subplot_top": 0.92,
    "subplot_wspace": 0.10,
    "subplot_hspace": 0.22,
    "column_header_y_shift": 0.14,
    "label_strategy": "cartopy",
    "longitude_ticks": GLOBAL_LONGITUDE_TICKS,
    "latitude_ticks": GLOBAL_LATITUDE_TICKS,
    "draw_left_labels": True,
    "draw_bottom_labels": True,
    "label_left_column_only": True,
    "label_bottom_row_only": True,
    "global_land_facecolor": "none",
    "global_land_edgecolor": "none",
    "global_land_linewidth": 0.0,
    "panel_title_pad": 3,
    "panel_title_loc": "center",
    "panel_title_y": 1.01,
    "year_title_y": 0.965,
    "year_title_linespacing": 1.05,
    "metric_box_x": 0.03,
    "metric_box_y": 0.05,
    "metric_box_ha": "left",
    "metric_box_va": "bottom",
    "metric_box_alpha": 0.85,
    "metric_box_pad": 0.18,
    "metric_box_linewidth": 0.5,
    "metric_box_two_lines": True,
    "colorbar_shrink": 0.78,
    "colorbar_pad": 0.02,
    "colorbar_fraction": 0.022,
}


def _profile(mode: str) -> dict[str, object]:
    token = mode.upper().strip()
    if token not in MODE_FIGURE_PROFILES:
        raise ValueError(f"No figure profile configured for mode {token!r}")
    return MODE_FIGURE_PROFILES[token]


def _regional_settings(mode: str, *, eof: bool) -> dict[str, object]:
    profile = _profile(mode)
    settings = deepcopy(SHARED_EOF_PATTERN_SETTINGS)
    settings.update(deepcopy(LAYOUT_PRESETS[str(profile["layout"])]))
    settings.update(deepcopy(profile.get("layout_overrides", {})))
    settings.update(deepcopy(MAP_PRESETS[str(profile["map"])]))
    levels = TEMPERATURE_LEVELS if profile.get("levels") == "temperature" else PRESSURE_LEVELS
    settings.update(levels=levels.copy(), contour_line_levels=levels.copy())
    if eof and profile.get("eof_style") == "compact":
        settings.update(deepcopy(COMPACT_REGIONAL_EOF))
    elif eof and profile.get("eof_style") == "atlantic_compact":
        settings.update(deepcopy(ATLANTIC_EOF))
    if eof:
        settings.update(deepcopy(profile.get("eof_overrides", {})))
    else:
        settings.update(deepcopy(SHARED_TELECONNECTION_OVERRIDES))
        settings.update(deepcopy(profile.get("teleconnection_overrides", {})))
    return settings


def get_eof_pattern_settings(
    mode: str, overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    settings = _regional_settings(mode, eof=True)
    if overrides:
        settings.update(overrides)
    return settings


def get_teleconnection_settings(
    mode: str, overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    settings = _regional_settings(mode, eof=False)
    if overrides:
        settings.update(overrides)
    return settings


ALBERS_PROJECTION_BY_MODE = {
    mode: deepcopy(profile["projection"])
    for mode, profile in MODE_FIGURE_PROFILES.items()
    if "projection" in profile
}


BASE_METRIC_SPECS = [
    ("corr", "ACC", (-0.5, 0.9)),
    ("rmse", "Normalized RMSE", (0.9, 1.2)),
]
MODE_METRIC_SPECS = {
    "NAM": [("corr", "ACC", (-0.5, 0.9)), ("rmse", "Normalized RMSE", (0.8, 1.3))],
    "NAO": [("corr", "ACC", (-0.5, 0.5)), ("rmse", "Normalized RMSE", (0.9, 1.2))],
    "EA": [("corr", "ACC", (-0.5, 0.5)), ("rmse", "Normalized RMSE", (0.9, 1.2))],
    "SCA": [("corr", "ACC", (-0.5, 0.5)), ("rmse", "Normalized RMSE", (0.9, 1.2))],
    "SAM": [("corr", "ACC", (-0.5, 0.9)), ("rmse", "Normalized RMSE", (0.8, 1.3))],
    "PNA": [("corr", "ACC", (-0.5, 0.9)), ("rmse", "Normalized RMSE", (0.6, 1.3))],
    "NPO": [("corr", "ACC", (-0.4, 0.9)), ("rmse", "Normalized RMSE", (0.8, 1.3))],
    "PDO": [("corr", "ACC", (0, 0.9)), ("rmse", "Normalized RMSE", (0.5, 1.1))],
    "NPGO": [("corr", "ACC", (-0.5, 0.9)), ("rmse", "Normalized RMSE", (0.6, 1.2))],
    "AMO": [("corr", "ACC", (-0.5, 0.9)), ("rmse", "Normalized RMSE", (0.8, 1.2))],
    "PSA1": [("corr", "ACC", (-0.5, 0.9)), ("rmse", "Normalized RMSE", (0.7, 1.3))],
    "PSA2": [("corr", "ACC", (-0.5, 0.9)), ("rmse", "Normalized RMSE", (0.7, 1.3))],
}


def metric_specs(mode: str) -> list[tuple[str, str, tuple[float, float]]]:
    return deepcopy(MODE_METRIC_SPECS.get(mode.upper().strip(), BASE_METRIC_SPECS))


def build_figure_setup(mode: str, *, include_nmme: bool = False) -> dict[str, object]:
    """Return the complete notebook figure setup with shared styles applied."""
    return {
        "common": {"fontsize": 18, "dpi": 300},
        "eof_patterns": get_eof_pattern_settings(mode),
        "teleconnections": get_teleconnection_settings(mode),
        "skill": {
            "fontsize": 24, "dpi": 300, "figsize_per_column": (9, 10),
            "figsize_init_panel_width": 7.0, "suptitle_scale": 1.0,
            "suptitle_linespacing": 1.15, "panel_title_scale": 1.0,
            "axis_label_scale": 0.95, "tick_label_scale": 0.90,
            "legend_font_scale": 0.85, "title_pad": 6, "marker_size": 16,
            "significance_zorder": 5, "grid_alpha": 0.45, "grid_linewidth": 0.5,
            "reference_linewidth": 1.0, "legend_init_linewidth": 2.5,
            "legend_columns": 5, "legend_frameon": True, "legend_handle_length": 2.5,
            "legend_column_spacing": 1.5, "legend_label_spacing": 0.35,
            "init_colors": {5: "green", 11: "black"},
            "init_labels": {5: "MAY init", 11: "NOV init"},
            "model_colors": {"E3SM-FOSIRL": "#0072B2", "E3SM-Reanalysis": "#D55E00",
                             "E3SM-4DEnVarOcn": "#CC79A7", "SMYLE": "#009E73",
                             "NMME": "#E69F00"},
            "model_linestyles": {"SMYLE": "-", "E3SM-FOSIRL": ":",
                                 "E3SM-Reanalysis": "--",
                                 "E3SM-4DEnVarOcn": (0, (3, 1, 1, 1, 1, 1)),
                                 "NMME": "-."},
            "model_linewidths": {"SMYLE": 3.0, "E3SM-FOSIRL": 3.5,
                                 "E3SM-Reanalysis": 3.5, "E3SM-4DEnVarOcn": 3.5,
                                 "NMME": 3.0},
            "season_markers": {1: "D", 4: "o", 7: "s", 10: "P"},
            "season_labels": {1: "DJF", 4: "MAM", 7: "JJA", 10: "SON"},
        },
        "pc_time_series": {
            "fontsize": 18,
            "fontsize_scales": {"title": 1.0, "tick": 0.9, "legend": 0.82,
                               "annotation": 0.72},
            "figsize_width": 15.0, "figsize_row_height": 4.0, "line_width": 3.0,
            "obs_marker_size": 10, "obs_color": "k", "grid_minor_alpha": 0.15,
            "annotation_box": None, "plot_ylim": [-4.0, 4.0],
            "annotation_band_height": 4.2 if include_nmme else 3.4,
            "annotation_band_facecolor": "white", "annotation_band_alpha": 1.0,
            "annotation_band_edgecolor": "0.75", "major_yticks": [-4, -2, 0, 2, 4],
            "minor_yticks_step": 0.5, "major_year_step": 10, "minor_year_step": 2,
            "hindcast_label": {11: "November init.", 5: "May init."},
            "hindcast_color": {11: "g", 5: "b"},
            "target_season_labels": {1: "DJF", 4: "MAM", 7: "JJA", 10: "SON"},
            "subplot_left": 0.075, "subplot_right": 0.975, "subplot_top": 0.930,
            "subplot_bottom": 0.090, "subplot_wspace": 0.13, "subplot_hspace": 0.36,
            "suptitle_scale": 1.25, "suptitle_y": 0.972, "tick_direction": "out",
            "ylabel_text": "Standardized PC", "legend_loc": "lower center",
            "legend_ncol": 5, "legend_bbox_to_anchor": (0.5, 0.022),
            "legend_frameon": True,
            "model_styles": {
                "E3SM-FOSIRL": {"label": "E3SMv3-FOSIRL", "annotation_label": "FOSIRL",
                                "linestyle": "-", "spread_alpha": 0.25,
                                "annotation_xy": (0.02, 0.22), "colors": "b"},
                "E3SM-Reanalysis": {"label": "E3SMv3-Reanalysis",
                                     "annotation_label": "Reanalysis", "linestyle": "-",
                                     "spread_alpha": 0.22, "annotation_xy": (0.02, 0.155),
                                     "colors": "tab:purple"},
                "SMYLE": {"label": "CESM-SMYLE", "annotation_label": "CESM-SMYLE",
                          "linestyle": "--", "spread_alpha": 0.20,
                          "annotation_xy": (0.02, 0.035), "colors": "tab:orange"},
                "E3SM-4DEnVarOcn": {"label": "E3SMv3-4DEnVarOcn",
                                    "annotation_label": "4DEnVarOcn", "linestyle": "-.",
                                    "spread_alpha": 0.18, "annotation_xy": (0.02, 0.095),
                                    "colors": "tab:brown"},
                "NMME": {"label": "NMME", "annotation_label": "NMME",
                         "linestyle": "-.", "spread_alpha": 0.18,
                         "annotation_xy": (0.02, 0.27), "colors": "tab:brown"},
            },
        },
    }


__all__ = [
    "ALBERS_PROJECTION_BY_MODE",
    "MODE_FIGURE_PROFILES",
    "build_figure_setup",
    "get_eof_pattern_settings",
    "get_teleconnection_settings",
    "metric_specs",
]
