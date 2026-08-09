"""
Utilities for computing tropical cyclone track density.

Two methods are available:

1. method="box"
   Simple grid-box count of TC track points. This matches the Yeager et al.
   NCL workflow that calls track_density(gridsize, 0.0, lat_points,
   lon_points, False) on flattened trajectory locations.

2. method="radius"
   Radius-based unique-track count. For each grid point, count how many
   unique tracks pass within a search radius. A single storm is counted only
   once per grid point. This is not the same diagnostic as counting track
   points inside each grid box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr


TrackDensityMethod = Literal["box", "radius"]


@dataclass(frozen=True)
class TrackDensityConfig:
    method: TrackDensityMethod = "box"

    # Used by method="box"
    box_grid_size: float = 5.0

    # Used by method="radius"
    radius_km: float = 350.0
    radius_dlat: float = 0.5
    radius_dlon: float = 0.5

    # Shared
    earth_radius_km: float = 6370.0
    lon_min: float = 0.0
    lon_max: float = 360.0
    lat_min: float = -90.0
    lat_max: float = 90.0


def _as_1d_float(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def _normalize_lon_360(lon: np.ndarray) -> np.ndarray:
    return lon % 360.0


def _validate_lat_lon(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    lat = _as_1d_float(lat)
    lon = _normalize_lon_360(_as_1d_float(lon))

    if lat.shape != lon.shape:
        raise ValueError(f"lat and lon must have the same shape, got {lat.shape} and {lon.shape}")

    valid = np.isfinite(lat) & np.isfinite(lon)
    return lat[valid], lon[valid]


def compute_track_density_box(
    lat,
    lon,
    grid_size: float = 5.0,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = 0.0,
    lon_max: float = 360.0,
    grid_lat=None,
    grid_lon=None,
) -> xr.DataArray:
    """
    Simple grid-box count of TC track points.

    Each valid track point contributes one count to one lat-lon grid box.
    This is fast and works without storm IDs.
    """

    lat, lon = _validate_lat_lon(lat, lon)

    if grid_lat is None:
        lat_centers = np.arange(lat_min, lat_max + 0.5 * grid_size, grid_size)
        lat_edges = np.arange(lat_min - grid_size / 2, lat_max + grid_size, grid_size)
    else:
        lat_centers = _as_1d_float(grid_lat)
        lat_edges = np.concatenate(
            [
                [lat_centers[0] - grid_size / 2],
                lat_centers[:-1] + 0.5 * np.diff(lat_centers),
                [lat_centers[-1] + grid_size / 2],
            ]
        )

    if grid_lon is None:
        lon_centers = np.arange(lon_min, lon_max, grid_size)
        lon_edges = np.arange(lon_min - grid_size / 2, lon_max + grid_size / 2, grid_size)
    else:
        lon_centers = _as_1d_float(grid_lon)
        lon_edges = np.concatenate(
            [
                [lon_centers[0] - grid_size / 2],
                lon_centers[:-1] + 0.5 * np.diff(lon_centers),
                [lon_centers[-1] + grid_size / 2],
            ]
        )

    # Treat longitude cyclically.
    lon_hist = lon.copy()
    lon_hist[lon_hist >= lon_edges[-1]] -= lon_max

    count, _, _ = np.histogram2d(lat, lon_hist, bins=[lat_edges, lon_edges])

    return xr.DataArray(
        count.astype("float32"),
        dims=("lat", "lon"),
        coords={"lat": lat_centers, "lon": lon_centers},
        name="track_density",
        attrs={
            "method": "box_count",
            "description": "Simple grid-box count of TC track points",
            "grid_size_degrees": grid_size,
        },
    )


def compute_track_density_radius(
    lat,
    lon,
    track_id,
    radius_km: float = 350.0,
    dlat: float = 0.5,
    dlon: float = 0.5,
    earth_radius_km: float = 6370.0,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    lon_min: float = 0.0,
    lon_max: float = 360.0,
    grid_lat=None,
    grid_lon=None,
) -> xr.DataArray:
    """
    Radius-based unique-track density.

    For each grid point, count how many unique tracks pass within radius_km.
    A single track is counted only once per grid point, even if multiple points
    from that storm pass within the search radius.
    """

    lat = _as_1d_float(lat)
    lon = _normalize_lon_360(_as_1d_float(lon))
    track_id = np.asarray(track_id).ravel()

    if lat.shape != lon.shape or lat.shape != track_id.shape:
        raise ValueError(
            "lat, lon, and track_id must have the same flattened shape; "
            f"got {lat.shape}, {lon.shape}, {track_id.shape}"
        )

    valid = np.isfinite(lat) & np.isfinite(lon)
    lat = lat[valid]
    lon = lon[valid]
    track_id = track_id[valid]

    if grid_lat is None:
        grid_lat = np.arange(lat_min, lat_max + 0.5 * dlat, dlat)
    else:
        grid_lat = _as_1d_float(grid_lat)

    if grid_lon is None:
        grid_lon = np.arange(lon_min, lon_max, dlon)
    else:
        grid_lon = _as_1d_float(grid_lon)

    lat2d, lon2d = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    count_total = np.zeros(lat2d.shape, dtype=np.int32)
    lat_grid_rad = np.deg2rad(lat2d)

    for tid in np.unique(track_id):
        ii = track_id == tid
        storm_lat = lat[ii]
        storm_lon = lon[ii]

        # Prevent one storm from being counted multiple times at one grid point.
        count_track = np.zeros(lat2d.shape, dtype=bool)

        for slat, slon in zip(storm_lat, storm_lon):
            slat_rad = np.deg2rad(slat)
            dlat_rad = slat_rad - lat_grid_rad

            # Cyclic shortest longitude distance.
            dlon_deg = ((slon - lon2d + 180.0) % 360.0) - 180.0
            dlon_rad = np.deg2rad(dlon_deg)

            a = (
                np.sin(0.5 * dlat_rad) ** 2
                + np.cos(lat_grid_rad)
                * np.cos(slat_rad)
                * np.sin(0.5 * dlon_rad) ** 2
            )
            a = np.clip(a, 0.0, 1.0)
            dist_km = 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))

            hit = (dist_km < radius_km) & (~count_track)
            count_track[hit] = True

        count_total += count_track.astype(np.int32)

    return xr.DataArray(
        count_total.astype("float32"),
        dims=("lat", "lon"),
        coords={"lat": grid_lat, "lon": grid_lon},
        name="track_density",
        attrs={
            "method": "radius_unique_track_count",
            "description": f"Number of unique tracks passing within {radius_km} km of each grid point",
            "radius_km": radius_km,
            "grid_dlat_degrees": dlat,
            "grid_dlon_degrees": dlon,
            "earth_radius_km": earth_radius_km,
        },
    )


def compute_track_density(
    lat,
    lon,
    track_id=None,
    config: TrackDensityConfig | None = None,
    method: TrackDensityMethod | None = None,
    grid_lat=None,
    grid_lon=None,
) -> xr.DataArray:
    """Compute track density with the selected method."""

    if config is None:
        config = TrackDensityConfig()

    selected_method = method or config.method

    if selected_method == "box":
        return compute_track_density_box(
            lat=lat,
            lon=lon,
            grid_size=config.box_grid_size,
            lat_min=config.lat_min,
            lat_max=config.lat_max,
            lon_min=config.lon_min,
            lon_max=config.lon_max,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
        )

    if selected_method == "radius":
        if track_id is None:
            raise ValueError("track_id is required when method='radius'")

        return compute_track_density_radius(
            lat=lat,
            lon=lon,
            track_id=track_id,
            radius_km=config.radius_km,
            dlat=config.radius_dlat,
            dlon=config.radius_dlon,
            earth_radius_km=config.earth_radius_km,
            lat_min=config.lat_min,
            lat_max=config.lat_max,
            lon_min=config.lon_min,
            lon_max=config.lon_max,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
        )

    raise ValueError(f"Unknown track-density method: {selected_method!r}")


def make_track_id_from_2d_or_3d_shape(shape, year_offset: int = 0, ensemble_offset: int = 0):
    """
    Create unique track IDs from 2D or 3D trajectory-array shapes.

    The last dimension is assumed to be time/step.
    """

    if len(shape) == 2:
        ntrack, _ = shape
        track = np.arange(ntrack)[:, None]
        return np.broadcast_to(track, shape)

    if len(shape) == 3:
        nouter, ntrack, _ = shape
        outer = np.arange(nouter)[:, None, None]
        track = np.arange(ntrack)[None, :, None]

        track_id = (
            ensemble_offset * 10_000_000
            + year_offset * 1_000_000
            + outer * 100_000
            + track
        )

        return np.broadcast_to(track_id, shape)

    raise ValueError(f"Unsupported shape {shape}; expected 2D or 3D trajectory arrays")


def read_stitch_nodes_tracks(track_files: list[str | Path], workflows_dir=None) -> pd.DataFrame:
    """
    Parse a list of TempestExtremes StitchNodes track files and combine them into a single DataFrame.
    """
    all_dfs = []

    for file_path in track_files:
        path = Path(file_path)
        if not path.exists():
            continue

        # Extract case and member from path parts:
        # e.g., .../case/member/post/atm/tc-analysis/file
        # case is at parts[-6], member is at parts[-5]
        try:
            member = path.parts[-5]
            case = path.parts[-6]
        except IndexError:
            # Fallback if the path structure is different:
            # try to parse from the filename which starts with case_member_
            filename = path.name
            match = re.search(r"_(EN\d+)_", filename)
            if match:
                member = match.group(1)
                case = filename.split(f"_{member}_")[0]
            else:
                member = "unknown"
                case = "unknown"

        # Parse case initialization time:
        try:
            match_init = re.search(r"_(\d{10})$", case)
            if match_init:
                tag = match_init.group(1)
                init_time = pd.Timestamp(
                    year=int(tag[:4]), month=int(tag[4:6]),
                    day=int(tag[6:8]), hour=int(tag[8:10]),
                )
            else:
                init_time = pd.NaT
        except Exception:
            init_time = pd.NaT

        rows = []
        storm_id = -1

        try:
            with open(path) as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.startswith("start"):
                        storm_id += 1
                        continue

                    parts = s.split()
                    if len(parts) < 10:
                        continue

                    rows.append({
                        "case": case,
                        "member": member,
                        "storm_id": storm_id,
                        "node": int(float(parts[0])),
                        "lon": float(parts[1]) % 360.0,
                        "lat": float(parts[2]),
                        "slp": float(parts[3]),
                        "wind": float(parts[4]),
                        "zs": float(parts[5]),
                        "time": pd.Timestamp(
                            year=int(parts[6]), month=int(parts[7]),
                            day=int(parts[8]), hour=int(parts[9]),
                        ),
                        "init_time": init_time,
                    })
        except Exception as e:
            print(f"Error parsing track file {path}: {e}")
            continue

        if rows:
            all_dfs.append(pd.DataFrame(rows))

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=[
            "case", "member", "storm_id", "node", "lon", "lat", "slp", "wind", "zs", "time", "init_time"
        ])
