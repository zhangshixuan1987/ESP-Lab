"""Build exact-period tropical-cyclone lead-time diagnostic caches."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr

from .inputs import ensure_tracks
from .track_density import TrackDensityConfig, compute_track_density


VALID_CACHE_MODES = {"auto", "require", "rebuild"}


def diagnostic_path(diag_root, case_key, case_prefix, parset, year_start, year_end):
    """Return the exact-period diagnostic cache path for one experiment."""
    return (
        Path(diag_root)
        / case_key
        / "tc_track"
        / f"{case_key}_tc_lead_track_density_{case_prefix}_{parset}_{year_start}_{year_end}.nc"
    )


def _case_init_time(case):
    match = re.search(r"_(\d{10})$", case)
    if match is None:
        raise ValueError(f"Could not find YYYYMMDDHH init tag at end of case name: {case}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H")


def _track_path(track_root, case, member, parset):
    return (
        Path(track_root)
        / case
        / member
        / "post"
        / "atm"
        / "tc-analysis"
        / f"{case}_{member}_{parset}_TCS_track.txt"
    )


def _read_track(path, case, member):
    rows = []
    storm_id = -1
    init_time = _case_init_time(case)
    with Path(path).open() as stream:
        for line in stream:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("start"):
                storm_id += 1
                continue
            fields = text.split()
            if len(fields) < 10:
                continue
            rows.append(
                {
                    "case": case,
                    "member": member,
                    "storm_id": storm_id,
                    "lon": float(fields[1]) % 360.0,
                    "lat": float(fields[2]),
                    "wind": float(fields[4]),
                    "time": pd.Timestamp(
                        year=int(fields[6]), month=int(fields[7]),
                        day=int(fields[8]), hour=int(fields[9]),
                    ),
                    "init_time": init_time,
                }
            )
    return pd.DataFrame(rows)


def _read_track_filtered(item, point_wind_min):
    """Read one metadata item and apply the optional point filter on-worker."""
    case, member, path = item
    frame = _read_track(path, case, member)
    if point_wind_min is not None and not frame.empty:
        frame = frame[frame.wind >= point_wind_min]
    return frame


def _read_track_frames(metadata, point_wind_min=None, dask_client=None):
    """Read track files serially or through a supplied distributed client."""
    if not metadata:
        return []
    if dask_client is None or len(metadata) < 2:
        frames = [_read_track_filtered(item, point_wind_min) for item in metadata]
    else:
        futures = dask_client.map(
            _read_track_filtered,
            metadata,
            [point_wind_min] * len(metadata),
            pure=False,
        )
        frames = dask_client.gather(futures)
    return [frame for frame in frames if not frame.empty]


def _add_season_and_lead(points, leads, years):
    if points.empty:
        return pd.DataFrame(columns=list(points.columns) + ["season", "season_year", "lead"])
    pieces = []
    nh = points[points.time.dt.month.between(6, 11)].copy()
    if not nh.empty:
        nh["season"] = "NH_JJASON"
        nh["season_year"] = nh.time.dt.year.astype(int)
        nh["season_start_month"] = 6
        pieces.append(nh)
    sh = points[(points.time.dt.month == 12) | (points.time.dt.month <= 5)].copy()
    if not sh.empty:
        sh["season"] = "SH_DJFMAM"
        sh["season_year"] = np.where(sh.time.dt.month == 12, sh.time.dt.year, sh.time.dt.year - 1)
        sh["season_start_month"] = 12
        pieces.append(sh)
    if not pieces:
        return pd.DataFrame(columns=list(points.columns) + ["season", "season_year", "lead"])
    result = pd.concat(pieces, ignore_index=True)
    result["lead"] = (
        (result.season_year - result.init_time.dt.year) * 12
        + result.season_start_month
        - result.init_time.dt.month
    ).astype(int)
    return result[
        result.lead.isin(leads) & result.season_year.isin(years)
    ].drop(columns="season_start_month")


def _add_density(target, points, sample_columns, dimension_columns, config, lat, lon):
    if points.empty:
        return
    for values, sample in points.groupby(sample_columns, sort=False):
        if len(sample_columns) == 1:
            values = (values,)
        lookup = dict(zip(sample_columns, values))
        location = {dim: lookup[column] for dim, column in dimension_columns.items()}
        field = compute_track_density(
            lat=sample.lat.to_numpy(), lon=sample.lon.to_numpy(),
            track_id=sample.storm_id.to_numpy(), config=config,
            method=config.method, grid_lat=lat, grid_lon=lon,
        )
        target.loc[location] += field.astype(np.float32)


def _basin_mask(basin_defs, seasons, lat, lon):
    mask = xr.DataArray(
        np.zeros((len(basin_defs), len(seasons), len(lat), len(lon)), dtype=np.int8),
        dims=("basin", "season", "lat", "lon"),
        coords={"basin": list(basin_defs), "season": seasons, "lat": lat, "lon": lon},
        name="basin_mask",
    )
    for basin, spec in basin_defs.items():
        lon0, lon1 = spec["lon"]
        lat0, lat1 = spec["lat"]
        mask.loc[
            dict(
                basin=basin, season=spec["season"],
                lat=(mask.lat >= lat0) & (mask.lat < lat1),
                lon=(mask.lon >= lon0) & (mask.lon < lon1),
            )
        ] = 1
    return mask


def _read_ibtracs(path, wind_min, time_step_hours):
    with xr.open_dataset(path) as dataset:
        lat_da = dataset["lat"]
        wind = xr.where(np.isfinite(dataset["wmo_wind"]), dataset["wmo_wind"], dataset["usa_wind"])
        storm_id = np.repeat(np.arange(lat_da.shape[0]), lat_da.shape[1])
        lat = lat_da.values.ravel()
        lon = dataset["lon"].values.ravel() % 360.0
        time = pd.to_datetime(dataset["time"].values.ravel())
        wind = wind.values.ravel()
    valid = ~pd.isna(time) & np.isfinite(lat) & np.isfinite(lon)
    if wind_min is not None:
        valid &= np.isfinite(wind) & (wind >= wind_min)
    result = pd.DataFrame(
        {"storm_id": storm_id[valid], "time": time[valid],
         "lat": lat[valid], "lon": lon[valid], "wind_kt": wind[valid]}
    )
    if time_step_hours is not None:
        result = result[
            result.time.dt.minute.eq(0)
            & result.time.dt.second.eq(0)
            & result.time.dt.hour.mod(time_step_hours).eq(0)
        ].copy()
    return result


def _add_observations(dataset, *, ibtracs_file, years, seasons, config,
                      lat, lon, wind_min, time_step_hours):
    observations = _read_ibtracs(ibtracs_file, wind_min, time_step_hours)
    pieces = []
    nh = observations[observations.time.dt.month.between(6, 11)].copy()
    nh["season"], nh["year"] = "NH_JJASON", nh.time.dt.year.astype(int)
    pieces.append(nh)
    sh = observations[(observations.time.dt.month == 12) | (observations.time.dt.month <= 5)].copy()
    sh["season"] = "SH_DJFMAM"
    sh["year"] = np.where(sh.time.dt.month == 12, sh.time.dt.year, sh.time.dt.year - 1)
    pieces.append(sh)
    obs = pd.concat(pieces, ignore_index=True)
    obs = obs[obs.year.isin(years)]
    density = xr.DataArray(
        np.zeros((len(seasons), len(years), len(lat), len(lon)), dtype=np.float32),
        dims=("season", "year", "lat", "lon"),
        coords={"season": seasons, "year": years, "lat": lat, "lon": lon},
        name="obs_track_density_count",
    )
    _add_density(density, obs, ["season", "year"], {"season": "season", "year": "year"}, config, lat, lon)
    sample_count = xr.ones_like(density.isel(lat=0, lon=0), dtype=np.int32).rename("obs_sample_count")
    storm_count = xr.zeros_like(sample_count, dtype=np.float32).rename("obs_storm_count")
    if not obs.empty:
        counts = obs.drop_duplicates(["season", "year", "storm_id"]).groupby(["season", "year"]).size()
        for (season, year), count in counts.items():
            storm_count.loc[dict(season=season, year=year)] = count
    basin_count = (density * dataset["basin_mask"]).sum(("lat", "lon")).rename("obs_basin_track_density_count")
    return dataset.assign(
        obs_track_density_count=density,
        obs_track_density_mean=density.rename("obs_track_density_mean"),
        obs_sample_count=sample_count,
        obs_storm_count=storm_count,
        obs_storm_count_mean=storm_count.rename("obs_storm_count_mean"),
        obs_basin_track_density_count=basin_count,
        obs_basin_track_density_mean=basin_count.rename("obs_basin_track_density_mean"),
    )


def ensure_experiment_diagnostic(
    *, case_key, spec, repo_root, track_root, diag_root, year_start, year_end,
    init_months, members, parset, leads, seasons, basin_defs, track_config,
    track_settings, ibtracs_file, obs_wind_min=35.0, obs_time_step_hours=6,
    point_wind_min=None, input_mode="auto", cache_mode="auto", dask_client=None,
):
    """Load or build one experiment's exact-period diagnostic cache."""
    if cache_mode not in VALID_CACHE_MODES:
        raise ValueError("TC diagnostic cache mode must be auto, require, or rebuild")
    output = diagnostic_path(
        diag_root, case_key, spec["case_prefix"], parset, year_start, year_end
    )
    if output.is_file() and cache_mode != "rebuild":
        return output, "reused"
    if cache_mode == "require":
        raise FileNotFoundError(f"Missing exact-period TC diagnostic: {output}")

    years = list(range(int(year_start), int(year_end) + 1))
    cases = [
        f"{spec['case_prefix']}_{year}{month:02d}0100"
        for month in init_months for year in years
    ]
    settings = dict(track_settings)
    settings.update(sim_dir=Path(spec["sim_dir"]), stream_tag=spec["stream_tag"])
    ensure_tracks(
        repo_root=repo_root, track_root=track_root, cases=cases, members=members,
        parsets=[parset], settings=settings, mode=input_mode,
    )
    metadata = [
        (case, member, _track_path(track_root, case, member, parset))
        for case in cases for member in members
    ]
    metadata = [(case, member, path) for case, member, path in metadata if path.is_file()]
    samples = xr.DataArray(
        np.zeros((len(seasons), len(leads), len(years)), dtype=np.int32),
        dims=("season", "lead", "year"),
        coords={"season": seasons, "lead": leads, "year": years}, name="sample_count",
    )
    for case, member, path in metadata:
        init = _case_init_time(case)
        for lead in leads:
            start = init + pd.DateOffset(months=int(lead))
            season = "NH_JJASON" if start.month == 6 else "SH_DJFMAM" if start.month == 12 else None
            if season and start.year in years:
                samples.loc[dict(season=season, lead=lead, year=start.year)] += 1
    frames = _read_track_frames(
        metadata, point_wind_min=point_wind_min, dask_client=dask_client
    )
    tracks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    points = _add_season_and_lead(tracks, leads, years)
    if track_config.method == "box":
        lat = np.arange(-90.0, 90.0 + track_config.box_grid_size / 2, track_config.box_grid_size)
        lon = np.arange(0.0, 360.0, track_config.box_grid_size)
    else:
        lat = np.arange(-90.0 + track_config.radius_dlat / 2, 90.0, track_config.radius_dlat)
        lon = np.arange(track_config.radius_dlon / 2, 360.0, track_config.radius_dlon)
    density = xr.DataArray(
        np.zeros((len(seasons), len(leads), len(years), len(lat), len(lon)), dtype=np.float32),
        dims=("season", "lead", "year", "lat", "lon"),
        coords={"season": seasons, "lead": leads, "year": years, "lat": lat, "lon": lon},
        name="tc_track_density_count",
    )
    _add_density(
        density, points, ["case", "member", "season", "lead", "season_year"],
        {"season": "season", "lead": "lead", "year": "season_year"}, track_config, lat, lon,
    )
    storm_count = xr.zeros_like(samples, dtype=np.float32).rename("tc_storm_count")
    if not points.empty:
        counts = points.drop_duplicates(
            ["case", "member", "season", "lead", "season_year", "storm_id"]
        ).groupby(["season", "lead", "season_year"]).size()
        for (season, lead, year), count in counts.items():
            storm_count.loc[dict(season=season, lead=lead, year=year)] = count
    mask = _basin_mask(basin_defs, seasons, lat, lon)
    mean_density = (density / samples.where(samples > 0)).rename("tc_track_density_mean")
    basin_count = (density * mask).sum(("lat", "lon")).rename("tc_basin_track_density_count")
    dataset = xr.Dataset(
        {
            "tc_track_density_count": density,
            "tc_track_density_mean": mean_density,
            "tc_storm_count": storm_count,
            "tc_storm_count_mean": (storm_count / samples.where(samples > 0)).rename("tc_storm_count_mean"),
            "sample_count": samples,
            "basin_mask": mask,
            "tc_basin_track_density_count": basin_count,
            "tc_basin_track_density_mean": (basin_count / samples.where(samples > 0)).rename("tc_basin_track_density_mean"),
        },
        attrs={"case_prefix": spec["case_prefix"], "years": f"{year_start}-{year_end}", "parset": parset},
    )
    if ibtracs_file is not None and Path(ibtracs_file).is_file():
        dataset = _add_observations(
            dataset, ibtracs_file=ibtracs_file, years=years, seasons=seasons,
            config=track_config, lat=lat, lon=lon, wind_min=obs_wind_min,
            time_step_hours=obs_time_step_hours,
        )
    elif ibtracs_file is not None:
        print(
            f"[TC diagnostics] IBTrACS file is unavailable; writing model-only cache: "
            f"{ibtracs_file}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output)
    dataset.close()
    return output, "built"
