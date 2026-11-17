"""Inputs, region means, summaries and figures shared by the sample-size notebooks.

The notebooks in ``jupyter/sample_size/`` (8a start years, 8b ensemble members,
8c lead-time climatology years) resample existing hindcast caches with the
scorers in :mod:`esp_lab.sample_size_skill`. This module holds what they share:

* :func:`load_inputs` reads, for every experiment and field, the longest cached
  record inside the experiment's start-year span (1a 5x5 prepared anomalies for
  the atmosphere, 3a seasonal SST-index series, 1a_lnd 5x5 prepared land
  fields) and pairs it with the observations at each verification season;
* :func:`region_reducers` turns score maps into cos-latitude-weighted region means;
* :func:`summarize` gives the median and 5-95 % range over resampled draws;
* :func:`plot_curves` draws one figure per group (rows: fields, columns: leads).

Inputs are only read (test root first, then production); nothing is recomputed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from esp_lab import sample_size_skill as sss
from esp_lab.utils import regrid_utils as regrid

SEASON = {1: "DJF", 4: "MAM", 7: "JJA", 10: "SON"}
GROUPS = ("atm", "sst", "lnd")

# Figure style: font sizes (pt), panel size (in), and axis labels for the sample-size axes.
FONT = dict(title=16, panel=14, label=14, tick=12, legend=13)
PANEL_SIZE = (5.0, 3.6)
X_LABELS = {"n_years": "Number of start years N", "members": "Ensemble members M",
            "clim_years": "Start years in the climatology K"}


# ----------------------------------------------------------------------------- inputs
def lead_labels(valid_time: xr.DataArray) -> list[str]:
    """Season name + year offset from the start year for each lead, e.g. JJA0, DJF1."""
    first = valid_time.isel(Y=0)
    y0 = sss.init_years(valid_time.Y.values[:1])[0]
    return [f"{SEASON.get(t.month, f'm{t.month:02d}')}{t.year - y0}" for t in first.values]


def _best(candidates: list[dict], span: tuple[int, int]) -> dict | None:
    """Longest record inside span; earlier read roots win ties."""
    inside = [c for c in candidates if span[0] <= c["y0"] and c["y1"] <= span[1]]
    return max(inside, key=lambda c: (c["ny"], -c["rank"])) if inside else None


def _atm_source(info, field, read_roots, init_month):
    rx = re.compile(r"_y(\d{4})-(\d{4})_ny(\d+)_")
    cands = []
    for rank, root in enumerate(read_roots):
        folder = root / info["tag"] / "leadtime_acc" / "prepared_skill" / "atm" / field
        for p in folder.glob(f"{info['tag']}_init{init_month:02d}_{field}_seasonal_anomaly_y*_5x5deg_skill_v1.nc"):
            y0, y1, ny = map(int, rx.search(p.name).groups())
            cands.append(dict(path=p, y0=y0, y1=y1, ny=ny, rank=rank))
    best = _best(cands, info["years"])
    if best is None:
        return None
    ds = xr.open_dataset(best["path"])
    return ds["anomaly"] + ds["climatology"], ds["time"], best["path"]


def _atm_obs(field, read_roots):
    paths = [p for root in read_roots for p in sorted(
        (root / "observations" / "leadtime_acc" / "prepared_skill" / "atm" / field)
        .glob(f"*_{field}_seasonal_5x5deg_skill_v1.nc"))]
    return xr.open_dataset(paths[0])["observation"] if paths else None


def _sst_source(info, index, read_roots, init_month):
    rx = re.compile(r"_y(\d{4})-(\d{4})_")
    cands = []
    for rank, root in enumerate(read_roots):
        folder = root / info["tag"] / "sst_index" / "timeseries"
        for p in folder.glob(f"{info['tag']}_init{init_month:02d}_TS_y*_M24_{index}SST_seas.nc"):
            y0, y1 = map(int, rx.search(p.name).groups())
            cands.append(dict(path=p, y0=y0, y1=y1, ny=y1 - y0 + 1, rank=rank))
    best = _best(cands, info["years"])
    if best is None:
        return None
    ds = xr.open_dataset(best["path"])
    return ds["sst"], ds["time"], best["path"]


def _sst_obs(index, read_roots):
    path = next((p for p in (root / "observations" / "sst_index" / "timeseries" / f"HadISST2_sst_{index}SST_seas.nc"
                             for root in read_roots) if p.is_file()), None)
    return xr.open_dataset(path)["sst"] if path else None


def _lnd_source(info, field, read_roots, init_month):
    cands = []
    for rank, root in enumerate(read_roots):
        folder = root / info["tag"] / "leadtime_acc" / "prepared_skill" / "lnd" / field
        for p in folder.glob(f"{info['tag']}_init{init_month:02d}_{field}*_seasonal_5x5deg_cell_centered.nc"):
            attrs = xr.open_dataset(p).attrs
            y0, y1 = int(attrs["initialization_year_start"]), int(attrs["initialization_year_end"])
            cands.append(dict(path=p, y0=y0, y1=y1, ny=y1 - y0 + 1, rank=rank))
    best = _best(cands, info["years"])
    if best is None:
        return None
    ds = xr.open_dataset(best["path"])
    return ds[field], ds["time"], best["path"]


def _lnd_obs(field, read_roots):
    paths = [p for root in read_roots for p in sorted(
        (root / "observations" / "leadtime_acc" / "prepared_skill" / "lnd" / field)
        .glob(f"*_{field}*_seasonal_5x5deg_cell_centered.nc"))]
    return xr.open_dataset(paths[0])[field] if paths else None


_LOADERS = {"atm": (_atm_source, _atm_obs), "sst": (_sst_source, _sst_obs), "lnd": (_lnd_source, _lnd_obs)}


def prepare(raw: xr.DataArray, valid_time: xr.DataArray, obs: xr.DataArray):
    """Members (Y, M, L, ...) and observations at verification (Y, L, ...), observed years only."""
    obs_at = sss.observations_at_valid_time(obs, valid_time)
    space = [d for d in obs_at.dims if d not in ("Y", "L")]
    covered = obs_at.notnull().any(space) if space else obs_at.notnull()
    keep = covered.all("L").values
    members = raw.isel(Y=keep).transpose("Y", "M", "L", ...).load().astype("float64")
    return members, obs_at.isel(Y=keep), lead_labels(valid_time)


def load_inputs(groups: Mapping[str, Sequence[str]], experiments: Mapping[str, Mapping], *,
                read_roots: Sequence[Path], init_month: int = 5, min_years: int = 6):
    """Load every group/field/experiment; return (data, inventory).

    ``groups`` maps "atm"/"sst"/"lnd" to field (or SST-index) names; each
    experiment needs ``tag`` (cache tag) and ``years`` (start-year span it
    covers; files outside it are ignored). ``data[group][field][experiment]``
    holds ``members`` (Y, M, L, ...), ``obs`` (Y, L, ...) and lead ``labels``;
    experiments with fewer than ``min_years`` observed start years are skipped.
    """
    unknown = set(groups) - set(_LOADERS)
    if unknown:
        raise ValueError(f"Unknown groups {sorted(unknown)}; choose from {GROUPS}")
    data, rows = {}, []
    for group, fields in groups.items():
        source_fn, obs_fn = _LOADERS[group]
        data[group] = {}
        for field in fields:
            obs = obs_fn(field, read_roots)
            if obs is None:
                print(f"[{group}] {field}: no observation cache; skipped")
                continue
            data[group][field] = {}
            for key, info in experiments.items():
                found = source_fn(info, field, read_roots, init_month)
                if found is None:
                    rows.append(dict(group=group, field=field, experiment=key, status="no cache"))
                    continue
                raw, valid_time, path = found
                members, obs_at, labels = prepare(raw, valid_time, obs)
                years = sss.init_years(members.Y.values)
                status = "used" if len(years) >= min_years else f"too few observed years ({len(years)})"
                rows.append(dict(group=group, field=field, experiment=key, status=status,
                                 years=f"{years.min()}-{years.max()}" if len(years) else "",
                                 n_years=len(years), members=members.sizes["M"], file=str(path)))
                if status == "used":
                    data[group][field][key] = dict(members=members, obs=obs_at, labels=labels)
    return data, pd.DataFrame(rows)


# ---------------------------------------------------------------------- region means
def make_reducer(regions: Mapping[str, tuple], landfrac: xr.DataArray | None = None) -> Callable:
    """Reducer turning (draw, L, lat, lon) scores into (draw, L, region) area means.

    ``regions`` maps names to (lat0, lat1, surface) with surface "all", "land"
    or "ocean" (land fraction >= / < 0.5). ``clim_error`` is averaged as RMS.
    Scores without space (SST indices) get a single "index" region.
    """
    def reduce(da):
        if "lat" not in da.dims:
            return da.expand_dims(region=["index"])
        lat = da.lat
        weights = np.cos(np.deg2rad(lat)).broadcast_like(da.isel(draw=0, L=0, drop=True))
        rms = da.name == "clim_error"
        values = da ** 2 if rms else da
        out = []
        for name, (lat0, lat1, surface) in regions.items():
            mask = (lat >= lat0) & (lat <= lat1)
            if surface != "all" and landfrac is not None:
                frac = landfrac.interp_like(da.isel(draw=0, L=0, drop=True), method="nearest")
                mask = mask & ((frac >= 0.5) if surface == "land" else (frac < 0.5))
            w = weights.where(mask)
            mean = values.weighted(w.fillna(0)).mean(("lat", "lon"), skipna=True)
            out.append(np.sqrt(mean) if rms else mean)
        return xr.concat(out, "region").assign_coords(region=list(regions))
    return reduce


def region_reducers(read_roots: Sequence[Path], atm_regions: Mapping[str, tuple],
                    land_regions: Mapping[str, tuple]) -> dict[str, Callable]:
    """Reducers per group; the atmosphere uses the E3SM land fraction on the 5x5 grid."""
    destgrid = regrid.make_latlon_grid(dlat=5.0, dlon=5.0)
    sftlf_path = next(p for root in read_roots for p in sorted(root.glob("*/fixed/sftlf.E3SM.nc")))
    sftlf = xr.open_dataset(sftlf_path)["sftlf"].astype("float64").to_dataset(name="sftlf")
    landfrac = regrid.make_regridder(sftlf, destgrid, method="conservative", periodic=True)(sftlf["sftlf"]).load()
    return {"atm": make_reducer(atm_regions, landfrac), "lnd": make_reducer(land_regions), "sst": make_reducer({})}


# ------------------------------------------------------------------------ summaries
def summarize(ds: xr.Dataset, variables: Sequence[str], **labels) -> list[dict]:
    """Median and 5-95 % range over draws as long-format rows (one per region and lead)."""
    rows = []
    for var in variables:
        q = ds[var].quantile([0.05, 0.5, 0.95], "draw")
        for region in ds.region.values:
            for i, lead in enumerate(ds.L.values):
                v = q.sel(region=region, L=lead).values
                rows.append(dict(labels, metric=var, region=str(region), lead=i,
                                 p05=float(v[0]), median=float(v[1]), p95=float(v[2]), draws=ds.sizes["draw"]))
    return rows


def year_members(members: xr.DataArray, n_members: int, seed: int) -> xr.DataArray:
    """A fixed random subset of ``n_members`` members (all members when there are fewer)."""
    if members.sizes["M"] <= n_members:
        return members
    pick = np.sort(np.random.default_rng(seed).choice(members.sizes["M"], n_members, replace=False))
    return members.isel(M=pick)


# -------------------------------------------------------------------------- figures
def plot_curves(table: pd.DataFrame, data: Mapping, group: str, x: str, ylabel: str, title: str,
                path: Path, *, experiments: Mapping[str, Mapping], plot_region: Mapping[str, str],
                plot_leads: Mapping[str, Sequence[int]], metric: str, variant: str | None = None,
                extra: Callable | None = None, ylim: tuple | None = None,
                font: Mapping = FONT, panel_size: tuple = PANEL_SIZE, show: bool = True):
    """Rows: fields; columns: ``plot_leads[group]``; median line + 5-95 % band per experiment.

    Constrained layout places the legend below the panels, so it never overlaps
    the title or panel titles regardless of the number of rows; y-axes are shared
    within each row so the leads of one field compare directly. ``extra(ax,
    group, field, lead, panel)`` adds guide lines. Returns the saved path, or
    None when the group has nothing to plot.
    """
    sub = table[(table.group == group) & (table.metric == metric) & (table.region == plot_region[group])]
    if variant is not None:
        sub = sub[sub.variant == variant]
    fields = [f for f in data.get(group, {}) if f in set(sub.field)]
    if not fields:
        print(f"No {group} data to plot")
        return None
    leads = plot_leads[group]
    fig, axes = plt.subplots(
        len(fields), len(leads), squeeze=False, sharex=True, sharey="row", layout="constrained",
        figsize=(panel_size[0] * len(leads), panel_size[1] * len(fields) + 1.2),
    )
    for r, field in enumerate(fields):
        for c, lead in enumerate(leads):
            ax = axes[r, c]
            panel = sub[(sub.field == field) & (sub.lead == lead)]
            for key, info in experiments.items():
                line = panel[panel.experiment == key].sort_values(x)
                if line.empty:
                    continue
                ax.plot(line[x], line["median"], color=info["color"], marker="o", ms=4, lw=1.8, label=info["label"])
                ax.fill_between(line[x], line["p05"], line["p95"], color=info["color"], alpha=0.12, lw=0)
            if extra is not None:
                extra(ax, group, field, lead, panel)
            any_exp = next(iter(data[group][field].values()))
            ax.set_title(f"{field}  {any_exp['labels'][lead]}", fontsize=font["panel"], fontweight="bold")
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=font["tick"])
            if r == len(fields) - 1:
                ax.set_xlabel(X_LABELS.get(x, x.replace("_", " ")), fontsize=font["label"])
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=font["label"])
    handles, names = [], []
    for ax in axes.flat:
        for h, n in zip(*ax.get_legend_handles_labels()):
            if n not in names:
                handles.append(h)
                names.append(n)
    fig.suptitle(title, fontsize=font["title"], fontweight="bold")
    fits = max(3, int(fig.get_figwidth() // 2.2))
    ncol = len(names) if len(names) <= fits else -(-len(names) // 2)   # one row, else two balanced rows
    fig.legend(handles, names, loc="outside lower center", ncol=ncol,
               frameon=False, fontsize=font["legend"], handlelength=2.2, columnspacing=1.6)
    path = Path(path)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path
