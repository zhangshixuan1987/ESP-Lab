#!/usr/bin/env python
"""Compare E3SM mode skill with NMME over one NMME cohort.

Reads mode-index products written by ``run_process_modes_of_variability.py``:
E3SM cases and the HadISST2/ERA5 reference from ``--diag-root`` (read only) and
NMME from one or more ``--nmme-root`` folders (one per model, as the Slurm
template writes them, or one multi-model folder). For each mode and start month
it writes a skill file, a figure, and a one-line-per-lead summary table:

* E3SM lines restricted to the NMME start years, so both are scored on the same
  years with the same skill function;
* the NMME line: skill of the equal-weight multi-model forecast;
* the NMME band: min-max of the individual models' skill, with the per-lead
  model count (leads where any model has ended are drawn lighter).

``--years`` and ``--models`` choose the cohort; the defaults (1982-2010, models
that cover every start) are the recommended production choice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from esp_lab import stats  # noqa: E402
from workflows.modes_of_variability.cross_source import (  # noqa: E402
    NMME_DEFAULT_YEARS,
    NMME_MIN_MEMBERS,
    _initialization_years,
    nmme_multimodel_mean,
    nmme_skill,
    select_nmme_cohort,
)

E3SM_CASES = {"E3SM-FOSIRL": "JRA55_FOSIRL", "E3SM-Reanalysis": "Reanalysis"}
COLORS = {"E3SM-FOSIRL": "#0072B2", "E3SM-Reanalysis": "#D55E00", "NMME": "#C0392B"}
STYLES = {"E3SM-FOSIRL": ":", "E3SM-Reanalysis": "--", "NMME": "-."}


def mode_index_path(root: Path, source_dir: str, mode: str, init_month: int) -> Path:
    lower = mode.lower()
    return root / source_dir / "modes_variability" / "modes" / lower / "indices" / f"{source_dir}_init{init_month:02d}_{lower}.nc"


def reference_path(root: Path, mode: str) -> Path:
    lower = mode.lower()
    matches = sorted((root / "observations" / "modes_variability" / "modes" / lower / "indices").glob(f"*_{lower}_reference.nc"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {mode} reference under {root / 'observations'}, found {matches}")
    return matches[0]


def skill_index_and_time(dataset: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Seasonal skill leads, as notebook 4a scores them, with integer start years."""
    dataset = dataset.assign_coords(Y=_initialization_years(dataset, str(dataset.attrs.get("source", "product"))))
    if {"mode_index_skill", "valid_time_skill"} <= set(dataset.data_vars):
        return dataset["mode_index_skill"].rename(skill_L="L"), dataset["valid_time_skill"].rename(skill_L="L")
    return dataset["mode_index"], dataset["valid_time"]


def load_nmme(roots: list[Path], mode: str, init_month: int) -> tuple[xr.DataArray, xr.DataArray]:
    indices, times = [], []
    for root in roots:
        path = mode_index_path(root, "NMME", mode, init_month)
        if not path.is_file():
            continue
        with xr.open_dataset(path) as dataset:
            index, time = skill_index_and_time(dataset.load())
        indices.append(index)
        times.append(time)
    if not indices:
        raise FileNotFoundError(f"No NMME {mode} init {init_month:02d} products under {[str(r) for r in roots]}")
    index = xr.concat(indices, dim="M", join="outer", combine_attrs="drop_conflicts")
    # Target dates are a property of start year and lead, so models agree where
    # they overlap; combine_first fills the years only some models have.
    time = times[0]
    for other in times[1:]:
        time = time.combine_first(other)
    return index, time.sel(Y=index.Y)


def recenter_by_model(index: xr.DataArray) -> xr.DataArray:
    """Remove each model's lead-dependent mean over the cohort years and members."""
    pieces = []
    for model in dict.fromkeys(label.split(":", 1)[0] for label in index.M.values.astype(str)):
        members = [label for label in index.M.values.astype(str) if label.startswith(model + ":")]
        part = index.sel(M=members)
        pieces.append(part - part.mean(("Y", "M"), skipna=True))
    return xr.concat(pieces, dim="M").assign_attrs(index.attrs, nmme_anomaly_period=index.attrs.get("nmme_years", ""))


def compare(args: argparse.Namespace, mode: str, init_month: int) -> Path:
    observation = xr.open_dataset(reference_path(args.diag_root, mode)).load()["mode_index"]
    raw_nmme, nmme_time = load_nmme(args.nmme_root, mode, init_month)
    nmme, dropped = select_nmme_cohort(raw_nmme, args.years, models=args.models, min_members=args.min_members)
    nmme = recenter_by_model(nmme)
    nmme_time = nmme_time.sel(Y=nmme.Y)
    for model, reason in dropped.items():
        print(f"{mode} init {init_month:02d}: NMME {model} left out ({reason})")

    sources = {}
    for label, source_dir in E3SM_CASES.items():
        with xr.open_dataset(mode_index_path(args.diag_root, source_dir, mode, init_month)) as dataset:
            index, time = skill_index_and_time(dataset.load())
        years = [year for year in nmme.Y.values if year in index.Y.values]
        sources[label] = (index.sel(Y=years), time.sel(Y=years))

    leads = sorted(set(map(int, nmme.L.values)).intersection(*(set(map(int, i.L.values)) for i, _ in sources.values())))
    cohorts = stats.common_valid_target_years_seasonal(
        {**{k: v[0].sel(L=leads) for k, v in sources.items()}, "NMME": nmme_multimodel_mean(nmme).sel(L=leads)},
        {**{k: v[1].sel(L=leads) for k, v in sources.items()}, "NMME": nmme_time.sel(L=leads)},
        observation, leads,
    )

    def score(index: xr.DataArray, time: xr.DataArray) -> xr.Dataset:
        return stats.compute_skill_seasonal(
            index.sel(L=leads), time.sel(L=leads), observation, args.clim_years[0], args.clim_years[1],
            nleadavg=1, nleads=len(leads), resamp=0, detrend=args.detrend, is_anomaly=True,
            target_years_by_lead=cohorts,
        ).load()

    skills = {label: score(index, time) for label, (index, time) in sources.items()}
    skills["NMME"] = nmme_skill(nmme, lambda index: score(index, nmme_time), model=args.single_model)
    label = "NMME" if args.single_model is None else f"NMME {args.single_model}"
    tag = f"{mode.lower()}_init{init_month:02d}_nmme{args.years[0]}_{args.years[1]}" + (
        f"_{args.single_model}" if args.single_model else "")
    outdir = args.outdir / mode.lower()
    outdir.mkdir(parents=True, exist_ok=True)

    combined = xr.concat([s[["corr", "rmse"]] for s in skills.values()], dim=xr.DataArray(list(skills), dims="source", name="source"))
    for name in ("corr_min", "corr_max", "rmse_min", "rmse_max", "model_count", "member_count"):
        if name in skills["NMME"]:
            combined[name] = skills["NMME"][name]
    combined.attrs.update({k: v for k, v in skills["NMME"].attrs.items() if k.startswith("nmme_")})
    combined.attrs.update(mode=mode, init_month=str(init_month), nmme_left_out="; ".join(f"{m}: {r}" for m, r in dropped.items()))
    skill_path = outdir / f"{tag}_skill.nc"
    combined.to_netcdf(skill_path)
    combined[["corr", "rmse"]].reset_coords(drop=True).to_dataframe().unstack("source").round(3).to_csv(outdir / f"{tag}_skill.csv")
    plot(combined, outdir / f"{tag}_skill.png", mode=mode, init_month=init_month, nmme_label=label, years=args.years)
    print(f"Wrote {skill_path}")
    return skill_path


def plot(skill: xr.Dataset, path: Path, *, mode: str, init_month: int, nmme_label: str, years) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count_name = "model_count" if "model_count" in skill else "member_count"
    count = skill[count_name]
    full = count == count.max()
    leads = skill.L.values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric, title in zip(axes, ("corr", "rmse"), ("ACC", "nRMSE")):
        axis.fill_between(leads, skill[f"{metric}_min"].where(full), skill[f"{metric}_max"].where(full),
                          color=COLORS["NMME"], alpha=0.18, linewidth=0)
        axis.fill_between(leads, skill[f"{metric}_min"], skill[f"{metric}_max"],
                          color=COLORS["NMME"], alpha=0.07, linewidth=0)
        for source in skill.source.values:
            name = "NMME" if source == "NMME" else str(source)
            values = skill[metric].sel(source=source)
            label = nmme_label if source == "NMME" else f"{source} ({years[0]}-{years[1]})"
            axis.plot(leads, values.where(full) if source == "NMME" else values, STYLES[name], marker="o",
                      color=COLORS[name], label=label)
            if source == "NMME":
                axis.plot(leads, values, STYLES[name], color=COLORS[name], alpha=0.35)
        axis.axhline(0.5 if metric == "corr" else 1.0, color="k", linewidth=0.8)
        axis.set_title(f"{mode} init {init_month:02d}  {title}")
        axis.set_xlabel("Forecast lead (months, 3-month means)")
        axis.set_xticks(leads)
        axis.grid(True, alpha=0.4)
    unit = "models" if count_name == "model_count" else "members"
    axes[0].fill_between([], [], [], color=COLORS["NMME"], alpha=0.18,
                         label=f"{nmme_label} range (min-max, {int(count.max())} {unit})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diag-root", type=Path, required=True, help="s2d_diag root holding the E3SM and observation products (read only).")
    parser.add_argument("--nmme-root", type=Path, nargs="+", required=True, help="Folder(s) holding NMME/modes_variability products.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["PDO", "NPGO", "AMO"])
    parser.add_argument("--init-months", type=int, nargs="+", default=[5, 11])
    parser.add_argument("--years", type=int, nargs=2, default=list(NMME_DEFAULT_YEARS), help="NMME start years (default 1982 2010).")
    parser.add_argument("--models", nargs="+", default=["auto"], help="'auto' (models covering every start) or model names.")
    parser.add_argument("--single-model", default=None, help="Show one NMME model (band = its members) instead of the multi-model line.")
    parser.add_argument("--min-members", type=int, default=NMME_MIN_MEMBERS)
    parser.add_argument("--clim-years", type=int, nargs=2, default=[1981, 2010])
    parser.add_argument("--detrend", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.models = "auto" if args.models == ["auto"] else args.models
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    failures = []
    for mode in args.modes:
        for init_month in args.init_months:
            try:
                compare(args, mode.upper(), init_month)
            except (OSError, ValueError, KeyError) as error:
                failures.append(f"{mode} init {init_month:02d}: {error}")
                print(f"FAILED {failures[-1]}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
