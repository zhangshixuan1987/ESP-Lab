"""Process and plot monthly change-in-SWE (ΔSWE) lead-time ACC.

ΔSWE at month t is approximated as SWE(t) - SWE(t-1) using consecutive
monthly-mean storage values. C3S gaps are expanded before differencing, so a
change is missing unless both adjacent calendar months are available.

Products use the source-first diagnostic layout shared with the 1a workflow:
``<diag-root>/<source-or-case>/leadtime_acc/{inputs,skill}/land/<field>``.
"""

from __future__ import annotations

import argparse
import glob
import os
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from esp_lab import data_access_e3sm, land_skill, stats
from esp_lab.paths import S2D_DIAG_ROOT, leadtime_acc_dir
from esp_lab.utils import mapplot_utils as maps
from esp_lab.utils import mov_utils as mov


FIELD = "DELTA_H2OSNO"
SOURCE_FIELD = "H2OSNO"
REFERENCE_PRODUCT = "C3S_SWE"
GRID_TAG = "1x1deg_cell_centered"
EVALUATION_PROTOCOL = "monthly_delta_swe_init1980-2018_clim1981-2010_v1"

RUN = {
    "years": (1980, 2018),
    "climatology_years": (1981, 2010),
    "init_months": [5, 11],
    "members": [f"EN{i:02d}" for i in range(10)],
    "monthly_nlead": 24,
    "detrend": True,
}

E3SM_CASES = {
    "E3SM-FOSIRL": {
        "case_prefix": "WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL",
        "cache_tag": "JRA55_FOSIRL",
        "display_name": "E3SMv3-FOSIRL",
    },
    "E3SM-Reanalysis": {
        "case_prefix": "WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_BruteForce",
        "cache_tag": "Reanalysis",
        "display_name": "E3SMv3-Reanalysis",
    },
}


def safe_to_netcdf(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        ds.to_netcdf(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def reference_path(root: Path) -> Path:
    return leadtime_acc_dir(
        REFERENCE_PRODUCT, "inputs", "land", FIELD, root=root
    ) / (
        f"{REFERENCE_PRODUCT}_{FIELD}_monthly_{GRID_TAG}.nc"
    )


def model_path(root: Path, cache_tag: str, init_month: int) -> Path:
    return leadtime_acc_dir(
        cache_tag, "inputs", "land", FIELD, root=root
    ) / (
        f"{cache_tag}{init_month:02d}_{FIELD}_monthly_{GRID_TAG}.nc"
    )


def skill_path(root: Path, cache_tag: str, init_month: int) -> Path:
    y0, y1 = RUN["climatology_years"]
    trend = "detrend" if RUN["detrend"] else "nodetrend"
    return leadtime_acc_dir(
        cache_tag, "skill", "land", FIELD, root=root
    ) / (
        f"{cache_tag}{init_month:02d}_{FIELD}_{REFERENCE_PRODUCT}_"
        f"skill_clim_{y0}_{y1}_{trend}.nc"
    )


def contract_attrs(source_kind: str, source_name: str) -> dict:
    return {
        "processing_stage": "analysis_ready_monthly_delta_swe_acc_input_v1",
        "field": FIELD,
        "source_field": SOURCE_FIELD,
        "source_kind": source_kind,
        "source_name": source_name,
        "units": "mm",
        "horizontal_grid": GRID_TAG,
        "temporal_resolution": "monthly",
        "change_definition": "SWE(t) - SWE(t-1) from consecutive monthly means",
        "change_timestamp": "later month t",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "initialization_years": f"{RUN['years'][0]}-{RUN['years'][1]}",
        "climatology_years": f"{RUN['climatology_years'][0]}-{RUN['climatology_years'][1]}",
    }


def assert_target_grid(da: xr.DataArray) -> None:
    expected_lat = np.arange(-89.5, 90.0, 1.0)
    expected_lon = np.arange(0.5, 360.0, 1.0)
    if not np.array_equal(da.lat.values, expected_lat):
        raise ValueError("latitude is not the expected cell-centered 1-degree grid")
    if not np.array_equal(da.lon.values, expected_lon):
        raise ValueError("longitude is not the expected cell-centered 1-degree grid")


def prepare_reference(root: Path, raw_pattern: str, force: bool) -> xr.DataArray:
    output = reference_path(root)
    if output.exists() and not force:
        return xr.open_dataset(output, chunks={"time": 24, "lat": 45, "lon": 90})[FIELD]

    paths = sorted(glob.glob(raw_pattern))
    if not paths:
        raise FileNotFoundError(f"No C3S SWE files match {raw_pattern!r}")
    ds = xr.open_mfdataset(paths, combine="by_coords")
    if "swe" not in ds:
        raise KeyError("C3S input does not contain 'swe'")
    ref = ds.swe.rename({"latitude": "lat", "longitude": "lon"}) if (
        "latitude" in ds.swe.dims or "longitude" in ds.swe.dims
    ) else ds.swe
    ref = ref.assign_coords(lon=(ref.lon % 360)).sortby("lon").sortby("lat")
    ref = land_skill.mask_c3s_swe_flags(ref).chunk(
        {"time": 24, "lat": 45, "lon": 90}
    )
    change = land_skill.complete_calendar_monthly_change(ref).rename(FIELD)
    assert_target_grid(change)
    change.attrs.update(contract_attrs("reference", REFERENCE_PRODUCT))
    change.attrs.update(
        {
            "supported_change_months": "November,December,January,February,March,April,May",
            "unsupported_change_months": "June,July,August,September,October",
            "reference_is_anomaly": "false",
        }
    )
    safe_to_netcdf(change.to_dataset(name=FIELD), output)
    print("Wrote reference:", output)
    return xr.open_dataset(output, chunks={"time": 24, "lat": 45, "lon": 90})[FIELD]


def prepare_models(root: Path, data_dir: str, reference: xr.DataArray, force: bool) -> None:
    years = np.arange(RUN["years"][0], RUN["years"][1] + 1)
    chunks = {"Y": 3, "L": 24, "M": 2, "lat": 45, "lon": 90}
    for case_name, cfg in E3SM_CASES.items():
        for init_month in RUN["init_months"]:
            output = model_path(root, cfg["cache_tag"], init_month)
            if output.exists() and not force:
                print("Reuse model:", output)
                continue
            monthly = land_skill.load_e3sm_land_monthly(
                data_dir=data_dir,
                case_prefix=cfg["case_prefix"],
                members=RUN["members"],
                init_tags=data_access_e3sm.build_init_tags(years, init_month),
                field=SOURCE_FIELD,
                nlead=RUN["monthly_nlead"],
                chunks=chunks,
            )
            delta_ds = land_skill.monthly_land_hindcast_change_dataset(
                monthly, SOURCE_FIELD
            )
            delta, valid_time, dropped = land_skill.retain_reference_supported_leads(
                delta_ds[FIELD], delta_ds.time, reference
            )
            assert_target_grid(delta)
            land_skill.validate_hindcast_evaluation_setup(
                delta,
                valid_time,
                init_month=init_month,
                initialization_years=RUN["years"],
                expected_members=RUN["members"],
                climatology_years=RUN["climatology_years"],
                require_complete_member_grid=True,
            )
            delta.attrs.update(contract_attrs("model", case_name))
            delta.attrs["dropped_reference_unsupported_leads"] = ",".join(map(str, dropped))
            ready = delta.to_dataset(name=FIELD)
            ready["time"] = valid_time
            ready.attrs.update(contract_attrs("model", case_name))
            safe_to_netcdf(ready, output)
            print("Wrote model:", output, "kept leads:", delta.L.values.tolist())


def load_models(root: Path, reference: xr.DataArray):
    forecasts = {}
    times = {}
    expected = {}
    for case_name, cfg in E3SM_CASES.items():
        forecasts[case_name] = {}
        times[case_name] = {}
        expected[case_name] = {}
        for init_month in RUN["init_months"]:
            path = model_path(root, cfg["cache_tag"], init_month)
            ds = xr.open_dataset(
                path, chunks={"Y": -1, "L": 2, "M": 2, "lat": 45, "lon": 90}
            )
            forecast = ds[FIELD]
            land_skill.validate_land_reference_compatibility(forecast, reference)
            valid_time = ds.time.load()
            expected_years = land_skill.validate_hindcast_evaluation_setup(
                forecast,
                valid_time,
                init_month=init_month,
                initialization_years=RUN["years"],
                expected_members=RUN["members"],
                climatology_years=RUN["climatology_years"],
                require_complete_member_grid=True,
            )
            forecasts[case_name][init_month] = forecast
            times[case_name][init_month] = valid_time
            expected[case_name][init_month] = expected_years
    return forecasts, times, expected


def compute_skills(root: Path, reference: xr.DataArray, force: bool):
    forecasts, times, expected = load_models(root, reference)
    skills = {case: {} for case in E3SM_CASES}
    common_by_month = {}
    for init_month in RUN["init_months"]:
        leads_by_case = {
            tuple(map(int, forecasts[case][init_month].L.values)) for case in E3SM_CASES
        }
        if len(leads_by_case) != 1:
            raise ValueError(f"cases have different leads for init {init_month}")
        leads = list(next(iter(leads_by_case)))
        first_expected = expected[next(iter(E3SM_CASES))][init_month]
        if any(expected[case][init_month] != first_expected for case in E3SM_CASES):
            raise ValueError(f"cases have different target cohorts for init {init_month}")
        land_skill.validate_reference_time_coverage(
            reference,
            first_expected,
            times[next(iter(E3SM_CASES))][init_month],
            RUN["climatology_years"],
        )
        common = stats.common_valid_target_years_seasonal(
            {case: forecasts[case][init_month] for case in E3SM_CASES},
            {case: times[case][init_month] for case in E3SM_CASES},
            reference,
            leads,
            require_all_members=True,
        )
        if common != first_expected:
            raise ValueError(
                f"incomplete common target cohort for init {init_month}: {common}"
            )
        common_by_month[init_month] = common

    for case_name, cfg in E3SM_CASES.items():
        for init_month in RUN["init_months"]:
            output = skill_path(root, cfg["cache_tag"], init_month)
            if output.exists() and not force:
                skill = xr.open_dataset(output).load()
                print("Reuse skill:", output)
            else:
                y0, y1 = RUN["climatology_years"]
                skill = land_skill.compute_land_monthly_acc_skill(
                    forecasts[case_name][init_month],
                    times[case_name][init_month],
                    reference,
                    y0,
                    y1,
                    detrend=RUN["detrend"],
                    target_years_by_lead=common_by_month[init_month],
                ).compute()
                skill.attrs.update(
                    {
                        "reference_product": REFERENCE_PRODUCT,
                        "evaluation_protocol": EVALUATION_PROTOCOL,
                        "initialization_years": f"{RUN['years'][0]}-{RUN['years'][1]}",
                        "ensemble_members": ",".join(RUN["members"]),
                        "sample_alignment": "identical model/reference target years by lead",
                    }
                )
                safe_to_netcdf(skill, output)
                print("Wrote skill:", output)
            skills[case_name][init_month] = skill
    return skills


def verification_month(init_month: int, lead: int) -> int:
    return ((init_month + lead - 2) % 12) + 1


def style_axis(ax, row: int, col: int, nrows: int) -> None:
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    ax.set_xticks([-160, -80, 0, 80, 160], crs=ccrs.PlateCarree())
    ax.set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(
        LongitudeFormatter(zero_direction_label=True) if row == nrows - 1
        else plt.NullFormatter()
    )
    ax.yaxis.set_major_formatter(LatitudeFormatter() if col == 0 else plt.NullFormatter())
    ax.tick_params(
        labelsize=10, length=2, width=0.5, top=False, right=False,
        labelbottom=(row == nrows - 1), labelleft=(col == 0),
    )
    ax.gridlines(linewidth=0.3, color="0.4", alpha=0.3, draw_labels=False)


def plot_skills(skills, figure_dir: Path) -> list[Path]:
    import cartopy.crs as ccrs
    from matplotlib.offsetbox import AnchoredText

    month_names = np.asarray(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )
    column_specs = [
        (init_month, case_name)
        for init_month in RUN["init_months"]
        for case_name in E3SM_CASES
    ]
    projection = ccrs.PlateCarree()
    outputs = []
    for page, leads in enumerate((range(1, 13), range(13, 25)), start=1):
        leads = list(leads)
        nrows, ncols = len(leads), len(column_specs)
        fig = plt.figure(figsize=(17.6, 26.4))
        mappable = None
        for row, lead in enumerate(leads):
            for col, (init_month, case_name) in enumerate(column_specs):
                subplot = row * ncols + col + 1
                skill = skills[case_name][init_month]
                title = (
                    f"{month_names[init_month-1]} {E3SM_CASES[case_name]['display_name']}"
                    if row == 0 else ""
                )
                if lead in skill.L.values:
                    corr = skill.corr.sel(L=lead)
                    corr = corr.where(skill.valid_sample_count.sel(L=lead) == skill.sample_count.sel(L=lead))
                    corr = corr.where(skill.pval.sel(L=lead) < 0.1)
                    ax, mappable = maps.map_pcolor_global_subplot(
                        fig, corr, skill.lon, skill.lat,
                        0.1, -1.0, 1.0, title, nrows, ncols, subplot, projection,
                        cmap="blue2red_acc", cutoff=0.5, fontsize=13,
                    )
                else:
                    ax = fig.add_subplot(nrows, ncols, subplot, projection=projection)
                    ax.set_global()
                    ax.coastlines(linewidth=0.5, color="0.45")
                    ax.set_facecolor("0.94")
                    ax.set_title(title, fontsize=13, fontweight="bold")
                    reason = (
                        "Requires prior model month" if lead == 1
                        else "No consecutive C3S\nSWE observations"
                    )
                    ax.text(
                        0.5, 0.5, reason, transform=ax.transAxes,
                        ha="center", va="center", fontsize=10,
                        color="0.35", fontweight="bold",
                    )
                style_axis(ax, row, col, nrows)
                if col == 0 or column_specs[col - 1][0] != init_month:
                    month = month_names[verification_month(init_month, lead) - 1]
                    badge = AnchoredText(
                        f"L{lead:02d}: {month}", loc="lower left",
                        prop={"size": 9, "weight": "bold", "family": "monospace"},
                        frameon=True, pad=0.15, borderpad=0.25,
                    )
                    badge.patch.set(facecolor=(1, 1, 1, 0.8), edgecolor="0.3", linewidth=0.4)
                    ax.add_artist(badge)

        if mappable is None:
            raise RuntimeError("No supported monthly ΔSWE leads were available to plot")
        fig.suptitle(
            f"Monthly ΔSWE ACC versus C3S ΔSWE, leads {leads[0]}–{leads[-1]}\n"
            "ΔSWE(t) ≈ monthly-mean SWE(t) − monthly-mean SWE(t−1); linear detrend",
            fontsize=16, fontweight="bold", y=0.998,
        )
        fig.tight_layout(rect=[0.0, 0.035, 1.0, 0.982])
        fig.subplots_adjust(hspace=0.06, wspace=0.025)
        colorbar_ax = fig.add_axes([0.28, 0.012, 0.44, 0.008])
        colorbar = fig.colorbar(mappable, cax=colorbar_ax, orientation="horizontal")
        colorbar.set_label("ACC", fontsize=13, fontweight="bold")
        figure_dir.mkdir(parents=True, exist_ok=True)
        output = figure_dir / (
            f"fig_delta_h2osno_{REFERENCE_PRODUCT}_monthly_acc_leads_"
            f"{leads[0]:02d}_{leads[-1]:02d}_sigmask_p10.png"
        )
        mov.save_figure(
            fig,
            output,
            mode="",
            metric="monthly_delta_swe_leadtime_acc",
            title=f"Monthly ΔSWE ACC, leads {leads[0]}–{leads[-1]}",
            caption=(
                "Monthly change in SWE approximated from consecutive monthly means. "
                "Gray panels identify unavailable adjacent-month observations."
            ),
            dpi=250,
        )
        plt.close(fig)
        outputs.append(output)
        print("Saved figure:", output)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diag-root", "--input-root",
        dest="diag_root",
        type=Path,
        default=Path(S2D_DIAG_ROOT),
        help=(
            "Top-level S2D diagnostic root. --input-root is retained as an "
            "alias for compatibility."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="/global/cfs/cdirs/e3sm/S2S2D/post_process",
    )
    parser.add_argument(
        "--reference-pattern",
        default="/global/cfs/cdirs/e3sm/zhan391/data/C3S_SWE/1x1/monthly/swe_*.nc",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("/global/cfs/cdirs/e3sm/www/zhan391/esp-lab_diag"),
    )
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-skill", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="reuse existing skill caches and regenerate only the figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        skills = {
            case_name: {
                init_month: xr.open_dataset(
                    skill_path(args.diag_root, cfg["cache_tag"], init_month)
                ).load()
                for init_month in RUN["init_months"]
            }
            for case_name, cfg in E3SM_CASES.items()
        }
        plot_skills(skills, args.figure_dir)
        return
    reference = prepare_reference(
        args.diag_root, args.reference_pattern, args.force_preprocess
    )
    prepare_models(
        args.diag_root, args.data_dir, reference, args.force_preprocess
    )
    skills = compute_skills(args.diag_root, reference, args.force_skill)
    plot_skills(skills, args.figure_dir)


if __name__ == "__main__":
    main()
