#!/usr/bin/env python3
"""Generate and save native MPAS-Ocean Equatorial Longitude Index (ELI) for E3SM."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esp_lab.diagnostics.native_eli import (
    DEFAULT_MPAS_MESH_FILE,
    DEFAULT_RAW_SIMULATION_DIR,
    process_native_eli_case,
)

LOG = logging.getLogger(__name__)

DEFAULT_S2D_DIAG_ROOT = Path("/global/cfs/cdirs/e3sm/S2S2D/s2d_diag")

KNOWN_CASES: dict[str, dict[str, str]] = {
    "JRA55_FOSIRL": {
        "display_name": "E3SM-FOSIRL",
        "data_dir": "/global/cfs/cdirs/e3smdata/simulations/S2S2D",
        "case_prefix": "WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL",
        "cache_tag": "JRA55_FOSIRL",
    },
    "Reanalysis": {
        "display_name": "E3SM-Reanalysis",
        "data_dir": "/global/cfs/cdirs/e3sm/S2S2D/simulation",
        "case_prefix": "WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_Reanalysis",
        "cache_tag": "Reanalysis",
    },
    "4DEnVarOcn": {
        "display_name": "E3SM-4DEnVarOcn",
        "data_dir": "/global/cfs/cdirs/e3sm/S2S2D/simulation",
        "case_prefix": "WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_4DEnVarOcn",
        "cache_tag": "4DEnVarOcn",
        "year_end": 2011,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        choices=list(KNOWN_CASES.keys()),
        help=f"Named presets to process. Known cases: {list(KNOWN_CASES.keys())}. If omitted, uses explicit --e3sm-* arguments.",
    )
    p.add_argument(
        "--outdir",
        default=str(DEFAULT_S2D_DIAG_ROOT),
        help=f"Base diagnostics output root. Default: {DEFAULT_S2D_DIAG_ROOT}",
    )
    p.add_argument(
        "--mesh-file",
        default=str(DEFAULT_MPAS_MESH_FILE),
        help=f"Path to MPAS-Ocean restart/mesh geometry file. Default: {DEFAULT_MPAS_MESH_FILE}",
    )
    p.add_argument(
        "--e3sm-data-dir",
        default=str(DEFAULT_RAW_SIMULATION_DIR),
        help=f"Raw simulation directory containing case runs. Default: {DEFAULT_RAW_SIMULATION_DIR}",
    )
    p.add_argument(
        "--e3sm-case-prefix",
        default="WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL",
        help="Case prefix name before the initialization timestamp.",
    )
    p.add_argument(
        "--e3sm-cache-tag",
        default="JRA55_FOSIRL",
        help="Subdirectory tag under --outdir (e.g. JRA55_FOSIRL).",
    )
    p.add_argument(
        "--e3sm-display-name",
        default=None,
        help="Display name stored in NetCDF attributes.",
    )
    p.add_argument(
        "--init-months",
        nargs="+",
        type=int,
        default=[5, 11],
        help="Initialization months to process. Default: 5 11",
    )
    p.add_argument(
        "--year-start",
        type=int,
        default=1980,
        help="Start year. Default: 1980",
    )
    p.add_argument(
        "--year-end",
        type=int,
        default=2018,
        help="End year. Default: 2018",
    )
    p.add_argument(
        "--nlead",
        type=int,
        default=24,
        help="Number of forecast lead months. Default: 24",
    )
    p.add_argument(
        "--e3sm-nens",
        type=int,
        default=10,
        help="Number of ensemble members. Default: 10",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker concurrency (default: 1, avoiding thread HDF5 lock errors).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite of existing files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    base_outdir = Path(args.outdir)

    if args.cases:
        case_list = [
            (case_name, KNOWN_CASES[case_name])
            for case_name in args.cases
        ]
    else:
        case_list = [
            (
                args.e3sm_cache_tag,
                {
                    "case_prefix": args.e3sm_case_prefix,
                    "data_dir": args.e3sm_data_dir,
                    "cache_tag": args.e3sm_cache_tag,
                    "display_name": args.e3sm_display_name or args.e3sm_cache_tag,
                },
            )
        ]

    for case_label, info in case_list:
        case_outdir = base_outdir / info["cache_tag"] / "sst_index" / "timeseries"
        LOG.info("Processing native MPAS-Ocean ELI for case: %s", case_label)
        LOG.info("  Data dir : %s", info["data_dir"])
        LOG.info("  Out dir  : %s", case_outdir)

        written = process_native_eli_case(
            case_prefix=info["case_prefix"],
            data_dir=info["data_dir"],
            outdir=case_outdir,
            mesh_path=args.mesh_file,
            cache_tag=info["cache_tag"],
            display_name=info.get("display_name"),
            init_months=args.init_months,
            year_start=args.year_start,
            year_end=info.get('year_end', args.year_end),
            nlead=args.nlead,
            nens=args.e3sm_nens,
            workers=args.workers,
            force=args.force,
        )
        for p in written:
            LOG.info("  -> %s", p)

    LOG.info("All native ELI processing completed successfully.")


if __name__ == "__main__":
    main()
