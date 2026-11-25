#!/usr/bin/env python
"""Build NMME regional SST indices and ELI for one model (cross-source method).

For each start month, reads the model's members from the NMME archive on their
native grid (land masked), keeps the tropical band, removes the model's own
lead-dependent drift, and writes every SST index used by notebooks 3a/5a:

    <outdir>/NMME/sst_index_by_model/<model>/NMME_<model>_init<MM>_sst_indices.nc

Run once per model (the Slurm template runs all models as an array job).
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.modes_of_variability import analysis  # noqa: E402
from workflows.modes_of_variability.cross_source import (  # noqa: E402
    NMME_SST_TROPICAL_BAND,
    build_nmme_model_sst_indices,
)

DEFAULT_NMME_ROOT = "/global/cfs/cdirs/e3sm/S2S2D/NMME/data_hindcast_by_member"


def output_path(outdir: Path, model: str, init_month: int) -> Path:
    return outdir / "NMME" / "sst_index_by_model" / model / f"NMME_{model}_init{init_month:02d}_sst_indices.nc"


def process(args: argparse.Namespace, init_month: int) -> Path:
    path = output_path(args.outdir, args.model, init_month)
    if path.is_file() and not args.force:
        with xr.open_dataset(path) as existing:
            current = "ELI_seas" in existing
        if current:
            print(f"Reusing {path}")
            return path
        print(f"Rebuilding {path} (no seasonal ELI)")
    loader_args = argparse.Namespace(
        nmme_root=args.nmme_root, nmme_models=[args.model], nmme_field="sst", nmme_chunks="",
        nmme_sst_land_mask=True, nmme_fixed_dir=args.fixed_dir,
        start_year=args.years[0], end_year=args.years[1], monthly_nlead=args.monthly_nlead,
    )
    raw = analysis.nmme_field_dataset(init_month, analysis.mode_settings("PDO"), loader_args)
    lat0, lat1 = NMME_SST_TROPICAL_BAND
    sst = raw["SST"].sel(lat=slice(lat0, lat1)).astype("float32").load()
    product = build_nmme_model_sst_indices(
        sst, raw["time"].load(), model=args.model, climatology_years=args.clim_years,
    )
    product.attrs.update(init_month=str(init_month), archive=str(args.nmme_root))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        product.to_netcdf(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {path} ({dict(product.sizes)})")
    return path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="NMME model folder name, e.g. CMC1-CanCM3.")
    parser.add_argument("--outdir", type=Path, required=True, help="Diagnostics root (NMME/... is created inside).")
    parser.add_argument("--nmme-root", type=Path, default=Path(DEFAULT_NMME_ROOT))
    parser.add_argument("--fixed-dir", type=Path, default=None, help="Per-model land masks (default <outdir>/NMME/fixed).")
    parser.add_argument("--init-months", type=int, nargs="+", default=[5, 11])
    parser.add_argument("--years", type=int, nargs=2, default=[1980, 2020], help="Start years to read (all available within).")
    parser.add_argument("--clim-years", type=int, nargs=2, default=[1981, 2010])
    parser.add_argument("--monthly-nlead", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.fixed_dir is None:
        args.fixed_dir = args.outdir / "NMME" / "fixed"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    for init_month in args.init_months:
        process(args, init_month)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
