#!/usr/bin/env python3
"""
run_process_tc_track_e3sm.py  --  TempestExtremes TC tracking for E3SM hindcasts.

Loops over hindcast cases (init tags) and ensemble members, running
DetectNodes → StitchNodes → HistogramNodes.

Grid modes
----------
TempestExtremes supports two grid modes, selected automatically:

* **Native unstructured grid** (e.g. ne30pg2): pass ``--connect-file`` pointing
  to the TempestExtremes connectivity/mesh file.  ``DetectNodes`` will receive
  ``--in_connect`` and operate on the ``ncol``-indexed output directly.

* **Structured lat-lon grid** (e.g. post-processed 180×360): omit
  ``--connect-file``.  ``DetectNodes`` uses standard lat/lon dimension
  indexing.

Data layout assumed
-------------------
  {sim_dir}/{case}/{member}/archive/atm/hist/{case}.{member}.{stream_tag}.*.nc

where ``stream_tag`` defaults to ``eam.h2`` (6-hourly TC fields) and
``eam.h0`` contains PHIS for the static surface-height file.

Example — native grid (ne30pg2, default)
-----------------------------------------
  python scripts/run_process_tc_track_e3sm.py \\
      --cases WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_BruteForce_1980110100 \\
      --members EN01 EN02 --parset set2 --dry-run

  # All members, real run (ne30pg2 connect file is the default):
  python scripts/run_process_tc_track_e3sm.py \\
      --cases WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_BruteForce_1980110100 \\
      --workers 10

Example — structured lat-lon grid (post-processed output)
----------------------------------------------------------
  python scripts/run_process_tc_track_e3sm.py \\
      --cases MY_LATLON_CASE \\
      --stream-tag eam.h2_180x360 \\
      --connect-file '' \\
      --parset set2 --workers 10
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from netCDF4 import Dataset

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SIM_DIR_DEFAULT = None  # must be supplied via --sim-dir
OUTDIR_DEFAULT  = None  # must be supplied via --outdir

# TempestExtremes commands (must be on PATH or supply full path via --te-bin)
CMD_DETECT    = "DetectNodes"
CMD_STITCH    = "StitchNodes"
CMD_HISTOGRAM = "HistogramNodes"

# Known connectivity files (on CFS, stable paths)
CONNECT_FILE_NE30PG2 = "/global/cfs/cdirs/e3sm/zhan391/TempestExtremes/grid_info/outCS_ne30pg2_connect.txt"

VAR_PSL  = "PSL"
VAR_U10  = "UBOT"
VAR_V10  = "VBOT"
VAR_ZS   = "PHIS"
VAR_LAT  = "lat"
VAR_LON  = "lon"

# Warm-core parameter sets.
# Current E3SM Z200/Z300/Z500 fields are geopotential height in m, so the
# Yeager-style Z threshold is -6.0 m.  If using geopotential in m2/s2 instead,
# use the equivalent threshold -6.0 * 9.8 = -58.8 m2/s2.
PARSETS = {
    "set1": dict(wc1="Z200", wc2="Z500", wc_mag=-6.0,  vc="_DIFF"),
    "set2": dict(wc1="T200", wc2="T500", wc_mag=-0.4,  vc="_AVG"),
    "set3": dict(wc1="Z300", wc2="Z500", wc_mag=-6.0,  vc="_DIFF"),
    "set4": dict(wc1="T300", wc2="T500", wc_mag=-0.4,  vc="_AVG"),
    "set5": dict(wc1="T400", wc2=None,   wc_mag=-0.4,  vc=None),
}


def warm_core_expr(parset: dict) -> str:
    """Return the TempestExtremes warm-core expression for a parset."""
    wc1 = parset["wc1"]
    wc2 = parset.get("wc2")
    varcmd = parset.get("vc")
    if varcmd is None:
        return wc1
    return f"{varcmd}({wc1},{wc2})"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- required / data location ---
    p.add_argument(
        "--sim-dir",
        default=None,
        required=True,
        help="Root simulation directory containing case subdirectories. Required.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        required=True,
        help="Output root directory. Results land in OUTDIR/CASE/MEMBER/post/atm/tc-analysis/. Required.",
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help=(
            "Hindcast case name(s) to process (subdirectory names under "
            "--sim-dir).  If omitted, all subdirectories of --sim-dir are "
            "used."
        ),
    )
    p.add_argument(
        "--members",
        nargs="+",
        default=None,
        help=(
            "Ensemble member(s) to process, e.g. EN01 EN02.  "
            "If omitted, all EN* subdirs found in the case directory are used."
        ),
    )
    p.add_argument(
        "--nens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of ensemble members to process per case.  "
            "Applied after --members (or auto-discovery): only the first N "
            "members (sorted) are used.  Useful for quick tests or partial runs."
        ),
    )
    p.add_argument(
        "--stream-tag",
        default=None,
        required=True,
        help="History stream tag for 6-hourly TC input files (e.g. 'eam.h2'). Required.",
    )
    p.add_argument(
        "--phis-stream-tag",
        default=None,
        required=True,
        help="History stream tag for the monthly file containing PHIS (e.g. 'eam.h0'). Required.",
    )

    # --- TempestExtremes setup ---
    p.add_argument(
        "--te-bin",
        default=None,
        required=True,
        help="Directory containing TempestExtremes binaries. Pass '' to use system PATH. Required.",
    )
    p.add_argument(
        "--connect-file",
        default=None,
        required=True,
        help=(
            "Path to TempestExtremes connectivity/mesh file for native "
            "unstructured-grid runs.  "
            "When provided, both DetectNodes and StitchNodes receive "
            "--in_connect and operate on the native ncol grid (lon=col2, lat=col3).  "
            "Pass an empty string ('') for structured lat-lon output, "
            "with lon=col3 and lat=col4 in DetectNodes output. Required."
        ),
    )
    p.add_argument(
        "--grid",
        default=None,
        required=True,
        help=(
            "SE grid identifier (e.g. 'ne30pg2', 'ne120pg2', 'ne30np4'). Required.  "
            "When --connect-file does not already exist, the script auto-generates "
            "the connectivity file using GenerateCSMesh / GenerateVolumetricMesh / "
            "GenerateConnectivityFile before running DetectNodes."
        ),
    )

    # --- detection parameters ---
    p.add_argument("--parset",   default=None, required=True, choices=list(PARSETS),
                   help="Warm-core variable set. Required.")
    p.add_argument("--wc1", default=None,
                   help=(
                       "Override the selected --parset's primary warm-core field "
                       "(e.g. Z200, T200).  Leave unset to use the PARSETS default."
                   ))
    p.add_argument("--wc2", default=None,
                   help=(
                       "Override the selected --parset's secondary warm-core field.  "
                       "Pass '' to force a single-field parset (like set5).  "
                       "Leave unset to use the PARSETS default."
                   ))
    p.add_argument("--wc-mag", type=float, default=None,
                   help=(
                       "Override the selected --parset's warm-core magnitude "
                       "threshold.  Leave unset to use the PARSETS default."
                   ))
    p.add_argument("--wc-vc", default=None, choices=["_DIFF", "_AVG", ""],
                   help=(
                       "Override the selected --parset's combination operator.  "
                       "Pass '' for a single-field parset.  "
                       "Leave unset to use the PARSETS default."
                   ))
    p.add_argument("--psl-fo-mag",    type=float, default=200.0,
                   help="PSL closed-contour magnitude (Pa).  Default: 200.0")
    p.add_argument("--psl-fo-dist",   type=float, default=8.0,
                   help="PSL closed-contour max distance (deg).  Default: 8.0")
    p.add_argument("--wc-fo-dist",    type=float, default=8.0,
                   help="Warm-core closed-contour max distance (deg).  Default: 8.0")
    p.add_argument("--wc-max-offset", type=float, default=3.0,
                   help="Max PSL/warm-core separation (deg).  Default: 3.0")
    p.add_argument("--merge-dist",    type=float, default=6.0,
                   help="Min distance between candidates (deg).  Default: 6.0")
    p.add_argument("--zs-factor",     type=float, default=9.81,
                   help="Factor to convert PHIS (m²/s²) to m.  Default: 9.81")
    p.add_argument("--time-filter",   default="6hr",
                   help="Sampling frequency filter for DetectNodes.  Default: 6hr")

    # --- stitching/filtering parameters ---
    p.add_argument("--traj-range",      type=float, default=10.0,
                   help="Max travel distance per 6 h (deg).  Default: 10.0")
    p.add_argument("--traj-min-length", default="10",
                   help="Min cyclone lifetime (steps).  Default: 10")
    p.add_argument("--traj-max-gap",    default="3",
                   help="Max allowable gap in trajectory (steps).  Default: 3")
    p.add_argument("--max-topo",        type=float, default=150.0,
                   help="Max surface height under PSL minimum (m).  Default: 150.0")
    p.add_argument("--max-lat",         type=float, default=50.0,
                   help="Max latitude of PSL minimum (deg).  Default: 50.0")
    p.add_argument("--min-wind",        type=float, default=8.0,
                   help="Min 10-m wind speed (m/s).  Default: 8.0")
    p.add_argument("--sci-dist",        type=int,   default=9,
                   help="Threshold checking distance (steps).  Default: 9")

    # --- NCO tools ---
    p.add_argument(
        "--nco-bin",
        default=None,
        help=(
            "Directory containing NCO binaries (ncks, ncwa). "
            "If omitted, the script searches PATH.  "
            "Useful when running from an environment that has NCO at a known location."
        ),
    )

    # --- run control ---
    p.add_argument(
        "--workers", type=int, default=None, required=True,
        help="Number of parallel member jobs (uses multiprocessing). Required.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without executing any commands.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging.",
    )

    args = p.parse_args()

    # --- validate argument ranges ---
    if args.workers < 1:
        p.error("--workers must be at least 1")
    if args.nens is not None and args.nens < 1:
        p.error("--nens must be at least 1")
    if not 0 < args.max_lat <= 90:
        p.error("--max-lat must be in the interval (0, 90]")
    for name in ("psl_fo_dist", "wc_fo_dist", "wc_max_offset", "merge_dist",
                 "traj_range", "max_topo"):
        if getattr(args, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be non-negative")

    return args


# ---------------------------------------------------------------------------
# Connectivity file generation
# ---------------------------------------------------------------------------

_GRID_RE = re.compile(r"^ne(?P<res>\d+)(?P<suffix>pg2|np4)$")


def generate_connect_file(
    grid: str,
    out_dir: Path,
    te_bin: str,
    dry_run: bool,
) -> Path:
    """
    Generate a TempestExtremes connectivity file for a given SE grid.

    Supports both pg2 (FV, v2/v3 production) and np4 (CGLL, v1 production)
    grids.  Steps mirror the E3SM post-processing workflow:

    pg2 grids (e.g. ne30pg2):
        GenerateCSMesh  --res N --alt  → outCSMeshneN.g
        GenerateVolumetricMesh --np 2 --uniform → outCSneN.g
        GenerateConnectivityFile --out_type FV → connect_CSneN_v2.dat

    np4 grids (e.g. ne30np4):
        GenerateCSMesh  --res N --alt  → outCSneN.g
        GenerateConnectivityFile --out_type CGLL → connect_CSneN_v2.dat

    Parameters
    ----------
    grid : str
        Grid identifier, e.g. 'ne30pg2', 'ne120pg2', 'ne30np4'.
    out_dir : Path
        Directory where mesh and connectivity files are written.
    te_bin : str
        Directory with TempestExtremes binaries (empty → use PATH).
    dry_run : bool
        If True, log commands but do not execute.

    Returns
    -------
    Path
        Path to the generated connectivity file.
    """
    log = logging.getLogger(__name__)

    m = _GRID_RE.match(grid)
    if not m:
        raise ValueError(
            f"Unsupported grid '{grid}'. "
            "Expected forms: ne30pg2, ne30np4, ne120pg2, ne120np4."
        )
    res    = m.group("res")
    suffix = m.group("suffix")
    is_pg2 = suffix == "pg2"
    out_type = "FV" if is_pg2 else "CGLL"

    out_dir.mkdir(parents=True, exist_ok=True)

    cs_mesh      = out_dir / f"outCSMeshne{res}.g"   # only for pg2
    vol_mesh     = out_dir / f"outCSne{res}.g"        # final mesh
    connect_file = out_dir / f"connect_CSne{res}_v2.dat"

    if connect_file.exists() and not dry_run:
        log.info("  Connect file already exists: %s", connect_file)
        return connect_file

    gen_cs   = _cmd(te_bin, "GenerateCSMesh")
    gen_vol  = _cmd(te_bin, "GenerateVolumetricMesh")
    gen_conn = _cmd(te_bin, "GenerateConnectivityFile")

    if is_pg2:
        # Step 1: spectral-element mesh
        _run([gen_cs, "--res", res, "--alt", "--file", str(cs_mesh)],
             dry_run, out_dir)
        # Step 2: volumetric (FV) mesh from SE mesh
        _run([gen_vol, "--in", str(cs_mesh),
              "--out", str(vol_mesh), "--np", "2", "--uniform"],
             dry_run, out_dir)
    else:
        # np4: GenerateCSMesh writes directly to vol_mesh path
        _run([gen_cs, "--res", res, "--alt", "--file", str(vol_mesh)],
             dry_run, out_dir)

    # Step 3: connectivity file
    _run([gen_conn,
          "--in_mesh",   str(vol_mesh),
          "--out_type",  out_type,
          "--out_connect", str(connect_file)],
         dry_run, out_dir)

    log.info("  Generated connectivity file: %s", connect_file)
    return connect_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cmd(bin_dir: str, name: str) -> str:
    """Return full path to a TempestExtremes binary."""
    if bin_dir:
        return str(Path(bin_dir) / name)
    full = shutil.which(name)
    if full is None:
        raise FileNotFoundError(
            f"TempestExtremes binary '{name}' not found on PATH.  "
            "Use --te-bin to specify the directory."
        )
    return full


def _run(cmd: list[str], dry_run: bool, cwd: Path) -> None:
    """Log and optionally execute a shell command."""
    log = logging.getLogger(__name__)
    log.info("  $ %s", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("STDOUT:\n%s", result.stdout)
        log.error("STDERR:\n%s", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _run_capture(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a command and return captured output."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def find_members(case_dir: Path) -> list[str]:
    """Return sorted EN* subdirectory names under case_dir."""
    members = sorted(p.name for p in case_dir.iterdir()
                     if p.is_dir() and p.name.startswith("EN"))
    return members


def phis_static_is_valid(zsfil: Path, ncks: str, var_zs: str) -> bool:
    """Return True when PHIS_static contains only a timeless PHIS field."""
    if not zsfil.exists():
        return False

    result = _run_capture([ncks, "-m", str(zsfil)], zsfil.parent)
    if result.returncode != 0:
        return False

    variable_pattern = re.compile(r"^\s+\w+\s+(\w+)(?:\(([^)]*)\))?\s*;")
    variables: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        match = variable_pattern.match(line)
        if not match:
            continue
        name, raw_dims = match.groups()
        variables[name] = [
            dim.strip() for dim in (raw_dims or "").split(",") if dim.strip()
        ]

    if "time" in variables or "time_bnds" in variables:
        return False

    pattern = re.compile(rf"\b{re.escape(var_zs)}\(([^)]*)\)")
    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        dims = [dim.strip() for dim in match.group(1).split(",") if dim.strip()]
        return "time" not in dims

    return False


def build_phis_static(
    case_dir: Path,
    member: str,
    work_dir: Path,
    phis_stream_tag: str,
    var_zs: str,
    dry_run: bool,
    nco_bin: str | None = None,
) -> Path:
    """
    Extract a time-invariant PHIS file from the first monthly h0 file.

    Returns the path to the static PHIS file.
    """
    import uuid
    log = logging.getLogger(__name__)
    hist_dir = case_dir / member / "archive" / "atm" / "hist"

    # Find first h0 file (any member; PHIS is model-state, same for all)
    pattern = str(hist_dir / f"*.{phis_stream_tag}.*.nc")
    h0_files = sorted(glob.glob(pattern))
    if not h0_files:
        raise FileNotFoundError(
            f"No {phis_stream_tag} files found in {hist_dir}"
        )
    phis_src = Path(h0_files[0])

    if nco_bin:
        ncks = str(Path(nco_bin) / "ncks")
        ncwa = str(Path(nco_bin) / "ncwa")
    else:
        ncks = shutil.which("ncks")
        ncwa = shutil.which("ncwa")
    if ncks is None or ncwa is None:
        raise FileNotFoundError(
            "ncks/ncwa not found on PATH.  Activate an environment that "
            "includes NCO (e.g. e3sm_unified)."
        )

    zsfil = work_dir / "PHIS_static.nc"
    if phis_static_is_valid(zsfil, ncks, var_zs) and not dry_run:
        log.debug("  PHIS static file already exists and is valid: %s", zsfil)
        return zsfil
    if zsfil.exists() and not dry_run:
        log.info("  Rebuilding invalid/stale PHIS static file: %s", zsfil)

    log.info("  Building PHIS static file from %s", phis_src)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Extract PHIS at the first time step and remove the time dimension.
    # DetectNodes reads PHIS from the semicolon-appended auxiliary file;
    # a scalar (timeless) field is required for the _DIV(PHIS,...) outputcmd.
    unique_id = uuid.uuid4().hex
    tmp_first = work_dir / f"PHIS_first_time.{unique_id}.nc"
    tmp_averaged = work_dir / f"PHIS_static_with_time.{unique_id}.nc"
    tmp_static = work_dir / f"PHIS_static.{unique_id}.tmp.nc"

    _run([ncks, "-O", "-v", var_zs, "-d", "time,0",
          str(phis_src), str(tmp_first)], dry_run, work_dir)
    _run([ncwa, "-O", "-a", "time",
          str(tmp_first), str(tmp_averaged)], dry_run, work_dir)
    _run([ncks, "-O", "-x", "-v", "time,time_bnds",
          str(tmp_averaged), str(tmp_static)], dry_run, work_dir)

    if not dry_run:
        if not phis_static_is_valid(tmp_static, ncks, var_zs):
            raise RuntimeError(
                f"Generated PHIS static file is invalid: {tmp_static}"
            )
        # Atomic replace moves the file, so no need to unlink tmp_static afterwards.
        os.replace(tmp_static, zsfil)
        log.debug("  Successfully moved %s to %s", tmp_static.name, zsfil.name)
        tmp_first.unlink(missing_ok=True)
        tmp_averaged.unlink(missing_ok=True)
    else:
        log.info("  [dry-run] Would create static PHIS file: %s", zsfil)

    return zsfil


def collect_h2_files(
    case_dir: Path,
    member: str,
    stream_tag: str,
) -> list[Path]:
    """Return sorted list of stream files for one member."""
    hist_dir = case_dir / member / "archive" / "atm" / "hist"
    pattern = str(hist_dir / f"*.{stream_tag}.*.nc")
    files = sorted(glob.glob(pattern))
    return [Path(f) for f in files]


def build_file_list(
    h2_files: list[Path],
    zsfil: Path,
    list_path: Path,
) -> None:
    """
    Write a TempestExtremes input file list.

    Each line: ``<h2_file>;<phis_static>``
    All TC variables (PSL, UBOT, VBOT, T200/T500, …) are in the same h2 file;
    PHIS lives in the static file appended via semicolon.
    """
    with list_path.open("w") as fh:
        for f in h2_files:
            fh.write(f"{f};{zsfil}\n")


def _time_record_count(path: Path) -> int | None:
    """Return the time dimension length, or None when the file has no time dim."""
    with Dataset(path, "r") as ds:
        dim = ds.dimensions.get("time")
        if dim is None:
            return None
        return len(dim)


def filter_time_bearing_files(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split stream files into usable files and files with zero time records."""
    usable: list[Path] = []
    empty: list[Path] = []
    for path in files:
        ntime = _time_record_count(path)
        if ntime == 0:
            empty.append(path)
        else:
            usable.append(path)
    return usable, empty


def detect_rank_files(detect_out: Path) -> list[Path]:
    """Return MPI rank-suffixed DetectNodes outputs for detect_out."""
    pattern = re.compile(rf"^{re.escape(detect_out.name)}\d{{6}}\.dat$")
    return sorted(
        p for p in detect_out.parent.glob(f"{detect_out.name}*.dat")
        if pattern.match(p.name)
    )


def clear_detect_outputs(detect_out: Path) -> None:
    """Remove stale DetectNodes outputs before a fresh run."""
    detect_out.unlink(missing_ok=True)
    for part in detect_rank_files(detect_out):
        part.unlink(missing_ok=True)
    for log_file in detect_out.parent.glob("log[0-9]*.txt"):
        log_file.unlink(missing_ok=True)


def log_detect_failures(work_dir: Path) -> None:
    """Log concise TempestExtremes worker diagnostics after an empty run."""
    log = logging.getLogger(__name__)
    diagnostics = []
    for log_file in sorted(work_dir.glob("log[0-9]*.txt")):
        text = log_file.read_text(errors="replace").strip()
        if text:
            diagnostics.append(f"{log_file.name}: {text}")
    if diagnostics:
        log.error("  DetectNodes worker diagnostics:\n%s", "\n".join(diagnostics))


def consolidate_detect_output(detect_out: Path) -> bool:
    """
    Ensure DetectNodes output exists at detect_out.

    Serial DetectNodes writes detect_out directly.  MPI DetectNodes writes
    rank-suffixed files like detect_out000000.dat; StitchNodes expects one
    candidate file, so collect all time blocks and write them in chronological
    order.  This keeps the downstream StitchNodes input consistent across
    serial and MPI DetectNodes runs.
    """
    sources: list[Path] = []
    if detect_out.exists() and detect_out.stat().st_size > 0:
        sources.append(detect_out)
    sources.extend(p for p in detect_rank_files(detect_out) if p.stat().st_size > 0)
    if not sources:
        return False

    try:
        blocks = read_detect_blocks(sources)
    except ValueError as exc:
        logging.getLogger(__name__).error("  Malformed DetectNodes output: %s", exc)
        return False

    if not blocks:
        logging.getLogger(__name__).warning("  No data blocks found in DetectNodes outputs.")
        return False

    # Chronological sort of data blocks
    blocks.sort(key=lambda item: item[0])
    with detect_out.open("w") as out_fh:
        for _, lines in blocks:
            out_fh.writelines(lines)

    return detect_out.stat().st_size > 0


def read_detect_blocks(paths: list[Path]) -> list[tuple[tuple[int, int, int, int, int], list[str]]]:
    """
    Read DetectNodes candidate files as sortable time blocks.

    Header lines have the form: year month day candidate_count hour.  Candidate
    rows follow the header and begin with whitespace.  The final sort key uses
    the source order as a tie-breaker for the unlikely case of duplicate headers.
    """
    blocks: list[tuple[tuple[int, int, int, int, int], list[str]]] = []
    current_key: tuple[int, int, int, int, int] | None = None
    current_expected_count: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_expected_count, current_lines
        if current_key is not None:
            found_count = sum(1 for line in current_lines[1:] if line.strip())
            if current_expected_count != found_count:
                year, month, day, hour, _sequence = current_key
                raise ValueError(
                    f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:00 "
                    f"expected {current_expected_count} candidate rows, "
                    f"found {found_count}"
                )
            blocks.append((current_key, current_lines))
        current_key = None
        current_expected_count = None
        current_lines = []

    sequence = 0
    for path in paths:
        with path.open() as fh:
            for line in fh:
                maybe_header = detect_header(line, sequence)
                if maybe_header is not None:
                    flush()
                    current_key, current_expected_count = maybe_header
                    sequence += 1
                    current_lines = [line]
                elif current_key is not None:
                    current_lines.append(line)
        flush()

    return blocks


def detect_header(line: str, sequence: int) -> tuple[tuple[int, int, int, int, int], int] | None:
    """Return a sortable key and expected row count for a DetectNodes header."""
    if not line or line[0].isspace():
        return None

    fields = line.split()
    if len(fields) < 5:
        return None

    try:
        year, month, day, count, hour = (int(field) for field in fields[:5])
    except ValueError:
        return None

    return (year, month, day, hour, sequence), count


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_member(
    case: str,
    member: str,
    sim_dir: Path,
    out_root: Path,
    args: argparse.Namespace,
) -> str:
    """
    Run the full TC tracking pipeline for one (case, member).

    Returns a status string: 'ok', 'skipped', 'no_data', 'failed'.
    """
    log = logging.getLogger(__name__)
    log.info("=== %s / %s ===", case, member)

    case_dir = sim_dir / case
    work_dir = out_root / case / member / "post" / "atm" / "tc-analysis"
    parset  = dict(PARSETS[args.parset])
    # Allow the caller (e.g. a notebook looping over parsets) to override the
    # warm-core definition for this run without editing the PARSETS table.
    if args.wc1 is not None:
        parset["wc1"] = args.wc1
    if args.wc2 is not None:
        parset["wc2"] = args.wc2 if args.wc2 != "" else None
    if args.wc_mag is not None:
        parset["wc_mag"] = args.wc_mag
    if args.wc_vc is not None:
        parset["vc"] = args.wc_vc if args.wc_vc != "" else None
    wc_expr = warm_core_expr(parset)
    wc_mag  = parset["wc_mag"]

    # Output filenames
    detect_out = work_dir / f"{case}_{member}_{args.parset}_TCS_detect.txt"
    track_out  = work_dir / f"{case}_{member}_{args.parset}_TCS_track.txt"
    hist_out   = work_dir / f"{case}_{member}_{args.parset}_TCS_hist.nc"
    track_tmp  = track_out.with_name(track_out.name + ".tmp")
    hist_tmp   = hist_out.with_name(hist_out.name + ".tmp")

    outputs_complete = (
        track_out.is_file() and track_out.stat().st_size > 0
        and hist_out.is_file() and hist_out.stat().st_size > 0
    )
    if outputs_complete and not args.force:
        log.info("  Skipping — complete outputs already exist: %s", track_out)
        return "skipped"

    # --- Collect h2 files ---
    h2_files = collect_h2_files(case_dir, member, args.stream_tag)
    if not h2_files:
        log.warning("  No %s files found for %s/%s", args.stream_tag, case, member)
        return "no_data"
    log.info("  Found %d %s files", len(h2_files), args.stream_tag)
    try:
        h2_files, empty_time_files = filter_time_bearing_files(h2_files)
    except Exception as exc:
        log.error("  Failed to inspect %s time dimensions: %s", args.stream_tag, exc)
        return "failed"
    if empty_time_files:
        log.warning(
            "  Skipping %d %s files with zero time records: %s",
            len(empty_time_files),
            args.stream_tag,
            ", ".join(p.name for p in empty_time_files),
        )
    if not h2_files:
        log.error("  No %s files with time records remain for %s/%s", args.stream_tag, case, member)
        return "no_data"

    if args.dry_run:
        log.info("  [dry-run] Would process %d files", len(h2_files))
        log.info("  [dry-run] Output: %s", track_out)
        return "dry_run"

    # Clear any partial outputs left over from a previous failed run.
    track_tmp.unlink(missing_ok=True)
    hist_tmp.unlink(missing_ok=True)

    work_dir.mkdir(parents=True, exist_ok=True)

    # --- PHIS static file (built once, shared across members via case work_dir) ---
    case_work = out_root / case
    try:
        zsfil = build_phis_static(
            case_dir, member, case_work,
            phis_stream_tag=args.phis_stream_tag,
            var_zs=VAR_ZS,
            dry_run=args.dry_run,
            nco_bin=args.nco_bin,
        )
    except Exception as exc:
        log.error("  Failed to build PHIS static: %s", exc)
        return "failed"

    # --- Build file list ---
    list_path = work_dir / f"filelist_{args.parset}.txt"
    build_file_list(h2_files, zsfil, list_path)
    log.info("  File list: %s (%d entries)", list_path, len(h2_files))

    # --- Resolve TempestExtremes binaries ---
    try:
        cmd_detect    = _cmd(args.te_bin, CMD_DETECT)
        cmd_stitch    = _cmd(args.te_bin, CMD_STITCH)
        cmd_histogram = _cmd(args.te_bin, CMD_HISTOGRAM)
    except FileNotFoundError as exc:
        log.error("  %s", exc)
        return "failed"

    # --- Base DetectNodes / StitchNodes arguments ---
    detect_args = [
        cmd_detect,
        "--in_data_list", str(list_path),
        "--verbosity", "0",
        "--closedcontourcmd",
            f"{VAR_PSL},{args.psl_fo_mag},{args.psl_fo_dist},0;"
            f"{wc_expr},{wc_mag},{args.wc_fo_dist},{args.wc_max_offset}",
        "--mergedist",    str(args.merge_dist),
        "--searchbymin",  VAR_PSL,
        "--outputcmd",
            f"{VAR_PSL},min,0;"
            f"_VECMAG({VAR_U10},{VAR_V10}),max,2;"
            f"_DIV({VAR_ZS},{args.zs_factor}),min,0",
        "--timefilter",   args.time_filter,
        "--out",          str(detect_out),
    ]

    # For native SE unstructured grid: pass --in_connect; lat/lon come from
    # the connectivity file so --latname/--lonname must NOT be given.
    # For structured lat-lon mode: pass --latname/--lonname instead.
    if args.connect_file:
        detect_args += ["--in_connect", args.connect_file]
    else:
        detect_args += ["--latname", VAR_LAT, "--lonname", VAR_LON]

    # Column indices in DetectNodes output differ by grid mode:
    #   native SE:        lon=col2, lat=col3  (run_process_e3sm_set2.csh)
    #   structured lat-lon: lon=col3, lat=col4
    if args.connect_file:
        iloncol, ilatcol = "2", "3"
    else:
        iloncol, ilatcol = "3", "4"

    stitch_args = [
        cmd_stitch,
        "--in",      str(detect_out),
        "--out",     str(track_tmp),
        "--in_fmt",  "lon,lat,slp,wind,zs",
        "--range",   str(args.traj_range),
        "--mintime", str(args.traj_min_length),
        "--maxgap",  str(args.traj_max_gap),
        "--threshold",
            f"wind,>=,{args.min_wind},{args.sci_dist};"
            f"lat,<=,{args.max_lat},{args.sci_dist};"
            f"lat,>=,-{args.max_lat},{args.sci_dist};"
            f"zs,<=,{args.max_topo},{args.sci_dist}",
    ]
    # StitchNodes also needs --in_connect for native grid
    if args.connect_file:
        stitch_args += ["--in_connect", args.connect_file]

    hist_args = [
        cmd_histogram,
        "--in",      str(track_tmp),
        "--iloncol", iloncol,
        "--ilatcol", ilatcol,
        "--out",     str(hist_tmp),
    ]

    # --- Execute ---
    try:
        clear_detect_outputs(detect_out)

        log.info("  Running DetectNodes …")
        _run(detect_args, dry_run=False, cwd=work_dir)

        if not consolidate_detect_output(detect_out):
            log_detect_failures(work_dir)
            log.error(
                "  DetectNodes produced no usable output: %s or %s*.dat",
                detect_out,
                detect_out,
            )
            return "failed"

        log.info("  Running StitchNodes …")
        _run(stitch_args, dry_run=False, cwd=work_dir)

        if not track_tmp.is_file() or track_tmp.stat().st_size == 0:
            raise RuntimeError(f"StitchNodes produced an empty track file: {track_tmp}")

        # HistogramNodes must run before track_tmp is renamed.
        log.info("  Running HistogramNodes …")
        _run(hist_args, dry_run=False, cwd=work_dir)

        if not hist_tmp.is_file() or hist_tmp.stat().st_size == 0:
            raise RuntimeError(f"HistogramNodes produced an empty histogram file: {hist_tmp}")

        # Publish both products only after the full pipeline succeeds.
        os.replace(track_tmp, track_out)
        os.replace(hist_tmp, hist_out)

    except (subprocess.CalledProcessError, RuntimeError) as exc:
        track_tmp.unlink(missing_ok=True)
        hist_tmp.unlink(missing_ok=True)
        log.error("  Command failed: %s", exc)
        return "failed"

    log.info("  Done → %s", track_out)
    return "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    sim_dir  = Path(args.sim_dir)
    out_root = Path(args.outdir)

    # --- Resolve cases ---
    if args.cases:
        cases = args.cases
    else:
        cases = sorted(p.name for p in sim_dir.iterdir() if p.is_dir())

    # --- Auto-generate connectivity file if needed ---
    if args.grid and args.connect_file:
        connect_path = Path(args.connect_file)
        if not connect_path.exists():
            log.info(
                "Connect file not found; generating for grid '%s' ...", args.grid
            )
            connect_dir = connect_path.parent
            try:
                generated = generate_connect_file(
                    args.grid, connect_dir, args.te_bin, args.dry_run
                )
                if not args.dry_run and (
                    not generated.is_file() or generated.stat().st_size == 0
                ):
                    raise RuntimeError(
                        f"Connectivity generation completed but produced no valid "
                        f"file: {generated}"
                    )
                # Update args so process_member picks up the correct path
                args.connect_file = str(generated)
            except Exception as exc:
                log.error("Failed to generate connectivity file: %s", exc)
                sys.exit(1)
        else:
            log.info("Using existing connect file: %s", args.connect_file)

    grid_mode = (
        f"native unstructured (connect={args.connect_file})"
        if args.connect_file
        else "structured lat-lon"
    )
    log.info("Grid mode  : %s", grid_mode)
    log.info("Stream tag : %s", args.stream_tag)
    log.info("Cases (%d): %s", len(cases), cases)

    # --- Build work list: (case, member) pairs ---
    work_items: list[tuple[str, str]] = []
    cases_members: dict[str, list[str]] = {}
    for case in cases:
        case_dir = sim_dir / case
        if not case_dir.is_dir():
            log.warning("Case directory not found: %s", case_dir)
            continue
        members = args.members if args.members else find_members(case_dir)
        if not members:
            log.warning("No EN* members found in %s", case_dir)
            continue
        if args.nens is not None:
            members = members[: args.nens]
            log.info("  --nens %d: using members %s", args.nens, members)
        if not members:
            log.warning("--nens %d leaves no members for case %s", args.nens, case)
            continue
        cases_members[case] = members
        for member in members:
            work_items.append((case, member))

    log.info("Total (case, member) pairs to process: %d", len(work_items))
    if args.dry_run:
        log.info("=== DRY RUN — no files will be written ===")

    # --- Build PHIS_static.nc once per case, in the parent process, before ---
    # --- starting the worker pool.  All members of a case share this file,  ---
    # --- so building it up front avoids redundant/concurrent NCO operations. ---
    failed_cases: set[str] = set()
    if not args.dry_run:
        for case, members in cases_members.items():
            case_dir  = sim_dir / case
            case_work = out_root / case
            last_exc: Exception | None = None
            for member in members:
                try:
                    build_phis_static(
                        case_dir, member, case_work,
                        phis_stream_tag=args.phis_stream_tag,
                        var_zs=VAR_ZS,
                        dry_run=args.dry_run,
                        nco_bin=args.nco_bin,
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    continue
            if last_exc is not None:
                log.error(
                    "  Could not build PHIS static for case %s: %s — "
                    "excluding this case from the run",
                    case, last_exc,
                )
                failed_cases.add(case)

    # --- Run ---
    counters = {"ok": 0, "skipped": 0, "dry_run": 0, "no_data": 0, "failed": 0}

    if failed_cases:
        excluded_items = [item for item in work_items if item[0] in failed_cases]
        work_items = [item for item in work_items if item[0] not in failed_cases]
        counters["failed"] += len(excluded_items)
        log.info(
            "Excluded %d case(s) (%d (case, member) pairs) due to PHIS static "
            "build failure; %d (case, member) pairs remain",
            len(failed_cases), len(excluded_items), len(work_items),
        )

    if args.workers > 1 and not args.dry_run:
        import multiprocessing as mp
        with mp.Pool(processes=args.workers) as pool:
            results = [
                pool.apply_async(
                    process_member,
                    (case, member, sim_dir, out_root, args),
                )
                for case, member in work_items
            ]
            for r in results:
                try:
                    status = r.get()
                except Exception:
                    log.exception("Unhandled worker exception")
                    status = "failed"
                counters[status] = counters.get(status, 0) + 1
    else:
        for case, member in work_items:
            try:
                status = process_member(case, member, sim_dir, out_root, args)
            except Exception:
                log.exception("Unhandled exception processing %s/%s", case, member)
                status = "failed"
            counters[status] = counters.get(status, 0) + 1

    # --- Summary ---
    log.info("")
    log.info("=== Summary ===")
    log.info("  OK        : %d", counters["ok"])
    log.info("  Skipped   : %d", counters["skipped"])
    log.info("  Dry-run   : %d", counters["dry_run"])
    log.info("  No data   : %d", counters["no_data"])
    log.info("  Failed    : %d", counters["failed"])

    if counters["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
