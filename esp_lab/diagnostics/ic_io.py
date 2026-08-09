"""
ic_io.py
========
Filesystem I/O bridge for the IC analysis workflow.

This is the *only* module in the IC analysis stack that touches the
filesystem, reads NetCDF files, or computes checksums.  All scientific
logic lives in ``ic_core.py`` and can be tested without NERSC access.

Sections
--------
1. Start-date discovery       (discover_start_dates, match_start_dates)
2. Checksum computation       (compute_sha256)
3. rpointer parsing           (read_rpointer)
4. NetCDF I/O                 (open_restart_file, read_nc_dims,
                               read_nc_global_attrs, read_sim_timestamp)
5. FileRecord builder         (build_file_record, build_file_records_for_pair)
6. Manifest construction      (build_file_manifest)
7. Audit output               (write_audit_csv, load_audit_csv)
"""

from __future__ import annotations

import csv
import hashlib
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import xarray as xr

from .ic_core import (
    AuditResult,
    ComponentSpec,
    FileRecord,
    ICConfig,
    compare_file_records,
)


# ---------------------------------------------------------------------------
# Named tuple-like container for a matched date pair
# ---------------------------------------------------------------------------


class MatchedPair:
    """A pair of start-date directories (ref + test) for a single date.

    Attributes
    ----------
    date_str:
        Start date string in ``"YYYY-MM-DD-00000"`` format.
    ref_dir:
        Path to the reference experiment's start-date directory.
    test_dir:
        Path to the test experiment's start-date directory.
    ref_exists:
        Whether ``ref_dir`` exists on the filesystem.
    test_exists:
        Whether ``test_dir`` exists on the filesystem.
    """

    def __init__(
        self,
        date_str: str,
        ref_dir: Path,
        test_dir: Path,
    ) -> None:
        self.date_str = date_str
        self.ref_dir = ref_dir
        self.test_dir = test_dir
        self.ref_exists = ref_dir.is_dir()
        self.test_exists = test_dir.is_dir()

    @property
    def both_exist(self) -> bool:
        return self.ref_exists and self.test_exists

    def __repr__(self) -> str:
        return (
            f"MatchedPair(date={self.date_str!r}, "
            f"ref_exists={self.ref_exists}, test_exists={self.test_exists})"
        )


# ===========================================================================
# Section 1 — Start-date discovery
# ===========================================================================

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{5}$")


def discover_start_dates(root: Path, season_month: Optional[int] = None) -> List[str]:
    """List all start-date directories found under ``root``.

    Parameters
    ----------
    root:
        Root directory that directly contains ``YYYY-MM-DD-00000/`` subdirs.
    season_month:
        If provided (5 or 11), return only dates with that calendar month.

    Returns
    -------
    Sorted list of date strings matching ``YYYY-MM-DD-00000`` pattern.
    """
    root = Path(root)
    if not root.is_dir():
        warnings.warn(f"discover_start_dates: {root} is not a directory.", stacklevel=2)
        return []

    dates = [
        d.name
        for d in sorted(root.iterdir())
        if d.is_dir() and _DATE_RE.match(d.name)
    ]

    if season_month is not None:
        dates = [d for d in dates if int(d.split("-")[1]) == season_month]

    return dates


def match_start_dates(
    ref_root: Path,
    test_root: Path,
    season_month: Optional[int] = None,
) -> List[MatchedPair]:
    """Return matched pairs of start-date directories present in both roots.

    Dates present in one experiment but not the other are warned about but
    not included in the returned list.

    Parameters
    ----------
    ref_root:
        Root directory for the reference experiment.
    test_root:
        Root directory for the test experiment.
    season_month:
        Filter to a specific calendar month (5 or 11) if provided.

    Returns
    -------
    List of ``MatchedPair`` objects, one per common date.
    """
    ref_dates = set(discover_start_dates(ref_root, season_month))
    test_dates = set(discover_start_dates(test_root, season_month))

    common = sorted(ref_dates & test_dates)
    only_ref = sorted(ref_dates - test_dates)
    only_test = sorted(test_dates - ref_dates)

    if only_ref:
        warnings.warn(
            f"match_start_dates: {len(only_ref)} date(s) exist only in ref "
            f"(e.g. {only_ref[0]}); skipping.",
            stacklevel=2,
        )
    if only_test:
        warnings.warn(
            f"match_start_dates: {len(only_test)} date(s) exist only in test "
            f"(e.g. {only_test[0]}); skipping.",
            stacklevel=2,
        )

    pairs = [
        MatchedPair(
            date_str=d,
            ref_dir=Path(ref_root) / d,
            test_dir=Path(test_root) / d,
        )
        for d in common
    ]
    return pairs


# ===========================================================================
# Section 2 — Checksum computation
# ===========================================================================


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 digest of a file.

    Parameters
    ----------
    path:
        Path to the file.
    chunk_size:
        Read chunk size in bytes (default 1 MiB).

    Returns
    -------
    Lowercase hex digest, or ``""`` if the file does not exist or is unreadable.
    """
    path = Path(path)
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                buf = fh.read(chunk_size)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except OSError as exc:
        warnings.warn(f"compute_sha256: cannot read {path}: {exc}", stacklevel=2)
        return ""


# ===========================================================================
# Section 3 — rpointer parsing
# ===========================================================================


def read_rpointer(path: Path) -> dict:
    """Parse an rpointer file and return its content.

    rpointer files typically contain one or two lines: the target filename and
    optionally a date string.

    Parameters
    ----------
    path:
        Path to the rpointer file.

    Returns
    -------
    Dict with keys ``"target"`` and ``"raw_lines"``.
    """
    path = Path(path)
    if not path.is_file():
        return {"target": "", "raw_lines": []}
    try:
        lines = path.read_text().strip().splitlines()
        return {"target": lines[0].strip() if lines else "", "raw_lines": lines}
    except OSError as exc:
        warnings.warn(f"read_rpointer: cannot read {path}: {exc}", stacklevel=2)
        return {"target": "", "raw_lines": []}


# ===========================================================================
# Section 4 — NetCDF I/O helpers
# ===========================================================================


def open_restart_file(
    path: Path,
    chunks: Optional[dict] = None,
    engine: str = "netcdf4",
) -> xr.Dataset:
    """Open a restart NetCDF file as an xr.Dataset.

    Parameters
    ----------
    path:
        Path to the NetCDF file.
    chunks:
        Dask chunking dict passed to ``xr.open_dataset``.
        Pass ``{}`` to let xarray choose chunk sizes.
    engine:
        NetCDF backend engine (default ``"netcdf4"``).

    Returns
    -------
    xr.Dataset (lazily loaded if chunks is not None).
    """
    path = Path(path)
    open_kw: dict = {"engine": engine}
    if chunks is not None:
        open_kw["chunks"] = chunks
    return xr.open_dataset(path, **open_kw)


def read_nc_dims(path: Path) -> Dict[str, int]:
    """Read only dimension sizes from a NetCDF file (no data loaded).

    Parameters
    ----------
    path:
        Path to the NetCDF file.

    Returns
    -------
    Dict mapping dimension name → size, or ``{}`` on failure.
    """
    try:
        with xr.open_dataset(path, engine="netcdf4", decode_cf=False) as ds:
            return dict(ds.sizes)
    except Exception as exc:
        warnings.warn(f"read_nc_dims: {path}: {exc}", stacklevel=2)
        return {}


def read_nc_global_attrs(path: Path) -> dict:
    """Read global attributes from a NetCDF file.

    Parameters
    ----------
    path:
        Path to the NetCDF file.

    Returns
    -------
    Dict of global attribute name → value, or ``{}`` on failure.
    """
    try:
        with xr.open_dataset(path, engine="netcdf4", decode_cf=False) as ds:
            return dict(ds.attrs)
    except Exception as exc:
        warnings.warn(f"read_nc_global_attrs: {path}: {exc}", stacklevel=2)
        return {}


def read_sim_timestamp(path: Path) -> str:
    """Attempt to read the simulation timestamp from a restart file.

    Tries, in order:
    1. The ``time`` coordinate (uses cftime or pandas parsing).
    2. The ``current_mday``, ``current_month``, ``current_year`` global attrs.
    3. The filename datestamp (``YYYY-MM-DD``-like pattern).

    Parameters
    ----------
    path:
        Path to the NetCDF file.

    Returns
    -------
    ISO-format timestamp string, or ``""`` if not determinable.
    """
    path = Path(path)
    # Strategy 1: time coordinate
    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            if "time" in ds.coords and ds.coords["time"].size > 0:
                t = ds.coords["time"].values[0]
                return str(t)
    except Exception:
        pass

    # Strategy 2: global attributes
    try:
        attrs = read_nc_global_attrs(path)
        yr = attrs.get("current_year") or attrs.get("year")
        mo = attrs.get("current_month") or attrs.get("month")
        dy = attrs.get("current_mday") or attrs.get("mday")
        if yr and mo and dy:
            return f"{int(yr):04d}-{int(mo):02d}-{int(dy):02d}"
    except Exception:
        pass

    # Strategy 3: filename pattern
    name = path.name
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)

    return ""


# ===========================================================================
# Section 5 — FileRecord builder
# ===========================================================================


def build_file_record(
    path: Path,
    component: str,
    experiment: str,
    start_date: str,
    member: str = "",
    compute_hash: bool = True,
    read_dims: bool = True,
    read_attrs: bool = False,
    read_timestamp: bool = True,
) -> FileRecord:
    """Build a ``FileRecord`` for a single restart file.

    Parameters
    ----------
    path:
        Path to the file (may or may not exist).
    component:
        Component name.
    experiment:
        Experiment label.
    start_date:
        Start-date string in ``"YYYY-MM-DD-00000"`` format.
    member:
        Ensemble member tag (``""`` for non-per-member files).
    compute_hash:
        Whether to compute SHA-256 (can be slow for large files).
    read_dims:
        Whether to read NetCDF dimension sizes.
    read_attrs:
        Whether to read global attributes.
    read_timestamp:
        Whether to attempt to read the simulation timestamp.

    Returns
    -------
    FileRecord
    """
    path = Path(path)
    exists = path.is_file()

    record = FileRecord(
        path=str(path),
        component=component,
        experiment=experiment,
        start_date=start_date,
        member=member,
        size_bytes=path.stat().st_size if exists else -1,
    )

    if exists:
        if compute_hash:
            record.sha256 = compute_sha256(path)
        if read_dims and path.suffix == ".nc":
            record.nc_dims = read_nc_dims(path)
        if read_attrs and path.suffix == ".nc":
            record.global_attrs = read_nc_global_attrs(path)
        if read_timestamp and path.suffix == ".nc":
            record.sim_timestamp = read_sim_timestamp(path)

    return record


def build_file_records_for_pair(
    pair: MatchedPair,
    component: ComponentSpec,
    ref_label: str,
    test_label: str,
    compute_hash: bool = True,
    hash_all_members: bool = True,
) -> Tuple[List[FileRecord], List[FileRecord]]:
    """Build FileRecord lists for both experiments for a single component.

    Handles per-member (atmospheric) and single-file (non-atmospheric)
    components.

    Parameters
    ----------
    pair:
        Matched start-date directory pair.
    component:
        ComponentSpec describing the file glob and member pattern.
    ref_label, test_label:
        Experiment labels.
    compute_hash:
        Whether to compute SHA-256 digests.
    hash_all_members:
        For per-member components, whether to hash every member file.
        If ``False``, only the first member is hashed.

    Returns
    -------
    (ref_records, test_records)
    """
    ref_records: List[FileRecord] = []
    test_records: List[FileRecord] = []

    for exp_label, exp_dir, records_list in [
        (ref_label, pair.ref_dir, ref_records),
        (test_label, pair.test_dir, test_records),
    ]:
        if not exp_dir.is_dir():
            # Directory missing — create a single MISSING record
            records_list.append(
                FileRecord(
                    path=str(exp_dir / component.file_glob),
                    component=component.name,
                    experiment=exp_label,
                    start_date=pair.date_str,
                    size_bytes=-1,
                )
            )
            continue

        matched_files = sorted(exp_dir.glob(component.file_glob))
        if not matched_files:
            records_list.append(
                FileRecord(
                    path=str(exp_dir / component.file_glob),
                    component=component.name,
                    experiment=exp_label,
                    start_date=pair.date_str,
                    size_bytes=-1,
                )
            )
            continue

        member_re = re.compile(component.member_pattern)

        for i, fpath in enumerate(matched_files):
            # Extract member tag
            m = member_re.search(fpath.name)
            member_tag = m.group(0) if m else ""

            # For per-member components, optionally skip non-first members
            do_hash = compute_hash and (hash_all_members or i == 0)

            record = build_file_record(
                path=fpath,
                component=component.name,
                experiment=exp_label,
                start_date=pair.date_str,
                member=member_tag,
                compute_hash=do_hash,
            )
            records_list.append(record)

            # For non-per-member components, only take the first file
            if not component.per_member:
                break

    return ref_records, test_records


# ===========================================================================
# Section 6 — Manifest construction
# ===========================================================================

AUDIT_MANIFEST_COLUMNS = [
    "start_date",
    "component",
    "member",
    "status",
    "ref_path",
    "test_path",
    "ref_size_bytes",
    "test_size_bytes",
    "ref_sha256",
    "test_sha256",
    "ref_timestamp",
    "test_timestamp",
    "ref_nc_dims",
    "test_nc_dims",
]


def build_file_manifest(
    pairs: List[MatchedPair],
    components: List[ComponentSpec],
    ref_label: str,
    test_label: str,
    compute_hash: bool = True,
    atm_hash_all_members: bool = True,
    non_atm_hash_per_member: bool = False,
) -> pd.DataFrame:
    """Build a complete file-level audit manifest for a list of matched pairs.

    For every (date, component, [member]) combination, compares the ref and
    test records and assigns an ``AuditResult``.

    Parameters
    ----------
    pairs:
        List of ``MatchedPair`` objects.
    components:
        List of ``ComponentSpec`` objects.
    ref_label, test_label:
        Experiment labels.
    compute_hash:
        Whether to compute SHA-256 digests.
    atm_hash_all_members:
        Hash every atmospheric member (needed for control check).
    non_atm_hash_per_member:
        Also hash every member for non-atmospheric components.

    Returns
    -------
    pd.DataFrame with one row per (date, component, member) combination.
    """
    rows: List[dict] = []

    for pair in pairs:
        for comp in components:
            hash_all = (
                atm_hash_all_members
                if comp.name == "atm"
                else non_atm_hash_per_member
            )

            ref_recs, test_recs = build_file_records_for_pair(
                pair=pair,
                component=comp,
                ref_label=ref_label,
                test_label=test_label,
                compute_hash=compute_hash,
                hash_all_members=hash_all,
            )

            # Pair ref and test records by member tag
            ref_by_member: Dict[str, FileRecord] = {r.member: r for r in ref_recs}
            test_by_member: Dict[str, FileRecord] = {r.member: r for r in test_recs}

            all_members = sorted(
                set(ref_by_member) | set(test_by_member)
            ) or [""]

            for member in all_members:
                ref_rec = ref_by_member.get(member)
                test_rec = test_by_member.get(member)

                # Build dummy records for missing side
                if ref_rec is None:
                    ref_rec = FileRecord(
                        path=str(pair.ref_dir / comp.file_glob),
                        component=comp.name,
                        experiment=ref_label,
                        start_date=pair.date_str,
                        member=member,
                        size_bytes=-1,
                    )
                if test_rec is None:
                    test_rec = FileRecord(
                        path=str(pair.test_dir / comp.file_glob),
                        component=comp.name,
                        experiment=test_label,
                        start_date=pair.date_str,
                        member=member,
                        size_bytes=-1,
                    )

                status = compare_file_records(ref_rec, test_rec)

                row = {
                    "start_date": pair.date_str,
                    "component": comp.name,
                    "member": member,
                    "status": status.value,
                    "ref_path": ref_rec.path,
                    "test_path": test_rec.path,
                    "ref_size_bytes": ref_rec.size_bytes,
                    "test_size_bytes": test_rec.size_bytes,
                    "ref_sha256": ref_rec.sha256,
                    "test_sha256": test_rec.sha256,
                    "ref_timestamp": ref_rec.sim_timestamp,
                    "test_timestamp": test_rec.sim_timestamp,
                    "ref_nc_dims": str(ref_rec.nc_dims),
                    "test_nc_dims": str(test_rec.nc_dims),
                }
                rows.append(row)

    # Preserve the public table contract even when ``pairs`` is empty.  This
    # lets notebook/reporting code inspect or group an empty result safely.
    return pd.DataFrame(rows, columns=AUDIT_MANIFEST_COLUMNS)


# ===========================================================================
# Section 7 — Audit output
# ===========================================================================


def write_audit_csv(
    df: pd.DataFrame,
    outdir: Path,
    prefix: str = "audit",
) -> Path:
    """Write the audit manifest DataFrame to a CSV file.

    Parameters
    ----------
    df:
        Audit manifest DataFrame.
    outdir:
        Output directory (created if it does not exist).
    prefix:
        Filename prefix; file is written as ``{prefix}.csv``.

    Returns
    -------
    Path to the written CSV file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{prefix}.csv"
    df.to_csv(outpath, index=False)
    return outpath


def load_audit_csv(path: Path) -> pd.DataFrame:
    """Load an audit manifest CSV previously written by ``write_audit_csv``.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path, dtype=str)


__all__ = [
    "AUDIT_MANIFEST_COLUMNS",
    # Section 1
    "MatchedPair",
    "discover_start_dates",
    "match_start_dates",
    # Section 2
    "compute_sha256",
    # Section 3
    "read_rpointer",
    # Section 4
    "open_restart_file",
    "read_nc_dims",
    "read_nc_global_attrs",
    "read_sim_timestamp",
    # Section 5
    "build_file_record",
    "build_file_records_for_pair",
    # Section 6
    "build_file_manifest",
    # Section 7
    "write_audit_csv",
    "load_audit_csv",
]
