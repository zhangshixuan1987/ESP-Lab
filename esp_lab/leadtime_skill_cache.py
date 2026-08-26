"""Reproducible provenance helpers for lead-time skill-cache products."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


SOURCE_FINGERPRINT_VERSION = "1"
RESAMPLING_ALGORITHM_VERSION = "numpy_choice_without_replacement_v1"
_SKILL_CACHE_SCHEMA_VERSION = "2"


def year_number(value: Any) -> int:
    """Extract a four-digit year from integer or datetime-like coordinates."""
    return int(str(value)[:4])


def verification_year_token(verification_years) -> str:
    """Return a readable token identifying an exact verification-year span."""
    years = [year_number(value) for value in verification_years]
    if not years:
        raise ValueError("verification_years must not be empty")
    return f"y{years[0]}-{years[-1]}_ny{len(years)}"


def expected_skill_cache_attrs(
    *,
    source: str,
    source_data_identity: str,
    case_prefix: str,
    variable: str,
    init_month: int,
    verification_years,
    climatology_years: tuple[int, int],
    observation_product: str,
    observation_variable: str,
    observation_data_identity: str,
    target_grid: str,
    regridding_method: str,
    ensemble_member_count: int,
    lead_start: int,
    lead_end: int,
    detrend: bool,
    unit_conversion_version: str,
    cache_kind: str = "skill",
) -> dict[str, str | int]:
    """Build the complete compatibility contract for a computed-skill cache."""
    years = [year_number(value) for value in verification_years]
    if not years:
        raise ValueError("verification_years must not be empty")
    clim_start, clim_end = climatology_years
    return {
        "skill_cache_version": _SKILL_CACHE_SCHEMA_VERSION,
        "cache_kind": str(cache_kind),
        "source": str(source),
        "source_data_identity": str(source_data_identity),
        "case_prefix": str(case_prefix),
        "variable": str(variable),
        "initialization_month": int(init_month),
        "verification_years": ",".join(str(year) for year in years),
        "verification_year_count": len(years),
        "climatology_start_year": int(clim_start),
        "climatology_end_year": int(clim_end),
        "observation_product": str(observation_product),
        "observation_variable": str(observation_variable),
        "observation_data_identity": str(observation_data_identity),
        "target_grid": str(target_grid),
        "regridding_method": str(regridding_method),
        "ensemble_member_count": int(ensemble_member_count),
        "lead_start": int(lead_start),
        "lead_end": int(lead_end),
        "detrend": int(bool(detrend)),
        "unit_conversion_version": str(unit_conversion_version),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} in cache provenance")


def canonical_json(value: Any) -> str:
    """Serialize provenance deterministically across processes and cache states."""
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def provenance_digest(value: Any, *, length: int = 16) -> str:
    """Return a stable SHA-256 prefix for a provenance payload."""
    if length < 8 or length > 64:
        raise ValueError("length must be between 8 and 64")
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def source_fingerprint(
    identity: Mapping[str, Any],
    *,
    source_revision: str,
) -> str:
    """Fingerprint a logical source plus an explicit in-place revision token.

    This intentionally avoids walking a remote archive during a cache-only run.
    ``source_revision`` must be changed when files are replaced in place; it can
    also be set to an upstream manifest checksum when one is available.
    """
    if not source_revision.strip():
        raise ValueError("source_revision must be a non-empty string")
    payload = {
        "fingerprint_version": SOURCE_FINGERPRINT_VERSION,
        "identity": dict(identity),
        "source_revision": source_revision,
    }
    return f"sha256:{provenance_digest(payload, length=32)}"


def file_inventory_digest(
    paths: list[str | Path],
    *,
    root: str | Path | None = None,
) -> str:
    """Digest an exact input inventory using path, size, and modification time.

    Callers should pass the files actually opened by upstream preprocessing.
    This is much cheaper than hashing multi-gigabyte contents while still
    detecting ordinary in-place replacements. An upstream content-manifest
    checksum remains preferable when available.
    """
    root_path = Path(root).resolve() if root is not None else None
    records = []
    for raw_path in sorted((Path(path).resolve() for path in paths), key=str):
        stat = raw_path.stat()
        if root_path is None:
            name = str(raw_path)
        else:
            try:
                name = str(raw_path.relative_to(root_path))
            except ValueError:
                name = str(raw_path)
        records.append(
            {
                "path": name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    if not records:
        raise ValueError("paths must contain at least one input file")
    return f"inventory-sha256:{provenance_digest(records, length=64)}"


def stable_resampling_seed(base_seed: int, identity: Mapping[str, Any]) -> int:
    """Derive a cache-order-independent NumPy seed for one comparison."""
    payload = {
        "algorithm": RESAMPLING_ALGORITHM_VERSION,
        "base_seed": int(base_seed),
        "identity": dict(identity),
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def generate_member_indices(
    *,
    population_size: int,
    sample_size: int,
    iterations: int,
    seed: int,
) -> np.ndarray:
    """Generate deterministic without-replacement member selections."""
    if population_size < 1:
        raise ValueError("population_size must be positive")
    if sample_size < 1 or sample_size > population_size:
        raise ValueError("sample_size must be between 1 and population_size")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.choice(population_size, size=sample_size, replace=False)
            for _ in range(iterations)
        ]
    ).astype(np.int32, copy=False)


def member_indices_checksum(member_indices: np.ndarray) -> str:
    """Return a checksum that includes selection shape and integer values."""
    values = np.asarray(member_indices, dtype="<i4")
    payload = values.shape.__repr__().encode("ascii") + values.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def member_selection_attrs(
    *,
    base_seed: int,
    derived_seed: int,
    identity: Mapping[str, Any],
    member_indices: np.ndarray,
) -> dict[str, str | int]:
    """Build the complete provenance contract for member selections."""
    values = np.asarray(member_indices)
    if values.ndim != 2:
        raise ValueError("member_indices must be two-dimensional")
    return {
        "resampling_algorithm_version": RESAMPLING_ALGORITHM_VERSION,
        "base_random_seed": int(base_seed),
        "derived_random_seed": str(int(derived_seed)),
        "resampling_identity": canonical_json(dict(identity)),
        "resampling_identity_digest": provenance_digest(identity, length=32),
        "member_selection_checksum": member_indices_checksum(values),
        "resampling_iterations": int(values.shape[0]),
        "resampled_member_count": int(values.shape[1]),
    }


def build_member_selection_dataset(
    member_indices: np.ndarray,
    *,
    attrs: Mapping[str, str | int],
) -> xr.Dataset:
    """Package auditable member selections as a small NetCDF dataset."""
    values = np.asarray(member_indices, dtype=np.int32)
    if values.ndim != 2:
        raise ValueError("member_indices must be two-dimensional")
    return xr.Dataset(
        {
            "member_indices": (
                ("iteration", "sample_member"),
                values,
            )
        },
        coords={
            "iteration": np.arange(values.shape[0], dtype=np.int32),
            "sample_member": np.arange(values.shape[1], dtype=np.int32),
        },
        attrs=dict(attrs),
    )


def validate_member_selection_dataset(
    dataset: xr.Dataset,
    *,
    expected_attrs: Mapping[str, str | int],
    population_size: int,
) -> None:
    """Validate member-selection metadata, shape, checksum, and bounds."""
    if "member_indices" not in dataset:
        raise ValueError("Member-selection cache is missing member_indices")
    missing_attrs = set(expected_attrs) - set(dataset.attrs)
    if missing_attrs:
        raise ValueError(f"Member-selection cache is missing attributes: {sorted(missing_attrs)}")
    mismatches = {
        key: (dataset.attrs.get(key), expected)
        for key, expected in expected_attrs.items()
        if dataset.attrs.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Member-selection metadata mismatch: {mismatches}")

    values = np.asarray(dataset["member_indices"].values, dtype=np.int32)
    expected_shape = (
        int(expected_attrs["resampling_iterations"]),
        int(expected_attrs["resampled_member_count"]),
    )
    if values.shape != expected_shape:
        raise ValueError(
            f"Member-selection shape {values.shape} does not match {expected_shape}"
        )
    if values.size and (values.min() < 0 or values.max() >= population_size):
        raise ValueError("Member-selection cache contains out-of-range indices")
    if any(np.unique(row).size != row.size for row in values):
        raise ValueError("Member-selection cache contains replacement within an iteration")
    checksum = member_indices_checksum(values)
    if checksum != expected_attrs["member_selection_checksum"]:
        raise ValueError("Member-selection checksum does not match metadata")


__all__ = [
    "RESAMPLING_ALGORITHM_VERSION",
    "SOURCE_FINGERPRINT_VERSION",
    "build_member_selection_dataset",
    "canonical_json",
    "expected_skill_cache_attrs",
    "generate_member_indices",
    "file_inventory_digest",
    "member_indices_checksum",
    "member_selection_attrs",
    "provenance_digest",
    "source_fingerprint",
    "stable_resampling_seed",
    "validate_member_selection_dataset",
    "verification_year_token",
    "year_number",
]
