"""Durable stores for compute-once diagnostic workflows.

Raw hindcast fields stay lazy while diagnostics are assembled.  This module is
the deliberate compute boundary: reduced diagnostic arrays are written once,
then plotting/reporting can reopen them without touching the raw campaign.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import xarray as xr


STORE_VERSION = 2
_LOCATION_ONLY_CONFIG_KEYS = {"output_root", "figure_outdir"}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def file_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    """Cheap source identity for cache invalidation (path, size, mtime)."""
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return {"path": str(source), "exists": False}
    stat = source.stat()
    return {
        "path": str(source), "exists": True,
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }


def dataframe_fingerprint(frame: Any) -> str:
    """Stable digest of an inventory DataFrame without serializing it to disk."""
    import pandas as pd
    ordered = frame.sort_index(axis=1).astype(str)
    if len(ordered.columns):
        ordered = ordered.sort_values(list(ordered.columns)).reset_index(drop=True)
    values = pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    return sha256(values).hexdigest()[:16]


def config_fingerprint(
    config: Any,
    *,
    variable: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> str:
    config_payload = _jsonable(config)
    if isinstance(config_payload, dict):
        config_payload = {
            key: value
            for key, value in config_payload.items()
            if key not in _LOCATION_ONLY_CONFIG_KEYS
        }
    payload = {
        "config": config_payload, "variable": variable,
        "context": _jsonable(context or {}), "store_version": STORE_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()[:16]


def diagnostic_store_path(
    output_root: str | Path,
    workflow: str,
    config: Any,
    variable: str,
    context: Mapping[str, Any] | None = None,
) -> Path:
    token = config_fingerprint(config, variable=variable, context=context)
    base = Path(output_root) / "diagnostics" / workflow / variable
    target = base / token
    if target.exists():
        return target

    # Stores written before output locations were excluded from the science
    # fingerprint retain a different directory token. Reuse one when its
    # manifest differs only by those location-only settings.
    for manifest_path in sorted(base.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if (
            manifest.get("workflow") == workflow
            and manifest.get("variable") == variable
            and _jsonable(manifest.get("context", {})) == _jsonable(context or {})
            and config_fingerprint(
                manifest.get("config", {}), variable=variable, context=context
            ) == token
        ):
            return manifest_path.parent
    return target


def _encode_month(
    month_results: Mapping[str, Any],
    month_bootstrap: Mapping[str, Any],
) -> xr.Dataset:
    arrays: dict[str, xr.DataArray] = {}
    metadata: dict[str, Any] = {}

    for key, value in month_results.items():
        if isinstance(value, xr.DataArray):
            arrays[f"result__{key}"] = value
        elif isinstance(value, Mapping):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, xr.DataArray):
                    arrays[f"window__{key}__{subkey}"] = subvalue
                else:
                    metadata[f"window__{key}__{subkey}"] = _jsonable(subvalue)
        else:
            metadata[f"result__{key}"] = _jsonable(value)

    for window, values in month_bootstrap.items():
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            if isinstance(value, xr.DataArray):
                arrays[f"bootstrap__{window}__{key}"] = value

    dataset = xr.Dataset(arrays)
    dataset.attrs["diagnostic_metadata"] = json.dumps(metadata, sort_keys=True)
    dataset.attrs["diagnostic_store_version"] = STORE_VERSION
    return dataset


def save_diagnostic_store(
    path: str | Path,
    results: Mapping[int, Mapping[str, Any]],
    bootstrap: Mapping[int, Mapping[str, Any]],
    *,
    config: Any,
    variable: str,
    workflow: str,
    context: Mapping[str, Any] | None = None,
    product_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Materialize reduced diagnostics atomically as compressed NetCDF files."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    month_files: list[str] = []

    for month, month_results in sorted(results.items()):
        dataset = _encode_month(month_results, bootstrap.get(month, {}))
        target = root / f"init_month_{int(month):02d}.nc"
        temporary = target.with_suffix(".nc.tmp")
        encoding = {
            name: {"zlib": True, "complevel": 2}
            for name in dataset.data_vars
        }
        dataset.to_netcdf(temporary, encoding=encoding)
        temporary.replace(target)
        month_files.append(target.name)

    manifest = {
        "store_version": STORE_VERSION,
        "workflow": workflow,
        "variable": variable,
        "config_fingerprint": config_fingerprint(config, variable=variable, context=context),
        "config": _jsonable(config),
        "context": _jsonable(context or {}),
        "product_metadata": _jsonable(product_metadata or {}),
        "month_files": month_files,
        "complete": True,
    }
    manifest_tmp = root / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    manifest_tmp.replace(root / "manifest.json")
    return root


def diagnostic_store_is_valid(
    path: str | Path, config: Any, *, variable: str,
    context: Mapping[str, Any] | None = None,
) -> bool:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    expected = config_fingerprint(config, variable=variable, context=context)
    manifest_expected = config_fingerprint(
        manifest.get("config", {}),
        variable=manifest.get("variable"),
        context=manifest.get("context", {}),
    )
    files = manifest.get("month_files", [])
    return (
        manifest.get("complete") is True
        and manifest.get("store_version") == STORE_VERSION
        and (
            manifest.get("config_fingerprint") == expected
            or manifest_expected == expected
        )
        and bool(files)
        and all((root / name).is_file() for name in files)
    )


def load_diagnostic_store(
    path: str | Path,
    *,
    chunks: Any = "auto",
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Lazily reopen diagnostics and reconstruct the workflow dictionaries."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete diagnostic store: {root}")

    results: dict[int, dict[str, Any]] = {}
    bootstrap: dict[int, dict[str, Any]] = {}
    for filename in manifest["month_files"]:
        month = int(Path(filename).stem.rsplit("_", 1)[-1])
        dataset = xr.open_dataset(root / filename, chunks=chunks)
        month_results: dict[str, Any] = {}
        month_bootstrap: dict[str, Any] = {}

        for name, array in dataset.data_vars.items():
            parts = name.split("__")
            if parts[0] == "result":
                month_results[parts[1]] = array
            elif parts[0] == "window":
                month_results.setdefault(parts[1], {})[parts[2]] = array
            elif parts[0] == "bootstrap":
                month_bootstrap.setdefault(parts[1], {})[parts[2]] = array

        metadata = json.loads(dataset.attrs.get("diagnostic_metadata", "{}"))
        for name, value in metadata.items():
            parts = name.split("__")
            if parts[0] == "result":
                month_results[parts[1]] = value
            elif parts[0] == "window":
                month_results.setdefault(parts[1], {})[parts[2]] = value

        results[month] = month_results
        bootstrap[month] = month_bootstrap

    return results, bootstrap


__all__ = [
    "config_fingerprint",
    "dataframe_fingerprint",
    "diagnostic_store_is_valid",
    "diagnostic_store_path",
    "file_fingerprint",
    "load_diagnostic_store",
    "save_diagnostic_store",
]
