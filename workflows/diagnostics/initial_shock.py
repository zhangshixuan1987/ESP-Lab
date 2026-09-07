"""Configurable NetCDF entry point: python -m workflows.diagnostics.initial_shock config.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from esp_lab.diagnostics.initial_shock import (
    VERSION, align_observation_months, compute_initial_shock_index, plot_std_ratio,
)
from esp_lab.paths import diagnostic_dir
from esp_lab.utils.netcdf_utils import atomic_to_netcdf, load_netcdf


def _inventory(spec):
    path = Path(spec["path"]).expanduser().resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _field(dataset, spec):
    data = dataset[spec["variable"]].rename(spec.get("rename", {}))
    data = data * spec.get("scale", 1.0) + spec.get("offset", 0.0)
    data.attrs["units"] = spec["units"]
    return data


def run_initial_shock(config):
    """Run configured cases from monthly Y/L caches and monthly observations.

    No ACC prepared anomalies are used. Each case input needs normalized
    verification_time(Y,L). Output cache identity includes file inventories,
    variables, conversions, options, and the algorithm version. Cache modes:
    auto (reuse or compute), rebuild, require (cache-only, still checks inventories).
    All cases must share the exact initialization labels for the comparison plot.
    """
    if not config.get("cases"):
        raise ValueError("Configure at least one case")
    mode = config.get("cache_mode", "auto")
    if mode not in {"auto", "rebuild", "require"}:
        raise ValueError("cache_mode must be auto, rebuild, or require")
    root = config["output_root"]
    results, paths = [], []
    for case, spec in config["cases"].items():
        inputs = {"model": _inventory(spec), "observation": _inventory(config["observation"])}
        if config.get("area"):
            inputs["area"] = _inventory(config["area"])
        identity = {"version": VERSION, "case": case, "inputs": inputs,
                    "model": spec, "observation": config["observation"],
                    "area": config.get("area"), "settings": config.get("settings", {})}
        payload = json.dumps(identity, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        path = diagnostic_dir(case, "initial_shock", "metrics", root=root) / f"std_index_{digest[:20]}.nc"
        result = None
        if mode != "rebuild" and path.exists():
            try:
                candidate = load_netcdf(path)
                required = {"std_ratio", "model_std", "observation_std", "paired_sample_count",
                            "model_index", "observation_index", "valid_ratio"}
                if candidate.attrs.get("identity_sha256") == digest and required <= set(candidate):
                    result = candidate
            except (OSError, ValueError):
                pass
        if result is None:
            if mode == "require":
                raise FileNotFoundError(f"Compatible initial-shock cache unavailable: {path}")
            with xr.open_dataset(inputs["model"]["path"], chunks={}) as model_ds, xr.open_dataset(
                inputs["observation"]["path"], chunks={}
            ) as obs_ds:
                model = _field(model_ds, spec)
                verification_time = model_ds[spec.get("verification_time", "verification_time")]
                verification_time = verification_time.rename({
                    k: v for k, v in spec.get("rename", {}).items()
                    if k in verification_time.dims or k in verification_time.coords
                })
                observation = _field(obs_ds, config["observation"])
                observation = align_observation_months(observation, verification_time)
                area = None
                if config.get("area"):
                    with xr.open_dataset(inputs["area"]["path"]) as area_ds:
                        area = area_ds[config["area"]["variable"]].load()
                result = compute_initial_shock_index(
                    model, observation, area=area, **config.get("settings", {})
                ).compute()
                if inputs["model"] != _inventory(spec) or inputs["observation"] != _inventory(config["observation"]):
                    raise RuntimeError("Source input changed during computation; cache not written")
                if config.get("area") and inputs["area"] != _inventory(config["area"]):
                    raise RuntimeError("Area input changed during computation; cache not written")
                result.attrs.update(case=case, identity_sha256=digest, provenance_json=payload)
                atomic_to_netcdf(result, path)
        results.append(result.expand_dims(case=[case]))
        paths.append(path)
    for result in results[1:]:
        for name in ("block_start_time", "block_end_time"):
            first = results[0][name]
            other = result[name]
            first_months = np.asarray(first.dt.year * 12 + first.dt.month)
            other_months = np.asarray(other.dt.year * 12 + other.dt.month)
            if not np.array_equal(first_months, other_months):
                raise ValueError("Cases have different verification months; compare identical windows")
    comparison = xr.concat(results, dim="case", join="exact", combine_attrs="drop_conflicts")
    # Retain the metric settings in the multi-case plot title.
    for name in ("block_months", "window_months"):
        comparison.attrs[name] = results[0].attrs[name]
    if config.get("figure_path"):
        fig = plot_std_ratio(comparison)
        try:
            output = Path(config["figure_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches="tight")
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)
    return comparison, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result, paths = run_initial_shock(json.loads(args.config.read_text()))
    print(result[["std_ratio", "paired_sample_count"]])
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
