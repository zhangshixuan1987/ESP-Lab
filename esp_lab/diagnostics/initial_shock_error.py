"""NCL-style RMSE and MAE of ensemble-mean global-index anomalies."""
from __future__ import annotations

import numpy as np
import xarray as xr

VERSION = "initial_shock_rmse_mae_v2"
NCL_RMSE_RANGES = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
NCL_MAE_RANGES = (0.13, 0.16, 0.19, 0.22, 0.25, 0.28)
NCL_COLORS = ("white", "orange", "#cd8500", "orangered", "#ff4500", "#cd3700", "#8b2500")


def compute_initial_shock_error_index(
    indices: xr.Dataset, *, min_samples: int | None = None,
) -> xr.Dataset:
    """Compute NCL-style anomaly RMSE and MAE for every initialization.

    ``indices`` must contain ``model_index`` and ``observation_index`` on
    ``(Y, block)`` plus any shared outer dimensions such as ``case``. As in the
    NCL script, each series is centered on its own mean over the full Y/block
    cohort. RMSE and MAE are then calculated over paired blocks within each Y.
    """
    required = {"model_index", "observation_index"}
    if not required <= set(indices.data_vars):
        raise ValueError(f"indices must contain {sorted(required)}")
    model, observation = xr.align(
        indices.model_index, indices.observation_index, join="exact"
    )
    if "Y" not in model.dims or "block" not in model.dims:
        raise ValueError("model_index and observation_index must contain Y and block")
    if model.dims != observation.dims:
        raise ValueError("model and observation indices must have identical dimensions")
    nblocks = model.sizes["block"]
    min_samples = nblocks if min_samples is None else min_samples
    if not isinstance(min_samples, (int, np.integer)) or not 1 <= min_samples <= nblocks:
        raise ValueError("min_samples must be between 1 and the number of blocks")
    model = model.where(np.isfinite(model))
    observation = observation.where(np.isfinite(observation))
    baseline_dims = ("Y", "block")
    model_climatology = model.mean(baseline_dims, skipna=True)
    observation_climatology = observation.mean(baseline_dims, skipna=True)
    model_anomaly = model - model_climatology
    observation_anomaly = observation - observation_climatology
    paired = model_anomaly.notnull() & observation_anomaly.notnull()
    error = (model_anomaly - observation_anomaly).where(paired)
    count = paired.sum("block")
    valid = count >= min_samples
    rmse = np.sqrt((error ** 2).sum("block", skipna=True) / count.where(count > 0)).where(valid)
    mae = (abs(error).sum("block", skipna=True) / count.where(count > 0)).where(valid)
    if "observation_climatology_std" in indices:
        observation_scale = indices.observation_climatology_std
    else:
        observation_scale = observation.isel(block=0).std("Y", ddof=1, skipna=True)
    observation_scale = observation_scale.where(
        np.isfinite(observation_scale) & (observation_scale > 0)
    )
    normalized_rmse = rmse / observation_scale
    normalized_mae = mae / observation_scale
    units = model.attrs.get("units", observation.attrs.get("units", ""))
    if model.attrs.get("units") and observation.attrs.get("units") and model.attrs["units"] != observation.attrs["units"]:
        raise ValueError("Model and observation index units differ")
    result = xr.Dataset({
        "rmse": rmse,
        "mae": mae,
        "normalized_rmse": normalized_rmse,
        "normalized_mae": normalized_mae,
        "observation_climatology_std": observation_scale,
        "paired_sample_count": count,
        "valid_metric": valid.astype("int8"),
        "model_climatology": model_climatology,
        "observation_climatology": observation_climatology,
        "model_anomaly": model_anomaly,
        "observation_anomaly": observation_anomaly,
        "error": error,
    })
    for name in ("rmse", "mae", "observation_climatology_std",
                 "model_climatology", "observation_climatology",
                 "model_anomaly", "observation_anomaly", "error"):
        result[name].attrs["units"] = units
    result.rmse.attrs["long_name"] = "Root mean square error of global-index anomalies"
    result.mae.attrs["long_name"] = "Mean absolute error of global-index anomalies"
    result.normalized_rmse.attrs.update(
        units="1",
        long_name="RMSE of global-index anomalies normalized by observed climatological standard deviation",
    )
    result.normalized_mae.attrs.update(
        units="1",
        long_name="MAE of global-index anomalies normalized by observed climatological standard deviation",
    )
    result.attrs.update(
        diagnostic_version=VERSION,
        min_samples=int(min_samples),
        anomaly_baseline="independent model and observation means over Y and block",
        error_reduction="paired blocks within each initialization; population mean",
        interpretation="Error amplitude after removing each series' full-cohort mean",
        normalized_error_denominator="sample std of observed first-block annual means across initialization years",
    )
    return result


def plot_error_heatmap(
    result: xr.Dataset,
    *,
    variable: str,
    levels,
    ax=None,
    add_colorbar: bool = True,
    add_invalid_legend: bool = True,
    title: str | None = None,
):
    """Plot one raw or normalized error metric with explicit color levels."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.patches import Patch

    if variable not in {"rmse", "mae", "normalized_rmse", "normalized_mae"}:
        raise ValueError("variable must be rmse, mae, normalized_rmse, or normalized_mae")
    if variable not in result:
        raise ValueError(f"result does not contain {variable!r}")
    bounds = np.asarray(levels, dtype=float)
    if bounds.ndim != 1 or bounds.size < 2:
        raise ValueError("levels must contain at least two one-dimensional boundaries")
    if not np.isfinite(bounds).all() or bounds[0] < 0 or not np.all(np.diff(bounds) > 0):
        raise ValueError("levels must be finite, nonnegative, and strictly increasing")

    values = result[variable]
    if "case" not in values.dims:
        values = values.expand_dims(case=[result.attrs.get("case", "model")])
    values = values.transpose("Y", "case")
    cmap = plt.colormaps["YlOrRd"].resampled(bounds.size - 1).with_extremes(
        bad="#bdbdbd", over="#4a0000"
    )
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(
            figsize=(max(6, values.sizes["case"] * 1.5), max(3, values.sizes["Y"] * .25))
        )
    else:
        fig = ax.figure
    mesh = ax.imshow(
        values.values, aspect="auto", cmap=cmap,
        norm=BoundaryNorm(bounds, cmap.N),
    )
    ax.set_xticks(
        np.arange(values.sizes["case"]), labels=values.case.values,
        rotation=30, ha="right",
    )
    ax.set_yticks(
        np.arange(values.sizes["Y"]), labels=[str(value) for value in values.Y.values]
    )
    ax.set_xticks(np.arange(-.5, values.sizes["case"], 1), minor=True)
    ax.set_yticks(np.arange(-.5, values.sizes["Y"], 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linewidth=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_ylabel("Initialization year")
    ax.set_title(title if title is not None else values.attrs.get("long_name", variable))
    if add_colorbar:
        units = "1" if variable.startswith("normalized_") else values.attrs.get("units", "")
        label = variable.replace("normalized_", "Normalized ").upper()
        if units and units != "1":
            label += f" ({units})"
        colorbar = fig.colorbar(mesh, ax=ax, ticks=bounds, label=label, extend="max")
        colorbar.ax.set_yticklabels([f"{value:g}" for value in bounds])
    if add_invalid_legend:
        ax.legend(
            handles=[Patch(facecolor=cmap.get_bad(), edgecolor="none", label="Invalid / missing")],
            loc="upper left", bbox_to_anchor=(1.01, 0), frameon=False, fontsize="small",
        )
    if owns_figure:
        fig.tight_layout()
    return fig


def plot_rmse_mae(
    result: xr.Dataset, *, rmse_ranges=NCL_RMSE_RANGES, mae_ranges=NCL_MAE_RANGES,
):
    """Plot RMSE and MAE panels using the color thresholds from the NCL figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if not {"rmse", "mae"} <= set(result):
        raise ValueError("result must contain rmse and mae")
    data = result if "case" in result.dims else result.expand_dims(case=["model"])
    if "Y" not in data.dims:
        raise ValueError("result must contain Y")
    checked_ranges = []
    for ranges in (rmse_ranges, mae_ranges):
        ranges = tuple(map(float, ranges))
        if (len(ranges) != len(NCL_COLORS) - 1 or
                not np.isfinite(ranges).all() or
                ranges[0] <= 0 or not np.all(np.diff(ranges) > 0)):
            raise ValueError("Plot ranges must contain six increasing positive finite values")
        checked_ranges.append(ranges)
    cmap = ListedColormap(NCL_COLORS).with_extremes(bad="#d9d9d9")
    fig, axes = plt.subplots(2, 1, figsize=(max(8, data.sizes["case"] * 1.2), 9), sharex=True)
    units = data.rmse.attrs.get("units", "")
    for ax, name, ranges, title in zip(
        axes, ("rmse", "mae"), checked_ranges,
        ("(a) Root Mean Square Error", "(b) Mean Absolute Error"), strict=True,
    ):
        values = data[name].transpose("Y", "case")
        finite = np.asarray(values).astype(float)
        finite = finite[np.isfinite(finite)]
        upper = max(float(ranges[-1]) + np.finfo(float).eps,
                    float(finite.max()) if finite.size else float(ranges[-1]) * 1.01)
        if upper <= ranges[-1]:
            upper = float(ranges[-1]) * 1.01
        boundaries = (0.0, *ranges, upper)
        mesh = ax.pcolormesh(
            np.arange(data.sizes["case"] + 1), np.arange(data.sizes["Y"] + 1), values,
            cmap=cmap, norm=BoundaryNorm(boundaries, cmap.N, clip=True),
            edgecolors="black", linewidth=.7,
        )
        ax.set_yticks(np.arange(data.sizes["Y"]) + .5, labels=[str(v) for v in data.Y.values])
        ax.set_ylabel("Initialization")
        ax.set_title(title, loc="left")
        fig.colorbar(mesh, ax=ax, ticks=ranges, label=f"{name.upper()} ({units})" if units else name.upper())
    axes[-1].set_xticks(
        np.arange(data.sizes["case"]) + .5,
        labels=[str(v) for v in data.case.values], rotation=45, ha="right",
    )
    fig.tight_layout()
    return fig
