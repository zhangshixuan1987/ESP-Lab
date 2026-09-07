"""Tests for reusable modes-of-variability teleconnection products."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from workflows.modes_of_variability import teleconnections


def _reference_products(tmp_path: Path) -> dict[str, dict[str, str]]:
    root = tmp_path / "ERA5" / "modes_variability"
    field_path = root / "fields" / "era5_psl.nc"
    index_path = root / "modes" / "nam" / "indices" / "era5_nam_reference.nc"
    field_path.parent.mkdir(parents=True)
    index_path.parent.mkdir(parents=True)
    time = pd.date_range("2000-01-01", "2004-12-01", freq="MS")
    signal = (
        np.sin(np.arange(time.size) * 2 * np.pi / 12)
        + np.repeat(np.arange(5, dtype=float), 12)
    )
    spatial = np.array([[1.0, 2.0], [-1.0, -2.0]])
    field = signal[:, None, None] * spatial[None, :, :]
    xr.Dataset(
        {"PSL_anom": (("time", "lat", "lon"), field)},
        coords={"time": time, "lat": [-30.0, 30.0], "lon": [0.0, 180.0]},
    ).to_netcdf(field_path)
    xr.Dataset(
        {
            "mode_index": ("time", signal),
            "mode_pattern": (
                ("target_month", "lat", "lon"), np.ones((12, 2, 2))
            ),
        },
        coords={
            "time": time, "target_month": np.arange(1, 13),
            "lat": [-30.0, 30.0], "lon": [0.0, 180.0],
        },
    ).to_netcdf(index_path)
    return {"NAM:reference": {"field": str(field_path), "index": str(index_path)}}


def test_auto_writes_and_reuses_reference_teleconnection(tmp_path):
    products = _reference_products(tmp_path)
    first = teleconnections.ensure_products(products, ensure_mode="auto", fdr=False)
    second = teleconnections.ensure_products(products, ensure_mode="auto", fdr=False)

    assert first == {"NAM:reference": "written"}
    assert second == {"NAM:reference": "reused"}
    details = products["NAM:reference"]
    path = teleconnections.output_path(
        "NAM:reference", Path(details["index"]), Path(details["field"])
    )
    with xr.open_dataset(path) as dataset:
        assert dataset["mode_global_regression_pattern"].dims == (
            "target_month", "lat", "lon"
        )
        assert bool(
            (dataset["mode_global_regression_sample_size"] >= 3).all()
        )


def test_require_rejects_stale_teleconnection(tmp_path):
    products = _reference_products(tmp_path)
    teleconnections.ensure_products(products, ensure_mode="auto", fdr=False)
    field_path = Path(products["NAM:reference"]["field"])
    field_path.touch()

    with pytest.raises(RuntimeError, match="incompatible"):
        teleconnections.ensure_products(products, ensure_mode="require", fdr=False)
