"""Defaults and anomaly re-centering of scripts/render_nmme_comparison.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_nmme_comparison.py"
SPEC = importlib.util.spec_from_file_location("render_nmme_comparison", SCRIPT)
render = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render)


def test_defaults_are_the_recommended_cohort():
    args = render.parse_args(["--diag-root", "d", "--nmme-root", "n", "--outdir", "o"])
    assert args.years == [1982, 2010]
    assert args.models == "auto"
    assert args.single_model is None
    assert args.init_months == [5, 11]
    assert args.min_members == 4


def test_explicit_models_and_years_pass_through():
    args = render.parse_args([
        "--diag-root", "d", "--nmme-root", "n", "--outdir", "o",
        "--years", "1991", "2009", "--models", "CMC1-CanCM3", "CanCM4i",
    ])
    assert args.years == [1991, 2009]
    assert args.models == ["CMC1-CanCM3", "CanCM4i"]


def test_recenter_removes_each_models_own_mean_over_the_cohort():
    values = np.array([[[1.0, 3.0, 10.0]], [[3.0, 5.0, 14.0]]])  # (Y=2, L=1, M=3)
    index = xr.DataArray(values, dims=("Y", "L", "M"),
                         coords={"Y": [1990, 1991], "L": [1], "M": ["A:1", "A:2", "B:1"]})
    centered = render.recenter_by_model(index)
    np.testing.assert_allclose(centered.sel(M=["A:1", "A:2"]).mean(("Y", "M")), 0.0)
    np.testing.assert_allclose(centered.sel(M="B:1").values.ravel(), [-2.0, 2.0])


def test_product_paths_follow_the_processor_layout(tmp_path):
    path = render.mode_index_path(tmp_path, "NMME", "PDO", 11)
    assert path == tmp_path / "NMME/modes_variability/modes/pdo/indices/NMME_init11_pdo.nc"
