"""Opt-in real-cache NMME source-adapter smoke test."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from workflows.modes_of_variability import analysis
from esp_lab.utils import calendar_utils as cal


def test_cmc1_prmsl_native_leads_are_not_padded():
    root = os.environ.get("ESP_LAB_NMME_SMOKE_ROOT")
    if not root:
        pytest.skip("Set ESP_LAB_NMME_SMOKE_ROOT to run the real NMME cache smoke test.")
    args = Namespace(
        nmme_root=Path(root), nmme_models=["CMC1-CanCM3"], nmme_field="auto",
        nmme_chunks="", nmme_sst_land_mask=True, nmme_fixed_dir=analysis.NMME_FIXED_DIR,
        start_year=1991, end_year=2009, monthly_nlead=12,
    )
    raw = analysis.nmme_field_dataset(2, analysis.mode_settings("NAO"), args)
    assert list(raw.L.values) == list(range(1, 13))
    assert list(raw.M.values) == [f"CMC1-CanCM3:M{i:03d}" for i in range(1, 11)]
    assert raw["PSL"].attrs["units"] == "Pa"

    seasonal = cal.mon_to_seas_dask(raw)
    seasonal, psl = analysis.drop_empty_leads(seasonal, seasonal["PSL"])
    assert list(psl.L.values) == [3, 6, 9]
    assert np.isfinite(psl.values).any()


def test_cmc1_sst_native_leads_and_mask_are_not_padded(tmp_path):
    """Exercise the notebook-confirmed temperature-mode NMME adapter path."""
    root = os.environ.get("ESP_LAB_NMME_SMOKE_ROOT")
    if not root:
        pytest.skip("Set ESP_LAB_NMME_SMOKE_ROOT to run the real NMME cache smoke test.")
    args = Namespace(
        nmme_root=Path(root), nmme_models=["CMC1-CanCM3"], nmme_field="auto",
        nmme_chunks="", nmme_sst_land_mask=True, nmme_fixed_dir=tmp_path,
        start_year=1991, end_year=2009, monthly_nlead=12,
    )
    raw = analysis.nmme_field_dataset(2, analysis.mode_settings("PDO"), args)

    assert list(raw.L.values) == list(range(1, 13))
    assert list(raw.M.values) == [f"CMC1-CanCM3:M{i:03d}" for i in range(1, 11)]
    assert raw["SST"].attrs["nmme_sst_land_mask"] == "True"
    assert "nmme_sst_mask_version" in raw["SST"].attrs

    seasonal = cal.mon_to_seas_dask(raw)
    seasonal, sst = analysis.drop_empty_leads(seasonal, seasonal["SST"])
    assert list(sst.L.values) == [3, 6, 9]
    assert np.isfinite(sst.values).any()
