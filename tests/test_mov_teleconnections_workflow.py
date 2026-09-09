"""Unit tests for MOV teleconnections diagnostics workflow."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from workflows.diagnostics import mov_teleconnections as mov_telecon


def test_mov_modes_registry():
    assert "NAM" in mov_telecon.MOV_MODES
    assert "NAO" in mov_telecon.MOV_MODES
    assert "SAM" in mov_telecon.MOV_MODES
    assert "PNA" in mov_telecon.MOV_MODES
    assert "PDO" in mov_telecon.MOV_MODES
    assert mov_telecon.MOV_MODES["NAM"]["kind"] == "pressure"
    assert mov_telecon.MOV_MODES["PDO"]["kind"] == "temperature"


def test_upstream_mov_paths_default():
    diag_root = Path("/tmp/mock_diag_root")
    fcst, obs = mov_telecon.upstream_mov_paths("E3SM-FOSIRL", 5, "NAM", diag_root=diag_root)
    assert fcst == diag_root / "JRA55_FOSIRL/modes_variability/modes/nam/indices/JRA55_FOSIRL_init05_nam.nc"
    assert obs == diag_root / "ERA5/modes_variability/modes/nam/indices/era5_nam_reference.nc"


def test_upstream_mov_paths_with_manifest(tmp_path):
    manifest_fcst = tmp_path / "custom_fcst.nc"
    manifest_obs = tmp_path / "custom_obs.nc"
    manifest_fcst.touch()
    manifest_obs.touch()

    manifest = {
        "products": {
            "NAM:JRA55_FOSIRL_init05": {"index": str(manifest_fcst)},
            "NAM:reference": {"index": str(manifest_obs)},
        }
    }
    fcst, obs = mov_telecon.upstream_mov_paths("E3SM-FOSIRL", 5, "NAM", manifest=manifest)
    assert fcst == manifest_fcst
    assert obs == manifest_obs


def test_mov_provenance_fingerprint():
    config = {
        "selection": {"upstream_mode": "NAM", "systems": ["E3SM-FOSIRL"]},
        "analysis": {"detrend": True},
    }
    with tempfile.NamedTemporaryFile() as tmp:
        p = Path(tmp.name)
        fp1 = mov_telecon.compute_mov_provenance_fingerprint(config, [p])
        fp2 = mov_telecon.compute_mov_provenance_fingerprint(config, [p])
        assert fp1 == fp2
        assert len(fp1) == 16


def test_build_inventory_skips_unsupported(tmp_path):
    config = {
        "paths": {"diag_root": str(tmp_path)},
        "selection": {
            "upstream_mode": "NAM",
            "systems": ["E3SM-4DEnVarOcn"],
            "init_months": [5],
            "downstream_variables": ["H2OSNO", "TREFHT"],
            "target_grid": "latlon_1.0x1.0_periodic-True",
        },
        "cache": {"allow_ambiguous_matches": False},
    }

    inv = mov_telecon.build_mov_teleconnection_inventory(config)
    assert len(inv) == 2

    # H2OSNO for 4DEnVarOcn should be skipped because 4DEnVarOcn does not produce land
    h2osno_row = inv.query("variable == 'H2OSNO'").iloc[0]
    assert h2osno_row["status"] == "skipped"
    assert "land diagnostics" in h2osno_row["detail"]

    # TREFHT should be skipped because upstream NAM index file does not exist in tmp_path
    trefht_row = inv.query("variable == 'TREFHT'").iloc[0]
    assert trefht_row["status"] == "skipped"
    assert "NAM index not computed" in trefht_row["detail"]

