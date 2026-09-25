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
    assert obs == diag_root / "observations/modes_variability/modes/nam/indices/era5_nam_reference.nc"


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


def test_mov_cache_path_is_descriptive_and_stable(tmp_path):
    config = {
        "paths": {"output_dir": str(tmp_path)},
        "selection": {
            "upstream_mode": "npo",
            "downstream_variable": "H2OSOI",
            "verification_years": [1981, 2011],
        },
    }

    assert mov_telecon.mov_teleconnection_cache_path(config) == (
        tmp_path / "teleconnection_NPO_H2OSOI_verify1981_2011_1x1deg.nc"
    )


def test_open_mov_cache_requires_matching_schema_and_fingerprint(tmp_path):
    cache = tmp_path / "teleconnection_NPO_H2OSOI_verify1981_2011.nc"
    xr.Dataset({"value": ("x", [1.0])}, attrs={
        "schema": "mov_teleconnection_metrics_v1",
        "fingerprint": "current",
    }).to_netcdf(cache)

    loaded = mov_telecon._open_compatible_mov_cache(cache, "current")
    assert loaded is not None
    loaded.close()
    assert mov_telecon._open_compatible_mov_cache(cache, "stale") is None


def test_build_inventory_reports_missing_upstream_index(tmp_path):
    assert mov_telecon.E3SM_CASES["E3SM-4DEnVarOcn"]["supports_land"] is True

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

    # A requested experiment must not disappear silently when its MOV index is absent.
    h2osno_row = inv.query("variable == 'H2OSNO'").iloc[0]
    assert h2osno_row["status"] == "missing"
    assert "NAM index not computed" in h2osno_row["detail"]

    # Atmospheric rows follow the same strict missing-input policy.
    trefht_row = inv.query("variable == 'TREFHT'").iloc[0]
    assert trefht_row["status"] == "missing"
    assert "NAM index not computed" in trefht_row["detail"]


def test_ensure_upstream_products_prepares_missing_downstream_fields(tmp_path, monkeypatch):
    index_forecast = tmp_path / "index-model.nc"
    index_observed = tmp_path / "index-obs.nc"
    index_forecast.touch()
    index_observed.touch()
    missing = pd.DataFrame([{
        "system": "E3SM-FOSIRL", "init_month": 5, "mode": "NAM",
        "variable": "TREFHT", "index_forecast": str(index_forecast),
        "index_observed": str(index_observed), "field_forecast": None,
        "field_observed": None, "status": "missing", "detail": "missing field",
    }])
    ready = missing.assign(
        field_forecast=str(tmp_path / "field-model.nc"),
        field_observed=str(tmp_path / "field-obs.nc"), status="ready", detail="",
    )
    inventories = iter([missing, ready])
    monkeypatch.setattr(
        mov_telecon, "build_mov_teleconnection_inventory", lambda config: next(inventories)
    )

    from workflows.diagnostics import teleconnection_inputs as preparation

    calls = []
    monkeypatch.setattr(
        preparation, "prepare_atmospheric_observation",
        lambda *args, **kwargs: calls.append(("obs", args, kwargs)),
    )
    monkeypatch.setattr(
        preparation, "prepare_atmospheric_model",
        lambda *args, **kwargs: calls.append(("model", args, kwargs)),
    )
    config = {"inputs": {"mode": "auto"}}

    result = mov_telecon.ensure_upstream_products(config)

    assert result.status.tolist() == ["ready"]
    assert [call[0] for call in calls] == ["obs", "model"]
