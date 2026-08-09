"""Regression tests for the 5x refactored workflow integration contracts."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from esp_lab.diagnostics.ic_core import DEFAULT_COMPONENTS
from esp_lab.diagnostics.ic_io import AUDIT_MANIFEST_COLUMNS, build_file_manifest
from esp_lab.paths import leadtime_acc_dir
from esp_lab.diagnostics.store import (
    diagnostic_store_is_valid,
    diagnostic_store_path,
    load_diagnostic_store,
    save_diagnostic_store,
)
from workflows.diagnostics.daily_drift.config import build_daily_config
from workflows.diagnostics.daily_drift.diagnostics import run_diagnostics as run_daily
from workflows.diagnostics.monthly_drift.config import build_monthly_config
from workflows.diagnostics.monthly_drift.config import DEFAULT_FIGURE_OUTDIR, DEFAULT_OUTPUT_ROOT
from workflows.diagnostics.monthly_drift.diagnostics import run_diagnostics as run_monthly
from workflows.diagnostics.monthly_drift import plotting as monthly_plotting
from workflows.diagnostics.physical_consistency.config import MEMBERS
from workflows.diagnostics.physical_consistency.run_physical import run as run_physical
from workflows.diagnostics.unified.inventory import run_inventory
from workflows.diagnostics.unified.run_unified import run as run_unified


def test_empty_ic_manifest_preserves_schema():
    manifest = build_file_manifest(
        pairs=[],
        components=DEFAULT_COMPONENTS,
        ref_label="BruteForce",
        test_label="JRA55-FOSIRL",
    )
    assert manifest.empty
    assert list(manifest.columns) == AUDIT_MANIFEST_COLUMNS


def _model_data(config, lead_dim, leads, years):
    coords = {
        "Y": years,
        "M": ["EN00"],
        lead_dim: leads,
        "lat": [0.0],
        "lon": [0.0],
    }
    values = np.zeros((len(years), 1, len(leads), 1, 1))
    labels = list(config.experiments)
    return {
        labels[0]: {5: xr.DataArray(values, dims=coords, coords=coords)},
        labels[1]: {5: xr.DataArray(values + 0.5, dims=coords, coords=coords)},
    }


def test_monthly_adjustment_uses_observation_evolution():
    config = build_monthly_config(
        pilot_years=[2000, 2001], pilot_months=[5], variable_filter=["TREFHT"]
    )
    config.leads = [1, 2]
    config.window_defs = {"two_months": (1, 2)}
    data = _model_data(config, "L", config.leads, config.active_years)
    times = pd.to_datetime(["2000-05-15", "2000-06-15", "2001-05-15", "2001-06-15"])
    obs = xr.DataArray(
        np.array([0.0, 2.0, 0.0, 2.0])[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
    )
    results = run_monthly(data, config, obs_da=obs, verbose=False)
    # Mean of [0, -2] after subtracting the lead-1 bias anchor.
    assert results[5]["two_months"]["adj_ref"].item() == -1.0
    assert results[5]["two_months"]["bias_ref"].item() == -1.0


def test_monthly_workflow_uses_fixed_diagnostic_and_figure_roots():
    config = build_monthly_config(variable_filter=["PRECT"])

    assert Path(config.output_root) == DEFAULT_OUTPUT_ROOT
    assert DEFAULT_OUTPUT_ROOT.parts[-4:] == (
        "multimodel", "leadtime_drift", "atm", "monthly_spatial"
    )
    assert DEFAULT_FIGURE_OUTDIR == Path(
        "/global/cfs/cdirs/e3sm/www/zhan391/esp-lab_diag"
    ) / "leadtime_drift" / "atm" / "monthly_spatial"


def test_land_acc_paths_follow_source_first_layout(tmp_path):
    model_inputs = leadtime_acc_dir(
        "JRA55_FOSIRL", "inputs", "land", "H2OSOI", root=tmp_path
    )
    reference_inputs = leadtime_acc_dir(
        "CPC_Soil_Moisture_V2", "inputs", "land", "H2OSOI", root=tmp_path
    )
    model_skill = leadtime_acc_dir(
        "JRA55_FOSIRL", "skill", "land", "H2OSOI", root=tmp_path
    )

    assert model_inputs == (
        tmp_path / "JRA55_FOSIRL" / "leadtime_acc" / "inputs" / "land" / "H2OSOI"
    )
    assert reference_inputs.parts[-6:] == (
        tmp_path.name,
        "CPC_Soil_Moisture_V2",
        "leadtime_acc",
        "inputs",
        "land",
        "H2OSOI",
    )
    assert model_skill == (
        tmp_path / "JRA55_FOSIRL" / "leadtime_acc" / "skill" / "land" / "H2OSOI"
    )


def test_atmospheric_acc_paths_include_realm(tmp_path):
    model_inputs = leadtime_acc_dir(
        "JRA55_FOSIRL", "inputs", "atm", "TREFHT", root=tmp_path
    )
    comparison = leadtime_acc_dir(
        "JRA55_FOSIRL", "comparison", "atm", "direct_rmse", root=tmp_path
    )

    assert model_inputs == (
        tmp_path / "JRA55_FOSIRL" / "leadtime_acc" / "inputs" / "atm" / "TREFHT"
    )
    assert comparison == (
        tmp_path
        / "JRA55_FOSIRL"
        / "leadtime_acc"
        / "comparison"
        / "atm"
        / "direct_rmse"
    )


def test_monthly_paired_diff_grid_combines_months_and_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(monthly_plotting, "_HAS_CARTOPY", False)
    lat = [-30.0, 30.0]
    lon = [0.0, 120.0, 240.0]
    field = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    results = {
        5: {
            "early": {"paired_diff": field},
            "late": {"paired_diff": field + 1},
        },
        11: {
            "early": {"paired_diff": -field},
            "late": {"paired_diff": -field - 1},
        },
    }
    outpath = tmp_path / "paired_diff_grid.png"

    monthly_plotting.plot_paired_diff_grid(
        results=results,
        boot_results={},
        window_defs={"early": (1, 3), "late": (10, 12)},
        title="Paired difference",
        outpath=outpath,
        units="K",
    )

    assert outpath.is_file()
    assert outpath.stat().st_size > 0


def test_daily_adjustment_uses_observation_evolution():
    config = build_daily_config(
        pilot_years=[2000, 2001], pilot_months=[5], variable_filter=["TREFHT"]
    )
    config.lead_days = [1, 2]
    config.window_defs = {"two_days": (1, 2)}
    data = _model_data(config, "d", config.lead_days, config.active_years)
    times = pd.to_datetime(["2000-05-02", "2000-05-03", "2001-05-02", "2001-05-03"])
    obs = xr.DataArray(
        np.array([0.0, 2.0, 0.0, 2.0])[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
    )
    results = run_daily(data, config, obs_da=obs, verbose=False)
    assert results[5]["two_days"]["adj_ref"].item() == -1.0


def test_diagnostic_store_round_trip(tmp_path):
    config = build_monthly_config(variable_filter=["PRECT"], n_bootstrap=5)
    field = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("Y", "lat", "lon"),
        coords={"Y": [1980, 1981], "lat": [-10.0, 10.0], "lon": [0.0, 1.0]},
    ).chunk({"lat": 1})
    results = {5: {"window": {"paired_diff_by_year": field, "init_years": [1980, 1981]}}}
    bootstrap = {5: {"window": {"lower": field.mean("Y"), "upper": field.mean("Y") + 1}}}
    path = diagnostic_store_path(tmp_path, "monthly_drift", config, "PRECT")
    save_diagnostic_store(
        path, results, bootstrap, config=config, variable="PRECT", workflow="monthly_drift"
    )
    assert diagnostic_store_is_valid(path, config, variable="PRECT")
    config.output_root = "/different/machine/location"
    assert diagnostic_store_path(tmp_path, "monthly_drift", config, "PRECT") == path
    assert diagnostic_store_is_valid(path, config, variable="PRECT")
    loaded_results, loaded_bootstrap = load_diagnostic_store(path)
    xr.testing.assert_allclose(
        loaded_results[5]["window"]["paired_diff_by_year"], field
    )
    assert loaded_results[5]["window"]["init_years"] == [1980, 1981]
    assert "lower" in loaded_bootstrap[5]["window"]


def test_physical_members_are_unique():
    assert MEMBERS == [f"EN{i:02d}" for i in range(10)]


def test_unified_inventory_does_not_claim_unchecked_sources(tmp_path):
    summary = run_inventory(output_root=str(tmp_path), verbose=False)
    assert not summary["all_required_ready"]
    assert set(summary["readiness"].values()) == {"NOT_CONFIGURED"}


def test_physical_orchestrator_executes_available_branches():
    coords = {"Y": [2000, 2001], "M": ["EN00"], "d": [1, 2], "lat": [0.0], "lon": [0.0]}
    base = xr.DataArray(np.ones((2, 1, 2, 1, 1)), dims=coords, coords=coords)
    fields = {
        "ref": {"LHFLX": base * 2, "SHFLX": base},
        "test": {"LHFLX": base * 3, "SHFLX": base},
    }
    result = run_physical(
        fields_by_experiment=fields,
        window_defs={"all": (1, 2)},
        verbose=False,
    )
    assert "flux_partitioning" in result
    assert "paired_diff_br" in result["flux_partitioning"]["all"]


def test_unified_orchestrator_runs_real_arrays(tmp_path):
    coords = {"Y": [2000, 2001], "M": ["EN00"], "L": [1, 2], "lat": [0.0], "lon": [0.0]}
    model = xr.DataArray(np.ones((2, 1, 2, 1, 1)), dims=coords, coords=coords)
    obs = xr.zeros_like(model.mean("M"))
    climatology = xr.zeros_like(model.mean(["Y", "M"]))
    result = run_unified(
        model_field=model,
        obs_ref=obs,
        e3sm_clim=climatology,
        output_root=str(tmp_path),
        verbose=False,
    )
    assert {"inventory", "field_drift", "model_attractor"} <= set(result)
    assert "windows" in result["model_attractor"]
