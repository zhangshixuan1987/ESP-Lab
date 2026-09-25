"""Tests for on-demand teleconnection input preparation orchestration."""

from workflows.diagnostics import teleconnection_inputs as inputs


def _config():
    return {
        "paths": {"diag_root": "/diagnostics"},
        "inputs": {
            "initialization_years": (1980, 2011),
            "raw_model_root": "/raw/e3sm",
            "ensemble_member_count": 10,
            "monthly_nlead": 24,
            "workers": 2,
            "sst_land_mask": True,
        },
        "selection": {
            "upstream_index": "Nino3.4",
            "init_months": [5, 11],
            "climatology_years": (1981, 2010),
        },
        "regrid": {
            "target_dlat": 5.0,
            "target_dlon": 5.0,
            "method": "conservative",
            "periodic": True,
        },
    }


def test_target_grid_identity_matches_prepared_atmospheric_contract():
    assert inputs._target_grid_identity(_config()) == "latlon_5.0x5.0_periodic-True"


def test_tws_reference_contract_matches_land_workflow():
    reference = inputs.LAND_REFERENCES["TWS"]

    assert reference["product"] == "C3S_TWSA"
    assert reference["variable"] == "twsa"
    assert reference["is_anomaly"] is True
    assert reference["restrict_model_to_reference_months"] is True


def test_atmospheric_expected_uses_selected_run_and_regrid_contract():
    path, expected = inputs._atmospheric_expected(
        _config(), "E3SM-FOSIRL", 5, "TREFHT"
    )

    assert path.name.startswith("JRA55_FOSIRL_init05_TREFHT_seasonal_anomaly_")
    assert expected["requested_year_count"] == 32
    assert expected["target_grid"] == "latlon_5.0x5.0_periodic-True"
    assert expected["regridding_method"] == "conservative"


def test_sst_index_preparation_invokes_only_requested_sources(monkeypatch):
    calls = []
    monkeypatch.setattr(
        inputs.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )

    inputs.ensure_sst_indices(
        _config(), ["E3SM-FOSIRL"], include_observation=False
    )

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[command.index("--sources") + 1] == "e3sm"
    assert command[command.index("--regions") + 1] == "Nino3.4"
    assert command[command.index("--e3sm-cache-tag") + 1] == "JRA55_FOSIRL"
    assert kwargs["check"] is True


def test_eli_preparation_passes_selected_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(
        inputs.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    config = _config()
    config["selection"]["upstream_index"] = "ELI"
    config["inputs"]["eli_grid"] = "native"
    config["inputs"]["eli_mesh_file"] = "/mesh.nc"
    config["inputs"]["eli_raw_model_root"] = "/raw/native"

    inputs.ensure_sst_indices(config, ["E3SM-FOSIRL"], include_observation=False)

    command = calls[0]
    assert command[command.index("--eli-grid") + 1] == "native"
    assert command[command.index("--mesh-file") + 1] == "/mesh.nc"
    assert command[command.index("--e3sm-raw-dir") + 1] == "/raw/native"
