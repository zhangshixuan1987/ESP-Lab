from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from esp_lab import land_prepared_skill


def _expected(**overrides):
    values = {
        "field": "H2OSNO",
        "source_kind": "model",
        "source_name": "case",
        "source_data_identity": "sha256:raw",
        "case_prefix": "prefix",
        "init_month": 5,
        "initialization_years": (2000, 2001),
        "climatology_years": (2000, 2001),
        "ensemble_members": ["EN00", "EN01"],
        "monthly_nlead": 6,
        "target_grid_name": "90x180deg_cell_centered",
        "regridding_method": "conservative",
        "output_units": "mm",
        "reference_is_anomaly": False,
    }
    values.update(overrides)
    return land_prepared_skill.expected_attrs(**values)


def _model_dataset(expected, grid):
    field = expected["field"]
    dataset = xr.Dataset(
        {
            field: (
                ("Y", "L", "M", "lat", "lon"),
                np.ones((2, 2, 2, grid.sizes["lat"], grid.sizes["lon"])),
            ),
            "time": (("Y", "L"), np.array([[1, 2], [3, 4]])),
        },
        coords={
            "Y": [2000, 2001], "L": [3, 6], "M": ["EN00", "EN01"],
            "lat": grid.lat, "lon": grid.lon,
        },
        attrs=expected,
    )
    dataset[field].attrs.update(expected)
    return dataset


def test_expected_attrs_change_with_requested_years_members_and_source():
    baseline = _expected()

    assert baseline != _expected(initialization_years=(2000, 2002))
    assert baseline != _expected(ensemble_members=["EN00"])
    assert baseline != _expected(source_data_identity="sha256:changed")
    assert baseline["initialization_month"] == 5


def test_prepared_cache_status_validates_metadata_shape_and_grid(tmp_path):
    grid = land_prepared_skill.target_grid(90, 180)
    expected = _expected()
    path = tmp_path / "prepared.nc"
    _model_dataset(expected, grid).to_netcdf(path)

    assert land_prepared_skill.prepared_cache_status(
        path, expected, grid=grid
    ) == (True, "compatible")

    changed = dict(expected, monthly_lead_count=12)
    ok, reason = land_prepared_skill.prepared_cache_status(path, changed, grid=grid)
    assert not ok
    assert "monthly_lead_count" in reason


def test_prepared_cache_status_rejects_wrong_member_count(tmp_path):
    grid = land_prepared_skill.target_grid(90, 180)
    expected = _expected()
    dataset = _model_dataset(expected, grid).isel(M=[0])
    path = tmp_path / "partial.nc"
    dataset.to_netcdf(path)

    ok, reason = land_prepared_skill.prepared_cache_status(path, expected, grid=grid)
    assert not ok
    assert "ensemble-member count" in reason


def test_reference_file_resolution_is_stable_and_nonempty(tmp_path):
    second = tmp_path / "b.nc"
    first = tmp_path / "a.nc"
    first.touch()
    second.touch()

    assert land_prepared_skill.resolve_reference_files(str(tmp_path / "*.nc")) == [
        first, second
    ]
    with pytest.raises(FileNotFoundError):
        land_prepared_skill.resolve_reference_files(str(tmp_path / "missing*.nc"))


def test_revision_identity_does_not_require_an_inventory():
    identity = land_prepared_skill.raw_source_identity(
        paths=[], logical_identity={"source": "test"},
        source_revision="revision-1", mode="revision",
    )
    assert identity.startswith("sha256:")


def test_inventory_snapshot_restart_preserves_exact_land_identity(tmp_path):
    source = tmp_path / "raw.nc"
    source.write_bytes(b"first")
    options = dict(
        logical_identity={"case": "test", "years": [2000, 2001]},
        source_revision="revision-1", inventory_root=tmp_path,
        snapshot_dir=tmp_path / "snapshots",
    )
    inventory = land_prepared_skill.raw_source_identity(
        paths=[source], mode="inventory", **options
    )
    source.unlink()
    assert land_prepared_skill.raw_source_identity(
        paths=[], mode="snapshot", **options
    ) == inventory
    changed = dict(options, logical_identity={"case": "test", "years": [2000, 2002]})
    with pytest.raises(RuntimeError, match="run once in inventory mode"):
        land_prepared_skill.raw_source_identity(paths=[], mode="snapshot", **changed)


def test_snapshot_rejects_tampered_record(tmp_path):
    source = tmp_path / "raw.nc"
    source.touch()
    options = dict(
        paths=[source], logical_identity={"source": "test"},
        source_revision="v1", inventory_root=tmp_path,
        snapshot_dir=tmp_path / "snapshots",
    )
    land_prepared_skill.raw_source_identity(mode="inventory", **options)
    record = next((tmp_path / "snapshots").glob("*.json"))
    record.write_text("{}")
    with pytest.raises(ValueError, match="Invalid land source"):
        land_prepared_skill.raw_source_identity(mode="snapshot", **options)


def test_prepared_data_identity_changes_with_contract():
    assert land_prepared_skill.prepared_data_identity(_expected()) != (
        land_prepared_skill.prepared_data_identity(
            _expected(source_data_identity="sha256:new")
        )
    )
