import numpy as np
import pytest
import xarray as xr

from esp_lab.prepared_skill import (
    build_prepared_skill_dataset,
    cache_status,
    expected_prepared_skill_attrs,
    open_prepared_skill_dataset,
    prepared_skill_cache_status,
    prepared_skill_path,
    write_prepared_skill_dataset,
)


PROVENANCE = {
    "source_data_identity": "/archive/case-a/TREFHT",
    "case_prefix": "case-a-prefix",
    "requested_years": (2000, 2001, 2002),
    "target_grid": "latlon_1.0x1.0_periodic",
    "regridding_method": "conservative",
    "ensemble_member_count": 2,
    "lead_count": 2,
    "unit_conversion_version": "kelvin_to_celsius_v1",
}


def _prepared_dataset():
    years = np.arange(2000, 2003)
    leads = np.arange(1, 3)
    members = np.arange(2)
    anomaly = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 2, 2),
        dims=("Y", "L", "M"),
        coords={"Y": years, "L": leads, "M": members},
        attrs={"units": "K"},
    )
    climatology = anomaly.mean(("Y", "M"))
    time = xr.DataArray(
        np.array(
            [
                ["2000-01-15", "2000-04-15"],
                ["2001-01-15", "2001-04-15"],
                ["2002-01-15", "2002-04-15"],
            ],
            dtype="datetime64[ns]",
        ),
        dims=("Y", "L"),
        coords={"Y": years, "L": leads},
    )
    return build_prepared_skill_dataset(
        anomaly,
        climatology,
        time,
        source="case-a",
        component="atm",
        variable="TREFHT",
        init_month=1,
        climatology_years=(2000, 2001),
        **PROVENANCE,
    )


def test_prepared_skill_round_trip(tmp_path):
    dataset = _prepared_dataset()
    path = prepared_skill_path(
        "case-a",
        "atm",
        "TREFHT",
        1,
        (2000, 2001),
        **PROVENANCE,
        root=tmp_path,
    )
    write_prepared_skill_dataset(dataset, path)

    expected = expected_prepared_skill_attrs(
        source="case-a",
        component="atm",
        variable="TREFHT",
        init_month=1,
        climatology_years=(2000, 2001),
        **PROVENANCE,
    )
    with open_prepared_skill_dataset(path, expected_attrs=expected) as loaded:
        xr.testing.assert_allclose(loaded["anomaly"], dataset["anomaly"])
        xr.testing.assert_allclose(loaded["climatology"], dataset["climatology"])
        xr.testing.assert_equal(loaded["time"], dataset["time"])


def test_prepared_skill_rejects_wrong_climatology_period(tmp_path):
    dataset = _prepared_dataset()
    path = tmp_path / "prepared.nc"
    write_prepared_skill_dataset(dataset, path)
    expected = expected_prepared_skill_attrs(
        source="case-a",
        component="atm",
        variable="TREFHT",
        init_month=1,
        climatology_years=(1981, 2010),
        **PROVENANCE,
    )

    with pytest.raises(ValueError, match="climatology_start_year"):
        open_prepared_skill_dataset(path, expected_attrs=expected)

    compatible, reason = prepared_skill_cache_status(path, expected_attrs=expected)
    assert not compatible
    assert "climatology_start_year" in reason


def test_prepared_skill_writer_rejects_missing_contract_metadata(tmp_path):
    dataset = _prepared_dataset()
    del dataset.attrs["source"]

    with pytest.raises(ValueError, match="missing attributes.*source"):
        write_prepared_skill_dataset(dataset, tmp_path / "prepared.nc")


def test_prepared_skill_writer_rejects_dimension_metadata_mismatch(tmp_path):
    dataset = _prepared_dataset()
    dataset.attrs["lead_count"] = 24

    with pytest.raises(ValueError, match="lead count"):
        write_prepared_skill_dataset(dataset, tmp_path / "prepared.nc")


def test_prepared_skill_writer_rejects_old_contract_version(tmp_path):
    dataset = _prepared_dataset()
    dataset.attrs["prepared_skill_version"] = "1"

    with pytest.raises(ValueError, match="prepared_skill_version"):
        write_prepared_skill_dataset(dataset, tmp_path / "prepared.nc")


@pytest.mark.parametrize(
    ("changed_key", "changed_value"),
    [
        ("requested_years", (2000, 2001)),
        ("target_grid", "latlon_2.0x2.0_periodic"),
        ("regridding_method", "bilinear"),
        ("ensemble_member_count", 10),
        ("lead_count", 8),
        ("unit_conversion_version", "kelvin_to_celsius_v2"),
        ("case_prefix", "different-case-prefix"),
        ("source_data_identity", "/archive/reprocessed/case-a/TREFHT"),
    ],
)
def test_prepared_skill_rejects_stale_provenance(tmp_path, changed_key, changed_value):
    dataset = _prepared_dataset()
    path = tmp_path / "prepared.nc"
    write_prepared_skill_dataset(dataset, path)
    changed = dict(PROVENANCE)
    changed[changed_key] = changed_value
    expected = expected_prepared_skill_attrs(
        source="case-a",
        component="atm",
        variable="TREFHT",
        init_month=1,
        climatology_years=(2000, 2001),
        **changed,
    )

    compatible, reason = prepared_skill_cache_status(path, expected_attrs=expected)
    assert not compatible
    reason_key = "requested_year" if changed_key == "requested_years" else changed_key
    assert reason_key in reason


def test_prepared_path_changes_with_provenance(tmp_path):
    first = prepared_skill_path(
        "case-a", "atm", "TREFHT", 1, (2000, 2001), root=tmp_path, **PROVENANCE
    )
    changed = dict(PROVENANCE)
    changed["requested_years"] = (2000, 2001)
    second = prepared_skill_path(
        "case-a", "atm", "TREFHT", 1, (2000, 2001), root=tmp_path, **changed
    )

    assert first != second
    assert "y2000-2002_ny3" in first.name
    assert "_v2" not in first.name


def test_generic_cache_status_validates_skill_metadata(tmp_path):
    path = tmp_path / "skill.nc"
    xr.Dataset(
        {"corr": ("L", [0.2])},
        attrs={"verification_years": "[2000,2001]"},
    ).to_netcdf(path)

    assert cache_status(
        path,
        expected_attrs={"verification_years": "[2000,2001]"},
        required_variables=("corr",),
    ) == (True, "compatible")
    compatible, reason = cache_status(
        path,
        expected_attrs={"verification_years": "[2000,2001,2002]"},
        required_variables=("corr",),
    )
    assert not compatible
    assert "verification_years" in reason
