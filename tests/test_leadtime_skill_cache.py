import numpy as np
import pytest

from esp_lab.leadtime_skill_cache import (
    build_member_selection_dataset,
    expected_skill_cache_attrs,
    file_inventory_digest,
    generate_member_indices,
    member_selection_attrs,
    source_fingerprint,
    stable_resampling_seed,
    validate_member_selection_dataset,
    verification_year_token,
    year_number,
)


IDENTITY = {
    "field": "PRECT",
    "case": "E3SM-FOSIRL",
    "init_month": 11,
    "verification_years": list(range(1980, 2004)),
    "sample_size": 10,
    "iterations": 100,
}


def _selection(identity=IDENTITY):
    seed = stable_resampling_seed(42, identity)
    indices = generate_member_indices(
        population_size=20,
        sample_size=10,
        iterations=100,
        seed=seed,
    )
    attrs = member_selection_attrs(
        base_seed=42,
        derived_seed=seed,
        identity=identity,
        member_indices=indices,
    )
    return seed, indices, attrs


def test_stable_seed_is_independent_of_mapping_order():
    reversed_identity = dict(reversed(list(IDENTITY.items())))

    assert stable_resampling_seed(42, IDENTITY) == stable_resampling_seed(
        42, reversed_identity
    )


def test_stable_seed_changes_for_case_month_and_years():
    baseline = stable_resampling_seed(42, IDENTITY)
    for key, value in (
        ("case", "E3SM-Reanalysis"),
        ("init_month", 5),
        ("verification_years", list(range(1980, 2019))),
    ):
        changed = dict(IDENTITY)
        changed[key] = value
        assert stable_resampling_seed(42, changed) != baseline


def test_member_indices_are_reproducible_and_without_replacement():
    seed, first, _ = _selection()
    second = generate_member_indices(
        population_size=20,
        sample_size=10,
        iterations=100,
        seed=seed,
    )

    np.testing.assert_array_equal(first, second)
    assert all(np.unique(row).size == 10 for row in first)


def test_member_selection_round_trip_validation():
    _, indices, attrs = _selection()
    dataset = build_member_selection_dataset(indices, attrs=attrs)

    validate_member_selection_dataset(
        dataset,
        expected_attrs=attrs,
        population_size=20,
    )


def test_member_selection_rejects_modified_values():
    _, indices, attrs = _selection()
    dataset = build_member_selection_dataset(indices, attrs=attrs)
    dataset["member_indices"][0, 0] = 19

    with pytest.raises(ValueError, match="checksum|replacement"):
        validate_member_selection_dataset(
            dataset,
            expected_attrs=attrs,
            population_size=20,
        )


def test_source_fingerprint_requires_revision_and_tracks_changes():
    identity = {"archive": "/data", "case_prefix": "case-a", "field": "PRECT"}
    first = source_fingerprint(identity, source_revision="manifest-abc")

    assert first == source_fingerprint(identity, source_revision="manifest-abc")
    assert first != source_fingerprint(identity, source_revision="manifest-def")
    with pytest.raises(ValueError, match="source_revision"):
        source_fingerprint(identity, source_revision="")


def test_file_inventory_digest_tracks_exact_files(tmp_path):
    first = tmp_path / "a.nc"
    second = tmp_path / "b.nc"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    baseline = file_inventory_digest([second, first], root=tmp_path)

    assert baseline == file_inventory_digest([first, second], root=tmp_path)
    second.write_bytes(b"replacement")
    assert baseline != file_inventory_digest([first, second], root=tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        file_inventory_digest([])


def test_expected_skill_cache_attrs_adds_internal_schema_contract():
    attrs = expected_skill_cache_attrs(
        source="case-a",
        source_data_identity="sha256:abc",
        case_prefix="case-a-prefix",
        variable="PRECT",
        init_month=11,
        verification_years=(2000, 2001),
        climatology_years=(1981, 2010),
        observation_product="GPCP",
        observation_variable="PRECT",
        observation_data_identity="sha256:obs",
        target_grid="latlon_1x1",
        regridding_method="conservative",
        ensemble_member_count=10,
        lead_start=1,
        lead_end=8,
        detrend=True,
        unit_conversion_version="precip_v1",
    )

    assert attrs["skill_cache_version"] == "2"
    assert attrs["verification_years"] == "2000,2001"
    assert attrs["verification_year_count"] == 2
    assert attrs["detrend"] == 1


def test_expected_skill_cache_attrs_requires_verification_years():
    with pytest.raises(ValueError, match="verification_years"):
        expected_skill_cache_attrs(
            source="case-a",
            source_data_identity="sha256:abc",
            case_prefix="case-a-prefix",
            variable="PRECT",
            init_month=11,
            verification_years=(),
            climatology_years=(1981, 2010),
            observation_product="GPCP",
            observation_variable="PRECT",
            observation_data_identity="sha256:obs",
            target_grid="latlon_1x1",
            regridding_method="conservative",
            ensemble_member_count=10,
            lead_start=1,
            lead_end=8,
            detrend=True,
            unit_conversion_version="precip_v1",
        )


def test_verification_year_helpers_accept_integer_and_datetime_coordinates():
    years = np.array(["2000-01-15", "2001-01-15"], dtype="datetime64[D]")

    assert year_number(years[0]) == 2000
    assert verification_year_token(years) == "y2000-2001_ny2"
    with pytest.raises(ValueError, match="verification_years"):
        verification_year_token([])
