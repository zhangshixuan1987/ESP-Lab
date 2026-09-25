from esp_lab import data_access_cesm_smyle
from esp_lab import data_access_e3sm


def test_e3sm_defers_concat_dimension_chunks():
    open_chunks, post_chunks = data_access_e3sm._split_open_and_post_chunks(
        {"Y": 1, "L": 12, "M": -1, "lat": 90, "lon": 180}
    )

    assert open_chunks == {"L": 12, "lat": 90, "lon": 180}
    assert post_chunks == {"Y": 1, "M": -1}


def test_cesm_smyle_defers_concat_dimension_chunks():
    open_chunks, post_chunks = data_access_cesm_smyle._split_open_and_post_chunks(
        {"Y": -1, "L": 12, "M": 1, "lat": 96, "lon": 144}
    )

    assert open_chunks == {"L": 12, "lat": 96, "lon": 144}
    assert post_chunks == {"Y": -1, "M": 1}


def test_none_chunks_use_native_open_chunks():
    assert data_access_e3sm._split_open_and_post_chunks(None) == ({}, {})
    assert data_access_cesm_smyle._split_open_and_post_chunks(None) == ({}, {})
