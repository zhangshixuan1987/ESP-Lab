import pytest

from esp_lab.utils.spatial_utils import _normalize_lon_180

global_land_mask = pytest.importorskip("global_land_mask", reason="global_land_mask not installed")
globe = global_land_mask.globe


def test_is_land_negative_and_positive_lon_equivalent():
    """Normalize 0-360 longitudes before calling global_land_mask."""
    assert globe.is_land(50, -10) == globe.is_land(50, _normalize_lon_180(350))
