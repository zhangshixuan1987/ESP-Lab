import numpy as np
import pytest
from esp_lab.utils.spatial_utils import _build_regionmask_land_source


def test_regionmask_land_source_has_geographic_coordinates_and_binary_values():
    pytest.importorskip("regionmask", reason="regionmask not installed")

    source = _build_regionmask_land_source(resolution=10.0)

    assert set(source.sftlf.dims) == {"lat", "lon"}
    assert float(source.lat.min()) > -90.0
    assert float(source.lat.max()) < 90.0
    assert float(source.lon.min()) >= 0.0
    assert float(source.lon.max()) < 360.0
    assert np.all(np.diff(source.lat) > 0)
    assert np.all(np.diff(source.lon) > 0)
    assert set(np.unique(source.sftlf)) == {0.0, 1.0}
