import numpy as np
import pytest
import xarray as xr

from esp_lab.utils import mov_map_utils as movmaps


def _pattern(lon=(-90, 30), lat=(20, 80)):
    return xr.DataArray(
        np.zeros((len(lat), len(lon))),
        coords={"lat": list(lat), "lon": list(lon)},
        dims=("lat", "lon"),
    )


def test_mode_extent_uses_polar_caps_for_annular_modes():
    nam_pattern = _pattern(lon=(-180, 180), lat=(10, 85))
    sam_pattern = _pattern(lon=(-180, 180), lat=(-85, -10))

    assert movmaps.mode_extent("NAM", nam_pattern) == [-180, 180, 20, 90]
    assert movmaps.mode_extent("SAM", sam_pattern) == [-180, 180, -90, -20]


def test_mode_extent_uses_pattern_bounds_for_regional_modes():
    pattern = _pattern(lon=(-90, 30), lat=(20, 80))

    assert movmaps.mode_extent("NAO", pattern) == [-90.0, 30.0, 20.0, 80.0]


def test_mode_extent_uses_pattern_bounds_for_amo():
    pattern = _pattern(lon=(-90, 30), lat=(20, 80))

    assert movmaps.mode_extent("AMO", pattern) == [-90.0, 30.0, 20.0, 80.0]


def test_mode_extent_uses_north_pacific_sector_for_npo_pdo_npgo():
    pattern = _pattern(lon=(120, 280), lat=(-20, 80))
    pna_pattern = _pattern(lon=(120, 240), lat=(20, 85))

    assert movmaps.mode_extent("PNA", pna_pattern) == [120, 240, 15, 85]
    assert movmaps.mode_extent("NPO", pattern) == [120, 240, 15, 75]
    assert movmaps.mode_extent("PDO", pattern) == [120, 240, 15, 75]
    assert movmaps.mode_extent("NPGO", pattern) == [120, 240, 15, 75]


def test_tick_values_uses_atlantic_ticks_for_atlantic_projection():
    settings = {
        "longitude_ticks": np.arange(-180, 181, 60),
        "latitude_ticks": np.arange(-90, 91, 30),
        "atlantic_longitude_ticks": np.arange(-90, 31, 30),
        "atlantic_latitude_ticks": np.arange(30, 81, 10),
    }

    lon_ticks, lat_ticks = movmaps.tick_values("atlantic", settings)

    np.testing.assert_array_equal(lon_ticks, [-90, -60, -30, 0, 30])
    np.testing.assert_array_equal(lat_ticks, [30, 40, 50, 60, 70, 80])


def test_tick_values_uses_global_ticks_for_global_projections():
    settings = {
        "longitude_ticks": np.arange(-180, 181, 60),
        "latitude_ticks": np.arange(-90, 91, 30),
        "atlantic_longitude_ticks": np.arange(-90, 31, 30),
        "atlantic_latitude_ticks": np.arange(30, 81, 10),
        "global_longitude_ticks": np.arange(-180, 181, 90),
        "global_latitude_ticks": np.arange(-60, 61, 30),
    }

    lon_ticks, lat_ticks = movmaps.tick_values("pacific_global", settings)

    np.testing.assert_array_equal(lon_ticks, [-180, -90, 0, 90, 180])
    np.testing.assert_array_equal(lat_ticks, [-60, -30, 0, 30, 60])


def test_tick_values_uses_north_pacific_ticks_for_sector_projection():
    settings = {
        "longitude_ticks": np.arange(-180, 181, 60),
        "latitude_ticks": np.arange(-90, 91, 30),
        "atlantic_longitude_ticks": np.arange(-90, 31, 30),
        "atlantic_latitude_ticks": np.arange(30, 81, 10),
        "global_longitude_ticks": np.arange(-180, 181, 90),
        "global_latitude_ticks": np.arange(-60, 61, 30),
        "north_pacific_longitude_ticks": np.arange(120, 241, 30),
        "north_pacific_latitude_ticks": np.arange(20, 81, 10),
    }

    lon_ticks, lat_ticks = movmaps.tick_values("north_pacific", settings)

    np.testing.assert_array_equal(lon_ticks, [120, 150, 180, 210, 240])
    np.testing.assert_array_equal(lat_ticks, [20, 30, 40, 50, 60, 70, 80])


def test_format_longitude_label():
    expected = {
        -180: "180°",
        -60: "60°W",
        0: "0°",
        30: "30°E",
        180: "180°",
    }

    for lon, label in expected.items():
        assert movmaps.format_longitude_label(lon) == label


class _RecordingAxis:
    def __init__(self):
        self.transAxes = object()
        self.text_calls = []

    def text(self, x, y, label, **kwargs):
        self.text_calls.append((x, y, label, kwargs))


def test_outside_polar_longitude_labels_use_axes_coordinates():
    axis = _RecordingAxis()
    data_projection = object()

    movmaps.add_polar_longitude_labels(
        axis,
        "PSA1",
        [-180, 180, -90, -20],
        data_projection,
        longitude_ticks=[-120, -60, 60, 120],
        label_offset=2.5,
        fontsize=10,
        label_position="outside_axes",
        axes_radius=0.625,
        central_longitude=0,
    )

    assert [call[2] for call in axis.text_calls] == [
        "120°W", "60°W", "60°E", "120°E"
    ]
    assert all(call[3]["transform"] is axis.transAxes for call in axis.text_calls)
    assert len({(round(call[0], 6), round(call[1], 6)) for call in axis.text_calls}) == 4


def test_side_polar_latitude_labels_use_axes_coordinates():
    axis = _RecordingAxis()

    movmaps.add_polar_latitude_labels(
        axis,
        "PSA1",
        latitude_ticks=[-85, -55, -25],
        label_longitude=0,
        data_projection=object(),
        fontsize=10,
        label_position="side",
    )

    assert all(call[3]["transform"] is axis.transAxes for call in axis.text_calls)


def test_make_projection_can_be_disabled_without_cartopy():
    projection, name = movmaps.make_projection("NAO", use_cartopy=False)

    assert projection is None
    assert name is None


def test_mask_pattern_for_plot_applies_spatial_ocean_mask():
    pattern = xr.DataArray(
        np.arange(6, dtype=float).reshape(2, 3),
        coords={"lat": [10, 20], "lon": [120, 130, 140]},
        dims=("lat", "lon"),
    )
    ocean_mask = xr.DataArray(
        [[True, False, True], [False, True, True]],
        coords=pattern.coords,
        dims=pattern.dims,
    )

    masked = movmaps.mask_pattern_for_plot(pattern, ocean_mask)

    assert np.isnan(masked.sel(lat=10, lon=130))
    assert np.isnan(masked.sel(lat=20, lon=120))
    assert masked.sel(lat=10, lon=120) == pattern.sel(lat=10, lon=120)


@pytest.mark.skipif(not movmaps.has_cartopy(), reason="Cartopy is unavailable")
def test_make_projection_uses_mode_defaults_when_cartopy_is_available():
    _, nao_name = movmaps.make_projection("NAO")
    _, pna_name = movmaps.make_projection("PNA")
    _, psa1_name = movmaps.make_projection("PSA1")
    _, psa2_name = movmaps.make_projection("PSA2")
    _, pdo_name = movmaps.make_projection("PDO")
    _, npo_name = movmaps.make_projection("NPO")
    _, npgo_name = movmaps.make_projection("NPGO")
    _, amo_name = movmaps.make_projection("AMO")

    assert nao_name == "atlantic"
    assert pna_name == "north_pacific"
    assert psa1_name == "south_polar"
    assert psa2_name == "south_polar"
    assert pdo_name == "north_pacific"
    assert npo_name == "north_pacific"
    assert npgo_name == "north_pacific"
    assert amo_name == "platecarree"
