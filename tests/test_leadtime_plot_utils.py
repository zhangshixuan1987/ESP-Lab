import numpy as np
import xarray as xr

from esp_lab.leadtime_plot_utils import (
    add_acc_comparison_markers,
    seasonal_label,
)


class _Axis:
    transAxes = object()

    def __init__(self):
        self.scatter_calls = []
        self.text_calls = []

    def scatter(self, *args, **kwargs):
        self.scatter_calls.append((args, kwargs))

    def text(self, *args, **kwargs):
        self.text_calls.append((args, kwargs))


def test_seasonal_label_matches_initialization_and_lead():
    assert seasonal_label(11, 1) == "DJF"
    assert seasonal_label(5, 1) == "JJA"


def test_comparison_marker_percentages_use_only_valid_area():
    coords = {"L": [1], "lat": [-30.0, 30.0], "lon": [0.0, 90.0]}
    base = xr.Dataset(
        {"corr": (("L", "lat", "lon"), [[[0.5, 0.5], [0.5, np.nan]]])},
        coords=coords,
    )
    superiority = xr.DataArray(
        [[[0.95, 0.05], [0.5, 0.95]]], dims=("L", "lat", "lon"), coords=coords
    )
    area = xr.DataArray(
        np.ones((2, 2)), dims=("lat", "lon"), coords={"lat": coords["lat"], "lon": coords["lon"]}
    )
    lon2d, lat2d = np.meshgrid(coords["lon"], coords["lat"])
    style = {
        "latlim": 80,
        "sig_level": 0.1,
        "marker_stride": 1,
        "marker_color": "black",
        "open_marker_size": 12,
        "filled_marker_size": 5,
        "marker_linewidth": 0.5,
    }

    smyle_fraction, e3sm_fraction = add_acc_comparison_markers(
        _Axis(), base, superiority, 0,
        area=area, latitude=area.lat, longitude_2d=lon2d, latitude_2d=lat2d,
        style=style, font_size=10, font_weight="bold", label_bbox={},
    )

    assert smyle_fraction == 1 / 3
    assert e3sm_fraction == 1 / 3
