import numpy as np
import xarray as xr

from esp_lab.leadtime_plot_utils import (
    add_acc_comparison_markers,
    add_pointwise_significance_markers,
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


def test_pointwise_significance_markers_use_complete_valid_area():
    coords = {"L": [1], "lat": [-60.0, 0.0], "lon": [0.0, 90.0]}
    skill = xr.Dataset(
        {
            "corr": (("L", "lat", "lon"), [[[0.5, 0.5], [0.5, 0.5]]]),
            "pval": (("L", "lat", "lon"), [[[0.01, 0.2], [0.01, 0.01]]]),
            "sample_count": (("L",), [10]),
            "valid_sample_count": (
                ("L", "lat", "lon"), [[[10, 10], [10, 9]]]
            ),
        },
        coords=coords,
    )
    lon2d, lat2d = np.meshgrid(coords["lon"], coords["lat"])
    axis = _Axis()

    fraction = add_pointwise_significance_markers(
        axis, skill, 0, longitude_2d=lon2d, latitude_2d=lat2d,
        significance_level=0.1,
        style={"latlim": 80, "marker_stride": 1, "marker_size": 4},
        font_size=10,
    )

    # Significant weights: cos(60) + cos(0); valid adds one more cos(60).
    assert np.isclose(fraction, 0.75)
    assert len(axis.scatter_calls[0][0][0]) == 2
    assert "75.0% sig." in axis.text_calls[0][0][2]


def test_subset_conus_handles_0_360_longitudes_and_descending_latitudes():
    import numpy as np
    import xarray as xr

    from esp_lab.leadtime_plot_utils import subset_conus

    data = xr.DataArray(
        np.arange(72 * 36, dtype=float).reshape(36, 72),
        coords={"lat": np.arange(87.5, -90, -5.0), "lon": np.arange(2.5, 360, 5.0)},
        dims=("lat", "lon"),
    )
    conus = subset_conus(data)
    assert float(conus.lon.min()) >= -125 and float(conus.lon.max()) <= -66
    assert float(conus.lat.min()) >= 24 and float(conus.lat.max()) <= 50
    assert conus.sizes["lon"] > 0 and conus.sizes["lat"] > 0


def test_plot_conus_rmse_panels_draws_available_seasons(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import xarray as xr

    from esp_lab.leadtime_plot_utils import plot_conus_rmse_panels

    lat = np.arange(-87.5, 90, 5.0)
    lon = np.arange(-177.5, 180, 5.0)
    field = xr.DataArray(
        np.random.default_rng(0).uniform(0.5, 1.5, (7, lat.size, lon.size)),
        coords={"L": np.arange(3, 22, 3), "lat": lat, "lon": lon}, dims=("L", "lat", "lon"),
    )
    fig = plot_conus_rmse_panels(
        {"A": {11: field}, "B": {11: field.isel(L=slice(0, 2))}},
        model_names={"A": "Case A"}, init_months=[11],
        levels=np.arange(0.5, 1.55, 0.1), colorbar_label="nRMSE", title="test", extend="both",
    )
    # 2 seasons x 2 forecast years x 2 models, plus the colorbar.
    assert len(fig.axes) == 9
    fig.savefig(tmp_path / "conus.png", dpi=30)
