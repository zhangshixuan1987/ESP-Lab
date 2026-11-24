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


# --- polar-cap (sea-ice) helpers -------------------------------------------

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from esp_lab.leadtime_plot_utils import (  # noqa: E402
    add_missing_map_panel,
    mask_weak_observed_variability,
    polar_cap,
    polar_projection,
    set_polar_extent,
    style_polar_map_axis,
)


def _skill(lat):
    coords = {"L": [3, 6], "lat": lat, "lon": [0.0, 180.0]}
    shape = (2, len(lat), 2)
    return xr.Dataset(
        {
            "corr": (("L", "lat", "lon"), np.full(shape, 0.5)),
            "sig_obs": (("L", "lat", "lon"), np.full(shape, 5.0)),
            "sample_count": (("L",), [39, 39]),
        },
        coords=coords,
    )


@pytest.mark.parametrize("lat", [[-75.0, -45.0, 45.0, 75.0], [75.0, 45.0, -45.0, -75.0]])
def test_polar_cap_selects_each_hemisphere_for_either_latitude_order(lat):
    skill = _skill(lat)
    assert sorted(polar_cap(skill, "NH", 50).lat.values) == [75.0]
    assert sorted(polar_cap(skill, "sh", -50).lat.values) == [-75.0]
    assert polar_cap(skill, "NH", 50).sample_count.dims == ("L",)


def test_polar_helpers_reject_bad_input():
    with pytest.raises(ValueError, match="hemisphere"):
        polar_projection("Arctic")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=polar_projection("NH"))
    with pytest.raises(ValueError, match="boundary_lat"):
        set_polar_extent(ax, "NH", 0)
    plt.close(fig)


def test_mask_weak_observed_variability_keeps_per_lead_scalars():
    skill = _skill([45.0, 75.0])
    skill["sig_obs"][:, 0, :] = 0.5
    masked = mask_weak_observed_variability(skill, 2.0)
    assert masked.corr.isel(lat=0).isnull().all()
    assert masked.corr.isel(lat=1).notnull().all()
    assert masked.sample_count.values.tolist() == [39, 39]
    assert mask_weak_observed_variability(skill, None) is skill
    with pytest.raises(ValueError, match="sig_obs"):
        mask_weak_observed_variability(skill.drop_vars("sig_obs"), 1.0)


@pytest.mark.parametrize("hemisphere", ["NH", "SH"])
def test_polar_panels_and_markers_render(tmp_path, hemisphere):
    style = {"grid_linewidth": 0.3, "grid_color": "0.4", "grid_alpha": 0.4}
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(1, 2, 1, projection=polar_projection(hemisphere))
    ax.set_aspect("auto")  # as map_pcolor_global_subplot leaves it
    style_polar_map_axis(ax, hemisphere=hemisphere, boundary_lat=50, style=style)
    assert ax.get_aspect() == 1.0
    sign = 1 if hemisphere == "NH" else -1
    lat = np.array([sign * 60.0, sign * 80.0])
    lon = np.array([0.0, 90.0])
    skill = xr.Dataset(
        {
            "corr": (("L", "lat", "lon"), np.full((1, 2, 2), 0.8)),
            "pval": (("L", "lat", "lon"), np.full((1, 2, 2), 0.01)),
        },
        coords={"L": [3], "lat": lat, "lon": lon},
    )
    lon2d, lat2d = np.meshgrid(lon, lat)
    fraction = add_pointwise_significance_markers(
        ax, skill, 0, longitude_2d=lon2d, latitude_2d=lat2d,
        significance_level=0.1, style={"latlim": 90}, font_size=6,
    )
    assert fraction == pytest.approx(1.0)
    # Markers are placed in lon/lat, so they land inside the polar cap.
    offsets = ax.collections[-1].get_offsets()
    assert len(offsets) == 4
    missing = add_missing_map_panel(
        fig, nrows=1, ncols=2, subplot=2, title="t", message="none",
        font_size=6, hemisphere=hemisphere, boundary_lat=50,
    )
    assert type(missing.projection) is type(polar_projection(hemisphere))
    fig.savefig(tmp_path / f"polar_{hemisphere}.png")
    plt.close(fig)
