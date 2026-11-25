"""Regional skill panels survive empty regions and keep lead labels aligned."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import xarray as xr

from workflows.leadtime_skill.regional_acc_skill_plot import plot_regional_acc_skill_template


def _skill(values_by_region, nlead):
    regions = list(values_by_region)
    data = np.array([values_by_region[r] for r in regions], dtype=float)
    step = 3 if nlead <= 8 else 1  # seasonal caches store L=3, 6, ...
    coords = {"region": regions, "L": np.arange(1, nlead + 1) * step}
    return xr.Dataset({"corr": (("region", "L"), data), "rmse": (("region", "L"), data)}, coords=coords)


def test_empty_region_panel_and_label_alignment():
    nan = np.nan
    seasonal = _skill({"global": [0.5, nan, 0.4, 0.3], "tropics": [nan] * 4}, 4)
    monthly = _skill({"global": [0.5] * 24, "tropics": [nan] * 24}, 24)
    seasons = {11: ("DJF", "MAM", "JJA", "SON")}

    fig, axes = plot_regional_acc_skill_template(
        {("CaseA", 11): seasonal}, monthly_skills_by_case_month={("CaseA", 11): monthly},
        region="global", init_months=(11,), season_labels=seasons,
    )
    labels = [t.get_text() for t in axes[0, 0].get_xticklabels()]
    assert labels == ["3:DJF", "9:JJA", "12:SON"]  # L=6 dropped, labels stay put

    fig, axes = plot_regional_acc_skill_template(
        {("CaseA", 11): seasonal}, monthly_skills_by_case_month={("CaseA", 11): monthly},
        region="tropics", init_months=(11,), season_labels=seasons,
    )
    assert any("no verifiable data" in t.get_text() for t in axes[0, 0].texts)


def test_nrmse_ignores_cells_without_observed_variability():
    from workflows.leadtime_skill.regional_acc import regional_skill_metrics

    lat, lon = np.array([-80.0, 0.0, 40.0]), np.array([0.0, 90.0])
    sig = np.array([[1e-23, 1e-23], [0.5, 0.4], [0.3, 0.6]])
    rmse = np.array([[1e20, 1e21], [1.0, 1.0], [1.0, 1.0]])
    skill = xr.Dataset(
        {"corr": (("L", "lat", "lon"), np.full((1, 3, 2), 0.5)),
         "rmse": (("L", "lat", "lon"), rmse[None]), "sig_obs": (("L", "lat", "lon"), sig[None])},
        coords={"L": [3], "lat": lat, "lon": lon},
    )
    region = {"global": {"bounds": (0.0, 360.0, -90.0, 90.0), "mask": None}}
    metrics = regional_skill_metrics(skill, region)
    assert np.isclose(float(metrics.rmse.sel(region="global", L=3)), 1.0)
    assert float(regional_skill_metrics(skill, region, min_obs_std_fraction=0).rmse.sel(region="global", L=3)) > 1e10


def test_axis_limits_follow_the_data():
    from workflows.leadtime_skill.regional_acc_skill_plot import acc_axis_limits, nrmse_axis_limits

    assert acc_axis_limits([0.1, 0.9]) == (-0.2, 1.0)
    assert acc_axis_limits([-0.34, 0.2]) == (-0.4, 1.0)
    assert nrmse_axis_limits([0.4, 0.9]) == (0.0, 1.25)
    assert nrmse_axis_limits([0.4, 1.61]) == (0.0, 1.75)
    assert nrmse_axis_limits([0.4, 9.0]) == (0.0, 2.5)


def test_weak_observed_leads_are_blank_and_large_nrmse_is_marked():
    nan = np.nan
    regions = ["conus"]
    skill = xr.Dataset(
        {
            "corr": (("region", "L"), [[0.5, 0.1, 0.4, 0.3]]),
            "rmse": (("region", "L"), [[0.8, 9.0, 3.2, 1.1]]),
            "sig_obs": (("region", "L"), [[10.0, 0.1, 8.0, 9.0]]),  # lead 6 barely varies
        },
        coords={"region": regions, "L": [3, 6, 9, 12]},
    )
    fig, axes = plot_regional_acc_skill_template(
        {("CaseA", 11): skill}, region="conus", init_months=(11,),
        season_labels={11: ("DJF", "MAM", "JJA", "SON")},
    )
    assert fig.blanked_leads == 2  # the 11-init row and the all-init row
    rmse_line = axes[0, 1].get_lines()[1].get_ydata()  # [0] is the 1.0 reference line
    assert np.isnan(rmse_line[1])                       # weak lead blanked, not 9.0
    assert axes[0, 1].get_ylim() == (0.0, 2.5)          # 3.2 is above the cap
    assert fig.off_scale_points == 2                    # 3.2 in both rows, drawn as markers
    assert np.isclose(fig.get_size_inches()[1], 2 * 2.9 + 1.6)  # height grows with rows only
