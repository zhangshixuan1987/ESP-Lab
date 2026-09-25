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
