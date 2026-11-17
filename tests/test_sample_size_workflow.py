import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from workflows.diagnostics import sample_size as ss


def _scores(n_draws=20, lat=(-10.0, 10.0, 50.0), lon=(0.0, 120.0)):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n_draws, 2, len(lat), len(lon)))
    return xr.Dataset(
        {"acc": (("draw", "L", "lat", "lon"), data), "clim_error": (("draw", "L", "lat", "lon"), np.abs(data))},
        coords={"L": [3, 6], "lat": list(lat), "lon": list(lon)},
    )


def test_best_prefers_longest_record_inside_span_then_earlier_root():
    cands = [dict(y0=1980, y1=2011, ny=32, rank=1), dict(y0=1980, y1=1995, ny=16, rank=0),
             dict(y0=1980, y1=2018, ny=39, rank=1), dict(y0=1980, y1=2018, ny=39, rank=0)]
    assert ss._best(cands, (1980, 2018)) == dict(y0=1980, y1=2018, ny=39, rank=0)
    assert ss._best(cands, (1980, 1995))["ny"] == 16
    assert ss._best(cands, (1990, 1995)) is None


def test_reducer_area_means_and_rms_for_climatology_error():
    scores = _scores()
    reduce = ss.make_reducer({"Tropics": (-20, 20, "all"), "All": (-60, 80, "all")})
    out = scores.map(reduce)
    assert list(out.region.values) == ["Tropics", "All"]
    tropics = scores.acc.sel(lat=[-10.0, 10.0]).mean(("lat", "lon"))
    np.testing.assert_allclose(out.acc.sel(region="Tropics"), tropics)
    rms = np.sqrt((scores.clim_error.sel(lat=[-10.0, 10.0]) ** 2).mean(("lat", "lon")))
    np.testing.assert_allclose(out.clim_error.sel(region="Tropics"), rms)
    index = xr.Dataset({"acc": (("draw", "L"), np.ones((3, 2)))}, coords={"L": [3, 6]}).map(ss.make_reducer({}))
    assert list(index.region.values) == ["index"]


def test_summarize_rows_and_year_members_subset():
    reduced = _scores().map(ss.make_reducer({"All": (-60, 80, "all")}))
    rows = pd.DataFrame(ss.summarize(reduced, ["acc"], group="atm", field="TS", experiment="x", n_years=5))
    assert len(rows) == 2 and set(rows.columns) >= {"p05", "median", "p95", "lead", "region", "n_years"}
    assert (rows.p05 <= rows["median"]).all() and (rows["median"] <= rows.p95).all()

    members = xr.DataArray(np.zeros((4, 20)), dims=("Y", "M"))
    picked = ss.year_members(members, 10, seed=1)
    assert picked.sizes["M"] == 10 and ss.year_members(members, 10, seed=1).M.equals(picked.M)
    assert ss.year_members(members, 30, seed=1).sizes["M"] == 20


def test_plot_curves_writes_one_figure(tmp_path):
    table = pd.DataFrame([
        dict(group="sst", field="Nino3.4", experiment="E3SM-FOSIRL", metric="acc", region="index",
             lead=lead, n_years=n, p05=0.4, median=0.6, p95=0.8)
        for lead in (0, 1) for n in (5, 10)
    ])
    data = {"sst": {"Nino3.4": {"E3SM-FOSIRL": {"labels": ["JJA0", "SON0"]}}}}
    experiments = {"E3SM-FOSIRL": dict(label="E3SMv3-FOSIRL", color="black")}
    path = ss.plot_curves(table, data, "sst", "n_years", "ACC", "test", tmp_path / "fig.png", metric="acc",
                          experiments=experiments, plot_region={"sst": "index"}, plot_leads={"sst": [0, 1]}, show=False)
    assert path.is_file()
    assert ss.plot_curves(table, data, "atm", "n_years", "ACC", "t", tmp_path / "none.png", metric="acc",
                          experiments=experiments, plot_region={"atm": "x"}, plot_leads={"atm": [0]}, show=False) is None


def test_load_inputs_rejects_unknown_groups():
    with pytest.raises(ValueError, match="Unknown groups"):
        ss.load_inputs({"ocn": ["SST"]}, {}, read_roots=[])
