"""Regression checks for lead selection and archive-free ACC restarts."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cftime
import numpy as np
import pytest
import xarray as xr

from esp_lab import env_paths, leadtime_prepared_cache, leadtime_skill_cache, stats
from esp_lab import leadtime_workflow as workflow
from esp_lab.data_access_obs import mon_to_seas_obs
from esp_lab.leadtime_validation import check_remove_drift_sample
from esp_lab.paths import leadtime_acc_dir
from esp_lab.utils import calendar_utils, unit_conversion
from esp_lab.utils.netcdf_utils import atomic_to_netcdf, load_netcdf
from esp_lab.utils.resource_utils import ResourceTracker


NOTEBOOK = Path(__file__).parents[1] / "jupyter/1a_atm_leadtime_acc_skill_map.ipynb"


# Stages are located by a line unique to their cell, not by position, so
# reorganizing the notebook (inserting or splitting cells) does not silently
# point the tests at different code.
IMPORT_CELL = "from esp_lab import leadtime_realm_skill"
CONFIG_CELL = 'E3SM_REFERENCE_CASE = "E3SM-FOSIRL"'
LAYOUT_CELL = "def e3sm_leadtime_dir(case_info, stage, *parts):"
PLAN_CELL = '"cleanup_temporary": WORKFLOW_SETTINGS["cache"].get("cleanup_temp_files", True),'
SEASONAL_CACHE_CELL = 'debug = WORKFLOW_SETTINGS["cache"]["debug_seasonal_cache"]'
PIPELINE_CELLS = (
    'target_dlat = WORKFLOW_SETTINGS["regrid"]["target_dlat"]',  # regrid E3SM
    'smyle_nens = WORKFLOW_SETTINGS["smyle"]["nens"]',  # CESM-SMYLE benchmark
    'obs_dir = WORKFLOW_SETTINGS["obs"]["data_dir"]',  # observations
    "init_month: e3sm_seas_by_case_month[case_key][init_month].time.load()",  # time axes
    'prepared_reuse = prepared_mode != "rebuild"',  # E3SM prepared anomalies
    "smyle05_time = smyle11_time = None",  # SMYLE prepared anomalies
    'if WORKFLOW_SETTINGS["diagnostics"].get("run_drift_check", False):',  # drift check
    'force_compute = WORKFLOW_SETTINGS["skill"]["force_compute"]',  # E3SM skill
    "smyle_overlap_skill_by_month = {}",  # SMYLE skill
    'compare_cfg = WORKFLOW_SETTINGS["finite_ensemble_compare"]',  # finite-ensemble compare
)


def _cell(key):
    """Return a code cell's source by position or by a line unique to that cell."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    if isinstance(key, int):
        cell = cells[key]
    else:
        matches = [
            c for c in cells
            if c["cell_type"] == "code" and key in "".join(c["source"])
        ]
        assert len(matches) == 1, f"{len(matches)} notebook cells contain {key!r}"
        cell = matches[0]
    return "".join(line for line in cell["source"] if not line.startswith("%"))


def _seasonal_data(members, *, missing=False):
    rng = np.random.default_rng(12)
    years = np.arange(2000, 2008)
    values = rng.normal(size=(8, 4, members, 2, 3))
    if missing:
        values[..., 0] = np.nan
    model = xr.DataArray(
        values, dims=("Y", "L", "M", "lat", "lon"),
        coords={"Y": [f"{y}110100" for y in years], "L": [3, 6, 9, 12],
                "M": np.arange(members), "lat": [-30., 30.], "lon": [0., 120., 240.]},
    )
    time = xr.DataArray(
        [[cftime.DatetimeNoLeap(int(y + 1), month, 15) for month in [1, 4, 7, 10]]
         for y in years], dims=("Y", "L"), coords={"Y": model.Y, "L": model.L},
    )
    return model, time


def _observations():
    rng = np.random.default_rng(25)
    times = [cftime.DatetimeNoLeap(y, m, 15) for y in range(1999, 2011) for m in range(1, 13)]
    return xr.DataArray(
        rng.normal(size=(len(times), 2, 3)), dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [-30., 30.], "lon": [0., 120., 240.]},
    )


@pytest.mark.parametrize("monthly_leads", [12, 24])
@pytest.mark.parametrize("reordered", [False, True])
def test_monthly_to_seasonal_cache_write_and_restart(tmp_path, monthly_leads, reordered):
    """Exercise the notebook's real aggregation, encoding, write, and reuse cell."""
    years = [2000, 2001, 2002]
    raw = xr.Dataset(
        {"PRECT": (("Y", "L", "M", "lat", "lon"),
                   np.random.default_rng(10).normal(size=(3, monthly_leads, 2, 2, 3)))},
        coords={"Y": [f"{year}110100" for year in years],
                "L": np.arange(1, monthly_leads + 1), "M": [0, 1],
                "lat": [-30., 30.], "lon": [0., 120., 240.]},
    )
    raw["time"] = xr.DataArray(
        [[cftime.DatetimeNoLeap(year + (10 + lead) // 12, (10 + lead) % 12 + 1, 15)
          for lead in range(monthly_leads)] for year in years],
        dims=("Y", "L"), coords={"Y": raw.Y, "L": raw.L},
    )
    if reordered:
        raw["PRECT"] = raw.PRECT.transpose("M", "Y", "L", "lon", "lat")
    chunks = {"Y": 2, "L": monthly_leads, "M": 1, "lat": 2, "lon": 3}
    raw = raw.chunk(chunks)
    expected = calendar_utils.mon_to_seas_dask(raw).compute()
    path = tmp_path / "seasonal.nc"
    attrs = {"cache_kind": "seasonal_e3sm_input"}
    tracker = ResourceTracker()
    ns = dict(
        xr=xr, cal=calendar_utils, field="PRECT", mchunk=chunks,
        WORKFLOW_SETTINGS={"cache": {"debug_seasonal_cache": False},
                           "e3sm": {"encoding_chunksizes": (1, 8, 1, 90, 180)}},
        seasonal_force_rewrite=False, E3SM_CASES={"case-a": {"cache_tag": "case-a"}},
        E3SM_REFERENCE_CASE="case-a", e3sm_months_to_prepare={"case-a": [11]},
        e3sm_raw_by_case_month={"case-a": {11: raw}},
        seasonal_input_specs_by_case_month={"case-a": {11: (path, attrs)}},
        e3sm_leadtime_dir=lambda *args: tmp_path,
        cache_status=leadtime_prepared_cache.cache_status, atomic_to_netcdf=atomic_to_netcdf,
        seasonal_cache_encoding=workflow.seasonal_cache_encoding,
        netcdf_write_options={}, workflow_resources=tracker,
    )
    try:
        exec(_cell(SEASONAL_CACHE_CELL), ns)
        with xr.open_dataset(path) as saved:
            xr.testing.assert_allclose(saved, expected)
            assert saved.attrs == attrs
            assert dict(zip(saved.PRECT.dims, saved.PRECT.encoding["chunksizes"])) == {
                "Y": 1, "L": monthly_leads // 3, "M": 1, "lat": 2, "lon": 3,
            }
        stamp = path.stat().st_mtime_ns
        tracker.close()
        # A restarted cell must reuse the file without accessing raw input.
        ns.update(e3sm_raw_by_case_month={}, cal=SimpleNamespace(mon_to_seas_dask=_forbid),
                  workflow_resources=ResourceTracker())
        exec(_cell(SEASONAL_CACHE_CELL), ns)
        assert path.stat().st_mtime_ns == stamp
        xr.testing.assert_allclose(ns["e3sm_seas_by_case_month"]["case-a"][11], expected)
    finally:
        tracker.close()
        ns["workflow_resources"].close()


@pytest.mark.parametrize("requested", [(1, 0, 1, 90, 180), (1, 8.5, 1, 90, 180), (1, 8)])
def test_seasonal_cache_rejects_invalid_chunks(requested):
    model, _ = _seasonal_data(2)
    with pytest.raises(ValueError, match="chunks|chunk sizes"):
        workflow.seasonal_cache_encoding(model, requested)


def test_selected_leads_match_full_skill_and_batch():
    model, time = _seasonal_data(3)
    obs = _observations()
    full = stats.compute_skill_seasonal(model, time, obs, "2000", "2007", nleads=4)
    selected = workflow.compute_skill_lead_range(model, time, obs, "2000", "2007", 2, 3)
    xr.testing.assert_allclose(selected, full.isel(L=slice(1, 3)))
    np.testing.assert_array_equal(selected.L, [6, 9])
    members = np.array([[0, 1], [1, 2]])
    batch_full = stats.compute_skill_seasonal_batch(
        model, time, obs, "2000", "2007", members, nleads=4, metrics=("corr",),
    )
    batch = workflow.compute_skill_lead_range_batch(
        model, time, obs, "2000", "2007", members, 2, 3, metrics=("corr",),
    )
    xr.testing.assert_allclose(batch, batch_full.isel(L=slice(1, 3)))


def test_monthly_skill_entry_point_uses_monthly_metric_contract(monkeypatch):
    """Monthly producer calls the shared metric kernel with monthly=True."""
    model = xr.DataArray(
        np.zeros((3, 4, 2)), dims=("Y", "L", "M"),
        coords={"Y": [2000, 2001, 2002], "L": [1, 2, 3, 4], "M": [0, 1]},
    )
    time = xr.DataArray(
        np.ones((3, 4)), dims=("Y", "L"), coords={"Y": model.Y, "L": model.L},
    )
    captured = {}

    def fake_compute(model_arg, time_arg, obs_arg, *args, **kwargs):
        captured.update(model=model_arg, time=time_arg, observations=obs_arg, kwargs=kwargs)
        return xr.Dataset({"corr": model_arg.mean(("Y", "M"))})

    monkeypatch.setattr(workflow.stats, "compute_skill_seasonal", fake_compute)
    result = workflow.compute_monthly_skill_lead_range(
        model, time, "observations", "1981", "2010", lead_start=2, lead_end=3,
    )
    assert captured["kwargs"]["monthly"] is True
    assert captured["kwargs"]["nleadavg"] == 1
    assert captured["kwargs"]["nleads"] == 2
    assert captured["model"].L.values.tolist() == [2, 3]
    assert result.corr.L.values.tolist() == [2, 3]


@pytest.mark.parametrize("start,end", [(0, 2), (3, 2), (1, 5), (1.5, 3), (True, 3)])
def test_invalid_lead_range_fails_before_computation(start, end):
    model, time = _seasonal_data(2)
    with pytest.raises(ValueError, match="lead"):
        workflow.select_lead_range(model, time, start, end)


def test_snapshot_preserves_inventory_without_archive_and_rejects_wrong_request(tmp_path):
    source = tmp_path / "source.nc"
    source.write_bytes(b"first")
    options = dict(identity={"field": "SST", "years": [2000, 2001]}, snapshot_dir=tmp_path / "snapshots")
    revision = workflow.resolve_source_revision("inventory", "v1", paths=[source], **options)
    source.unlink()
    assert workflow.resolve_source_revision("snapshot", "v1", **options) == revision
    for changed in ({**options, "identity": {"field": "PRECT"}}, options):
        token = "v1" if changed != options else "v2"
        with pytest.raises(RuntimeError, match="inventory mode"):
            workflow.resolve_source_revision("snapshot", token, **changed)
    source.write_bytes(b"replacement")
    updated = workflow.resolve_source_revision("inventory", "v1", paths=[source], **options)
    assert updated != revision
    assert workflow.resolve_source_revision("snapshot", "v1", **options) == updated


def test_mask_identity_tracks_values_and_grid():
    mask = xr.DataArray([[True, False]], dims=("lat", "lon"), coords={"lat": [0.], "lon": [0., 1.]})
    identity = workflow.ocean_mask_identity(mask)
    assert workflow.ocean_mask_identity(mask.transpose()) == identity
    assert workflow.ocean_mask_identity(~mask) != identity
    assert workflow.ocean_mask_identity(mask.assign_coords(lon=[1., 2.])) != identity


def _forbid(*args, **kwargs):
    raise AssertionError("Cached restart accessed archives or recomputed skill")


@pytest.fixture
def notebook_run(tmp_path):
    """Execute real workflow cells with small arrays and an identity regridder.

    Archive lookup/loading and spatial remapping are test doubles. Provenance,
    prepared bundles, observation I/O, drift removal, ACC, and resampling are real.
    """
    source_paths = {}
    for name in ("case-a", "case-b", "smyle", "obs"):
        source_paths[name] = tmp_path / f"{name}.nc"
        source_paths[name].write_bytes(name.encode())
    trackers = []

    def run(field="TS", mode="inventory", cases=("case-a", "case-b"), cached=False, has_smyle=True):
        ns = {}
        for module in (leadtime_prepared_cache, leadtime_skill_cache, workflow, unit_conversion):
            ns.update({k: v for k, v in vars(module).items() if not k.startswith("_")})
        # Run the notebook's own import cell so the namespace tracks its imports;
        # archive access and regridding are replaced with test doubles below.
        exec(_cell(IMPORT_CELL), ns)
        tracker = ResourceTracker()
        trackers.append(tracker)
        ns.update(
            np=np, xr=xr, cftime=cftime, os=os, Path=Path, stats=stats,
            env_paths=env_paths,
            workflow_resources=tracker, leadtime_acc_dir=leadtime_acc_dir,
            atomic_to_netcdf=atomic_to_netcdf, load_netcdf=load_netcdf,
            check_remove_drift_sample=check_remove_drift_sample,
        )
        exec(_cell(CONFIG_CELL), ns)
        ns["field"] = field
        ns["cfg"] = dict(ns["VAR_CONFIG"][field], has_smyle_benchmark=has_smyle)
        ns["analysis_component"] = ns["cfg"].get("component", "atm")
        ns["e3sm_field"] = ns["cfg"].get("e3sm_field", field)
        ns["smyle_field"] = ns["cfg"].get("smyle_field", field)
        ns["E3SM_CASES"] = {
            name: {"case_prefix": name, "cache_tag": name, "source_revision": "v1"}
            for name in cases
        }
        ns["E3SM_REFERENCE_CASE"] = "case-a"
        settings = ns["WORKFLOW_SETTINGS"]
        settings["paths"].update(s2d_diag_root=str(tmp_path / field), figure_outdir=str(tmp_path / "figures"))
        settings["run"].update(years=(2000, 2007), init_months=[11], climatology_years=(2000, 2007))
        settings["e3sm"].update(data_dir=str(tmp_path), nens=2, nlead=12)
        settings["smyle"].update(benchmark_dir=str(tmp_path), nens=3, nlead=12)
        settings["obs"].update(data_dir=str(tmp_path), chunks={})
        # Pin the notebook's user-facing recompute toggles off so the test does
        # not depend on how the notebook was last configured for a real run.
        settings["prepared_skill"].update(
            mode="require" if mode == "snapshot" else "auto", force_recompute=False, chunks={}
        )
        settings["cache"]["source_identity_mode"] = mode
        settings["skill"].update(
            force_compute=False, lead_start=2, lead_end=3, model_chunks={}, obs_chunks={}
        )
        settings["finite_ensemble_compare"].update(
            force_recompute=False, iteration_batch_size=2, model_chunks={}, obs_chunks={}
        )
        settings["finite_ensemble_compare"]["modes"]["final"] = dict(init_months=[11], iterations=3, lead_start=2, lead_end=3)
        settings["diagnostics"]["run_drift_check"] = True
        ns["data_access"] = SimpleNamespace(
            build_init_tags=lambda years, month: [f"{y}{month:02d}0100" for y in years],
            nested_file_list_by_init=(
                _forbid if cached else lambda **kw: ([[source_paths[kw["case_prefix"]]]], kw["init_tags"])
            ),
        )
        smyle, smyle_time = _seasonal_data(3)
        ns["smyle_access"] = SimpleNamespace(
            # The planning cell's inventory-identity code calls benchmark_path()
            # unconditionally to build a cache key, independent of whether the
            # field actually has a CESM-SMYLE benchmark; only load_benchmark()
            # (the real data access) is gated by has_smyle_benchmark.
            benchmark_path=_forbid if cached else lambda *a, **kw: source_paths["smyle"],
            load_benchmark=_forbid if (cached or not has_smyle) else lambda **kw: xr.Dataset({ns["smyle_field"]: smyle, "time": smyle_time}),
        )
        ns["obs_access"] = SimpleNamespace(
            # Listing the configured obs_path_pattern is allowed on restart (it
            # only inventories files); loading the data is not.
            resolve_glob_files=lambda pattern: [source_paths["obs"]],
            get_monthly_data_from_pattern=(
                _forbid if cached
                else lambda *a, **kw: _observations().to_dataset(name=ns["cfg"]["obs_var"])
            ),
            mon_to_seas_obs=mon_to_seas_obs,
        )
        grid = xr.Dataset(coords={"lat": smyle.lat, "lon": smyle.lon})
        grid["area"] = xr.ones_like(smyle.isel(Y=0, M=0, L=0, drop=True))
        ns["regrid"] = SimpleNamespace(
            make_latlon_grid=lambda **kw: grid,
            make_regridder=_forbid if cached else lambda *a, **kw: lambda data, **opts: data,
        )
        if cached:
            ns["compute_skill_lead_range"] = _forbid
            ns["compute_skill_lead_range_batch"] = _forbid
        exec(_cell(LAYOUT_CELL), ns)
        exec(_cell(PLAN_CELL), ns)
        # Inject already aggregated archive arrays at the regrid-stage boundary.
        ns["e3sm_seas_by_case_month"] = {}
        for case, months in ns["e3sm_months_to_prepare"].items():
            if months:
                model, time = _seasonal_data(2)
                ns["e3sm_seas_by_case_month"][case] = {11: xr.Dataset({field: model, "time": time})}
        for marker in PIPELINE_CELLS:
            exec(compile(_cell(marker), f"1a_cell[{marker[:30]}]", "exec"), ns)
        return ns

    yield run, source_paths
    for tracker in trackers:
        tracker.close()


@pytest.mark.parametrize("field", ["PRECT", "TS"])
def test_notebook_inventory_to_snapshot_restart(notebook_run, field, capsys):
    run, sources = notebook_run
    first = run(field)
    expected = first["accpval_by_case_month"]["case-a"][11].copy(deep=True)
    paths = first["prepared_specs_by_case_month"]
    first["workflow_resources"].close()
    for path in sources.values():
        path.unlink()
    restarted = run(field, mode="snapshot", cached=True)
    assert restarted["prepared_specs_by_case_month"] == paths
    xr.testing.assert_identical(restarted["accpval_by_case_month"]["case-a"][11], expected)
    np.testing.assert_array_equal(restarted["skill_by_month"][11].L, [6, 9])
    assert "Skipping independent drift check for cached" in capsys.readouterr().out


def test_has_smyle_benchmark_false_skips_smyle_without_archive_access(notebook_run, capsys):
    """A field with has_smyle_benchmark=False must never touch smyle_access.

    This is the contract land fields will rely on: skip every CESM-SMYLE cell
    cleanly, leave its outputs as empty/None fallbacks, and still produce the
    normal E3SM-only skill result.
    """
    run, _ = notebook_run
    ns = run("PRECT", cases=("case-a",), has_smyle=False)

    assert ns["smyle_anom_by_month"] == {}
    assert ns["smyle_time_by_month"] == {}
    assert ns["smyle05_anom"] is None and ns["smyle11_anom"] is None
    assert ns["smyle_skill_by_month"] == {}
    assert ns["smyle_overlap_skill_by_month"] == {}
    assert ns["skill_delta_by_case_month"] == {"case-a": {}}

    # The E3SM-only skill result is still produced normally.
    assert bool(ns["skill_by_month"][11].corr.notnull().any())

    skipped = capsys.readouterr().out
    assert "Skipping the CESM-SMYLE benchmark load" in skipped
    assert "Skipping the CESM-SMYLE benchmark skill computation" in skipped
    assert "Skipping the CESM-SMYLE vs E3SM significance comparison" in skipped


def test_snapshot_missing_observations_fails_without_archive_fallback(notebook_run):
    run, _ = notebook_run
    first = run("PRECT", cases=("case-a",))
    first["workflow_resources"].close()
    first["obs_cache_path"].unlink()
    with pytest.raises(RuntimeError, match="Required observation cache is unavailable"):
        run("PRECT", mode="snapshot", cases=("case-a",), cached=True)


def test_interrupted_resampling_reuses_completed_batch(notebook_run):
    run, _ = notebook_run
    first = run("PRECT", cases=("case-a",))
    expected = first["accpval_by_case_month"]["case-a"][11].copy(deep=True)
    first["workflow_resources"].close()
    layout = first["comparison_cache_layout"]
    smyle, _, fraction = layout.result_paths("case-a", 11, 2, first["compare_years"])
    kept_batch = layout.batch_path(smyle, 0, 2)
    stamp = kept_batch.stat().st_mtime_ns
    smyle.unlink()
    fraction.unlink()
    layout.batch_path(smyle, 2, 3).unlink()
    resumed = run("PRECT", cases=("case-a",))
    assert kept_batch.stat().st_mtime_ns == stamp
    xr.testing.assert_identical(resumed["accpval_by_case_month"]["case-a"][11], expected)


def test_all_notebook_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(_cell(index), filename=f"1a_cell_{index}")
