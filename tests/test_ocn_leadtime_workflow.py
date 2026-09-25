"""Regression checks for the ocean-realm companion notebook (1a_ocn).

Mirrors the fixture pattern in test_leadtime_workflow.py (which covers
1a_atm), scoped to the ocean-specific behavior that moved out of 1a when SST
was split into its own notebook: the common ocean-mask domain restriction and
its interaction with cache restarts.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import cftime
import numpy as np
import pytest
import xarray as xr

from esp_lab import env_paths, leadtime_prepared_cache, leadtime_realm_skill, leadtime_skill_cache, stats
from esp_lab import leadtime_workflow as workflow
from esp_lab.data_access_obs import mon_to_seas_obs
from esp_lab.leadtime_validation import check_remove_drift_sample
from esp_lab.paths import leadtime_acc_dir
from esp_lab.utils import unit_conversion
from esp_lab.utils.netcdf_utils import atomic_to_netcdf, load_netcdf
from esp_lab.utils.resource_utils import ResourceTracker
from tests.test_leadtime_workflow import _forbid, _observations, _seasonal_data


NOTEBOOK = Path(__file__).parents[1] / "jupyter/1a_ocn_leadtime_acc_skill_map.ipynb"


def _cell(index):
    cell = json.loads(NOTEBOOK.read_text())["cells"][index]
    return "".join(line for line in cell["source"] if not line.startswith("%"))


def _cell_index(marker):
    """Index of the one code cell containing marker, so inserted cells don't shift tests."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    hits = [
        i for i, cell in enumerate(cells)
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    ]
    assert len(hits) == 1, (marker, hits)
    return hits[0]


@pytest.fixture
def notebook_run(tmp_path):
    """Execute real 1a_ocn cells with small arrays and an identity regridder.

    Same construction as test_leadtime_workflow.py's fixture; kept as a
    separate copy since it's bound to a different NOTEBOOK path.
    """
    source_paths = {}
    for name in ("case-a", "case-b", "smyle", "obs"):
        source_paths[name] = tmp_path / f"{name}.nc"
        source_paths[name].write_bytes(name.encode())
    trackers = []

    def run(field="SST", mode="inventory", cases=("case-a", "case-b"), cached=False, has_smyle=True):
        ns = {}
        for module in (leadtime_prepared_cache, leadtime_skill_cache, workflow, unit_conversion):
            ns.update({k: v for k, v in vars(module).items() if not k.startswith("_")})
        tracker = ResourceTracker()
        trackers.append(tracker)
        ns.update(
            np=np, xr=xr, cftime=cftime, os=os, Path=Path, stats=stats,
            env_paths=env_paths, leadtime_realm_skill=leadtime_realm_skill,
            workflow_resources=tracker, leadtime_acc_dir=leadtime_acc_dir,
            atomic_to_netcdf=atomic_to_netcdf, load_netcdf=load_netcdf,
            check_remove_drift_sample=check_remove_drift_sample,
        )
        exec(_cell(7), ns)
        ns["field"] = field
        ns["cfg"] = dict(
            ns["VAR_CONFIG"][field], has_smyle_benchmark=has_smyle,
            obs_path_pattern=str(source_paths["obs"]),
        )
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
        settings["prepared_skill"].update(mode="require" if mode == "snapshot" else "auto", chunks={})
        settings["cache"]["source_identity_mode"] = mode
        settings["skill"].update(lead_start=2, lead_end=3, model_chunks={}, obs_chunks={})
        settings["finite_ensemble_compare"].update(iteration_batch_size=2, model_chunks={}, obs_chunks={})
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
            benchmark_path=_forbid if cached else lambda *a, **kw: source_paths["smyle"],
            load_benchmark=_forbid if (cached or not has_smyle) else lambda **kw: xr.Dataset({ns["smyle_field"]: smyle, "time": smyle_time}),
        )
        ns["obs_access"] = SimpleNamespace(
            resolve_glob_files=_forbid if cached else lambda pattern: [Path(pattern)],
            get_monthly_data_from_pattern=_forbid if cached else lambda *a, **kw: _observations().to_dataset(name=ns["cfg"]["obs_var"]),
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
        exec(_cell(9), ns)
        # Inject already aggregated archive arrays at the boundary of cell 16.
        ns["e3sm_seas_by_case_month"] = {}
        for case, months in ns["e3sm_months_to_prepare"].items():
            if months:
                model, time = _seasonal_data(2, missing=(field == "SST" and case == "case-b"))
                ns["e3sm_seas_by_case_month"][case] = {11: xr.Dataset({field: model, "time": time})}
        smyle_significance = _cell_index("CESM-SMYLE vs each E3SM hindcast ACC significance figure")
        for index in (15, 17, 20, 23, 24, 25, 26, 27, 28, smyle_significance):
            exec(compile(_cell(index), f"1a_ocn_cell_{index}", "exec"), ns)
        return ns

    yield run, source_paths
    for tracker in trackers:
        tracker.close()


def test_ocn_notebook_inventory_to_snapshot_restart(notebook_run, capsys):
    run, sources = notebook_run
    first = run("SST")
    expected = first["accpval_by_case_month"]["case-a"][11].copy(deep=True)
    paths = first["prepared_specs_by_case_month"]
    first["workflow_resources"].close()
    for path in sources.values():
        path.unlink()
    restarted = run("SST", mode="snapshot", cached=True)
    assert restarted["prepared_specs_by_case_month"] == paths
    xr.testing.assert_identical(restarted["accpval_by_case_month"]["case-a"][11], expected)
    np.testing.assert_array_equal(restarted["skill_by_month"][11].L, [6, 9])
    assert "Skipping independent drift check for cached" in capsys.readouterr().out


def test_sst_case_removal_recovers_domain_without_rebuilding_prepared_inputs(notebook_run):
    run, _ = notebook_run
    first = run()
    mask_identity = first["analysis_domain_identity"]
    prepared_path = first["prepared_specs_by_case_month"]["case-a"][11][0]
    with xr.open_dataset(prepared_path) as ds:
        assert bool(ds.anomaly.isel(lon=0).notnull().all())
    stamp = prepared_path.stat().st_mtime_ns
    first["workflow_resources"].close()
    reduced = run(cases=("case-a",))
    assert reduced["e3sm_months_to_prepare"] == {"case-a": []}
    assert prepared_path.stat().st_mtime_ns == stamp
    assert reduced["analysis_domain_identity"] != mask_identity
    assert bool(reduced["ocean_mask"].all())
    assert bool(reduced["skill_by_month"][11].corr.isel(lon=0).notnull().all())
