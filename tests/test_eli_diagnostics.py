import json
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from dask.base import is_dask_collection

from esp_lab import eli_diagnostics
from esp_lab.utils.resource_utils import ResourceTracker


_eli_matches = list((Path(__file__).parents[1] / "jupyter").rglob("5a_eli_skill_ts.ipynb"))
ELI_NOTEBOOK = _eli_matches[0] if _eli_matches else Path(__file__).parents[1] / "jupyter" / "5a_eli_skill_ts.ipynb"
_eli_diag_matches = list((Path(__file__).parents[1] / "jupyter").rglob("5b_eli_diagnostics.ipynb"))
ELI_DIAGNOSTICS_NOTEBOOK = (
    _eli_diag_matches[0] if _eli_diag_matches else Path(__file__).parents[1] / "jupyter" / "5b_eli_diagnostics.ipynb"
)


def test_eli_notebook_resolves_index_in_model_filename_templates():
    notebook = json.loads(ELI_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert 'f"E3SMLE{{init_month:02d}}_{INDEX}' in source
    assert 'f"BSMYLE{{init_month:02d}}_{INDEX}' in source
    assert '"E3SMLE{init_month:02d}_{INDEX}' not in source
    assert '"BSMYLE{init_month:02d}_{INDEX}' not in source
    assert '"cache_mode": "auto"' in source
    assert "run_process_native_eli.py" in source
    assert "REPO_ROOT = Path(eli_tools.__file__).resolve().parents[1]" in source
    assert "str(NATIVE_ELI_SCRIPT)" in source
    assert "subprocess.run(command, check=True)" in source


def test_eli_notebook_uses_explicit_compact_figure_layouts():
    notebook = json.loads(ELI_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count('"fig_width": 16.0') == 2
    assert '"fig_height": 12.5' in source
    assert '"fig_height": 8.5' in source
    assert 'fig = plt.figure(figsize=(fig_width, fig_height))' in source
    assert 'row_height_scale' not in source
    assert source.count('fig.suptitle(') == 2
    assert 'AnchoredOffsetbox(' in source
    assert 'child=HPacker(children=metric_columns' in source
    assert 'annotation_start_y' not in source
    assert 'seasonal_skill, seasonal_common_target_years' in source
    assert 'retain_complete_seasonal_leads(' in source
    assert 'scores.L if month is None else scores.L - plot_cfg["seasonal_lead_offset"]' in source
    assert 'season_labels_by_month' in source
    assert 'Seasonal skill by initialization and overall monthly skill' in source
    assert '"plot_months": [11, 5]' in source
    assert '"target_leads": {11: [3, 15], 5: [9, 21]}' in source
    assert 'reference_times = eli_seas_time[reference_model][month].sel(L=lead)' in source
    assert 'values = eli_seas_dd[model][month].sel(L=lead)' in source
    assert 'score = seasonal_skill[model][month].sel(L=lead)' in source
    assert 'nmme_eli_seas_timeseries[month]["values"].sel(L=lead)' in source
    assert 'label="HadISST2"' in source
    assert 'NMME_STYLE = {"label": "NMME"' in source
    assert '"title": f"{INDEX} Seasonal Skill Scores ({START_YEAR}-{END_YEAR})"' in source
    assert '"title": f"{INDEX} DJF Anomaly Time Series"' in source
    assert '"legend_ncol": 4' in source
    assert 'def _reorder_legend_row_major(' in source
    assert '"legend_facecolor": "#fcfcfc"' in source
    assert '"legend_edgecolor": "#d0d0d0"' in source
    assert '"annotation_loc": "lower left"' in source
    assert '"annotation_bbox": (0.015, 0.035)' in source
    assert 'loc=plot_cfg["annotation_loc"]' in source


def test_eli_diagnostics_reorganized_nmme_helper_is_complete_and_explicit():
    notebook = json.loads(ELI_DIAGNOSTICS_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "def _compute_all_available_skill(init_month, config, obs_lookup, rng):" in source
    assert "return xr.Dataset(" in source
    assert "_compute_all_available_skill(month, plot_cfg, obs_lookup, rng)" in source
    assert 'if not config["detrend"]' in source
    assert 'ELI_CACHE_MODE = "auto"' in source
    assert '"--year-end", str(END_YEAR)' in source
    assert 'subprocess.run(command, check=True)' in source
    assert '"E3SM-4DEnVarOcn": "4DEnVarOcn"' in source
    assert 'NMME_ELI_BENCHMARK_AVAILABLE = not missing_nmme_benchmark_files' in source
    assert 'raise FileNotFoundError(\n        "Missing NMME benchmark inputs:' not in source


def test_eli_diagnostics_defines_figure_filename_before_first_plot():
    notebook = json.loads(ELI_DIAGNOSTICS_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    definition = 'def figure_filename(*parts, ext="png"):'
    first_use = 'figpath = FIGURE_OUTDIR / figure_filename('
    assert source.count(definition) == 1
    assert source.index(definition) < source.index(first_use)


def test_eli_nino34_seasonal_observations_avoid_short_dask_chunks():
    notebook = json.loads(ELI_DIAGNOSTICS_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert (
        'nino34_raw = open_tracked_dataset(NINO34_OBS_FILE, chunks=None)["sst"].load()'
        in source
    )
    assert '.rolling(time=window, center=True, min_periods=window)' in source


def test_eli_nino34_figures_do_not_draw_top_explanatory_note():
    notebook = json.loads(ELI_DIAGNOSTICS_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert '"anomaly_note"' not in source
    assert '"anomaly_note_y"' not in source
    assert '"anomaly_note_font_scale"' not in source
    assert '"figure_title_y": 0.985' in source


def test_eli_nino34_figures_use_one_bottom_figure_legend():
    notebook = json.loads(ELI_DIAGNOSTICS_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert '"legend_loc": "lower center"' in source
    assert '"legend_bbox": (0.5, 0.015)' in source
    assert '"legend_ncol": 3' in source
    assert 'legend_handles = [\n        line_eli,\n        line_nino34,' in source
    assert 'fig.legend(\n        handles=legend_handles,' in source
    assert 'ax_eli.legend(' not in source
    assert '"layout_rect": [0.035, 0.065, 0.965, 0.94]' in source


def test_drift_reference_lines_distinguish_eli_and_nino34_sources():
    notebook = json.loads(ELI_DIAGNOSTICS_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert '"eli_obs_color": "tab:green"' in source
    assert '"eli_obs_linestyle": "--"' in source
    assert '"nino34_obs_color": "tab:red"' in source
    assert '"nino34_obs_linestyle": ":"' in source
    assert 'color=diagnostic["obs_color"]' in source
    assert 'linestyle=diagnostic["obs_linestyle"]' in source
    assert 'color=plot_cfg["obs_color"]' not in source


def test_drift_figure_uses_two_row_bottom_legend_without_header_rules():
    notebook = json.loads(ELI_DIAGNOSTICS_NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert '"subplot_adjust": {"top": 0.85, "bottom": 0.19' in source
    assert '"legend_bbox": (0.50, 0.012)' in source
    assert '"legend_ncol": 4' in source
    assert 'legend_nrow = (len(legend_elements) + legend_ncol - 1) // legend_ncol' in source
    assert 'bbox_to_anchor=plot_cfg["legend_bbox"]' in source
    assert 'col_line = plt.Line2D(' not in source
    assert 'fig.add_artist(col_line)' not in source


def _hindcast(years, missing=None):
    leads = [1, 2]
    values = np.arange(len(years) * 2 * 2, dtype=float).reshape(len(years), 2, 2)
    data = xr.DataArray(
        values,
        dims=("Y", "L", "M"),
        coords={"Y": years, "L": leads, "M": [0, 1]},
    )
    if missing is not None:
        data.loc[missing] = np.nan
    time = xr.DataArray(
        [[cftime.DatetimeNoLeap(year, lead, 15) for lead in leads] for year in years],
        dims=("Y", "L"),
        coords={"Y": years, "L": leads},
    )
    return data, time


def test_initialization_years_supports_numeric_and_tagged_coordinates():
    np.testing.assert_array_equal(eli_diagnostics.initialization_years([1980, 1981]), [1980, 1981])
    np.testing.assert_array_equal(
        eli_diagnostics.initialization_years(["1980050100", "1981050100"]), [1980, 1981]
    )


def test_observation_alignment_supports_numpy_and_cftime_dates():
    observation = xr.DataArray(
        [1.0, 2.0],
        dims="time",
        coords={"time": pd.to_datetime(["2000-01-01", "2000-02-01"])},
    )

    result = eli_diagnostics.observations_for_times(
        observation,
        [cftime.DatetimeNoLeap(2000, 2, 15), cftime.DatetimeNoLeap(2000, 1, 15)],
    )

    np.testing.assert_array_equal(result, [2.0, 1.0])


def test_common_target_years_are_intersected_per_lead():
    first, first_time = _hindcast([2000, 2001, 2002, 2003])
    second, second_time = _hindcast(
        [2000, 2001, 2002, 2003], missing={"Y": 2001, "L": 1}
    )
    observation = xr.DataArray(
        np.arange(8, dtype=float),
        dims="time",
        coords={
            "time": [
                cftime.DatetimeNoLeap(year, month, 15)
                for year in [2000, 2001, 2002, 2003]
                for month in [1, 2]
            ]
        },
    )

    cohorts = eli_diagnostics.common_target_years_by_lead(
        {"first": {5: first}, "second": {5: second}},
        {"first": {5: first_time}, "second": {5: second_time}},
        observation,
        5,
    )

    assert cohorts == {1: [2000, 2002, 2003], 2: [2000, 2001, 2002, 2003]}


def test_model_hindcasts_can_remain_dask_backed(tmp_path):
    values, valid_time = _hindcast([2000, 2001, 2002, 2003])
    path = tmp_path / "model_05_2_2.nc"
    xr.Dataset({"eli": values, "time": valid_time}).to_netcdf(path)
    spec = eli_diagnostics.ELIModelSpec(
        key="model",
        label="Model",
        root=tmp_path,
        filename_template="model_{init_month:02d}_{nens}_{nlead}.nc",
        ensemble_size=2,
        color="black",
        marker="o",
    )
    tracker = ResourceTracker()

    data, times, _ = eli_diagnostics.load_model_hindcasts(
        {"model": spec},
        [5],
        nlead=2,
        start_year=2000,
        end_year=2003,
        chunks={"Y": -1, "L": 1, "M": -1},
        resource_tracker=tracker,
    )

    assert is_dask_collection(data["model"][5].data)
    assert not is_dask_collection(times["model"][5].data)
    np.testing.assert_array_equal(data["model"][5].compute(), values)
    tracker.close()
