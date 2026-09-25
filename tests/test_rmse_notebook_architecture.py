import json
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell


NOTEBOOKS = (
    Path("jupyter/1b_atm_leadtime_rmse_skill_map.ipynb"),
    Path("jupyter/1b_ocn_leadtime_rmse_skill_map.ipynb"),
    Path("jupyter/1c_atm_leadtime_rmse_compare.ipynb"),
    Path("jupyter/1c_ocn_leadtime_rmse_compare.ipynb"),
)

LAND_NOTEBOOKS = (
    Path("jupyter/1b_lnd_leadtime_rmse_skill_map.ipynb"),
    Path("jupyter/1c_lnd_leadtime_rmse_compare.ipynb"),
)

# (skill-map notebook, compare notebook) pairs, one per realm.
REALM_PAIRS = (
    (Path("jupyter/1b_atm_leadtime_rmse_skill_map.ipynb"), Path("jupyter/1c_atm_leadtime_rmse_compare.ipynb")),
    (Path("jupyter/1b_ocn_leadtime_rmse_skill_map.ipynb"), Path("jupyter/1c_ocn_leadtime_rmse_compare.ipynb")),
)


def _source(path):
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_rmse_notebooks_follow_finalized_leadtime_architecture():
    for path in NOTEBOOKS:
        source = _source(path)
        for required in (
            "WORKFLOW_SETTINGS",
            "source_identity_mode",
            "file_inventory_digest",
            "source_fingerprint",
            "stable_resampling_seed",
            "generate_member_indices",
            "load_or_write_member_selection",
            "cache_status",
            "atomic_to_netcdf",
            "ResourceTracker",
            "restart_notebook_cluster",
            "close_notebook_resources(globals())",
            "figure_filename",
        ):
            assert required in source, f"{path} is missing {required}"
        assert "np.random.default_rng" not in source
        assert "except:\n" not in source
        assert "dask_workers = 4" in source
        assert 'memory_limit="4GB"' in source
        assert "dask_workers = 30" not in source
        assert "e3sm_seas = e3sm_seas.persist()" not in source
        assert "resolve_source_revision" in source
        assert "'snapshot'" in source
        assert "mode='require'" in source


def test_rmse_notebooks_delegate_metric_specific_logic():
    for normalized_path, direct_path in REALM_PAIRS:
        normalized = _source(normalized_path)
        direct = _source(direct_path)

        assert "rmse_utils.absolute_rmse_from_skill" in normalized
        assert "rmse_utils.valid_area_weighted_fraction" in normalized
        assert "rmse_compare_helper.direct_rmse_cache_attrs" in direct
        assert "rmse_compare_helper.direct_rmse_cache_path" in direct
        assert "valid_area_weighted_fraction" in direct


def test_normalized_rmse_uses_one_common_cohort_and_metric_definition():
    for normalized_path, _ in REALM_PAIRS:
        normalized = _source(normalized_path)

        assert "common_skill_years_by_month" in normalized
        assert "compact_skill_year_tag" in normalized
        assert "verification_years = common_skill_years_by_month[init_month]" in normalized
        assert "compare_detrend = WORKFLOW_SETTINGS['skill']['detrend']" in normalized
        assert (
            "compare_iteration_batch_size = "
            "WORKFLOW_SETTINGS['significance']['iteration_batch_size']"
        ) in normalized
        assert "use_prepared_skill_inputs" in normalized
        assert "prepared_rmse_observations_v1" in normalized


def test_direct_rmse_has_archive_free_raw_value_inputs():
    for _, direct_path in REALM_PAIRS:
        direct = _source(direct_path)

        assert "use_prepared_direct_inputs" in direct
        assert "prepared_direct_rmse_model_v1" in direct
        assert "prepared_direct_rmse_observations_v1" in direct
        assert "required_variables=('model', 'verification_time')" in direct
        assert "WORKFLOW_SETTINGS['paths']['e3sm_data_dir']" in direct


def _var_config_block(source):
    """Extract the VAR_CONFIG dict literal's own text, ignoring unrelated
    per-plot colorbar fallback dicts elsewhere in the notebook that legitimately
    keep listing other fields as explicit overrides while gracefully defaulting
    for fields they don't know about (e.g. CONUS_RMSE_COLORBAR_CONFIG.get(field, ...))."""
    start = source.index("VAR_CONFIG = {")
    end = source.index("\nif field not in VAR_CONFIG:", start)
    return source[start:end]


def test_ocn_rmse_notebooks_are_ocean_scoped_and_atm_notebooks_are_not():
    for path in (
        Path("jupyter/1b_ocn_leadtime_rmse_skill_map.ipynb"),
        Path("jupyter/1c_ocn_leadtime_rmse_compare.ipynb"),
    ):
        config = _var_config_block(_source(path))
        assert '"SST"' in config
        assert '"TREFHT"' not in config
        assert '"PRECT"' not in config
        assert '"PSL"' not in config

    for path in (
        Path("jupyter/1b_atm_leadtime_rmse_skill_map.ipynb"),
        Path("jupyter/1c_atm_leadtime_rmse_compare.ipynb"),
    ):
        config = _var_config_block(_source(path))
        assert '"SST"' not in config
        assert '"TREFHT"' in config


def test_rmse_notebook_code_cells_compile():
    shell = InteractiveShell.instance()
    for path in NOTEBOOKS + LAND_NOTEBOOKS:
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            transformed = shell.input_transformer_manager.transform_cell(source)
            compile(transformed, f"{path}:cell-{index}", "exec")


def test_land_direct_rmse_scores_raw_values_and_skips_anomaly_reference(tmp_path, capsys):
    import numpy as np
    import pandas as pd
    import pytest
    import xarray as xr

    from esp_lab.paths import leadtime_acc_dir

    notebook = json.loads(LAND_NOTEBOOKS[1].read_text())
    source = next(
        ''.join(cell['source']) for cell in notebook['cells']
        if cell['cell_type'] == 'code'
        and 'Finished direct-RMSE calculation.' in ''.join(cell['source'])
    )
    source = source.removeprefix('%%time\n')
    times = pd.to_datetime(['2000-07-01', '2001-07-01'])
    # Member means are 3 and 7; observations are 1 and 3: RMSE = sqrt(10).
    model = xr.DataArray(
        np.array([[2., 4.], [6., 8.]]).reshape(2, 2, 1, 1, 1),
        dims=('Y', 'M', 'L', 'lat', 'lon'),
        coords={'Y': [2000, 2001], 'M': [0, 1], 'L': [3], 'lat': [0.], 'lon': [0.]},
    )
    observed = xr.DataArray(
        np.array([1., 3.]).reshape(2, 1, 1), dims=('time', 'lat', 'lon'),
        coords={'time': times, 'lat': [0.], 'lon': [0.]},
    )
    model_time = xr.DataArray(
        times.values[:, None], dims=('Y', 'L'), coords={'Y': [2000, 2001], 'L': [3]},
    )
    written = []
    ns = {
        'DIRECT_RMSE_UNDEFINED': True, 'SKIP_MESSAGE': 'direct RMSE is not defined',
        'GRID_TAG': '5x5', 'E3SM_CASES': {'case': {'cache_tag': 'case', 'case_prefix': 'case'}},
        'RUN': {'init_months': [5], 'ensemble_members': [0, 1], 'force_compute': False},
        'forecast_by_case_month': {'case': {5: model}},
        'valid_time_by_case_month': {'case': {5: model_time}},
        'model_data_identity_by_case_month': {'case': {5: 'model-v1'}},
        'reference': observed, 'REFERENCE_PRODUCT': 'absolute-reference',
        'reference_cfg': {'variable': 'swe', 'output_units': 'mm'},
        'reference_data_identity': 'obs-v1', 'REGRID': {'method': 'conservative'},
        'field': 'H2OSNO', 'S2D_DIAG_ROOT': tmp_path,
        'leadtime_acc_dir': leadtime_acc_dir,
        'cache_status': lambda *args, **kwargs: (False, 'missing'),
        'atomic_to_netcdf': lambda ds, path, **kwargs: written.append(ds),
        'netcdf_write_options': {},
    }
    # Anomaly-only reference: the cell skips cleanly instead of failing.
    exec(compile(source, '<land-direct-rmse-cell>', 'exec'), ns)
    assert 'direct RMSE is not defined' in capsys.readouterr().out
    assert not written and 'direct_rmse_by_case_month' not in ns
    ns['DIRECT_RMSE_UNDEFINED'] = False
    exec(compile(source, '<land-direct-rmse-cell>', 'exec'), ns)
    result = ns['direct_rmse_by_case_month']['case'][5]
    assert result.rmse.item() == pytest.approx(np.sqrt(10))
    assert result.bias.item() == pytest.approx(3)
    assert result.n_years.item() == 2
    assert result.L.values.tolist() == [3]
    assert len(written) == 1
