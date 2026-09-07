import json
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell


NOTEBOOKS = (
    Path("jupyter/2a_refactor_leadtime_rmse_skill_map.ipynb"),
    Path("jupyter/2b_refactor_leadtime_rmse_compare.ipynb"),
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
    normalized = _source(NOTEBOOKS[0])
    direct = _source(NOTEBOOKS[1])

    assert "rmse_utils.absolute_rmse_from_skill" in normalized
    assert "rmse_utils.valid_area_weighted_fraction" in normalized
    assert "rmse_compare_helper.direct_rmse_cache_attrs" in direct
    assert "rmse_compare_helper.direct_rmse_cache_path" in direct
    assert "valid_area_weighted_fraction" in direct


def test_normalized_rmse_uses_one_common_cohort_and_metric_definition():
    normalized = _source(NOTEBOOKS[0])

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
    direct = _source(NOTEBOOKS[1])

    assert "use_prepared_direct_inputs" in direct
    assert "prepared_direct_rmse_model_v1" in direct
    assert "prepared_direct_rmse_observations_v1" in direct
    assert "required_variables=('model', 'verification_time')" in direct
    assert "WORKFLOW_SETTINGS['paths']['e3sm_data_dir']" in direct


def test_rmse_notebook_code_cells_compile():
    shell = InteractiveShell.instance()
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            transformed = shell.input_transformer_manager.transform_cell(source)
            compile(transformed, f"{path}:cell-{index}", "exec")
