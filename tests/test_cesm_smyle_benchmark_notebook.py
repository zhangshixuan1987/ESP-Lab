import json
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell


NOTEBOOK = (
    Path(__file__).parents[1]
    / "jupyter"
    / "0_run_cesm_smyle_benchmark.ipynb"
)


def _notebook_source():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_benchmark_notebook_has_a_bounded_restartable_driver():
    source = _notebook_source()

    assert '(candidate / "esp_lab").is_dir()' in source
    assert 'sys.path.insert(0, str(REPO_ROOT))' in source
    assert 'import esp_lab' not in source
    assert "/global/homes/" not in source
    assert "restart_notebook_cluster" in source
    assert "close_notebook_resources(globals())" in source
    assert "use_dask = True" in source
    assert "workers = 4" in source
    assert 'memory_limit = "4GB"' in source
    assert "workers = 32" not in source
    assert "prep.existing_benchmark_issues" in source
    assert "if unsuccessful:" in source
    assert "Benchmark output validation failed" in source


def test_benchmark_notebook_code_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text())
    shell = InteractiveShell.instance()

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        transformed = shell.transform_cell("".join(cell.get("source", [])))
        compile(transformed, f"benchmark notebook cell {index}", "exec")
