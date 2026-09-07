"""Structural checks for the optional SST preprocessing notebook drivers."""

import json
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell


NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1] / "jupyter" / "preprocessing" / "sst"
)
NOTEBOOKS = (
    "0_run_sigmod_sst_index.ipynb",
    "0_run_nmme_sst_index.ipynb",
)


def test_sst_preprocessing_drivers_are_grouped_and_documented():
    assert (NOTEBOOK_DIR / "README.md").is_file()
    assert {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")} == set(NOTEBOOKS)
    assert (
        NOTEBOOK_DIR.parents[1] / "0_run_cesm_smyle_benchmark.ipynb"
    ).is_file()


def test_sst_preprocessing_drivers_resolve_nested_checkout_and_compile():
    shell = InteractiveShell.instance()
    for name in NOTEBOOKS:
        notebook = json.loads((NOTEBOOK_DIR / name).read_text())
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "_working_directory.parents" in source
        assert '(candidate / "esp_lab").is_dir()' in source
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            transformed = shell.transform_cell("".join(cell.get("source", [])))
            compile(transformed, f"{name} cell {index}", "exec")
