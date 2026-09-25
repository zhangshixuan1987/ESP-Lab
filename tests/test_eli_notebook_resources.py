import json
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]


def _source(name):
    notebook = json.loads((REPO_ROOT / "jupyter" / "s2d_skill" / name).read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_eli_notebooks_use_explicit_bounded_dask_settings():
    for name in (
        "5a_eli_skill_ts.ipynb",
        "5b_eli_diagnostics.ipynb",
        "5c_eli_telecon.ipynb",
    ):
        source = _source(name)
        assert '"workers": 4' in source
        assert '"cores": 1' in source
        assert '"memory_limit": "4GB"' in source
        assert 'os.environ.get("CLUSTER_TYPE", "local")' in source
        assert '"workers": 12' not in source
        assert "restart_notebook_cluster(" in source
        assert "close_notebook_resources(globals())" in source


def test_teleconnection_notebook_skips_cluster_for_exact_cache_hit():
    source = _source("5c_eli_telecon.ipynb")

    assert "telecon.teleconnection_cache_path(CONFIG, inventory)" in source
    assert 'CONFIG["cache"]["mode"] != "require"' in source
    assert 'CONFIG["cache"]["mode"] == "rebuild"' in source
    assert "not expected_cache.is_file()" in source
    assert "if needs_distributed_compute else (None, None)" in source
    assert "metrics_ds.load()" in source
    assert source.index("metrics_ds.load()") < source.index(
        'print("Closed teleconnection Dask resources after cache preparation.")'
    )

