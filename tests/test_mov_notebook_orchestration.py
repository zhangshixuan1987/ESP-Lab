"""Checks for standalone orchestration in the refactored MOV notebook."""

import json
from pathlib import Path

import pytest

from workflows.modes_of_variability import orchestration


NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "jupyter" / "4_refactor_mov_analysis.ipynb"
)


def _settings(tmp_path, **overrides):
    values = {
        "ensure_mode": "auto",
        "mode": "NAM",
        "outdir": tmp_path / "diagnostics",
        "years": (1980, 2018),
        "climatology_years": (1981, 2010),
        "observation_years": (1979, 2019),
        "eof_reference_years": (1979, 2019),
        "init_months": [5, 11],
        "monthly_nlead": 24,
        "target_dlat": 2.5,
        "target_dlon": 2.5,
        "workers": 4,
        "smyle_benchmark_dir": tmp_path / "smyle-benchmark",
        "smyle_data_dir": tmp_path / "smyle-raw",
        "smyle_nens": 20,
        "obs_dir": tmp_path / "obs",
        "eof_bootstrap_iterations": 0,
        "model_sst_land_mask": True,
        "e3sm_cases": {
            "a": {
                "data_dir": tmp_path / "e3sm",
                "case_prefix": "case-a",
                "cache_tag": "case_a",
                "display_name": "Case A",
            },
            "b": {
                "data_dir": tmp_path / "e3sm",
                "case_prefix": "case-b",
                "cache_tag": "case_b",
                "display_name": "Case B",
            },
        },
        "include_nmme": False,
    }
    values.update(overrides)
    return values


def test_notebook_declares_standalone_input_policy():
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
    )
    assert 'mov_input_mode = "auto"' in source
    assert "ensure_mode_products(MOV_INPUT_SETTINGS)" in source
    assert "0_run*` notebook is a prerequisite" in source


def test_legacy_mov_drivers_are_grouped_as_optional_preprocessing():
    root = NOTEBOOK.parents[1]
    preprocessing = NOTEBOOK.parent / "preprocessing" / "mov"
    assert not (NOTEBOOK.parent / "0_run_sigmod_emov.ipynb").exists()
    assert not (NOTEBOOK.parent / "0_run_nmme_emov.ipynb").exists()
    assert (preprocessing / "0_run_sigmod_emov.ipynb").is_file()
    assert (preprocessing / "0_run_nmme_emov.ipynb").is_file()
    assert (preprocessing / "README.md").is_file()
    assert (root / "scripts" / "run_process_modes_of_variability.py").is_file()


@pytest.mark.parametrize(
    "ensure_mode, expected_runs, forced",
    [("auto", 3, False), ("require", 0, False), ("rebuild", 3, True)],
)
def test_core_sources_are_ensured_per_policy(
    tmp_path, monkeypatch, ensure_mode, expected_runs, forced
):
    benchmark_calls = []
    processor_args = []
    monkeypatch.setattr(
        orchestration,
        "_ensure_smyle_benchmarks",
        lambda settings, mode: benchmark_calls.append(mode),
    )
    monkeypatch.setattr(
        orchestration.mode_processor, "run", lambda args: processor_args.append(args)
    )
    monkeypatch.setattr(
        orchestration.mode_processor, "expected_product_issues", lambda args: []
    )
    monkeypatch.setattr(
        orchestration.mode_processor, "expected_products", lambda args: {}
    )
    teleconnection_calls = []
    monkeypatch.setattr(
        orchestration.teleconnections,
        "ensure_products",
        lambda products, **kwargs: teleconnection_calls.append(kwargs) or {},
    )

    report = orchestration.ensure_mode_products(
        _settings(tmp_path, ensure_mode=ensure_mode)
    )

    assert benchmark_calls == [ensure_mode]
    assert len(processor_args) == expected_runs
    assert report["processor_calls"] == 3
    assert teleconnection_calls == [{
        "ensure_mode": ensure_mode,
        "alpha": 0.05,
        "standardize_index": True,
        "fdr": True,
    }]
    for args in processor_args:
        assert args.modes == ["NAM"]
        assert args.force is forced
    if processor_args:
        assert processor_args[0].sources == ["obs", "smyle"]
        assert [args.e3sm_cache_tag for args in processor_args[1:]] == [
            "case_a", "case_b"
        ]


def test_require_mode_reports_incompatible_products(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestration, "_ensure_smyle_benchmarks", lambda *args: None)
    monkeypatch.setattr(
        orchestration.mode_processor,
        "expected_product_issues",
        lambda args: ["missing expected.nc"] if "smyle" in args.sources else [],
    )
    with pytest.raises(RuntimeError, match="missing expected.nc"):
        orchestration.ensure_mode_products(
            _settings(tmp_path, ensure_mode="require")
        )


def test_auto_builds_only_stale_smyle_month(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    processed = []

    def benchmark_path(field, month, root, **kwargs):
        return Path(root) / f"{month:02d}_{field}_{kwargs['freq']}.nc"

    def issues(path, **kwargs):
        return ["stale"] if kwargs["init_month"] == 11 else []

    monkeypatch.setattr(
        orchestration.analysis.smyle_access, "benchmark_path", benchmark_path
    )
    monkeypatch.setattr(
        orchestration.smyle_benchmark, "existing_benchmark_issues", issues
    )
    monkeypatch.setattr(
        orchestration.smyle_benchmark,
        "process_one",
        lambda **kwargs: processed.append(kwargs["init_month"]) or "ok",
    )
    calls = {11: 0}

    def issues_after_build(path, **kwargs):
        month = kwargs["init_month"]
        if month == 11:
            calls[11] += 1
            return ["stale"] if calls[11] == 1 else []
        return []

    monkeypatch.setattr(
        orchestration.smyle_benchmark,
        "existing_benchmark_issues",
        issues_after_build,
    )

    orchestration._ensure_smyle_benchmarks(settings, "auto")
    assert processed == [11]
