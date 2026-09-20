"""Checks for standalone orchestration in the refactored MOV notebook."""

import json
from pathlib import Path

import pytest

from workflows.modes_of_variability import orchestration
from workflows.modes_of_variability import figure_config


_mov_matches = list((Path(__file__).resolve().parents[1] / "jupyter").rglob("4a_mov_analysis.ipynb"))
NOTEBOOK = _mov_matches[0] if _mov_matches else (Path(__file__).resolve().parents[1] / "jupyter" / "4a_mov_analysis.ipynb")


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


def test_notebook_defaults_to_cache_only_analysis_policy():
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
    )
    assert 'mov_input_mode = "require"' in source
    assert "force_recompute_skill = False" in source
    assert "ensure_mode_products(MOV_INPUT_SETTINGS)" in source
    assert "0_run*` notebook is a prerequisite" in source


def test_global_teleconnection_figure_separates_initializations():
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
    )
    assert "build_pattern_cases(*, include_all_initializations=False)" in source
    assert "build_pattern_cases(include_all_initializations=True)" in source
    assert "cases_by_initialization = {" in source
    assert 'f"{month_names[init_month]} initialization: Global {selected_mode} "' in source


def test_skill_figure_separates_initializations_into_panels():
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
    )
    assert "skill_panel_specs = [" in source
    assert "for init_month in init_months" in source
    assert "nskillcols = len(skill_panel_specs)" in source
    assert "for col, (method, panel_init_month) in enumerate(skill_panel_specs)" in source
    assert "for init_month in (panel_init_month,)" in source


def test_north_pacific_eof_maps_use_nonoverlapping_layout():
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
    )

    for mode in ("PNA", "NPO"):
        settings = figure_config.get_eof_pattern_settings(mode)
        assert settings["figsize_width"] == 15.5
        assert settings["panel_title_two_lines"] is False
        assert settings["metric_box_two_lines"] is True
    assert 'eof_plot.get("figsize_min_height", 4.0)' in source
    assert 'title.replace(" (", "\\n(", 1)' in source


def test_regional_eof_contours_transform_before_cartopy_rendering():
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
    )

    assert 'transform_first = eof_plot.get(' in source
    assert '"transform_first", map_projection_name == "north_pacific"' in source
    assert 'transform_first = teleconnection_plot.get("transform_first", False)' in source


def test_notebook_uses_centralized_figure_configuration():
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
    )

    assert "build_figure_setup(selected_mode, include_nmme=include_nmme)" in source
    assert "MODE_FIGURE_PROFILES" in source
    assert "STIPPLE_STYLE" in source
    assert "FIGURE_OVERRIDES" in source
    assert '"stipple_stride"' in source
    assert '"stipple_size"' in source
    assert '"stipple_alpha"' in source
    assert "FIGURE_SETUP[section].update(overrides)" in source
    for legacy_block in (
        "BASE_EOF_PATTERN_SETTINGS",
        "MODE_LAYOUT_CONFIG",
        "MODE_MAP_LABEL_CONFIG",
        "EOF_PATTERN_ONLY_REFINEMENTS",
        "BASE_TELECONNECTION_SETTINGS",
    ):
        assert legacy_block not in source




@pytest.mark.parametrize(
    "ensure_mode, expected_runs, forced",
    [("auto", 0, False), ("require", 0, False), ("rebuild", 3, True)],
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
        orchestration.mode_processor, "expected_product_issues", lambda args, **kwargs: []
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
    assert report["processor_runs"] == expected_runs
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
        if ensure_mode == "rebuild":
            assert [args.e3sm_cache_tag for args in processor_args[1:]] == [
                "case_a", "case_b"
            ]


def test_auto_runs_only_incompatible_processor_call(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestration, "_ensure_smyle_benchmarks", lambda *args: None)
    completed = set()
    processor_args = []

    def call_key(args):
        return getattr(args, "e3sm_cache_tag", "obs-smyle")

    def issues(args, **kwargs):
        key = call_key(args)
        return [f"stale {key}"] if key == "case_a" and key not in completed else []

    def run(args):
        processor_args.append(args)
        completed.add(call_key(args))

    monkeypatch.setattr(orchestration.mode_processor, "expected_product_issues", issues)
    monkeypatch.setattr(orchestration.mode_processor, "run", run)
    monkeypatch.setattr(orchestration.mode_processor, "expected_products", lambda args: {})
    monkeypatch.setattr(orchestration.teleconnections, "ensure_products", lambda *args, **kwargs: {})

    report = orchestration.ensure_mode_products(_settings(tmp_path, ensure_mode="auto"))

    assert [call_key(args) for args in processor_args] == ["case_a"]
    assert report["processor_calls"] == 3
    assert report["processor_runs"] == 1


def test_require_mode_reports_incompatible_products(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestration, "_ensure_smyle_benchmarks", lambda *args: None)
    monkeypatch.setattr(
        orchestration.mode_processor,
        "expected_product_issues",
        lambda args, **kwargs: ["missing expected.nc"] if "smyle" in args.sources else [],
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


def test_mov_validation_allows_smyle_year_superset(tmp_path, monkeypatch):
    settings = _settings(tmp_path, years=(1980, 2011))
    calls = []

    monkeypatch.setattr(
        orchestration.analysis.smyle_access,
        "benchmark_path",
        lambda field, month, root, **kwargs: Path(root) / f"{month:02d}.nc",
    )

    def issues(path, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(
        orchestration.smyle_benchmark, "existing_benchmark_issues", issues
    )

    _, _, found = orchestration._smyle_benchmark_issues(settings)

    assert found == {}
    assert len(calls) == 2
    assert all(call["allow_year_superset"] is True for call in calls)
