from pathlib import Path

import pytest

from esp_lab.paths import (
    diagnostic_dir,
    figure_output_dir,
    join_below,
    leadtime_acc_dir,
    normalize_root,
)


def test_normalize_root_rejects_missing_and_blank_values():
    with pytest.raises(ValueError, match="explicitly configured"):
        normalize_root(None)

    with pytest.raises(ValueError, match="must not be blank"):
        normalize_root("   ")


def test_normalize_root_expands_user_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    assert normalize_root("~/diagnostics") == tmp_path / "diagnostics"


def test_join_below_rejects_path_escape(tmp_path):
    unsafe_components = [
        "/tmp/outside",
        "../outside",
        "safe/../../outside",
        Path("/tmp/outside"),
    ]
    for component in unsafe_components:
        with pytest.raises(ValueError, match="Unsafe path component"):
            join_below(tmp_path, component)


def test_join_below_rejects_blank_component(tmp_path):
    with pytest.raises(ValueError, match="must not be blank"):
        join_below(tmp_path, "")


def test_path_builders_use_explicit_roots(tmp_path):
    assert leadtime_acc_dir(
        "JRA55_FOSIRL", "skill", "atm", "TREFHT", root=tmp_path
    ) == tmp_path / "JRA55_FOSIRL" / "leadtime_acc" / "skill" / "atm" / "TREFHT"
    assert diagnostic_dir(
        "JRA55_FOSIRL", "leadtime_drift", "atm", root=tmp_path
    ) == tmp_path / "JRA55_FOSIRL" / "leadtime_drift" / "atm"
    assert figure_output_dir(
        "leadtime_drift", "atm", root=tmp_path
    ) == tmp_path / "leadtime_drift" / "atm"


def test_path_builders_require_an_explicit_root():
    with pytest.raises(TypeError, match="required keyword-only argument: 'root'"):
        leadtime_acc_dir("JRA55_FOSIRL", "skill")
