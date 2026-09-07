from pathlib import Path
from subprocess import CompletedProcess

from scripts import run_process_tc_track_e3sm as tc


def test_warm_core_expr_for_pair_parset():
    assert tc.warm_core_expr(tc.PARSETS["set2"]) == "_AVG(T200,T500)"


def test_warm_core_expr_for_single_variable_parset():
    assert tc.warm_core_expr(tc.PARSETS["set5"]) == "T400"


def test_phis_static_rejects_scalar_time_metadata(tmp_path, monkeypatch):
    path = tmp_path / "PHIS_static.nc"
    path.touch()
    metadata = """
  dimensions:
    ncol = 21600 ;
    nbnd = 2 ;
  variables:
    float PHIS(ncol) ;
    double time ;
    double time_bnds(nbnd) ;
"""
    monkeypatch.setattr(
        tc,
        "_run_capture",
        lambda command, cwd: CompletedProcess(command, 0, metadata, ""),
    )

    assert not tc.phis_static_is_valid(path, "ncks", "PHIS")


def test_phis_static_accepts_timeless_phis(tmp_path, monkeypatch):
    path = tmp_path / "PHIS_static.nc"
    path.touch()
    metadata = """
  dimensions:
    ncol = 21600 ;
  variables:
    float PHIS(ncol) ;
"""
    monkeypatch.setattr(
        tc,
        "_run_capture",
        lambda command, cwd: CompletedProcess(command, 0, metadata, ""),
    )

    assert tc.phis_static_is_valid(path, "ncks", "PHIS")


def test_build_phis_static_removes_time_metadata(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    hist_dir = case_dir / "EN00" / "archive" / "atm" / "hist"
    hist_dir.mkdir(parents=True)
    (hist_dir / "case.EN00.eam.h0.1980-05.nc").touch()
    commands = []

    monkeypatch.setattr(tc, "_run", lambda command, dry_run, cwd: commands.append(command))
    monkeypatch.setattr(tc, "phis_static_is_valid", lambda *args: False)

    tc.build_phis_static(
        case_dir,
        "EN00",
        tmp_path / "work",
        "eam.h0",
        "PHIS",
        dry_run=True,
        nco_bin="/nco",
    )

    assert commands[-1][1:5] == ["-O", "-x", "-v", "time,time_bnds"]
