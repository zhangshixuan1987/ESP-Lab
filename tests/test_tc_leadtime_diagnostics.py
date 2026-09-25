from pathlib import Path

import pytest
import xarray as xr

from workflows.tropical_cyclones.leadtime_diagnostics import (
    diagnostic_path,
    ensure_experiment_diagnostic,
)
from workflows.tropical_cyclones.track_density import TrackDensityConfig


SPEC = {
    "case_prefix": "example_case",
    "sim_dir": "/unused/simulation",
    "stream_tag": "eam.h2",
}


def _ensure(tmp_path, mode):
    return ensure_experiment_diagnostic(
        case_key="experiment", spec=SPEC, repo_root=tmp_path,
        track_root=tmp_path / "tracks", diag_root=tmp_path / "diagnostics",
        year_start=1980, year_end=2011, init_months=[5, 11],
        members=["EN01"], parset="set3", leads=[1],
        seasons=["NH_JJASON", "SH_DJFMAM"], basin_defs={},
        track_config=None, track_settings={}, ibtracs_file=None,
        cache_mode=mode,
    )


def test_diagnostic_path_uses_exact_period(tmp_path):
    path = diagnostic_path(
        tmp_path, "experiment", "case", "set3", 1980, 2011
    )
    assert path == (
        tmp_path / "experiment" / "tc_track"
        / "experiment_tc_lead_track_density_case_set3_1980_2011.nc"
    )


def test_auto_reuses_exact_existing_cache_without_processing(tmp_path):
    path = diagnostic_path(
        tmp_path / "diagnostics", "experiment", "example_case", "set3", 1980, 2011
    )
    path.parent.mkdir(parents=True)
    path.touch()

    actual, status = _ensure(tmp_path, "auto")

    assert actual == path
    assert status == "reused"


def test_require_rejects_missing_exact_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="1980_2011"):
        _ensure(tmp_path, "require")


def test_cache_mode_is_validated(tmp_path):
    with pytest.raises(ValueError, match="auto, require, or rebuild"):
        _ensure(tmp_path, "invalid")


def test_auto_builds_each_season_for_exact_period(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "workflows.tropical_cyclones.leadtime_diagnostics.ensure_tracks",
        lambda **kwargs: {"reused": 2, "built": 0},
    )
    track_root = tmp_path / "tracks"
    for month, point_month in [(5, 6), (11, 12)]:
        case = f"example_case_1980{month:02d}0100"
        path = (
            track_root / case / "EN01" / "post" / "atm" / "tc-analysis"
            / f"{case}_EN01_set3_TCS_track.txt"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            f"start\n0 150 10 99000 20 0 1980 {point_month} 1 0\n"
        )

    class InlineClient:
        def __init__(self):
            self.map_calls = 0

        def map(self, function, *iterables, **kwargs):
            self.map_calls += 1
            return [function(*arguments) for arguments in zip(*iterables)]

        def gather(self, futures):
            return futures

    client = InlineClient()
    output, status = ensure_experiment_diagnostic(
        case_key="experiment", spec=SPEC, repo_root=tmp_path,
        track_root=track_root, diag_root=tmp_path / "diagnostics",
        year_start=1980, year_end=1980, init_months=[5, 11],
        members=["EN01"], parset="set3", leads=[1],
        seasons=["NH_JJASON", "SH_DJFMAM"],
        basin_defs={
            "TEST": {
                "season": "NH_JJASON", "lon": (100, 180), "lat": (0, 40),
                "long_name": "Test basin",
            }
        },
        track_config=TrackDensityConfig(method="box", box_grid_size=30),
        track_settings={}, ibtracs_file=None, cache_mode="auto",
        dask_client=client,
    )

    assert status == "built"
    assert client.map_calls == 1
    with xr.open_dataset(output) as dataset:
        assert dataset.attrs["years"] == "1980-1980"
        assert dataset.sample_count.sel(lead=1, year=1980).values.tolist() == [1, 1]
        assert float(dataset.tc_track_density_count.sum()) == 2
