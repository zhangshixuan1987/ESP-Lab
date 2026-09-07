import cftime
import numpy as np
import pytest
import xarray as xr

from esp_lab import data_access_cesm_smyle as smyle
from esp_lab import data_access_e3sm as e3sm


def _benchmark_dataset(time_values):
    return xr.Dataset(
        {"TREFHT": (("Y", "L", "M"), np.ones((2, 3, 1)))},
        coords={
            "Y": ["1980110100", "1981110100"],
            "L": [3, 6, 9],
            "M": ["EN01"],
            "time": (("Y", "L"), time_values),
        },
    )


def test_expected_verification_time_uses_init_tag_and_lead():
    result = smyle.expected_verification_time(
        ["1980110100", "1981110100"],
        [3, 6, 9],
    )

    assert list(result.isel(Y=0).values) == [
        cftime.DatetimeNoLeap(1981, 1, 15),
        cftime.DatetimeNoLeap(1981, 4, 15),
        cftime.DatetimeNoLeap(1981, 7, 15),
    ]
    assert result.isel(Y=1, L=0).item() == cftime.DatetimeNoLeap(1982, 1, 15)


def test_e3sm_verification_time_uses_init_tag_and_lead():
    result = e3sm._verification_time_from_init_tags(
        ["1980050100", "1981050100"],
        [3, 6, 9],
    )

    assert list(result.isel(Y=0).values) == [
        cftime.DatetimeNoLeap(1980, 7, 15),
        cftime.DatetimeNoLeap(1980, 10, 15),
        cftime.DatetimeNoLeap(1981, 1, 15),
    ]


def test_ensure_verification_time_repairs_scrambled_benchmark_dates():
    bad_time = np.array(
        [
            [
                cftime.DatetimeNoLeap(2010, 1, 15),
                cftime.DatetimeNoLeap(2010, 4, 15),
                cftime.DatetimeNoLeap(2010, 7, 15),
            ],
            [
                cftime.DatetimeNoLeap(1985, 1, 15),
                cftime.DatetimeNoLeap(1985, 4, 15),
                cftime.DatetimeNoLeap(1985, 7, 15),
            ],
        ],
        dtype=object,
    )
    dataset = _benchmark_dataset(bad_time)

    with pytest.warns(UserWarning, match="Repairing 6 inconsistent"):
        result = smyle.ensure_verification_time(dataset, init_month=11)

    assert smyle.verification_time_mismatch_count(result, init_month=11) == 0
    assert result.time.isel(Y=0, L=0).item() == cftime.DatetimeNoLeap(1981, 1, 15)
    assert result.attrs["verification_time_repaired_count"] == 6


def test_benchmark_path_resolves_standard_diagnostic_layout(tmp_path):
    filename = smyle.benchmark_filename("TREFHT", 11)
    expected = tmp_path / "leadtime_acc" / "inputs" / "atm" / "TREFHT" / filename
    expected.parent.mkdir(parents=True)
    expected.touch()

    assert smyle.benchmark_path("TREFHT", 11, tmp_path) == expected


def test_benchmark_path_reports_all_candidates_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Benchmark file not found"):
        smyle.benchmark_path("TREFHT", 11, tmp_path)
