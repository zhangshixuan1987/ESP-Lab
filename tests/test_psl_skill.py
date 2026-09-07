from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from esp_lab import psl_skill


def _write_psl_csv(path: Path, values):
    times = pd.date_range("2000-01-01", periods=len(values), freq="MS")
    frame = pd.DataFrame({"Date": times.strftime("%Y-%m-%d"), " TEST missing value -9999": values})
    frame.to_csv(path, index=False)


def test_load_psl_index_reads_packaged_csv(tmp_path, monkeypatch):
    _write_psl_csv(tmp_path / "iod.HadISST.PSL.csv", [1.0, -9999.0, 2.5])
    monkeypatch.setitem(psl_skill.PSL_INDEX_FILES, "IOD", "iod.HadISST.PSL.csv")

    da = psl_skill.load_psl_index("IOD", external_dir=tmp_path)

    assert da.name == "iod_psl"
    assert da.attrs["is_anomaly"] == "true"
    assert da.time.values[0] == cftime.DatetimeNoLeap(2000, 1, 15)
    np.testing.assert_allclose(da.values, [1.0, np.nan, 2.5])


def test_seasonal_centered_mean_keeps_season_center_months():
    time = [cftime.DatetimeNoLeap(2000, month, 15) for month in range(1, 13)]
    da = xr.DataArray(np.arange(12, dtype=float), dims="time", coords={"time": time})

    out = psl_skill.seasonal_centered_mean(da)

    assert [int(t.month) for t in out.time.values] == [4, 7, 10]
    np.testing.assert_allclose(out.values, [3.0, 6.0, 9.0])


def test_seasonal_centered_mean_rejects_all_missing_input():
    time = [cftime.DatetimeNoLeap(2000, month, 15) for month in range(1, 13)]
    da = xr.DataArray(np.full(12, np.nan), dims="time", coords={"time": time})

    with pytest.raises(ValueError, match="contains no finite values"):
        psl_skill.seasonal_centered_mean(da)


def test_retain_complete_seasonal_leads_drops_only_all_missing_endpoint():
    data = xr.DataArray(
        np.ones((2, 8, 3)),
        dims=("Y", "L", "M"),
        coords={"Y": [2000, 2001], "L": [3, 6, 9, 12, 15, 18, 21, 24], "M": range(3)},
    )
    data.loc[{"L": 24}] = np.nan
    # A partially missing lead remains a valid, complete-season coordinate.
    data.loc[{"Y": 2000, "L": 21, "M": 0}] = np.nan
    valid_time = xr.DataArray(
        np.tile(np.arange(8), (2, 1)),
        dims=("Y", "L"),
        coords={"Y": data.Y, "L": data.L},
    )

    kept, kept_time, dropped = psl_skill.retain_complete_seasonal_leads(data, valid_time)

    assert kept.L.values.tolist() == [3, 6, 9, 12, 15, 18, 21]
    assert kept_time.L.values.tolist() == kept.L.values.tolist()
    assert dropped == [24]


def test_retain_complete_seasonal_leads_rejects_mismatched_coordinates():
    data = xr.DataArray([1.0, np.nan], dims="L", coords={"L": [3, 6]})
    valid_time = xr.DataArray([0, 1], dims="L", coords={"L": [3, 9]})

    with pytest.raises(ValueError, match="identical L coordinates"):
        psl_skill.retain_complete_seasonal_leads(data, valid_time)


def test_subset_hindcast_initialization_years_preserves_later_valid_times():
    years = np.arange(1999, 2004)
    valid_time = xr.DataArray(
        [
            [
                cftime.DatetimeNoLeap(int(year), 11, 15),
                cftime.DatetimeNoLeap(int(year + 1), 2, 15),
            ]
            for year in years
        ],
        dims=("Y", "L"),
        coords={"Y": years, "L": [1, 4]},
    )
    data = xr.DataArray(
        np.arange(10).reshape(5, 2),
        dims=("Y", "L"),
        coords=valid_time.coords,
    )

    selected, selected_time = psl_skill.subset_hindcast_initialization_years(
        data, valid_time, 2000, 2002
    )

    assert selected.Y.values.tolist() == [2000, 2001, 2002]
    assert selected_time.isel(Y=-1, L=-1).item() == cftime.DatetimeNoLeap(2003, 2, 15)


def test_subset_hindcast_initialization_years_rejects_incomplete_cohort():
    years = [2000, 2002]
    valid_time = xr.DataArray(
        [cftime.DatetimeNoLeap(year, 5, 15) for year in years],
        dims="Y",
        coords={"Y": years},
    )
    data = xr.DataArray([1.0, 2.0], dims="Y", coords={"Y": years})

    with pytest.raises(ValueError, match="Expected every initialization year"):
        psl_skill.subset_hindcast_initialization_years(data, valid_time, 2000, 2002)


def test_observation_agreement_removes_local_monthly_climatology():
    time = [
        cftime.DatetimeNoLeap(year, month, 15)
        for year in range(2000, 2004)
        for month in range(1, 13)
    ]
    month_values = np.array([t.month for t in time], dtype=float)
    signal = np.array([t.year - 2001.5 for t in time], dtype=float)
    local_raw = xr.DataArray(month_values + signal, dims="time", coords={"time": time})
    psl_anom = xr.DataArray(signal, dims="time", coords={"time": time})

    result = psl_skill.observation_agreement(
        local_raw,
        psl_anom,
        2000,
        2003,
    )

    assert float(result["corr"]) == pytest.approx(1.0)
    assert float(result["rmse"]) == pytest.approx(0.0)
    assert int(result["n"]) == len(time)


def test_add_reference_sensitivity_adds_deltas_and_robust_metrics():
    skill = xr.Dataset(
        {
            "corr": (("reference", "L"), [[0.5, 0.2], [0.3, 0.4]]),
            "rmse": (("reference", "L"), [[1.0, 1.1], [1.2, 0.9]]),
        },
        coords={"reference": ["HadISST2", "PSL"], "L": [3, 6]},
    )

    out = psl_skill.add_reference_sensitivity(skill)

    np.testing.assert_allclose(out["dacc_ref"], [-0.2, 0.2])
    np.testing.assert_allclose(out["dnrmse_ref"], [0.2, -0.2])
    np.testing.assert_allclose(out["corr_robust"], [0.3, 0.2])
    np.testing.assert_allclose(out["rmse_robust"], [1.2, 1.1])


def test_compute_reference_skill_accepts_raw_and_anomaly_references():
    years = np.arange(2000, 2008)
    signal = xr.DataArray(
        np.linspace(-1.0, 1.0, years.size),
        dims="Y",
        coords={"Y": years},
    )
    model = xr.concat([signal - 0.1, signal + 0.1], dim=xr.DataArray([0, 1], dims="M", name="M"))
    model = model.expand_dims(L=[3]).transpose("Y", "L", "M")
    model_time = xr.DataArray(
        [[cftime.DatetimeNoLeap(int(year), 1, 15)] for year in years],
        dims=("Y", "L"),
        coords={"Y": years, "L": [3]},
    )
    obs_time = [cftime.DatetimeNoLeap(int(year), 1, 15) for year in years]
    obs_values = signal.values + np.array([0.0, 0.08, -0.03, 0.04, -0.02, 0.07, -0.04, 0.02])
    psl_anom = xr.DataArray(obs_values, dims="time", coords={"time": obs_time})
    local_raw = xr.DataArray(obs_values + 10.0, dims="time", coords={"time": obs_time})

    skill = psl_skill.compute_reference_skill(
        model,
        model_time,
        {
            "HadISST2": (local_raw, False),
            "PSL": (psl_anom, True),
        },
        2000,
        2007,
        detrend=False,
    )

    assert list(skill.reference.values) == ["HadISST2", "PSL"]
    assert np.all(np.isfinite(skill["corr"].values))
    assert float(skill["corr"].min()) > 0.98


def test_november_seasonal_cohort_uses_initialization_year():
    years = np.arange(1980, 2013)
    time = xr.DataArray(
        [[cftime.DatetimeNoLeap(int(y + 1), 1, 15),
          cftime.DatetimeNoLeap(int(y + 2), 1, 15)] for y in years],
        dims=('Y', 'L'), coords={'Y': [f'{y}110100' for y in years], 'L': [3, 15]},
    )
    data = xr.ones_like(time, dtype=float)
    selected, selected_time = psl_skill.subset_hindcast_initialization_years(
        data, time, 1981, 2011
    )
    assert selected.Y.values[[0, -1]].tolist() == ['1981110100', '2011110100']
    assert selected_time.isel(Y=0, L=0).item().year == 1982
    assert selected_time.isel(Y=-1, L=-1).item().year == 2013
