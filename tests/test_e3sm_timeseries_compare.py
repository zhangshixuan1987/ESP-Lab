import numpy as np
import xarray as xr

from workflows.e3sm_analysis.timeseries_comparison import (
    _format_time_label,
    _find_landfrac_file,
    _get_initial_value_mask,
    _get_time_indices,
    _get_time_values,
    _select_matching_file,
    _time_values_to_plot_axis,
    load_observation_timeseries,
    load_simulation_timeseries,
    plot_timeseries_comparison,
)


def _write_ts(path, values, start="1980-05", var_name="TREFHT"):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            var_name: (
                ("time", "lat", "lon"),
                np.asarray(values, dtype=float)[:, None, None],
            )
        },
        coords={
            "time": np.arange(
                np.datetime64(start),
                np.datetime64(start) + len(values),
                dtype="datetime64[M]",
            ),
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    ds.to_netcdf(path)


def test_select_matching_file_prefers_init_tag_month():
    files = [
        "/tmp/TREFHT_198006_198205.nc",
        "/tmp/TREFHT_198005_198204.nc",
    ]

    assert _select_matching_file(files, init_tag="1980050100").endswith(
        "TREFHT_198005_198204.nc"
    )


def test_format_time_label_handles_numpy_datetime64():
    assert _format_time_label(np.datetime64("1980-05")) == "1980-05"


def test_load_simulation_timeseries_skips_incomplete_member(tmp_path):
    sim_dir = tmp_path / "run"
    ts_rel = "post/atm/180x360_aave/ts/monthly/2yr"
    _write_ts(sim_dir / "EN00" / ts_rel / "TREFHT_198005_198204.nc", [280.0])
    _write_ts(
        sim_dir / "EN01" / ts_rel / "TREFHT_198005_198204.nc",
        [280.0, 281.0],
    )

    result = load_simulation_timeseries(
        tmp_path,
        {"case": {"run_name": "run", "ens": 2}},
        "TREFHT",
        init_tag="1980050100",
        nlead=2,
    )

    np.testing.assert_allclose(result["case"]["members"], [[6.85, 7.85]])
    assert result["case"]["members"].shape == (1, 2)
    assert result["case"]["source_grids"] == ["180x360_aave"]


def test_load_simulation_timeseries_keeps_time_per_simulation(tmp_path):
    ts_rel = "post/atm/180x360_aave/ts/monthly/2yr"
    _write_ts(
        tmp_path / "may_run" / ts_rel / "TREFHT_198005_198006.nc",
        [280.0, 281.0],
        start="1980-05",
    )
    _write_ts(
        tmp_path / "nov_run" / ts_rel / "TREFHT_198011_198012.nc",
        [282.0, 283.0],
        start="1980-11",
    )

    result = load_simulation_timeseries(
        tmp_path,
        {
            "may": {"run_name": "may_run", "ens": 1},
            "nov": {"run_name": "nov_run", "ens": 1},
        },
        "TREFHT",
        init_tag=None,
        nlead=2,
    )

    assert _format_time_label(result["may"]["time"][0]) == "1980-05"
    assert _format_time_label(result["nov"]["time"][0]) == "1980-11"


def test_load_simulation_timeseries_skips_mismatched_member_time(tmp_path):
    sim_dir = tmp_path / "run"
    ts_rel = "post/atm/180x360_aave/ts/monthly/2yr"
    _write_ts(
        sim_dir / "EN00" / ts_rel / "TREFHT_198005_198006.nc",
        [280.0, 281.0],
        start="1980-05",
    )
    _write_ts(
        sim_dir / "EN01" / ts_rel / "TREFHT_198011_198012.nc",
        [290.0, 291.0],
        start="1980-11",
    )

    result = load_simulation_timeseries(
        tmp_path,
        {"case": {"run_name": "run", "ens": 2}},
        "TREFHT",
        init_tag=None,
        nlead=2,
    )

    np.testing.assert_allclose(result["case"]["members"], [[6.85, 7.85]])
    assert _format_time_label(result["case"]["time"][0]) == "1980-05"


def test_land_masked_variable_does_not_fallback_to_glb(tmp_path):
    ts_rel = "post/atm/glb/ts/monthly/2yr"
    _write_ts(
        tmp_path / "run" / ts_rel / "TREFHT_198005_198006.nc",
        [290.0, 291.0],
        start="1980-05",
    )

    result = load_simulation_timeseries(
        tmp_path,
        {"case": {"run_name": "run", "ens": 1}},
        "TREFHT",
        init_tag="1980050100",
        nlead=2,
    )

    assert result == {}


def test_load_simulation_timeseries_uses_requested_frequency(tmp_path):
    ts_rel = "post/atm/180x360_aave/ts/6hourly/2yr"
    _write_ts(
        tmp_path / "run" / ts_rel / "PRECT_198005_198006.nc",
        [1.0e-8, 2.0e-8],
        start="1980-05",
        var_name="PRECT",
    )

    result = load_simulation_timeseries(
        tmp_path,
        {"case": {"run_name": "run", "ens": 1}},
        "PRECT",
        init_tag="1980050100",
        nlead=2,
        frequency="6hourly",
    )

    assert result["case"]["frequency"] == "6hourly"
    assert result["case"]["source_grids"] == ["180x360_aave"]
    np.testing.assert_allclose(result["case"]["members"], [[0.864, 1.728]])


def test_find_landfrac_file_falls_back_to_monthly_for_6hourly(tmp_path):
    six_hourly_dir = tmp_path / "post/atm/180x360_aave/ts/6hourly/2yr"
    monthly_dir = tmp_path / "post/atm/180x360_aave/ts/monthly/2yr"
    monthly_dir.mkdir(parents=True)
    landfrac = monthly_dir / "LANDFRAC_198005_198204.nc"
    landfrac.touch()

    assert _find_landfrac_file(six_hourly_dir, init_tag="1980050100") == str(landfrac)


def test_find_landfrac_file_falls_back_to_sibling_member_monthly(tmp_path):
    six_hourly_dir = tmp_path / "run" / "EN00" / "post/atm/180x360_aave/ts/6hourly/2yr"
    sibling_monthly_dir = tmp_path / "run" / "EN03" / "post/atm/180x360_aave/ts/monthly/2yr"
    six_hourly_dir.mkdir(parents=True)
    sibling_monthly_dir.mkdir(parents=True)
    landfrac = sibling_monthly_dir / "LANDFRAC_198005_198204.nc"
    landfrac.touch()

    assert _find_landfrac_file(six_hourly_dir, init_tag="1980050100") == str(landfrac)


def test_time_values_to_plot_axis_preserves_distinct_dates():
    may_x, may_is_time = _time_values_to_plot_axis(
        np.array(["1980-05", "1980-06"], dtype="datetime64[M]"),
        2,
    )
    nov_x, nov_is_time = _time_values_to_plot_axis(
        np.array(["1980-11", "1980-12"], dtype="datetime64[M]"),
        2,
    )

    assert may_is_time
    assert nov_is_time
    assert may_x[0] != nov_x[0]


def test_time_values_to_plot_axis_preserves_subdaily_time():
    x, is_time = _time_values_to_plot_axis(
        np.array(["1980-05-01T00:30", "1980-05-01T06:30"], dtype="datetime64[m]"),
        2,
    )

    assert is_time
    assert x[0] != x[1]


def test_plot_timeseries_comparison_uses_each_simulation_time(tmp_path, monkeypatch):
    captured_x = []

    def capture_plot(x, *args, **kwargs):
        captured_x.append(np.asarray(x))
        return []

    monkeypatch.setattr(
        "workflows.e3sm_analysis.timeseries_comparison.plt.plot",
        capture_plot,
    )
    monkeypatch.setattr(
        "workflows.e3sm_analysis.timeseries_comparison.plt.legend",
        lambda *args, **kwargs: None,
    )

    sim_data = {
        "may": {
            "members": np.array([[1.0, 2.0]]),
            "mean": np.array([1.0, 2.0]),
            "std": np.array([0.0, 0.0]),
            "time": np.array(["1980-05", "1980-06"], dtype="datetime64[M]"),
        },
        "nov": {
            "members": np.array([[3.0, 4.0]]),
            "mean": np.array([3.0, 4.0]),
            "std": np.array([0.0, 0.0]),
            "time": np.array(["1980-11", "1980-12"], dtype="datetime64[M]"),
        },
    }

    plot_timeseries_comparison(sim_data, "TREFHT", tmp_path)

    assert len(captured_x) == 2
    assert captured_x[0][0] != captured_x[1][0]


def test_load_observation_timeseries_matches_reference_months(tmp_path):
    obs_path = tmp_path / "obs.nc"
    ds = xr.Dataset(
        {
            "t2m": (
                ("time", "latitude", "longitude"),
                np.array(
                    [
                        [[280.0, 282.0], [284.0, 286.0]],
                        [[281.0, 283.0], [285.0, 287.0]],
                        [[282.0, 284.0], [286.0, 288.0]],
                    ]
                ),
            )
        },
        coords={
            "time": np.array(
                ["1980-04-01", "1980-05-01", "1980-06-01"],
                dtype="datetime64[D]",
            ),
            "latitude": [-60.0, 60.0],
            "longitude": [0.0, 180.0],
        },
    )
    ds.to_netcdf(obs_path)

    result = load_observation_timeseries(
        "TREFHT",
        reference_time=np.array(["1980-05", "1980-06"], dtype="datetime64[M]"),
        obs_info={
            "TREFHT": {
                "path": str(obs_path),
                "file_var": "t2m",
                "label": "test obs",
                "conversion": lambda x: x - 273.15,
            }
        },
    )

    np.testing.assert_allclose(result["mean"], [10.85, 11.85])
    assert result["label"] == "test obs"


def test_plot_timeseries_comparison_overlays_observations(tmp_path, monkeypatch):
    labels = []

    def capture_plot(*args, **kwargs):
        if "label" in kwargs:
            labels.append(kwargs["label"])
        return []

    monkeypatch.setattr(
        "workflows.e3sm_analysis.timeseries_comparison.plt.plot",
        capture_plot,
    )
    monkeypatch.setattr(
        "workflows.e3sm_analysis.timeseries_comparison.plt.legend",
        lambda *args, **kwargs: None,
    )

    sim_data = {
        "case": {
            "members": np.array([[1.0, 2.0]]),
            "mean": np.array([1.0, 2.0]),
            "std": np.array([0.0, 0.0]),
            "time": np.array(["1980-05", "1980-06"], dtype="datetime64[M]"),
        },
    }
    obs_data = {
        "mean": np.array([1.5, 2.5]),
        "time": np.array(["1980-05", "1980-06"], dtype="datetime64[M]"),
        "label": "ERA5",
    }

    plot_timeseries_comparison(sim_data, "TREFHT", tmp_path, obs_data=obs_data)

    assert "ERA5" in labels


def test_6hourly_time_values_prefer_interval_start_bounds():
    ds = xr.Dataset(
        coords={
            "time": np.array(
                ["1980-05-01T06:00", "1980-05-01T12:00"],
                dtype="datetime64[m]",
            ),
            "nbnd": [0, 1],
        },
        data_vars={
            "time_bnds": (
                ("time", "nbnd"),
                np.array(
                    [
                        ["1980-05-01T00:00", "1980-05-01T06:00"],
                        ["1980-05-01T06:00", "1980-05-01T12:00"],
                    ],
                    dtype="datetime64[m]",
                ),
            )
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"

    time_indices = _get_time_indices(ds, nlead=2, frequency="6hourly")
    result = _get_time_values(ds, time_indices, frequency="6hourly")

    np.testing.assert_array_equal(time_indices, [0, 1])
    np.testing.assert_array_equal(
        result,
        np.array(["1980-05-01T00:00", "1980-05-01T06:00"], dtype="datetime64[m]"),
    )


def test_6hourly_time_indices_discard_and_align_zero_duration_initial_sample():
    ds = xr.Dataset(
        coords={
            "time": np.array(
                ["1980-05-01T00:30", "1980-05-01T06:30", "1980-05-01T12:30"],
                dtype="datetime64[m]",
            ),
            "nbnd": [0, 1],
        },
        data_vars={
            "time_bnds": (
                ("time", "nbnd"),
                np.array(
                    [
                        ["1980-05-01T00:30", "1980-05-01T00:30"],
                        ["1980-05-01T00:30", "1980-05-01T06:30"],
                        ["1980-05-01T06:30", "1980-05-01T12:30"],
                    ],
                    dtype="datetime64[m]",
                ),
            )
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"

    time_indices = _get_time_indices(ds, nlead=2, frequency="6hourly")
    result = _get_time_values(ds, time_indices, frequency="6hourly")
    initial_mask = _get_initial_value_mask(ds, time_indices, frequency="6hourly")

    np.testing.assert_array_equal(time_indices, [1, 2])
    np.testing.assert_array_equal(
        result,
        np.array(
            ["1980-05-01T00:00", "1980-05-01T06:00"],
            dtype="datetime64[m]",
        ),
    )
    np.testing.assert_array_equal(initial_mask, [False, False])


def test_load_6hourly_keeps_shorter_series_after_discarding_initial_sample(tmp_path):
    ts_rel = "post/atm/180x360_aave/ts/6hourly/2yr"
    path = tmp_path / "run" / ts_rel / "PRECT_198005_198006.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "PRECT": (
                ("time", "lat", "lon"),
                np.asarray([1.0e-8, 2.0e-8, 3.0e-8], dtype=float)[:, None, None],
            ),
            "time_bnds": (
                ("time", "nbnd"),
                np.array(
                    [
                        ["1980-05-01T00:30", "1980-05-01T00:30"],
                        ["1980-05-01T00:30", "1980-05-01T06:30"],
                        ["1980-05-01T06:30", "1980-05-01T12:30"],
                    ],
                    dtype="datetime64[m]",
                ),
            ),
        },
        coords={
            "time": np.array(
                ["1980-05-01T00:30", "1980-05-01T06:30", "1980-05-01T12:30"],
                dtype="datetime64[m]",
            ),
            "lat": [0.0],
            "lon": [0.0],
            "nbnd": [0, 1],
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"
    ds.to_netcdf(path)

    result = load_simulation_timeseries(
        tmp_path,
        {"case": {"run_name": "run", "ens": 1}},
        "PRECT",
        init_tag="1980050100",
        nlead=3,
        frequency="6hourly",
    )

    np.testing.assert_allclose(result["case"]["members"], [[1.728, 2.592]])
    np.testing.assert_array_equal(
        result["case"]["time"],
        np.array(["1980-05-01T00:00", "1980-05-01T06:00"], dtype="datetime64[m]"),
    )


def test_plot_timeseries_comparison_uses_output_tag(tmp_path, monkeypatch):
    saved_paths = []

    monkeypatch.setattr(
        "workflows.e3sm_analysis.timeseries_comparison.plt.savefig",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )

    sim_data = {
        "case": {
            "members": np.array([[1.0, 2.0]]),
            "mean": np.array([1.0, 2.0]),
            "std": np.array([0.0, 0.0]),
            "time": np.array(["1980-05-01T00:30", "1980-05-01T06:30"], dtype="datetime64[m]"),
        }
    }

    plot_timeseries_comparison(
        sim_data,
        "TREFHT",
        tmp_path,
        frequency="6hourly",
        output_tag="6hourly_first_month",
    )

    assert saved_paths == [
        str(tmp_path / "e3sm_timeseries_6hourly_first_month_TREFHT.png")
    ]


def test_first_month_6hourly_plot_uses_day_ticks(tmp_path):
    time = np.arange(
        np.datetime64("1980-05-01T00:30"),
        np.datetime64("1980-06-01T00:30"),
        np.timedelta64(6, "h"),
    )
    sim_data = {
        "case": {
            "members": np.arange(time.size, dtype=float)[None, :],
            "mean": np.arange(time.size, dtype=float),
            "std": np.zeros(time.size, dtype=float),
            "time": time,
        }
    }

    plot_timeseries_comparison(
        sim_data,
        "TREFHT",
        tmp_path,
        frequency="6hourly",
        output_tag="6hourly_first_month",
    )

    assert (tmp_path / "e3sm_timeseries_6hourly_first_month_TREFHT.png").exists()
