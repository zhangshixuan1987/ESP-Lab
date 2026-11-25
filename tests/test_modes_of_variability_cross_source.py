import numpy as np
import pytest
import xarray as xr
from cftime import DatetimeNoLeap

from workflows.modes_of_variability.cross_source import (
    align_common_mode_products,
    build_common_skill_cache,
    build_ensemble_spread_native_skill_cache,
    build_ensemble_spread_common_skill_cache,
    build_native_skill_cache,
    build_nao_station_eof_skill_cache,
    build_projected_mode_skill_cache,
    build_projected_pc_timeseries,
    plot_common_skill_figure,
    plot_native_skill_figure,
    plot_ensemble_spread_native_skill_figure,
    plot_cross_source_teleconnection_figure,
    plot_mode_domain_regression_figure,
    plot_projected_mode_skill_figure,
    plot_projected_pc_timeseries,
    skill_summary_dataframe,
    write_common_skill_cache,
    write_skill_summary_table,
)


def _product(
    *, leads=(3, 6, 9), lon=(0.0, 2.5), reference="ERA5", time_shift=0,
    padded=False, year_labels=None,
):
    years = np.array([1991, 1992, 1993, 1994])
    year_labels = years if year_labels is None else np.asarray(year_labels)
    leads = np.asarray(leads)
    values = np.ones((years.size, leads.size, 1), dtype=float)
    if padded:
        values[:, -1, :] = np.nan
    valid_time = np.array(
        [[np.datetime64(f"{year}-04-15") + np.timedelta64(time_shift, "D") for _ in leads] for year in years]
    )
    return xr.Dataset(
        {
            "mode_index": (("Y", "L", "M"), values),
            "valid_time": (("Y", "L"), valid_time),
            "target_month": ("L", np.full(leads.size, 4)),
        },
        coords={
            "Y": year_labels,
            "L": leads,
            "M": ["member"],
            "lat": np.asarray([-1.25, 1.25]),
            "lon": np.asarray(lon),
        },
        attrs={
            "eof_strategy": "fixed_obs_projection",
            "common_basis_reference": reference,
            "eof_reference_period": "1979-2019",
        },
    )


def test_align_common_mode_products_returns_only_shared_native_leads():
    aligned = align_common_mode_products({"e3sm": _product(leads=(3, 6, 9, 12)), "nmme": _product()})

    assert list(aligned["e3sm"].L.values) == [3, 6, 9]
    assert list(aligned["nmme"].L.values) == [3, 6, 9]


def test_align_common_mode_products_normalizes_source_native_year_labels():
    aligned = align_common_mode_products({
        "e3sm": _product(
            year_labels=["1991040100", "1992040100", "1993040100", "1994040100"]
        ),
        "nmme": _product(),
    })

    np.testing.assert_array_equal(aligned["e3sm"].Y, [1991, 1992, 1993, 1994])
    np.testing.assert_array_equal(aligned["nmme"].Y, [1991, 1992, 1993, 1994])


def test_align_common_mode_products_rejects_padded_nmme_lead():
    with pytest.raises(ValueError, match="padded leads"):
        align_common_mode_products({"e3sm": _product(leads=(3, 6, 9, 12)), "nmme": _product(leads=(3, 6, 9, 12), padded=True)})


def test_align_common_mode_products_rejects_unavailable_requested_lead():
    with pytest.raises(ValueError, match="will not be padded"):
        align_common_mode_products({"e3sm": _product(leads=(3, 6, 9, 12)), "nmme": _product()}, required_leads=(3, 6, 9, 12))


def test_align_common_mode_products_accepts_calendar_day_difference():
    align_common_mode_products({"e3sm": _product(), "nmme": _product(time_shift=1)})


def test_align_common_mode_products_rejects_target_time_mismatch():
    with pytest.raises(ValueError, match="Target-time mismatch"):
        align_common_mode_products({"e3sm": _product(), "nmme": _product(time_shift=31)})


def test_align_common_mode_products_rejects_grid_mismatch():
    with pytest.raises(ValueError, match="Incompatible target grid"):
        align_common_mode_products({"e3sm": _product(), "nmme": _product(lon=(0.0, 5.0))})


def test_align_common_mode_products_rejects_reference_mismatch():
    with pytest.raises(ValueError, match="Incompatible fixed reference"):
        align_common_mode_products({"e3sm": _product(), "nmme": _product(reference="HadISST2")})


def test_build_common_skill_cache_records_shared_cohort_and_coverage():
    products = {"e3sm": _product(leads=(3, 6, 9, 12)), "nmme": _product()}
    observation = xr.DataArray(
        [1.0, 1.0, 1.0, 1.0],
        dims="time",
        coords={"time": np.array(["1991-04-15", "1992-04-15", "1993-04-15", "1994-04-15"], dtype="datetime64[ns]")},
    )

    result = build_common_skill_cache(products, observation)

    assert list(result.source.values) == ["e3sm", "nmme"]
    assert list(result.L.values) == [3, 6, 9]
    np.testing.assert_array_equal(result.sample_count, 4)
    np.testing.assert_allclose(result.coverage, 1.0)
    assert result.attrs["source_only_longer_leads"] == "excluded from this comparison cache"




def test_native_skill_cache_retains_source_periods_and_native_leads(tmp_path):
    e3sm = _product(leads=(3, 6, 9, 12))
    nmme = _product(leads=(3, 6, 9)).isel(Y=slice(1, None))
    for product in (e3sm, nmme):
        values = np.broadcast_to(
            np.arange(product.sizes["Y"], dtype=float)[:, None, None],
            product["mode_index"].shape,
        ).copy()
        product["mode_index"] = xr.DataArray(
            values, dims=("Y", "L", "M"), coords=product["mode_index"].coords
        )
    observation = xr.DataArray(
        [0.0, 1.0, 2.0, 3.0], dims="time",
        coords={"time": np.array([
            "1991-04-15", "1992-04-15", "1993-04-15", "1994-04-15",
        ], dtype="datetime64[ns]")},
    )

    cache = build_native_skill_cache({"E3SM experiment": e3sm, "NMME": nmme}, observation)
    path = plot_native_skill_figure(cache, tmp_path / "pdo_native_skill.png", mode="PDO")

    assert list(cache.L.values) == [3, 6, 9, 12]
    assert int(cache.sample_count.sel(source="E3SM experiment", L=12)) == 4
    assert int(cache.sample_count.sel(source="NMME", L=3)) == 3
    assert np.isnan(cache.sample_count.sel(source="NMME", L=12))
    assert int(cache.target_year_start.sel(source="NMME", L=3)) == 1992
    assert cache.attrs["period_alignment"].startswith("source-native")
    assert path.is_file() and path.stat().st_size > 0

def test_write_common_skill_cache_round_trips(tmp_path):
    products = {"e3sm": _product(), "nmme": _product()}
    observation = xr.DataArray(
        [1.0, 1.0, 1.0, 1.0], dims="time",
        coords={"time": np.array(["1991-04-15", "1992-04-15", "1993-04-15", "1994-04-15"], dtype="datetime64[ns]")},
    )
    path = write_common_skill_cache(products, observation, tmp_path / "skill.nc")

    with xr.open_dataset(path) as result:
        assert list(result.source.values) == ["e3sm", "nmme"]
        assert set(result.data_vars) >= {"corr", "pval", "rmse", "msss", "rpc", "sample_count", "coverage"}


def test_plot_common_skill_figure_marks_source_only_extension(tmp_path):
    products = {"e3sm": _product(), "nmme": _product()}
    observation = xr.DataArray(
        [1.0, 1.0, 1.0, 1.0], dims="time",
        coords={"time": np.array(["1991-04-15", "1992-04-15", "1993-04-15", "1994-04-15"], dtype="datetime64[ns]")},
    )
    cache = build_common_skill_cache(products, observation)
    extension = xr.Dataset({
        "corr": ("L", [0.1]),
        "rmse": ("L", [0.8]),
    }, coords={"L": [12]})

    path = plot_common_skill_figure(
        cache, tmp_path / "nao_common_skill.png", mode="NAO", source_only_skill={"e3sm": extension}
    )

    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_common_skill_figure_supports_distinct_model_labels(tmp_path):
    products = {"CMC1-CanCM3": _product(), "CMC2-CanCM4": _product()}
    observation = xr.DataArray(
        [1.0, 1.0, 1.0, 1.0], dims="time",
        coords={"time": np.array(["1991-04-15", "1992-04-15", "1993-04-15", "1994-04-15"], dtype="datetime64[ns]")},
    )
    path = plot_common_skill_figure(
        build_common_skill_cache(products, observation), tmp_path / "pdo_models.png", mode="PDO"
    )

    assert path.is_file() and path.stat().st_size > 0




def test_ensemble_spread_native_skill_uses_members_and_model_means(tmp_path):
    def product(labels, offset):
        base = _product(leads=(3, 6)).isel(M=[0]).reindex(M=labels)
        values = np.empty((base.sizes["Y"], base.sizes["L"], len(labels)))
        for member, _ in enumerate(labels):
            values[:, :, member] = np.arange(base.sizes["Y"])[:, None] + offset * member
        base["mode_index"] = xr.DataArray(
            values, dims=("Y", "L", "M"), coords={"Y": base.Y, "L": base.L, "M": labels}
        )
        return base

    products = {
        "E3SM": product(["e1", "e2"], 0.2),
        "CESM-SMYLE": product(["s1", "s2"], 0.3),
        "NMME": product(["CMC1:m1", "CMC1:m2", "CMC2:m1", "CMC2:m2"], 0.4),
    }
    observation = xr.DataArray(
        [0.0, 1.0, 2.0, 3.0], dims="time",
        coords={"time": np.array([
            "1991-04-15", "1992-04-15", "1993-04-15", "1994-04-15",
        ], dtype="datetime64[ns]")},
    )
    cache = build_ensemble_spread_native_skill_cache(products, observation)
    path = plot_ensemble_spread_native_skill_figure(
        cache, tmp_path / "pdo_ensemble_spread.png", mode="PDO"
    )

    assert list(cache.source.values) == ["E3SM", "CESM-SMYLE", "NMME"]
    assert int(cache.spread_unit_count.sel(source="E3SM").isel(L=0)) == 2
    assert int(cache.spread_unit_count.sel(source="CESM-SMYLE").isel(L=0)) == 2
    assert int(cache.spread_unit_count.sel(source="NMME").isel(L=0)) == 2
    assert "per-model" in str(cache.spread_definition.sel(source="NMME").item())
    assert set(cache.data_vars) >= {"corr_spread", "rmse_spread", "msss_spread", "rpc_spread"}
    assert path.is_file() and path.stat().st_size > 0

    single_model_products = dict(products)
    single_model_products["NMME CMC1"] = product(["CMC1:m1", "CMC1:m2"], 0.4)
    del single_model_products["NMME"]
    single_cache = build_ensemble_spread_native_skill_cache(
        single_model_products, observation, model_ensemble_sources=("NMME CMC1",)
    )
    assert int(single_cache.spread_unit_count.sel(source="NMME CMC1").isel(L=0)) == 2
    assert "single NMME model" in str(
        single_cache.spread_definition.sel(source="NMME CMC1").item()
    )

def test_projected_pc_timeseries_preserves_native_leads_and_target_time(tmp_path):
    product = _product(leads=(3, 6))
    product = product.drop_vars("mode_index").assign_coords(M=["a", "b"])
    product["mode_index"] = xr.DataArray(
        np.array([[[1.0, 3.0], [2.0, 4.0]]] * 4),
        dims=("Y", "L", "M"), coords={"Y": product.Y, "L": product.L, "M": product.M},
    )
    product["valid_time"] = xr.DataArray(
        np.array([[np.datetime64(f"{year}-04-15"), np.datetime64(f"{year}-05-15")] for year in product.Y.values]),
        dims=("Y", "L"), coords={"Y": product.Y, "L": product.L},
    )
    product["target_month"] = xr.DataArray([4, 5], dims="L", coords={"L": product.L})
    product.attrs["source"] = "NMME"
    reference = xr.Dataset({"mode_index": ("time", np.arange(8, dtype=float))}, coords={"time": np.array([
        "1991-04-15", "1991-05-15", "1992-04-15", "1992-05-15",
        "1993-04-15", "1993-05-15", "1994-04-15", "1994-05-15",
    ], dtype="datetime64[ns]")})

    cache = build_projected_pc_timeseries(product, reference)
    cache["valid_time"] = xr.DataArray(
        np.array([DatetimeNoLeap(year, month, 15) for year in product.Y.values for month in (4, 5)], dtype=object).reshape(4, 2),
        dims=("Y", "L"), coords={"Y": product.Y, "L": product.L},
    )
    path = plot_projected_pc_timeseries(cache, tmp_path / "nao_pc.png", mode="NAO", init_month=2, lead=3)

    assert list(cache.L.values) == [3, 6]
    np.testing.assert_allclose(cache.forecast_mean.sel(L=3), 2.0)
    np.testing.assert_allclose(cache.forecast_std.sel(L=6), 1.0)
    np.testing.assert_allclose(cache.observation.sel(L=3), [0.0, 2.0, 4.0, 6.0])
    np.testing.assert_array_equal(cache.valid_time.sel(L=6).dt.month, 5)
    assert cache.attrs["native_leads"] == "only populated product leads are retained"
    assert path.is_file() and path.stat().st_size > 0


def test_projected_pc_timeseries_rejects_missing_observation_time_dimension():
    with pytest.raises(ValueError, match=r"exactly the \('time',\) dimension"):
        build_projected_pc_timeseries(_product(), xr.DataArray([1.0], dims="Y"))


def test_mode_domain_regression_figure_uses_stored_metrics_and_significance(tmp_path):
    lat, lon = [20.0, 30.0], [-40.0, -30.0]
    product = xr.Dataset(
        {
            "mode_regression_pattern": (("L", "lat", "lon"), [[[1.0, -1.0], [0.5, -0.5]]]),
            "mode_regression_significant": (("L", "lat", "lon"), [[[1, 0], [0, 1]]]),
            "target_month": ("L", [4]),
            "mode_pattern_rmse_reference": ("L", [0.25]),
            "mode_pattern_pcc_reference": ("L", [0.75]),
        },
        coords={"L": [3], "lat": lat, "lon": lon}, attrs={"source": "NMME"},
    )
    reference = xr.Dataset(
        {"mode_pattern": (("target_month", "lat", "lon"), [[[0.8, -0.8], [0.4, -0.4]]])},
        coords={"target_month": [4], "lat": lat, "lon": lon},
    )

    path = plot_mode_domain_regression_figure(
        product, reference, tmp_path / "nao_pattern.png", mode="NAO", init_month=2, lead=3
    )

    assert path.is_file() and path.stat().st_size > 0


def test_mode_domain_regression_figure_rejects_grid_mismatch(tmp_path):
    product = xr.Dataset(
        {
            "mode_regression_pattern": (("L", "lat", "lon"), [[[1.0]]]),
            "mode_regression_significant": (("L", "lat", "lon"), [[[1]]]),
            "target_month": ("L", [4]),
            "mode_pattern_rmse_reference": ("L", [0.25]),
            "mode_pattern_pcc_reference": ("L", [0.75]),
        }, coords={"L": [3], "lat": [20.0], "lon": [-40.0]},
    )
    reference = xr.Dataset(
        {"mode_pattern": (("target_month", "lat", "lon"), [[[1.0]]])},
        coords={"target_month": [4], "lat": [25.0], "lon": [-40.0]},
    )
    with pytest.raises(ValueError, match="grids differ"):
        plot_mode_domain_regression_figure(
            product, reference, tmp_path / "unused.png", mode="NAO", init_month=2, lead=3
        )


def test_cross_source_teleconnection_figure_handles_reference_and_model_axes(tmp_path):
    lat, lon = [-30.0, 30.0], [0.0, 180.0]
    fields = {
        "mode_global_regression_pattern": (("axis", "lat", "lon"), [[[1.0, -1.0], [0.5, -0.5]]]),
        "mode_global_regression_significant": (("axis", "lat", "lon"), [[[1, 0], [0, 1]]]),
    }
    reference = xr.Dataset(fields, coords={"axis": [4], "lat": lat, "lon": lon}).rename(axis="target_month")
    model = xr.Dataset(fields, coords={"axis": [3], "lat": lat, "lon": lon}).rename(axis="L")

    path = plot_cross_source_teleconnection_figure(
        {"ERA5 reference": reference, "NMME": model},
        tmp_path / "nao_global_teleconnection.png",
        mode="NAO", lead=3, target_month=4, stipple_stride=1,
    )

    assert path.is_file() and path.stat().st_size > 0


def test_projected_mode_skill_uses_product_skill_axis_and_plots(tmp_path):
    years, leads = [1991, 1992, 1993, 1994], [3, 6]
    valid_time = xr.DataArray(
        np.array([[np.datetime64(f"{year}-04-15"), np.datetime64(f"{year}-07-15")] for year in years]),
        dims=("Y", "skill_L"), coords={"Y": years, "skill_L": leads},
    )
    product = xr.Dataset(
        {
            "mode_index_skill": (("Y", "skill_L", "M"), np.array([[[0.1, 0.3], [0.2, 0.4]]] * 4)),
            "valid_time_skill": valid_time,
            "target_month_skill": ("skill_L", [4, 7]),
        }, coords={"Y": years, "skill_L": leads, "M": ["a", "b"]}, attrs={"mode": "PDO", "source": "NMME"},
    )
    reference = xr.Dataset(
        {"mode_index": ("time", np.arange(8, dtype=float))},
        coords={"time": np.array([
            "1991-04-15", "1991-07-15", "1992-04-15", "1992-07-15",
            "1993-04-15", "1993-07-15", "1994-04-15", "1994-07-15",
        ], dtype="datetime64[ns]")},
    )

    cache = build_projected_mode_skill_cache(product, reference, detrend=False)
    path = plot_projected_mode_skill_figure(cache, tmp_path / "pdo_skill.png", mode="PDO", init_month=2)

    assert list(cache.L.values) == leads
    assert list(cache.target_month.values) == [4, 7]
    assert cache.attrs["skill_axis"].startswith("product mode_index_skill")
    assert path.is_file() and path.stat().st_size > 0


def test_nao_station_eof_skill_and_summary_match_notebook_shape(tmp_path):
    years, leads = [1991, 1992, 1993, 1994], [3]
    valid_time = xr.DataArray(
        np.array([[np.datetime64(f"{year}-04-15")] for year in years]),
        dims=("Y", "L"), coords={"Y": years, "L": leads},
    )
    eof = np.array([0.2, 0.4, 0.8, 1.0])[:, None, None]
    station = np.array([1.0, 0.7, 0.3, 0.1])[None, :, None]
    product = xr.Dataset({
        "mode_index": (("Y", "L", "M"), eof),
        "nao_station": (("M", "Y", "L"), station),
        "valid_time": valid_time,
        "target_month": ("L", [4]),
    }, attrs={"source": "NMME"})
    reference = xr.Dataset({
        "mode_index": ("time", [0.1, 0.5, 0.7, 1.1]),
        "nao_station": ("time", [0.9, 0.8, 0.2, 0.0]),
    }, coords={"time": np.array(["1991-04-15", "1992-04-15", "1993-04-15", "1994-04-15"], dtype="datetime64[ns]")})

    cache = build_nao_station_eof_skill_cache(product, reference, detrend=False)
    summary = skill_summary_dataframe(cache, init_month=2)
    csv_path, text_path = write_skill_summary_table(cache, tmp_path, init_month=2)

    assert list(cache.method.values) == ["eof", "station"]
    assert list(summary.method) == ["eof", "station"]
    assert set(summary) >= {"acc", "pval", "nrmse", "msss", "rpc", "sample_count"}
    assert csv_path.is_file() and text_path.is_file()


def test_ensemble_spread_common_skill_uses_exact_common_cohort():
    def product(labels, offset):
        base = _product(leads=(1, 2, 3)).isel(M=[0]).reindex(M=labels)
        values = np.empty((base.sizes["Y"], base.sizes["L"], len(labels)))
        for member in range(len(labels)):
            values[:, :, member] = np.arange(base.sizes["Y"])[:, None] + offset * member
        base["mode_index"] = xr.DataArray(values, dims=("Y", "L", "M"), coords={"Y": base.Y, "L": base.L, "M": labels})
        return base
    products = {
        "E3SM": product(["e1", "e2"], 0.1),
        "CESM-SMYLE": product(["s1", "s2"], 0.2),
        "NMME": product(["CMC1:m1", "CMC1:m2", "CMC2:m1", "CMC2:m2"], 0.3),
    }
    observation = xr.DataArray([0.0, 1.0, 2.0, 3.0], dims="time", coords={"time": np.array(["1991-04-15", "1992-04-15", "1993-04-15", "1994-04-15"], dtype="datetime64[ns]")})
    cache = build_ensemble_spread_common_skill_cache(products, observation)
    assert cache.attrs["period_alignment"] == "identical source-intersection target times by lead"
    assert list(cache.L.values) == [1, 2, 3]
    assert int(cache.spread_unit_count.sel(source="NMME").isel(L=0)) == 2
    assert np.all(cache.sample_count.sel(source="NMME").values == 4)


# ---------------------------------------------------------------------------
# Regional SST indices (build_regional_sst_index_product / _observation)
# ---------------------------------------------------------------------------
from esp_lab.diagnostics.sst_index import BASIC_REGIONS  # noqa: E402
from workflows.modes_of_variability.cross_source import (  # noqa: E402
    DERIVED_SST_INDICES,
    build_regional_sst_index_observation,
    build_regional_sst_index_product,
    write_ensemble_spread_native_skill_cache,
)

_SST_LAT = np.arange(-28.75, 30.0, 2.5)
_SST_LON = np.arange(1.25, 360.0, 2.5)
_SST_YEARS = [1991, 1992, 1993, 1994, 1995]
_SST_LEADS = np.arange(1, 6)
_SST_REFERENCE = {
    "eof_strategy": "fixed_obs_projection",
    "common_basis_reference": "obs",
    "eof_reference_period": "1950-2020",
}


def _sst_field(members, seed=0):
    rng = np.random.default_rng(seed)
    shape = (len(_SST_YEARS), _SST_LEADS.size, len(members), _SST_LAT.size, _SST_LON.size)
    return xr.DataArray(
        rng.normal(size=shape), dims=("Y", "L", "M", "lat", "lon"),
        coords={"Y": _SST_YEARS, "L": _SST_LEADS, "M": list(members), "lat": _SST_LAT, "lon": _SST_LON},
        attrs={"units": "degC"},
    )


def _sst_valid_time():
    return xr.DataArray(
        [[DatetimeNoLeap(y + (10 + lead) // 12, (10 + lead) % 12 + 1, 15) for lead in _SST_LEADS] for y in _SST_YEARS],
        dims=("Y", "L"), coords={"Y": _SST_YEARS, "L": _SST_LEADS},
    )


def _sst_forecast_cache():
    return xr.Dataset(
        {"SST_anom": _sst_field(["1", "2", "3"]), "valid_time": _sst_valid_time()}, attrs=dict(_SST_REFERENCE)
    )


_NMME_METHOD = "per-model lead-dependent drift removed over the climatology years"


def _sst_nmme_cache():
    raw = _sst_field(["A:1", "A:2", "B:1", "B:2"], seed=1) + 20.0
    raw = raw + xr.DataArray([0.0, 0.0, 3.0, 3.0], dims="M", coords={"M": raw.M})  # model B is 3 K warmer
    anomaly = xr.concat(
        [raw.sel(M=members) - raw.sel(M=members).mean(("Y", "M")) for members in (["A:1", "A:2"], ["B:1", "B:2"])],
        dim="M",
    ).assign_attrs(units="degC", nmme_anomaly_method=_NMME_METHOD)
    return xr.Dataset(
        {"SST_anom": anomaly, "SST": raw, "valid_time": _sst_valid_time()},
        attrs={**_SST_REFERENCE, "source": "NMME"},
    )


def _sst_observation_cache(lat=_SST_LAT):
    rng = np.random.default_rng(2)
    time = [DatetimeNoLeap(1990 + i // 12, i % 12 + 1, 15) for i in range(60)]
    data = xr.DataArray(
        rng.normal(size=(60, _SST_LAT.size, _SST_LON.size)), dims=("time", "lat", "lon"),
        coords={"time": time, "lat": _SST_LAT, "lon": _SST_LON}, attrs={"units": "degC"},
    )
    return xr.Dataset({"SST_anom": data.sel(lat=lat)})


def _manual_region_mean(field, index):
    lon_min, lon_max, lat_min, lat_max = BASIC_REGIONS[index]["lonlat"]
    lon_mask = (
        (field.lon >= lon_min) & (field.lon <= lon_max)
        if lon_min <= lon_max else (field.lon >= lon_min) | (field.lon <= lon_max)
    )
    region = field.where((field.lat >= lat_min) & (field.lat <= lat_max) & lon_mask)
    return region.weighted(np.cos(np.deg2rad(field.lat))).mean(("lat", "lon"))


@pytest.mark.parametrize("index", ["Nino3.4", "TSA", "AtlNino"])
def test_regional_sst_index_matches_cosine_weighted_region_mean(index):
    cache = _sst_forecast_cache()
    product = build_regional_sst_index_product(cache, index=index)
    expected = _manual_region_mean(cache["SST_anom"], index).transpose("Y", "L", "M")
    np.testing.assert_allclose(product["mode_index"].values, expected.values, rtol=1e-12)
    assert product["mode_index"].dims == ("Y", "L", "M")
    assert product.attrs["index"] == index
    assert product.attrs["regional_anomaly_method"] == "source SST_anom field"
    assert product.attrs["eof_reference_period"] == "1950-2020"


def test_regional_sst_index_accepts_north_to_south_latitudes():
    ascending = build_regional_sst_index_observation(_sst_observation_cache(), index="Nino3.4")
    descending = build_regional_sst_index_observation(
        _sst_observation_cache(lat=_SST_LAT[::-1]), index="Nino3.4"
    )
    np.testing.assert_allclose(descending.values, ascending.values, rtol=1e-12)


def test_iod_is_west_minus_east_and_oni_smooths_nino34_along_lead():
    cache = _sst_forecast_cache()
    iod = build_regional_sst_index_product(cache, index="IOD")["mode_index"]
    assert iod.attrs["index"] == "IOD"
    nino34 = build_regional_sst_index_product(cache, index="Nino3.4")["mode_index"]
    oni = build_regional_sst_index_product(cache, index="ONI")["mode_index"]
    expected = nino34.rolling(L=3, center=True, min_periods=1).mean()
    np.testing.assert_allclose(oni.values, expected.transpose("Y", "L", "M").values, rtol=1e-12)
    assert oni.attrs["index"] == "ONI"


def test_oni_observation_smooths_along_time():
    cache = _sst_observation_cache()
    nino34 = build_regional_sst_index_observation(cache, index="Nino3.4")
    oni = build_regional_sst_index_observation(cache, index="ONI")
    np.testing.assert_allclose(oni.values, nino34.rolling(time=3, center=True, min_periods=1).mean().values)


def test_nmme_index_uses_each_models_own_drift_corrected_anomaly():
    product = build_regional_sst_index_product(_sst_nmme_cache(), index="Nino3.4")
    index = product["mode_index"]
    for model in ("A", "B"):
        members = [m for m in index.M.values if str(m).startswith(model + ":")]
        np.testing.assert_allclose(index.sel(M=members).mean(("Y", "M")).values, 0.0, atol=1e-12)
    assert product.attrs["regional_anomaly_method"].startswith("per-model")


def test_nmme_index_requires_model_member_labels():
    cache = _sst_nmme_cache().assign_coords(M=["1", "2", "3", "4"])
    with pytest.raises(ValueError, match="MODEL:MEMBER"):
        build_regional_sst_index_product(cache, index="Nino3.4")


def test_nmme_index_rejects_a_pooled_multi_model_anomaly():
    cache = _sst_nmme_cache()
    cache["SST_anom"].attrs.pop("nmme_anomaly_method")
    with pytest.raises(ValueError, match="drift removed separately"):
        build_regional_sst_index_product(cache, index="Nino3.4")


@pytest.mark.parametrize("index", DERIVED_SST_INDICES)
@pytest.mark.parametrize("cache_factory", [_sst_forecast_cache, _sst_nmme_cache])
def test_derived_indices_build_for_products_and_observations(index, cache_factory):
    cache = cache_factory()
    product = build_regional_sst_index_product(cache, index=index)
    assert product["mode_index"].dims == ("Y", "L", "M")
    assert np.isfinite(product["mode_index"].values).all()
    assert product.attrs["index"] == index
    expected_method = "per-model" if cache.attrs.get("source") == "NMME" else "source SST_anom field"
    assert product.attrs["regional_anomaly_method"].startswith(expected_method)
    observation = build_regional_sst_index_observation(_sst_observation_cache(), index=index)
    assert observation.dims == ("time",) and observation.attrs["index"] == index


def test_tni_is_standardized_difference_smoothed_over_five_months():
    cache = _sst_observation_cache()
    nino12 = build_regional_sst_index_observation(cache, index="Nino12")
    nino4 = build_regional_sst_index_observation(cache, index="Nino4")
    expected = (nino12 / nino12.std("time") - nino4 / nino4.std("time")).rolling(
        time=5, center=True, min_periods=1
    ).mean()
    tni = build_regional_sst_index_observation(cache, index="TNI")
    np.testing.assert_allclose(tni.values, expected.values, rtol=1e-12)


def test_regional_sst_index_rejects_unknown_index_and_missing_anomaly():
    with pytest.raises(ValueError, match="not a supported regional SST definition"):
        build_regional_sst_index_product(_sst_forecast_cache(), index="NotARegion")
    with pytest.raises(ValueError, match="SST_anom and valid_time"):
        build_regional_sst_index_product(_sst_forecast_cache().drop_vars("SST_anom"), index="Nino3.4")
    with pytest.raises(ValueError, match="lacks SST_anom"):
        build_regional_sst_index_observation(xr.Dataset(), index="TNI")


def test_ensemble_spread_cache_write_is_atomic_and_leaves_no_temporary(tmp_path, monkeypatch):
    from pathlib import Path

    import workflows.modes_of_variability.cross_source as cross_source

    cache = xr.Dataset({"acc": ("L", [0.5, 0.4])}, coords={"L": [1, 2]})
    monkeypatch.setattr(cross_source, "build_ensemble_spread_native_skill_cache", lambda *a, **k: cache)
    path = tmp_path / "nested" / "spread.nc"
    assert write_ensemble_spread_native_skill_cache({}, None, path) == path
    xr.testing.assert_identical(xr.open_dataset(path).load(), cache)
    assert sorted(p.name for p in path.parent.iterdir()) == ["spread.nc"]

    def partial_write(self, target, *args, **kwargs):
        Path(target).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", partial_write)
    with pytest.raises(OSError, match="disk full"):
        write_ensemble_spread_native_skill_cache({}, None, path)
    # the failed write leaves neither a temporary file nor a damaged destination
    assert sorted(p.name for p in path.parent.iterdir()) == ["spread.nc"]
    monkeypatch.undo()
    xr.testing.assert_identical(xr.open_dataset(path).load(), cache)
