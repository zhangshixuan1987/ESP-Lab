import json
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr
import cftime

from esp_lab.paths import NMME_DIAG_DIR, NMME_FIXED_DIR
from workflows import modes_of_variability_core as core
from scripts.run_process_modes_of_variability import (
    add_skill_lead_subset,
    cached_product_matches,
    configuration_signature,
    validate_args,
    write_manifest,
)


def _args(**overrides):
    values = {
        "start_year": 1980,
        "end_year": 2018,
        "obs_start_year": 1979,
        "obs_end_year": 2019,
        "eof_reference_start_year": 1980,
        "eof_reference_end_year": 2018,
        "eof_reference_source": "obs",
        "clim_start": 1981,
        "clim_end": 2010,
        "init_months": [5, 11],
        "monthly_nlead": 24,
        "target_dlat": 2.5,
        "target_dlon": 2.5,
        "regrid_method": "conservative",
        "no_periodic": False,
        "dask_workers": 8,
        "e3sm_nens": 10,
        "smyle_nens": 20,
        "e3sm_data_dir": "/data/e3sm",
        "e3sm_case_prefix": "case",
        "e3sm_engine": "netcdf4",
        "e3sm_grid": "180x360_aave",
        "smyle_benchmark_dir": "/data/smyle",
        "nmme_root": "/data/nmme",
        "nmme_models": ["model-a"],
        "nmme_field": "auto",
        "nmme_chunks": "",
        "nmme_sst_land_mask": True,
        "nmme_fixed_dir": "/data/fixed",
        "obs_dir": "/data/obs",
        "eof_scaling": True,
        "remove_domain_mean": True,
        "eof_bootstrap_iterations": 0,
        "eof_bootstrap_seed": 42,
        "eof_bootstrap_confidence": 0.90,
        "eof_strategy": "fixed_obs_projection",
        "regression_confidence": 0.95,
        "legacy_nao_layout": False,
        "modes": ["NAO"],
        "sources": ["obs"],
    }
    values.update(overrides)
    return Namespace(**values)


def _settings():
    return {
        "mode": "NAO",
        "eof_number": 1,
        "domain_mode": "NAO",
        "field": "PSL",
        "frequency": "seasonal",
        "obs_product": "ERA5",
        "obs_var": "psl",
    }


def test_configuration_signature_invalidates_changed_inputs():
    baseline = configuration_signature(
        "e3sm", _settings(), _args(), include_mode=True
    )
    changed = configuration_signature(
        "e3sm", _settings(), _args(monthly_nlead=18), include_mode=True
    )
    assert baseline != changed


def test_nmme_sst_mask_version_invalidates_only_temperature_fields():
    pressure = json.loads(
        configuration_signature("nmme", _settings(), _args(), include_mode=False)
    )
    temperature_settings = {
        **_settings(),
        "mode": "PDO",
        "domain_mode": "PDO",
        "field": "SST",
        "frequency": "monthly",
        "obs_product": "HadISST2",
        "obs_var": "sst",
    }
    temperature = json.loads(
        configuration_signature(
            "nmme", temperature_settings, _args(), include_mode=False
        )
    )

    assert "nmme_sst_mask_version" not in pressure
    assert temperature["nmme_sst_mask_version"] == (
        core.nmme_access.NMME_SST_MASK_VERSION
    )
    assert temperature["nmme_sst_land_mask"] is True
    assert temperature["nmme_fixed_dir"] == "/data/fixed"


def test_nmme_chunk_choice_does_not_invalidate_products():
    native = configuration_signature(
        "nmme", _settings(), _args(nmme_chunks=""), include_mode=False
    )
    custom = configuration_signature(
        "nmme",
        _settings(),
        _args(nmme_chunks="S:1,L:1,Y:45,X:90"),
        include_mode=False,
    )

    assert native == custom


def test_cached_product_accepts_legacy_nmme_chunk_signature(tmp_path):
    expected = configuration_signature(
        "nmme", _settings(), _args(), include_mode=False
    )
    legacy_payload = json.loads(expected)
    legacy_payload["nmme_chunks"] = "S:12,L:12,Y:45,X:72"
    path = tmp_path / "legacy-nmme-field.nc"
    xr.Dataset(
        {"value": ("x", [1.0])},
        attrs={"field_configuration": json.dumps(legacy_payload)},
    ).to_netcdf(path)

    assert cached_product_matches(path, "field_configuration", expected)


def test_nmme_products_use_canonical_uppercase_directory(tmp_path):
    settings = {
        **_settings(),
        "mode": "PDO",
        "field": "SST",
        "frequency": "monthly",
    }

    field_path, index_path = core.product_paths(
        tmp_path,
        "PDO",
        "nmme",
        2,
        "2p5x2p5deg",
        settings,
        False,
    )

    expected_root = tmp_path / "NMME" / "modes_variability"
    assert field_path.parent == expected_root / "fields"
    assert index_path.parent == expected_root / "modes" / "pdo" / "indices"
    assert field_path.name.startswith("nmme_init02_")


def test_nmme_fixed_fields_live_under_canonical_nmme_directory():
    assert NMME_FIXED_DIR == NMME_DIAG_DIR / "fixed"


def test_nmme_fields_are_materialized_one_model_at_a_time(monkeypatch):
    calls = []

    def fake_nmme_field_dataset(init_month, settings, args):
        model = args.nmme_models[0]
        calls.append(list(args.nmme_models))
        years = np.array([2000, 2001])
        values = np.full((2, 1, 1, 2, 2), len(calls), dtype="float32")
        valid_time = xr.DataArray(
            np.array(
                [[cftime.DatetimeNoLeap(int(year), init_month, 15)] for year in years],
                dtype=object,
            ),
            dims=("Y", "L"),
            coords={"Y": years, "L": [1]},
        )
        return xr.Dataset(
            {
                "SST": xr.DataArray(
                    values,
                    dims=("Y", "L", "M", "lat", "lon"),
                    coords={
                        "Y": years,
                        "L": [1],
                        "M": [f"{model}:M001"],
                        "lat": [-1.0, 1.0],
                        "lon": [0.0, 2.0],
                    },
                    attrs={"units": "degC"},
                ),
                "time": valid_time,
            }
        )

    class IdentityRegridder:
        def __call__(self, dataset):
            return dataset

    monkeypatch.setattr(core, "nmme_field_dataset", fake_nmme_field_dataset)
    monkeypatch.setattr(
        core.regrid,
        "make_regridder",
        lambda *args, **kwargs: IdentityRegridder(),
    )

    settings = {
        **_settings(),
        "mode": "PDO",
        "field": "SST",
        "archive_field": "TS",
        "frequency": "monthly",
    }
    args = _args(
        start_year=2000,
        end_year=2001,
        clim_start=2000,
        clim_end=2001,
        monthly_nlead=1,
        nmme_models=["model-a", "model-b"],
    )
    destination = xr.Dataset(coords={"lat": [-1.0, 1.0], "lon": [0.0, 2.0]})

    result = core.model_field_dataset("nmme", 5, settings, args, destination)

    assert calls == [["model-a"], ["model-b"]]
    assert list(result.M.values) == ["model-a:M001", "model-b:M001"]
    assert result["SST"].dtype == np.float32


def test_reference_ocean_mask_accepts_cli_resolution_tokens(monkeypatch):
    requested = []

    class FakeLand:
        def mask(self, lon, lat):
            return xr.DataArray(
                np.full((lat.size, lon.size), np.nan),
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
            )

    class FakeNaturalEarth:
        def __getattr__(self, name):
            requested.append(name)
            return FakeLand()

    class FakeDefinedRegions:
        natural_earth_v5_0_0 = FakeNaturalEarth()

    class FakeRegionmask:
        defined_regions = FakeDefinedRegions()

    monkeypatch.setitem(__import__("sys").modules, "regionmask", FakeRegionmask())
    data = xr.DataArray(
        np.ones((2, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [-1.0, 1.0], "lon": [0.0, 2.0]},
    )

    for resolution in ("110m", "50m", "10m"):
        mask = core.generate_reference_ocean_mask(
            data,
            natural_earth_resolution=resolution,
        )
        assert bool(mask.all())

    assert requested == ["land_110", "land_50", "land_10"]


def test_add_skill_lead_subset_keeps_full_monthly_leads():
    dataset = xr.Dataset(
        {
            "mode_index": (
                ("Y", "L", "M"),
                np.arange(2 * 12 * 3, dtype=float).reshape(2, 12, 3),
            ),
            "valid_time": (
                ("Y", "L"),
                np.array(
                    [
                        np.arange("2000-11", "2001-11", dtype="datetime64[M]"),
                        np.arange("2001-11", "2002-11", dtype="datetime64[M]"),
                    ]
                ),
            ),
            "target_month": ("L", [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        },
        coords={"Y": [2000, 2001], "L": np.arange(1, 13), "M": [1, 2, 3]},
    )

    result = add_skill_lead_subset(dataset, "NPGO")

    assert result.sizes["L"] == 12
    assert result.sizes["skill_L"] == 4
    np.testing.assert_array_equal(result["mode_index_skill"].skill_L, [3, 6, 9, 12])
    np.testing.assert_array_equal(result["target_month_skill"], [1, 4, 7, 10])
    xr.testing.assert_equal(
        result["mode_index_skill"].rename(skill_L="L"),
        dataset["mode_index"].sel(L=[3, 6, 9, 12]),
    )


def test_cached_product_requires_matching_signature(tmp_path):
    signature = configuration_signature(
        "obs", _settings(), _args(), include_mode=False
    )
    path = tmp_path / "field.nc"
    xr.Dataset({"value": ("x", [1.0])}, attrs={"field_configuration": signature}).to_netcdf(
        path
    )

    assert cached_product_matches(path, "field_configuration", signature)
    assert not cached_product_matches(
        path, "field_configuration", signature + "-changed"
    )


def test_validate_args_rejects_incompatible_ranges():
    validate_args(_args())

    for invalid in (
        _args(clim_start=1970),
        _args(obs_end_year=2000),
        _args(monthly_nlead=0),
        _args(target_dlat=0),
        _args(e3sm_nens=0),
        _args(eof_bootstrap_iterations=-1),
        _args(eof_bootstrap_confidence=1),
        _args(eof_reference_start_year=1970),
        _args(eof_reference_end_year=2025),
    ):
        try:
            validate_args(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected validation failure for {invalid}")


def test_atomic_to_netcdf_uses_independent_temporary_files(tmp_path):
    destination = tmp_path / "shared.nc"
    temporary_paths = []

    class FakeDataset:
        def to_netcdf(self, path, encoding):
            temporary_paths.append(Path(path))
            Path(path).write_text("complete")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(core.atomic_to_netcdf, FakeDataset(), destination)
            for _ in range(4)
        ]
        for future in futures:
            future.result()

    assert destination.read_text() == "complete"
    assert len(set(temporary_paths)) == 4
    assert not list(tmp_path.glob("*.tmp.*"))


def test_manifest_merge_preserves_parallel_task_products(tmp_path):
    args = _args(
        outdir=str(tmp_path),
        sources=["obs", "e3sm", "smyle"],
        merge_manifest=True,
    )
    station = core.StationNaoDefinition()
    nao = _settings()
    pdo = {
        **_settings(),
        "mode": "PDO",
        "field": "SST",
        "frequency": "monthly",
        "domain_mode": "PDO",
        "obs_product": "HadISST2",
        "obs_var": "sst",
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                write_manifest,
                args,
                {"NAO": nao},
                {"NAO:reference": {"status": "ok"}},
                station,
            ),
            executor.submit(
                write_manifest,
                args,
                {"PDO": pdo},
                {"PDO:reference": {"status": "ok"}},
                station,
            ),
        ]
        paths = [future.result() for future in futures]

    manifest = json.loads(paths[0].read_text())
    assert set(manifest["modes"]) == {"NAO", "PDO"}
    assert set(manifest["products"]) == {"NAO:reference", "PDO:reference"}


def test_eof_analysis_retries_svd_nonconvergence(monkeypatch):
    attempts = []

    def fake_eof_analysis():
        attempts.append(True)
        if len(attempts) == 1:
            raise ValueError("error encountered in SVD")
        values = np.linalg.svd(np.eye(2), compute_uv=False)
        return values

    def failing_numpy_svd(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(core, "eof_analysis_get_variance_mode", fake_eof_analysis)
    monkeypatch.setattr(core, "_NUMPY_SVD", failing_numpy_svd)

    result = core.eof_analysis_with_svd_fallback()

    assert len(attempts) == 2
    np.testing.assert_allclose(result, [1.0, 1.0])


def test_eof_ready_dataset_applies_complete_case_spatial_mask():
    field = xr.DataArray(
        [
            [[1.0, 2.0, np.inf, 8.0]],
            [[3.0, np.nan, 4.0, 9.0]],
            [[5.0, 6.0, 7.0, 10.0]],
        ],
        dims=("time", "lat", "lon"),
        coords={
            "time": [0, 1, 2],
            "lat": [0.0],
            "lon": [0.0, 1.0, 2.0, 3.0],
        },
        name="mode_field",
    )

    result = core.eof_ready_dataset(field.to_dataset(), "mode_field")

    assert result["mode_field"].sel(lon=0.0).notnull().all()
    assert result["mode_field"].sel(lon=1.0).isnull().all()
    assert result["mode_field"].sel(lon=2.0).isnull().all()
    assert result["mode_field"].sel(lon=3.0).notnull().all()
