"""Unit tests for native MPAS-Ocean ELI calculation and workflows."""

from pathlib import Path
from types import SimpleNamespace

import cftime
import numpy as np
import pytest
import xarray as xr

from esp_lab.diagnostics.native_eli import (
    build_eli_mesh_subsets,
    build_native_eli_datasets,
    compute_native_eli_member,
    eli_from_sst_on_union,
    load_mpas_mesh_geometry,
    native_eli_output_is_current,
    process_native_eli_case,
    read_surface_sst_on_cells,
    weighted_mean_skipna_1d,
)


def _create_mock_mesh_file(path: Path, ncells: int = 100) -> Path:
    """Create a minimal mock MPAS mesh NetCDF file."""
    # Place some cells in equatorial Pacific (lat ~ 0, lon ~ 180 = pi rad)
    # and some in extra-tropical regions
    lat = np.linspace(-10.0, 10.0, ncells) * np.pi / 180.0
    lon = np.linspace(110.0, 300.0, ncells) * np.pi / 180.0
    area = np.full(ncells, 1.0e8, dtype=np.float64)

    ds = xr.Dataset(
        {
            "latCell": ("nCells", lat),
            "lonCell": ("nCells", lon),
            "areaCell": ("nCells", area),
        }
    )
    ds.to_netcdf(path)
    return path


def test_load_mpas_mesh_geometry(tmp_path):
    mesh_path = _create_mock_mesh_file(tmp_path / "mock_mesh.nc", ncells=20)
    lat, lon, area = load_mpas_mesh_geometry(mesh_path)

    assert lat.shape == (20,)
    assert lon.shape == (20,)
    assert area.shape == (20,)
    assert np.all(area > 0)
    assert np.all(lon >= 0) and np.all(lon < 360)


def test_build_eli_mesh_subsets(tmp_path):
    mesh_path = _create_mock_mesh_file(tmp_path / "mock_mesh.nc", ncells=50)
    lat, lon, area = load_mpas_mesh_geometry(mesh_path)

    (
        idx_union,
        eq_on_union,
        tropics_on_union,
        lon_eq,
        area_eq,
        area_tr,
    ) = build_eli_mesh_subsets(lat, lon, area)

    assert len(idx_union) > 0
    assert len(lon_eq) > 0
    assert len(area_eq) == len(lon_eq)
    assert len(area_tr) > 0
    assert np.all(lon_eq >= 120.0) and np.all(lon_eq <= 290.0)


def test_weighted_mean_skipna_1d():
    vals = np.array([20.0, 30.0, np.nan, 40.0])
    weights = np.array([1.0, 2.0, 5.0, 1.0])
    # Mean: (20*1 + 30*2 + 40*1) / (1 + 2 + 1) = (20 + 60 + 40) / 4 = 120 / 4 = 30.0
    result = weighted_mean_skipna_1d(vals, weights)
    assert np.isclose(result, 30.0)

    # All NaN returns NaN
    assert np.isnan(weighted_mean_skipna_1d(np.array([np.nan, np.nan]), np.array([1.0, 2.0])))


def test_eli_from_sst_on_union():
    # Construct a small union of 4 cells:
    # Cell 0: in tropics & in eq Pacific
    # Cell 1: in tropics & in eq Pacific
    # Cell 2: in tropics outside eq Pacific
    # Cell 3: in tropics outside eq Pacific
    eq_on_union = np.array([True, True, False, False])
    tropics_on_union = np.array([True, True, True, True])
    lon_eq = np.array([150.0, 200.0], dtype=np.float32)
    area_eq = np.array([1.0, 1.0], dtype=np.float64)
    area_tr = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # Tc = mean(sst_tr) = (25 + 30 + 20 + 25) / 4 = 25.0
    # In eq region: cell 0 has 25.0 (not > Tc), cell 1 has 30.0 (> Tc)
    # So warm cell is cell 1 only! Centroid = lon_eq[1] = 200.0
    sst = np.array([25.0, 30.0, 20.0, 25.0], dtype=np.float32)
    eli = eli_from_sst_on_union(sst, eq_on_union, tropics_on_union, lon_eq, area_eq, area_tr)
    assert np.isclose(eli, 200.0)


def test_read_surface_sst_on_cells(tmp_path):
    sst_file = tmp_path / "sample.mpaso.nc"
    ncells = 10
    # Create 3D dataset (Time, nVertLevels, nCells) in degC
    time = [0]
    levels = [0, 1, 2]
    sst_vals = np.full((1, 3, ncells), 27.0, dtype=np.float32)
    sst_vals[0, 0, 3] = -999.0  # unphysical value to be masked

    ds = xr.Dataset(
        {
            "timeMonthly_avg_activeTracers_temperature": (
                ("Time", "nVertLevels", "nCells"),
                sst_vals,
                {"units": "degC"},
            )
        }
    )
    ds.to_netcdf(sst_file)

    idx_union = np.array([0, 1, 2, 3])
    result = read_surface_sst_on_cells(sst_file, idx_union)
    assert result.shape == (4,)
    assert np.isclose(result[0], 27.0, atol=0.01)
    assert np.isnan(result[3])  # Unphysical value masked to NaN


def test_build_native_eli_datasets():
    lead_years = [1980, 1981]
    members = ["EN00", "EN01"]
    nlead = 6
    seasonal_leads = np.array([1, 4])  # 0-indexed center indices
    eli_mon = np.ones((2, nlead, 2), dtype=np.float32) * 180.0
    eli_seas = np.ones((2, len(seasonal_leads), 2), dtype=np.float32) * 180.0

    attrs = {"case_prefix": "TEST_CASE", "test_attr": 42}
    ds_mon, ds_seas = build_native_eli_datasets(
        eli_mon,
        eli_seas,
        lead_years=lead_years,
        init_month=5,
        members=members,
        nlead=nlead,
        seasonal_lead_indices=seasonal_leads,
        attrs_common=attrs,
    )

    assert ds_mon["eli"].shape == (2, 6, 2)
    assert ds_mon.attrs["frequency"] == "monthly"
    assert ds_mon.attrs["test_attr"] == 42
    assert "time" in ds_mon
    assert "member_id" in ds_mon
    assert ds_seas["eli"].shape == (2, 2, 2)
    assert ds_seas.attrs["frequency"] == "seasonal"


def test_native_eli_output_is_current(tmp_path):
    path = tmp_path / "test_eli.nc"
    ds = xr.Dataset({"eli": xr.DataArray([180.0], dims="Y")})
    ds.attrs = {"case_prefix": "CASE_A", "run_nens": 10}
    ds.to_netcdf(path)

    assert native_eli_output_is_current(path, {"case_prefix": "CASE_A", "run_nens": 10})
    assert not native_eli_output_is_current(path, {"case_prefix": "CASE_B", "run_nens": 10})
    assert not native_eli_output_is_current(tmp_path / "non_existent.nc", {"case_prefix": "CASE_A"})


def test_process_native_eli_case_end_to_end(tmp_path):
    # Setup mock mesh
    mesh_path = _create_mock_mesh_file(tmp_path / "mesh.nc", ncells=20)

    # Setup mock raw simulation dir
    sim_dir = tmp_path / "simulations"
    case_prefix = "CASE_MOCK"
    year = 1980
    init_month = 5
    case_name = f"{case_prefix}_{year}{init_month:02d}0100"
    member_dir = sim_dir / case_name / "EN00" / "archive" / "ocn" / "hist"
    member_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 monthly history files
    for month in range(1, 4):
        fpath = member_dir / f"{case_name}.EN00.mpaso.hist.am.timeSeriesStatsMonthly.1980-{month:02d}-01.nc"
        sst = np.full((1, 1, 20), 28.0 + month, dtype=np.float32)
        ds = xr.Dataset(
            {
                "timeMonthly_avg_activeTracers_temperature": (
                    ("Time", "nVertLevels", "nCells"),
                    sst,
                )
            }
        )
        ds.to_netcdf(fpath)

    outdir = tmp_path / "out" / "sst_index" / "timeseries"

    written = process_native_eli_case(
        case_prefix=case_prefix,
        data_dir=sim_dir,
        outdir=outdir,
        mesh_path=mesh_path,
        init_months=[5],
        year_start=1980,
        year_end=1980,
        nlead=3,
        nens=1,
        workers=1,
        force=True,
    )

    assert len(written) == 2
    mon_file, seas_file = written
    assert mon_file.is_file()
    assert seas_file.is_file()

    with xr.open_dataset(mon_file) as ds:
        assert ds["eli"].shape == (1, 3, 1)
        assert ds.attrs["frequency"] == "monthly"
        assert ds.attrs["source_grid"] == "native MPAS-Ocean"
