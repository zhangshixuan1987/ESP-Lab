"""Shared SST normalization and fixed land-mask utilities."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import xarray as xr


SST_PREPROCESSING_VERSION = 1
LAND_MASK_VERSION = 1
DEFAULT_MIN_SST_DEGC = -13.15
DEFAULT_MAX_SST_DEGC = 56.85

_KELVIN_UNITS = {
    "k",
    "degk",
    "kelvin",
    "kelvin_scale",
    "degree_k",
    "degrees_k",
}
_CELSIUS_UNITS = {
    "c",
    "degc",
    "celsius",
    "degree_c",
    "degrees_c",
    "degree_celsius",
    "degrees_celsius",
}


def _unit_token(units: object) -> str:
    return str(units or "").strip().lower().replace(" ", "_").replace("°", "deg")


def _representative_value(data: xr.DataArray) -> float:
    """Return a cheap representative value for unit inference."""
    indexer = {
        dim: np.unique(
            np.linspace(0, size - 1, num=min(size, 4), dtype=int)
        ).tolist()
        for dim, size in data.sizes.items()
    }
    sample = data.isel(indexer).where(np.isfinite(data.isel(indexer)))
    # Dask does not implement a full-array nanmedian.  A mean is sufficient
    # here because Kelvin and Celsius SST differ by 273.15 degrees and the
    # sampled points are distributed across every dimension.
    return float(sample.mean(skipna=True).compute())


def normalize_sst_to_degc(data: xr.DataArray) -> xr.DataArray:
    """Convert SST to degrees Celsius without assuming that ``TS`` means Kelvin."""
    units = _unit_token(data.attrs.get("units"))
    if units in _KELVIN_UNITS:
        is_kelvin = True
        inference = "units"
    elif units in _CELSIUS_UNITS:
        is_kelvin = False
        inference = "units"
    else:
        sample = _representative_value(data)
        if not np.isfinite(sample):
            raise ValueError("Cannot infer SST units because the sampled values are all missing.")
        is_kelvin = sample > 150.0
        inference = f"sample_mean={sample:g}"

    result = data - 273.15 if is_kelvin else data.copy()
    result.attrs.update(data.attrs)
    result.attrs.update(
        units="degC",
        original_units=str(data.attrs.get("units", "unknown")),
        sst_unit_inference=inference,
    )
    return result


def mask_nonphysical_sst(
    data: xr.DataArray,
    *,
    minimum: float = DEFAULT_MIN_SST_DEGC,
    maximum: float = DEFAULT_MAX_SST_DEGC,
) -> xr.DataArray:
    """Mask missing and nonphysical SST after conversion to degrees Celsius."""
    result = data.where(np.isfinite(data) & (data >= minimum) & (data <= maximum))
    result.attrs.update(data.attrs)
    result.attrs.update(
        valid_sst_range=f"{minimum:g} <= SST <= {maximum:g} degC",
        sst_preprocessing_version=SST_PREPROCESSING_VERSION,
    )
    return result


def _coordinate_names(data: xr.DataArray) -> tuple[str, str]:
    lon_name = next(
        (name for name in ("lon", "longitude", "X") if name in data.coords),
        None,
    )
    lat_name = next(
        (name for name in ("lat", "latitude", "Y") if name in data.coords),
        None,
    )
    if lon_name is None or lat_name is None:
        raise ValueError("SST must provide longitude and latitude coordinates.")
    if data[lon_name].ndim != 1 or data[lat_name].ndim != 1:
        raise ValueError("Fixed SST land masks currently require a rectilinear grid.")
    return lon_name, lat_name


def _cached_mask_matches(
    dataset: xr.Dataset,
    data: xr.DataArray,
    *,
    source: str,
    lon_name: str,
    lat_name: str,
) -> bool:
    if "sftlf" not in dataset:
        return False
    if dataset.attrs.get("land_mask_version") != LAND_MASK_VERSION:
        return False
    if dataset.attrs.get("source_id") != source:
        return False
    for name in (lon_name, lat_name):
        if name not in dataset.coords or dataset[name].shape != data[name].shape:
            return False
        if not np.allclose(dataset[name], data[name], rtol=0.0, atol=1.0e-10):
            return False
    return True


def _natural_earth_land_mask(
    data: xr.DataArray,
    *,
    lon_name: str,
    lat_name: str,
) -> xr.DataArray:
    import regionmask

    regions = regionmask.defined_regions.natural_earth_v5_0_0
    land = regions.land_110
    mask = land.mask(data[lon_name], data[lat_name], wrap_lon=360)
    return mask.notnull().astype("int8").rename("sftlf")


def load_or_create_land_mask(
    data: xr.DataArray,
    path: str | Path,
    *,
    source: str,
    force: bool = False,
) -> xr.DataArray:
    """Load or atomically create a source- and grid-validated land mask."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lon_name, lat_name = _coordinate_names(data)
    lock_path = path.with_suffix(path.suffix + ".lock")

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if path.exists() and not force:
            try:
                with xr.open_dataset(path) as cached:
                    if _cached_mask_matches(
                        cached,
                        data,
                        source=source,
                        lon_name=lon_name,
                        lat_name=lat_name,
                    ):
                        return cached["sftlf"].load().astype(bool)
            except Exception:
                pass

        land_mask = _natural_earth_land_mask(
            data,
            lon_name=lon_name,
            lat_name=lat_name,
        )
        land_mask.attrs.update(
            long_name="Natural Earth land mask on the source SST grid",
            units="1",
            flag_values=np.asarray([0, 1], dtype="int8"),
            flag_meanings="ocean land",
        )
        output = land_mask.to_dataset()
        output.attrs.update(
            source_id=source,
            source="Natural Earth land_110",
            land_mask_version=LAND_MASK_VERSION,
        )
        temporary = path.with_name(f".{path.name}.tmp.{uuid4().hex}")
        try:
            output.to_netcdf(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return land_mask.astype(bool)


def prepare_sst(
    data: xr.DataArray,
    *,
    apply_land_mask: bool = True,
    land_mask_path: str | Path | None = None,
    source: str = "unknown",
    force_land_mask: bool = False,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Normalize, sanity-check, and optionally land-mask an SST field."""
    result = mask_nonphysical_sst(normalize_sst_to_degc(data))
    land_mask = None
    if apply_land_mask:
        if land_mask_path is None:
            raise ValueError("land_mask_path is required when apply_land_mask=True.")
        land_mask = load_or_create_land_mask(
            result,
            land_mask_path,
            source=source,
            force=force_land_mask,
        )
        prepared_attrs = dict(result.attrs)
        result = result.where(~land_mask)
        result.attrs.update(prepared_attrs)
        result.attrs.update(
            units="degC",
            valid_sst_range=(
                f"{DEFAULT_MIN_SST_DEGC:g} <= SST <= {DEFAULT_MAX_SST_DEGC:g} degC"
            ),
            sst_preprocessing_version=SST_PREPROCESSING_VERSION,
            sst_land_mask="true",
            sst_land_mask_file=str(Path(land_mask_path)),
        )
    return result, land_mask
