"""Lazy unit conversions shared by prediction-skill workflows."""

from __future__ import annotations


def convert_kelvin_to_celsius(data):
    """Convert kelvin to degrees Celsius while preserving lazy evaluation."""
    converted = data - 273.15
    converted.attrs["units"] = r"$^\circ$C"
    return converted


def convert_precip_mps_to_mmday(data):
    """Convert precipitation from metres per second to millimetres per day."""
    converted = data * (1000.0 * 86400.0)
    converted.attrs["units"] = "mm/day"
    return converted


def convert_pa_to_hpa(data):
    """Convert pressure from pascals to hectopascals."""
    converted = data * 1.0e-2
    converted.attrs["units"] = "hPa"
    return converted


def no_unit_conversion(data, units=None):
    """Return a lazy copy and optionally standardize its unit attribute."""
    converted = data * 1.0
    if units is not None:
        converted.attrs["units"] = units
    return converted


__all__ = [
    "convert_kelvin_to_celsius",
    "convert_pa_to_hpa",
    "convert_precip_mps_to_mmday",
    "no_unit_conversion",
]
