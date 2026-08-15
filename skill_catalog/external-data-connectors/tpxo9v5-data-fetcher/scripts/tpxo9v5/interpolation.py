"""Complex-phasor interpolation for TPXO harmonic fields."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from .coordinates import unwrap_targets
from .io import HarmonicField


def complex_to_amplitude_phase(coefficient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert native A*exp(-i*phase) coefficients to amplitude and phase lag."""

    values = np.asarray(coefficient)
    amplitude = np.abs(values)
    phase = np.mod(np.rad2deg(np.arctan2(-values.imag, values.real)), 360.0)
    invalid = ~(np.isfinite(values.real) & np.isfinite(values.imag))
    amplitude = np.asarray(amplitude, dtype=float)
    phase = np.asarray(phase, dtype=float)
    amplitude[invalid] = np.nan
    phase[invalid] = np.nan
    return amplitude, phase


def interpolate_complex_field(
    field: HarmonicField,
    target_longitude: np.ndarray,
    target_latitude: np.ndarray,
    nearest_wet_max_degrees: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate complex coefficients with bounded nearest-wet fallback.

    Flags are 0 for linear interpolation, 1 for nearest-wet fallback, and 2 for
    unresolved or outside-coverage values.
    """

    target_lon = np.asarray(target_longitude, dtype=float).ravel()
    target_lat = np.asarray(target_latitude, dtype=float).ravel()
    if target_lon.shape != target_lat.shape:
        raise ValueError("Target longitude and latitude must have identical shapes.")
    if not np.all(np.isfinite(target_lon) & np.isfinite(target_lat)):
        raise ValueError("Target coordinates must be finite.")
    if nearest_wet_max_degrees < 0:
        raise ValueError("nearest_wet_max_degrees must be non-negative")

    lon = np.asarray(field.longitude, dtype=float)
    lat = np.asarray(field.latitude, dtype=float)
    coefficient = np.asarray(field.coefficient)
    lon_order = np.argsort(lon, kind="stable")
    lat_order = np.argsort(lat, kind="stable")
    lon = lon[lon_order]
    lat = lat[lat_order]
    coefficient = coefficient[:, lat_order, :][:, :, lon_order]
    if lon.size < 2 or lat.size < 2:
        raise ValueError("At least two source coordinates are required on each axis.")
    if np.any(np.diff(lon) <= 0) or np.any(np.diff(lat) <= 0):
        raise ValueError("Source coordinates must be unique after longitude unwrapping.")

    target_unwrapped = unwrap_targets(target_lon, lon)
    query = np.column_stack((target_lat, target_unwrapped))
    output = np.full((coefficient.shape[0], target_lon.size), np.nan + 1j * np.nan, dtype=np.complex128)
    flags = np.full(output.shape, 2, dtype=np.int8)
    source_lon, source_lat = np.meshgrid(lon, lat)

    for index in range(coefficient.shape[0]):
        values = coefficient[index]
        real = RegularGridInterpolator(
            (lat, lon), values.real, method="linear", bounds_error=False, fill_value=np.nan
        )(query)
        imag = RegularGridInterpolator(
            (lat, lon), values.imag, method="linear", bounds_error=False, fill_value=np.nan
        )(query)
        linear_valid = np.isfinite(real) & np.isfinite(imag)
        output[index, linear_valid] = real[linear_valid] + 1j * imag[linear_valid]
        flags[index, linear_valid] = 0

        unresolved = ~linear_valid
        wet = np.isfinite(values.real) & np.isfinite(values.imag)
        if unresolved.any() and wet.any() and nearest_wet_max_degrees > 0:
            latitude_scale = max(0.05, float(np.cos(np.deg2rad(np.nanmean(target_lat)))))
            source_points = np.column_stack((source_lat[wet], source_lon[wet] * latitude_scale))
            target_points = np.column_stack((target_lat[unresolved], target_unwrapped[unresolved] * latitude_scale))
            tree = cKDTree(source_points)
            distance, nearest = tree.query(target_points, k=1)
            bounded = distance <= nearest_wet_max_degrees
            unresolved_indices = np.where(unresolved)[0]
            wet_values = values[wet]
            chosen = unresolved_indices[bounded]
            output[index, chosen] = wet_values[nearest[bounded]]
            flags[index, chosen] = 1

    return output, flags
