"""Longitude, latitude, and rectangular TPXO grid helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisSelection:
    """Indices and monotonically increasing coordinates for one axis."""

    indices: np.ndarray
    values: np.ndarray


def normalize_longitude_360(values: np.ndarray | float) -> np.ndarray:
    """Normalize longitude to [0, 360), preserving arrays."""

    return np.mod(np.asarray(values, dtype=float), 360.0)


def normalize_longitude_180(values: np.ndarray | float) -> np.ndarray:
    """Normalize longitude to [-180, 180)."""

    return (normalize_longitude_360(values) + 180.0) % 360.0 - 180.0


def unwrap_interval(west: float, east: float, padding: float = 0.0) -> tuple[float, float]:
    """Return a positive, possibly dateline-crossing interval in unwrapped degrees."""

    if padding < 0:
        raise ValueError("padding must be non-negative")
    if abs(float(east) - float(west)) >= 359.999999:
        return 0.0, 360.0
    start = float(normalize_longitude_360(west))
    stop = float(normalize_longitude_360(east))
    if stop < start:
        stop += 360.0
    start -= padding
    stop += padding
    if stop - start >= 359.999999:
        return 0.0, 360.0
    return start, stop


def minimal_longitude_interval(values: np.ndarray, padding: float = 0.0) -> tuple[float, float]:
    """Find the shortest circular interval containing all finite longitudes."""

    lon = np.unique(normalize_longitude_360(np.asarray(values, dtype=float)[np.isfinite(values)]))
    if lon.size == 0:
        raise ValueError("No finite target longitudes were supplied.")
    if lon.size == 1:
        return float(lon[0] - padding), float(lon[0] + padding)
    ordered = np.sort(lon)
    gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
    gap_index = int(np.argmax(gaps))
    start = float(ordered[(gap_index + 1) % ordered.size])
    stop = float(ordered[gap_index])
    if stop < start:
        stop += 360.0
    start -= padding
    stop += padding
    if stop - start >= 359.999999:
        return 0.0, 360.0
    return start, stop


def select_longitudes(
    coordinates: np.ndarray,
    west: float,
    east: float,
    padding: float = 0.0,
) -> AxisSelection:
    """Select native longitude indices and return ordered unwrapped coordinates."""

    raw = np.asarray(coordinates, dtype=float).ravel()
    start, stop = unwrap_interval(west, east, padding)
    normalized = normalize_longitude_360(raw)
    if stop - start >= 359.999999:
        unwrapped = normalized.copy()
        # TPXO commonly stores the final longitude as 360 rather than zero.
        unwrapped[(np.isclose(normalized, 0.0)) & (raw > 180.0)] = 360.0
        order = np.argsort(unwrapped, kind="stable")
        return AxisSelection(order.astype(int), unwrapped[order])

    candidates = normalized + 360.0 * np.ceil((start - normalized) / 360.0)
    selected = np.where((candidates >= start - 1e-9) & (candidates <= stop + 1e-9))[0]
    if selected.size == 0:
        raise ValueError(f"Requested longitude interval {west}..{east} does not intersect the source grid.")
    order = np.argsort(candidates[selected], kind="stable")
    selected = selected[order]
    return AxisSelection(selected.astype(int), candidates[selected])


def select_latitudes(
    coordinates: np.ndarray,
    south: float,
    north: float,
    padding: float = 0.0,
) -> AxisSelection:
    """Select and order latitude coordinates."""

    if south > north:
        raise ValueError("south must not exceed north")
    raw = np.asarray(coordinates, dtype=float).ravel()
    selected = np.where((raw >= south - padding - 1e-9) & (raw <= north + padding + 1e-9))[0]
    if selected.size == 0:
        raise ValueError(f"Requested latitude interval {south}..{north} does not intersect the source grid.")
    order = np.argsort(raw[selected], kind="stable")
    selected = selected[order]
    return AxisSelection(selected.astype(int), raw[selected])


def unwrap_targets(values: np.ndarray, source_longitudes: np.ndarray) -> np.ndarray:
    """Map target longitudes onto the continuous branch used by a source subset."""

    target = normalize_longitude_360(values)
    source = np.asarray(source_longitudes, dtype=float)
    center = 0.5 * (float(np.nanmin(source)) + float(np.nanmax(source)))
    return target + 360.0 * np.round((center - target) / 360.0)


def spatial_span(longitude: np.ndarray, latitude: np.ndarray) -> dict[str, float]:
    """Return finite native and conventional longitude/latitude bounds."""

    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    lon180 = normalize_longitude_180(lon)
    return {
        "longitude_min_native": float(np.nanmin(lon)),
        "longitude_max_native": float(np.nanmax(lon)),
        "longitude_min_180": float(np.nanmin(lon180)),
        "longitude_max_180": float(np.nanmax(lon180)),
        "latitude_min": float(np.nanmin(lat)),
        "latitude_max": float(np.nanmax(lat)),
    }
