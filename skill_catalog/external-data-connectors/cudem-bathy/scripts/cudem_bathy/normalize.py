"""Normalize CUDEM elevations into FVCOM-friendly bathymetry fields."""

from __future__ import annotations

import numpy as np


def elevation_to_depth(elevation_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-down depth and wet mask from elevation in meters."""

    elevation = np.asarray(elevation_m, dtype=np.float64)
    finite = np.isfinite(elevation)
    wet = finite & (elevation < 0.0)
    depth = np.where(finite, np.where(wet, -elevation, 0.0), np.nan)
    return depth.astype(np.float32), wet


def finite_coverage_fraction(values: np.ndarray) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(arr)) / arr.size)
