"""Domain-boundary construction for smooth FVCOM offshore boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bathymetry import BathymetryGrid


@dataclass(frozen=True)
class DomainBoundary:
    """Closed boundary polyline and open-boundary node positions."""

    lon: np.ndarray
    lat: np.ndarray
    open_indices: np.ndarray
    offshore_side: str

    @property
    def points(self) -> np.ndarray:
        return np.column_stack([self.lon, self.lat])


SIDE_ANGLES = {
    "east": 0.0,
    "north": 0.5 * np.pi,
    "west": np.pi,
    "south": 1.5 * np.pi,
}


def infer_offshore_side(bathy: BathymetryGrid, samples: int = 80) -> str:
    """Choose the bbox side with the largest median positive-down depth."""
    lon_min, lat_min, lon_max, lat_max = bathy.bbox
    sides = {
        "west": (
            np.full(samples, lon_min),
            np.linspace(lat_min, lat_max, samples),
        ),
        "east": (
            np.full(samples, lon_max),
            np.linspace(lat_min, lat_max, samples),
        ),
        "south": (
            np.linspace(lon_min, lon_max, samples),
            np.full(samples, lat_min),
        ),
        "north": (
            np.linspace(lon_min, lon_max, samples),
            np.full(samples, lat_max),
        ),
    }
    scores = {}
    for side, (lon, lat) in sides.items():
        depth = bathy.sample(lon, lat)
        finite = depth[np.isfinite(depth)]
        scores[side] = float(np.nanmedian(finite)) if finite.size else -np.inf
    return max(scores, key=scores.get)


def build_elliptical_domain(
    bathy: BathymetryGrid,
    offshore_side: str | None = None,
    buffer_fraction: float = 0.18,
    n_boundary: int = 192,
    open_arc_fraction: float = 0.28,
) -> DomainBoundary:
    """Build a smooth closed ellipse and tag the offshore arc as the open boundary."""
    offshore_side = offshore_side or infer_offshore_side(bathy)
    if offshore_side not in SIDE_ANGLES:
        raise ValueError(f"offshore_side must be one of {sorted(SIDE_ANGLES)}")

    lon_min, lat_min, lon_max, lat_max = bathy.bbox
    lon_c = 0.5 * (lon_min + lon_max)
    lat_c = 0.5 * (lat_min + lat_max)
    lon_radius = 0.5 * (lon_max - lon_min) * (1.0 + buffer_fraction)
    lat_radius = 0.5 * (lat_max - lat_min) * (1.0 + buffer_fraction)

    # Use endpoint=False so the polygon does not duplicate the first node.
    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    lon = lon_c + lon_radius * np.cos(theta)
    lat = lat_c + lat_radius * np.sin(theta)

    center_angle = SIDE_ANGLES[offshore_side]
    angular_distance = np.abs(np.angle(np.exp(1j * (theta - center_angle))))
    open_width = open_arc_fraction * np.pi
    open_indices = np.flatnonzero(angular_distance <= open_width)
    open_indices = _order_open_indices(open_indices, n_boundary)

    return DomainBoundary(
        lon=lon.astype(float),
        lat=lat.astype(float),
        open_indices=open_indices.astype(int),
        offshore_side=offshore_side,
    )


def _order_open_indices(indices: np.ndarray, n_boundary: int) -> np.ndarray:
    """Return contiguous arc indices in boundary order even when wrapping zero."""
    if len(indices) == 0:
        return indices
    indices = np.asarray(indices, dtype=int)
    gaps = np.diff(np.r_[indices, indices[0] + n_boundary])
    split = int(np.argmax(gaps))
    ordered = np.r_[indices[split + 1 :], indices[: split + 1]]
    return ordered % n_boundary
