"""Audit compatibility between boundary targets and an interior size field."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .size_field import recorded_size_interpolator


SCHEMA_VERSION = "fvcom_boundary_interior_size_contract_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        label: float(np.quantile(values, quantile))
        for label, quantile in (
            ("minimum", 0.0),
            ("p01", 0.01),
            ("p05", 0.05),
            ("p50", 0.50),
            ("p95", 0.95),
            ("p99", 0.99),
            ("maximum", 1.0),
        )
    }


def diagnose_boundary_size_contract(
    boundary_geojson_path: str | Path,
    size_field_netcdf_path: str | Path,
    *,
    ratio_tolerance: float = 2.0,
) -> dict[str, Any]:
    """Compare source boundary spacing with the 2-D field at source vertices."""

    boundary_path = Path(boundary_geojson_path).expanduser().resolve()
    field_path = Path(size_field_netcdf_path).expanduser().resolve()
    if ratio_tolerance <= 1.0:
        raise ValueError("ratio_tolerance must be greater than one")
    document = json.loads(boundary_path.read_text(encoding="utf-8"))
    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("boundary GeoJSON has no point features")
    coordinates = np.asarray(
        [feature["geometry"]["coordinates"][:2] for feature in features],
        dtype=float,
    )
    boundary_target = np.asarray(
        [feature["properties"]["target_spacing_m"] for feature in features],
        dtype=float,
    )
    if (
        coordinates.shape != (len(features), 2)
        or not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(boundary_target))
        or np.any(boundary_target <= 0.0)
    ):
        raise ValueError("boundary coordinates or targets are invalid")

    with xr.open_dataset(field_path) as dataset:
        if dataset.attrs.get("schema_version") != "fvcom_size_field_v4":
            raise ValueError("size field is not fvcom_size_field_v4")
        if "mesh_size_m" not in dataset:
            raise ValueError("size field has no mesh_size_m variable")
        lon = np.asarray(dataset["lon"].values, dtype=float)
        lat = np.asarray(dataset["lat"].values, dtype=float)
        field = np.asarray(
            dataset["mesh_size_m"].transpose("lat", "lon").values,
            dtype=float,
        )
        coverage = (
            np.asarray(
                dataset["size_field_coverage_mask"]
                .transpose("lat", "lon")
                .values,
                dtype=bool,
            )
            if "size_field_coverage_mask" in dataset
            else np.isfinite(field) & (field > 0.0)
        )
        domain = (
            np.asarray(
                dataset["model_domain_mask"]
                .transpose("lat", "lon")
                .values,
                dtype=bool,
            )
            if "model_domain_mask" in dataset
            else coverage.copy()
        )
        sampling_interface_schema = str(
            dataset.attrs.get(
                "sampling_interface_schema_version",
                "legacy_unspecified",
            )
        ).strip()
    if field.shape != (len(lat), len(lon)):
        raise ValueError("mesh_size_m dimensions do not match lat/lon")
    if len(lon) < 2 or len(lat) < 2:
        raise ValueError("size-field grid must be at least 2 by 2")
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        field = field[:, ::-1]
        coverage = coverage[:, ::-1]
        domain = domain[:, ::-1]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[::-1, :]
        coverage = coverage[::-1, :]
        domain = domain[::-1, :]
    sampler = recorded_size_interpolator(
        lat,
        lon,
        field,
        coverage,
        domain,
        sampling_interface_schema,
    )
    interior_target = np.asarray(
        sampler.sample(
            np.column_stack([coordinates[:, 1], coordinates[:, 0]])
        ),
        dtype=float,
    )
    if (
        not np.all(np.isfinite(interior_target))
        or np.any(interior_target <= 0.0)
    ):
        raise ValueError("interior field is invalid at source boundary vertices")

    ratio = boundary_target / interior_target
    upper = float(ratio_tolerance)
    lower = 1.0 / upper
    coarser = ratio > upper
    finer = ratio < lower
    conflict = coarser | finer
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "conflict_detected" if np.any(conflict) else "pass",
        "source_boundary_vertex_count": int(len(features)),
        "ratio_definition": "boundary_target_m / interior_field_at_vertex_m",
        "ratio_tolerance": upper,
        "acceptable_ratio_interval": [lower, upper],
        "boundary_target_m": _quantiles(boundary_target),
        "interior_field_at_source_vertex_m": _quantiles(interior_target),
        "boundary_over_interior_ratio": _quantiles(ratio),
        "boundary_coarser_than_interior_count": int(np.count_nonzero(coarser)),
        "boundary_finer_than_interior_count": int(np.count_nonzero(finer)),
        "conflict_vertex_count": int(np.count_nonzero(conflict)),
        "conflict_vertex_fraction": float(np.mean(conflict)),
        "target_size_attribution_valid": bool(not np.any(conflict)),
        "interpretation": (
            "When this contract conflicts, boundary-adjacent L/h failures mix "
            "the 1-D representation policy with the 2-D field and cannot be "
            "attributed to the triangle algorithm alone."
        ),
        "inputs": {
            "boundary_geojson": {
                "path": str(boundary_path),
                "sha256": _sha256(boundary_path),
            },
            "size_field_netcdf": {
                "path": str(field_path),
                "sha256": _sha256(field_path),
            },
        },
    }


def write_boundary_size_contract(
    output_path: str | Path,
    report: dict[str, Any],
) -> Path:
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
