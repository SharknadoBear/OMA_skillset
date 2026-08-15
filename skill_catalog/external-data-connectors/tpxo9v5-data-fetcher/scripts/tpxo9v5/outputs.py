"""Atomic, model-neutral TPXO product writers and health validation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4 as nc4
import numpy as np

from .interpolation import complex_to_amplitude_phase
from .io import HarmonicField


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a source before an optional staged-file cleanup."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_attr(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _set_global_attributes(ds: nc4.Dataset, metadata: dict[str, Any]) -> None:
    ds.Conventions = "CF-1.10"
    ds.product_schema = "tpxo9v5_harmonics_v1"
    ds.title = "Model-neutral TPXO9v5 harmonic constants"
    ds.history = f"Created {datetime.now(timezone.utc).isoformat()}"
    for key, value in metadata.items():
        setattr(ds, key, _json_attr(value) if isinstance(value, (dict, list, tuple)) else value)


def _write_constituents(ds: nc4.Dataset, names: list[str]) -> None:
    ds.createDimension("constituent", len(names))
    variable = ds.createVariable("constituent_name", str, ("constituent",))
    variable.long_name = "tidal constituent name"
    variable[:] = np.asarray(names, dtype=object)


def _write_harmonics(
    ds: nc4.Dataset,
    prefix: str,
    dimensions: tuple[str, ...],
    coefficient: np.ndarray,
    units: str,
    flags: np.ndarray | None = None,
) -> None:
    values = np.asarray(coefficient)
    amplitude, phase = complex_to_amplitude_phase(values)
    fill = np.float32(np.nan)
    for suffix, data, long_name, variable_units in (
        ("real", values.real, "real part of harmonic coefficient", units),
        ("imaginary", values.imag, "imaginary part of harmonic coefficient", units),
        ("amplitude", amplitude, "harmonic amplitude", units),
        ("phase_lag", phase, "Greenwich phase lag", "degree"),
    ):
        variable = ds.createVariable(
            f"{prefix}_{suffix}", "f4", dimensions, zlib=True, complevel=2, fill_value=fill
        )
        variable.long_name = f"{prefix.replace('_', ' ')} {long_name}"
        variable.units = variable_units or "unknown"
        if suffix == "phase_lag":
            variable.phase_convention = "coefficient = amplitude * exp(-i * phase_lag)"
        variable[:] = np.asarray(data, dtype=np.float32)
    if flags is not None:
        flag = ds.createVariable(f"{prefix}_interpolation_flag", "i1", dimensions, zlib=True, complevel=2)
        flag.long_name = f"{prefix.replace('_', ' ')} interpolation method"
        flag.flag_values = np.asarray([0, 1, 2], dtype=np.int8)
        flag.flag_meanings = "linear nearest_wet unresolved"
        flag[:] = np.asarray(flags, dtype=np.int8)


def _atomic_path(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.with_name(f".{output.name}.partial")


def write_native_product(
    output: str | Path,
    fields: Iterable[HarmonicField],
    metadata: dict[str, Any],
) -> Path:
    """Write bounded harmonics on their native staggered grids."""

    destination = Path(output).resolve()
    partial = _atomic_path(destination)
    if partial.exists():
        partial.unlink()
    field_list = list(fields)
    if not field_list:
        raise ValueError("At least one field is required.")
    names = field_list[0].constituents
    if any(field.constituents != names for field in field_list):
        raise ValueError("All output fields must have the same constituent ordering.")
    try:
        with nc4.Dataset(partial, "w", format="NETCDF4") as ds:
            _set_global_attributes(ds, metadata)
            _write_constituents(ds, names)
            written_grids: set[str] = set()
            for field in field_list:
                lat_dim = f"latitude_{field.grid}"
                lon_dim = f"longitude_{field.grid}"
                if field.grid not in written_grids:
                    ds.createDimension(lat_dim, field.latitude.size)
                    ds.createDimension(lon_dim, field.longitude.size)
                    lat_var = ds.createVariable(lat_dim, "f8", (lat_dim,))
                    lon_var = ds.createVariable(lon_dim, "f8", (lon_dim,))
                    lat_var.units = "degrees_north"
                    lon_var.units = "degrees_east"
                    lon_var.longitude_convention = "continuous unwrapped [0, 720) branch for regional subsets"
                    lat_var[:] = field.latitude
                    lon_var[:] = field.longitude
                    if field.depth is not None:
                        depth = ds.createVariable(
                            f"depth_{field.grid}", "f4", (lat_dim, lon_dim), zlib=True, complevel=2
                        )
                        depth.units = "m"
                        depth[:] = np.asarray(field.depth, dtype=np.float32)
                    if field.mask is not None:
                        mask = ds.createVariable(
                            f"mask_{field.grid}", "i1", (lat_dim, lon_dim), zlib=True, complevel=2
                        )
                        mask.long_name = "native wet mask"
                        mask[:] = np.asarray(np.isfinite(field.mask) & (field.mask > 0), dtype=np.int8)
                    written_grids.add(field.grid)
                prefix = "elevation" if field.name == "elevation" else field.name.replace("velocity_", "velocity_")
                if field.name in {"u", "v"}:
                    prefix = f"transport_{field.name}"
                _write_harmonics(
                    ds,
                    prefix,
                    ("constituent", lat_dim, lon_dim),
                    field.coefficient,
                    field.units,
                )
        os.replace(partial, destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return destination


def write_point_product(
    output: str | Path,
    fields: Iterable[tuple[HarmonicField, np.ndarray, np.ndarray]],
    target_longitude: np.ndarray,
    target_latitude: np.ndarray,
    target_shape: tuple[int, ...],
    metadata: dict[str, Any],
) -> Path:
    """Write harmonics interpolated to arbitrary flattened targets."""

    destination = Path(output).resolve()
    partial = _atomic_path(destination)
    if partial.exists():
        partial.unlink()
    field_list = list(fields)
    if not field_list:
        raise ValueError("At least one field is required.")
    names = field_list[0][0].constituents
    if any(field.constituents != names for field, _, _ in field_list):
        raise ValueError("All output fields must have the same constituent ordering.")
    lon = np.asarray(target_longitude, dtype=float).ravel()
    lat = np.asarray(target_latitude, dtype=float).ravel()
    try:
        with nc4.Dataset(partial, "w", format="NETCDF4") as ds:
            _set_global_attributes(ds, {**metadata, "target_shape": list(target_shape)})
            _write_constituents(ds, names)
            ds.createDimension("point", lon.size)
            lon_var = ds.createVariable("longitude", "f8", ("point",))
            lat_var = ds.createVariable("latitude", "f8", ("point",))
            lon_var.units = "degrees_east"
            lat_var.units = "degrees_north"
            lon_var[:] = lon
            lat_var[:] = lat
            for field, values, flags in field_list:
                prefix = "elevation" if field.name == "elevation" else field.name
                if field.name in {"u", "v"}:
                    prefix = f"transport_{field.name}"
                _write_harmonics(
                    ds,
                    prefix,
                    ("constituent", "point"),
                    values,
                    field.units,
                    flags,
                )
        os.replace(partial, destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return destination


def validate_product(path: str | Path) -> dict[str, Any]:
    """Validate structure and finite harmonic coverage for a generated product."""

    source = Path(path).resolve()
    problems: list[str] = []
    variables: dict[str, Any] = {}
    if not source.is_file() or source.stat().st_size == 0:
        return {"schema_version": "tpxo9v5_health_v1", "status": "fail", "problems": ["Output is missing or empty."]}
    try:
        with nc4.Dataset(source) as ds:
            if getattr(ds, "product_schema", "") != "tpxo9v5_harmonics_v1":
                problems.append("Unexpected or missing product_schema attribute.")
            if "constituent" not in ds.dimensions or len(ds.dimensions["constituent"]) == 0:
                problems.append("No constituents are present.")
            amplitude_names = [name for name in ds.variables if name.endswith("_amplitude")]
            if not amplitude_names:
                problems.append("No harmonic amplitude variables are present.")
            for name in amplitude_names:
                values = np.ma.asarray(ds[name][:])
                finite = np.isfinite(np.asarray(np.ma.filled(values, np.nan), dtype=float))
                fraction = float(finite.mean()) if finite.size else 0.0
                variables[name] = {
                    "shape": [int(value) for value in values.shape],
                    "finite_fraction": fraction,
                }
                if fraction == 0.0:
                    problems.append(f"{name} contains no finite values.")
    except (OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
        problems.append(f"NetCDF read failed: {exc}")
    return {
        "schema_version": "tpxo9v5_health_v1",
        "input_basename": source.name,
        "size_bytes": source.stat().st_size,
        "status": "pass" if not problems else "fail",
        "problems": problems,
        "variables": variables,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a UTF-8 JSON report."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
