#!/usr/bin/env python3
"""Shared ROMS C-grid inspection, vertical transforms, extraction, and QA."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Package import.
    from . import ofs_archive_sources as archive_sources
    from .roms_aws_core import (
        BUCKET, S3_ENDPOINT, ModelConfig, canonical_json_sha256, expected_times,
        iso_utc, json_clean, manifest_paths, normalize_request_input, parse_utc,
        read_json, sha256_file, verify_transfers, write_json_atomic,
    )
except ImportError:  # Direct script execution.
    import ofs_archive_sources as archive_sources
    from roms_aws_core import (
        BUCKET, S3_ENDPOINT, ModelConfig, canonical_json_sha256, expected_times,
        iso_utc, json_clean, manifest_paths, normalize_request_input, parse_utc,
        read_json, sha256_file, verify_transfers, write_json_atomic,
    )

UTC = timezone.utc
COMPACT_SCHEMA_VERSION = "roms_compact_fields_v1"


def _modules():
    try:
        import netCDF4
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy and netCDF4 are required for ROMS processing") from exc
    return netCDF4, np


def _filled(value: Any, *, dtype: Any = None):
    _, np = _modules()
    array = np.ma.filled(value, np.nan)
    return np.asarray(array, dtype=dtype)


def _scalar(variable: Any, default: Any = None) -> Any:
    if variable is None:
        return default
    value = variable[...]
    try:
        return value.item()
    except (AttributeError, ValueError):
        return value


def _dataset_scalar(ds: Any, name: str, default: Any = None) -> Any:
    if name in ds.variables:
        return _scalar(ds.variables[name], default)
    return getattr(ds, name, default)


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return datetime(
        int(value.year), int(value.month), int(value.day),
        int(value.hour), int(value.minute), int(value.second),
        int(getattr(value, "microsecond", 0)), tzinfo=UTC,
    )


CALENDAR_ALIASES = {
    # Historical NOAA/NCEI ROMS files use this non-CF spelling.
    "gregorian_proleptic": "proleptic_gregorian",
}


def ocean_time_metadata(ds: Any) -> dict[str, Any]:
    """Return lossless ocean_time metadata plus the decoder-safe calendar."""
    if "ocean_time" not in ds.variables:
        raise ValueError("NetCDF has no ocean_time variable")
    variable = ds.variables["ocean_time"]
    units = getattr(variable, "units", None)
    if not units:
        raise ValueError("ocean_time has no units")
    source_calendar = str(getattr(variable, "calendar", "standard"))
    decoder_calendar = CALENDAR_ALIASES.get(source_calendar.strip().casefold(),
                                            source_calendar)
    return {
        "source_time_units": str(units),
        "source_calendar": source_calendar,
        "decoder_calendar": decoder_calendar,
        "calendar_alias_applied": decoder_calendar != source_calendar,
    }


def decode_ocean_times(ds: Any) -> list[datetime]:
    netCDF4, np = _modules()
    metadata = ocean_time_metadata(ds)
    variable = ds.variables["ocean_time"]
    raw = np.atleast_1d(variable[:])
    decoded = netCDF4.num2date(raw, units=metadata["source_time_units"],
                              calendar=metadata["decoder_calendar"],
                              only_use_cftime_datetimes=False)
    return [_datetime(value) for value in np.atleast_1d(decoded)]


def normalize_time(value: datetime, cadence_seconds: int, tolerance_seconds: float = 60.0) -> tuple[datetime, float]:
    rounded = round(value.timestamp() / cadence_seconds) * cadence_seconds
    normalized = datetime.fromtimestamp(rounded, tz=UTC)
    offset = (value - normalized).total_seconds()
    return (normalized, offset) if abs(offset) <= tolerance_seconds else (value, 0.0)


def _digest_array(value: Any) -> str:
    _, np = _modules()
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


STATIC_NAMES = (
    "lon_rho", "lat_rho", "mask_rho", "h", "angle",
    "lon_u", "lat_u", "mask_u", "lon_v", "lat_v", "mask_v",
    "s_rho", "s_w", "Cs_r", "Cs_w",
)

ANGLE_CONVENTION = "xi_axis_counterclockwise_from_east_radians"


def _strict_mask(ds: Any, name: str, shape: tuple[int, int]):
    _, np = _modules()
    if name not in ds.variables:
        raise ValueError(f"ROMS native fields require {name}; masks are never synthesized")
    variable = ds.variables[name]
    expected_dimensions = {
        "mask_rho": ("eta_rho", "xi_rho"),
        "mask_u": ("eta_u", "xi_u"),
        "mask_v": ("eta_v", "xi_v"),
    }[name]
    if tuple(variable.dimensions) != expected_dimensions:
        raise ValueError(f"{name} dimensions {variable.dimensions} != {expected_dimensions}")
    value = _variable_data(variable)
    if value.shape != shape:
        raise ValueError(f"{name} shape {value.shape} != {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with binary 0/1 semantics")
    unique = sorted(float(item) for item in np.unique(value))
    if any(item not in {0.0, 1.0} for item in unique):
        raise ValueError(f"{name} must use binary 0/1 semantics, found {unique}")
    return value.astype(bool), unique


def _strict_vertical_vector(ds: Any, name: str, expected_length: int | None = None):
    _, np = _modules()
    if name not in ds.variables:
        raise ValueError(f"ROMS vertical metadata is missing {name}")
    variable = ds.variables[name]
    expected_dimension = "s_rho" if name in {"s_rho", "Cs_r"} else "s_w"
    if tuple(variable.dimensions) != (expected_dimension,):
        raise ValueError(f"{name} dimensions {variable.dimensions} != {(expected_dimension,)}")
    value = _variable_data(variable)
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if expected_length is not None and value.size != expected_length:
        raise ValueError(f"{name} length {value.size} != {expected_length}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    difference = np.diff(value)
    if not (np.all(difference > 0) or np.all(difference < 0)):
        raise ValueError(f"{name} must be strictly monotonic")
    return value


def _strict_numeric_scalar(ds: Any, name: str, *, integer: bool = False):
    _, np = _modules()
    if name in ds.variables and ds.variables[name].shape != ():
        raise ValueError(f"ROMS vertical metadata {name} must be a scalar variable")
    raw = _dataset_scalar(ds, name, None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"ROMS vertical metadata {name} must be a finite scalar") from None
    if not np.isfinite(value):
        raise ValueError(f"ROMS vertical metadata {name} must be a finite scalar")
    if integer and value != round(value):
        raise ValueError(f"ROMS vertical metadata {name} must be an integer scalar")
    return int(round(value)) if integer else value


def _vertical_contract(ds: Any) -> dict[str, Any]:
    s_rho = _strict_vertical_vector(ds, "s_rho")
    s_w = _strict_vertical_vector(ds, "s_w")
    cs_r = _strict_vertical_vector(ds, "Cs_r", len(s_rho))
    cs_w = _strict_vertical_vector(ds, "Cs_w", len(s_w))
    if len(s_w) != len(s_rho) + 1:
        raise ValueError("s_w/Cs_w must have exactly one more level than s_rho/Cs_r")
    hc = _strict_numeric_scalar(ds, "hc")
    if hc < 0:
        raise ValueError("ROMS vertical metadata hc must be nonnegative")
    vtransform = _strict_numeric_scalar(ds, "Vtransform", integer=True)
    if vtransform not in {1, 2}:
        raise ValueError(f"ROMS Vtransform must be 1 or 2, found {vtransform}")
    vstretching = _strict_numeric_scalar(ds, "Vstretching", integer=True)
    if vstretching < 1:
        raise ValueError(f"ROMS Vstretching must be a positive integer, found {vstretching}")
    return {
        "s_rho": s_rho, "s_w": s_w, "Cs_r": cs_r, "Cs_w": cs_w,
        "hc": hc, "Vtransform": vtransform, "Vstretching": vstretching,
    }


def _angle_contract(ds: Any, rho_shape: tuple[int, int], wet: Any) -> dict[str, str]:
    _, np = _modules()
    variable = ds.variables["angle"]
    if tuple(variable.dimensions) != ("eta_rho", "xi_rho"):
        raise ValueError("angle must use the rho-grid dimensions eta_rho, xi_rho")
    value = _variable_data(variable)
    if value.shape != rho_shape:
        raise ValueError(f"angle shape {value.shape} != rho grid {rho_shape}")
    wet_values = value[np.asarray(wet, dtype=bool)]
    if wet_values.size == 0 or not np.isfinite(wet_values).all():
        raise ValueError("angle must be finite at every wet rho point")
    if np.any(np.abs(wet_values) > 2 * np.pi + 1e-6):
        raise ValueError("angle wet values exceed the plausible radian range +/-2*pi")
    units = str(getattr(variable, "units", "")).strip().lower()
    if units not in {"rad", "radian", "radians"}:
        raise ValueError("angle units must explicitly be rad, radian, or radians")
    standard_name = str(getattr(variable, "standard_name", "")).strip().lower()
    long_name = str(getattr(variable, "long_name", "")).strip().lower()
    standard_ok = standard_name == "grid_angle_of_rotation_from_east_to_y"
    long_ok = all(token in long_name for token in ("angle", "xi", "east"))
    if not (standard_ok or long_ok):
        raise ValueError("angle metadata does not identify the XI-axis rotation from east")
    return {
        "units": "radians",
        "convention": ANGLE_CONVENTION,
        "semantic_source": "standard_name" if standard_ok else "long_name",
        "source_units": str(getattr(variable, "units", "")).strip(),
        "source_standard_name": str(getattr(variable, "standard_name", "")).strip(),
        "source_long_name": str(getattr(variable, "long_name", "")).strip(),
    }


def requested_variable_schema(ds: Any, names: Sequence[str]) -> dict[str, Any]:
    """Validate and fingerprint dynamic variables before concatenation."""
    signatures: dict[str, Any] = {}
    for name in names:
        if name not in ds.variables:
            raise ValueError(f"requested variable {name!r} is unavailable")
        variable = ds.variables[name]
        dimensions = tuple(variable.dimensions)
        if name == "u":
            allowed = (("ocean_time", "s_rho", "eta_u", "xi_u"),)
        elif name == "v":
            allowed = (("ocean_time", "s_rho", "eta_v", "xi_v"),)
        elif name == "zeta":
            allowed = (("ocean_time", "eta_rho", "xi_rho"),)
        else:
            allowed = (
                ("ocean_time", "eta_rho", "xi_rho"),
                ("ocean_time", "s_rho", "eta_rho", "xi_rho"),
            )
        if dimensions not in allowed:
            raise ValueError(
                f"requested variable {name!r} has incompatible ROMS dimensions "
                f"{dimensions}; expected one of {allowed}"
            )
        if variable.shape[0] != len(ds.dimensions["ocean_time"]):
            raise ValueError(f"requested variable {name!r} time axis disagrees with ocean_time")
        signatures[name] = {
            "dimensions": list(dimensions),
            "shape_without_time": list(variable.shape[1:]),
            "dtype": str(variable.dtype),
            "grid": getattr(variable, "grid", None),
            "location": getattr(variable, "location", None),
        }
    return signatures


def geometry_snapshot(ds: Any) -> dict[str, Any]:
    _, np = _modules()
    required = (
        "lon_rho", "lat_rho", "h", "angle", "mask_rho", "mask_u", "mask_v",
        "s_rho", "s_w", "Cs_r", "Cs_w",
    )
    missing = [name for name in required if name not in ds.variables]
    if missing:
        raise ValueError(f"ROMS geometry is missing variables: {', '.join(missing)}")
    h = _variable_data(ds.variables["h"])
    if h.ndim != 2 or min(h.shape) < 2:
        raise ValueError(f"ROMS h must define a two-dimensional rho grid, got {h.shape}")
    rho_shape = h.shape
    for name in ("lon_rho", "lat_rho", "angle"):
        if _variable_data(ds.variables[name]).shape != rho_shape:
            raise ValueError(f"{name} must be rho-shaped {rho_shape}")
    mask_rho, rho_values = _strict_mask(ds, "mask_rho", rho_shape)
    _, u_values = _strict_mask(ds, "mask_u", (rho_shape[0], rho_shape[1] - 1))
    _, v_values = _strict_mask(ds, "mask_v", (rho_shape[0] - 1, rho_shape[1]))
    vertical = _vertical_contract(ds)
    angle = _angle_contract(ds, rho_shape, mask_rho)
    result: dict[str, Any] = {"variables": {}, "dimensions": {name: len(dim) for name, dim in ds.dimensions.items()}}
    for name in STATIC_NAMES:
        if name in ds.variables:
            array = _variable_data(ds.variables[name])
            result["variables"][name] = {"shape": list(array.shape), "digest": _digest_array(array)}
    result.update({name: vertical[name] for name in ("hc", "Vtransform", "Vstretching")})
    result["angle_contract"] = angle
    for name in ("eta_rho", "xi_rho", "eta_u", "xi_u", "eta_v", "xi_v", "s_rho", "s_w"):
        if name in ds.dimensions:
            result[name] = len(ds.dimensions[name])
    result["grid_hash"] = hashlib.sha256(
        "|".join(f"{name}:{meta['digest']}" for name, meta in sorted(result["variables"].items())).encode()
    ).hexdigest()
    result["mask_values"] = {"mask_rho": rho_values, "mask_u": u_values, "mask_v": v_values}
    return result


def assert_geometry(reference: Mapping[str, Any], candidate: Mapping[str, Any], path: str | Path) -> None:
    if reference["grid_hash"] != candidate["grid_hash"]:
        raise ValueError(f"ROMS geometry/schema drift in {path}")
    for name in ("hc", "Vtransform", "Vstretching", "angle_contract"):
        if reference.get(name) != candidate.get(name):
            raise ValueError(f"ROMS {name} drift in {path}: {candidate.get(name)} != {reference.get(name)}")


def inspect_file(path: str | Path, product: str | None = None) -> dict[str, Any]:
    netCDF4, _ = _modules()
    source = Path(path)
    with netCDF4.Dataset(source) as ds:
        variables = {
            name: {
                "dimensions": list(variable.dimensions),
                "shape": list(variable.shape),
                "dtype": str(variable.dtype),
                "standard_name": getattr(variable, "standard_name", None),
                "units": getattr(variable, "units", None),
                "fill_value": getattr(variable, "_FillValue", None),
            }
            for name, variable in ds.variables.items()
        }
        try:
            times = [iso_utc(value) for value in decode_ocean_times(ds)]
        except Exception as exc:
            times = []
            time_error = f"{type(exc).__name__}: {exc}"
        else:
            time_error = None
        geometry = geometry_snapshot(ds) if product in {None, "fields"} and all(name in ds.variables for name in ("lon_rho", "lat_rho", "h", "angle", "s_rho", "s_w", "Cs_r", "Cs_w")) else None
        return {
            "path": str(source.resolve()), "size": source.stat().st_size,
            "sha256": sha256_file(source), "format": getattr(ds, "data_model", None),
            "dimensions": {name: len(dim) for name, dim in ds.dimensions.items()},
            "variables": variables, "times_utc": times, "time_error": time_error,
            "geometry": geometry,
        }


def inspect_paths(paths: Sequence[str | Path], *, product: str | None = None, output: str | Path | None = None) -> dict[str, Any]:
    report = {
        "schema_version": "roms_inspection_v1",
        "created_utc": iso_utc(datetime.now(UTC)),
        "files": [inspect_file(path, product=product) for path in paths],
    }
    if output:
        write_json_atomic(output, report)
    return report


def _variable_data(variable: Any, index: int | None = None):
    _, np = _modules()
    # NetCDF4 masks valid_min/valid_max as well as missing values. NOAA Cs_w
    # can have harmless endpoint roundoff beyond valid_max, so disable that
    # policy and honor only explicit source fill/missing sentinels below.
    variable.set_auto_mask(False)
    try:
        value = variable[index] if index is not None and variable.dimensions and variable.dimensions[0] in {"ocean_time", "time"} else variable[:]
    finally:
        variable.set_auto_mask(True)
    array = np.asarray(value, dtype=np.float64)
    fill = getattr(variable, "_FillValue", None)
    missing = getattr(variable, "missing_value", None)
    for sentinel in (fill, missing):
        if sentinel is not None:
            array[np.isclose(array, float(sentinel), rtol=0.0, atol=0.0)] = np.nan
    return array


def average_adjacent(array: Any, axis: int, wet: Any | None = None):
    """Finite-aware adjacent average from rho points to a staggered grid."""
    _, np = _modules()
    values = np.asarray(array, dtype=float)
    left = np.take(values, indices=range(values.shape[axis] - 1), axis=axis)
    right = np.take(values, indices=range(1, values.shape[axis]), axis=axis)
    left_ok, right_ok = np.isfinite(left), np.isfinite(right)
    numerator = np.where(left_ok, left, 0.0) + np.where(right_ok, right, 0.0)
    denominator = left_ok.astype(float) + right_ok.astype(float)
    result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    if wet is not None:
        result = np.where(np.asarray(wet, dtype=bool), result, np.nan)
    return result


def destagger_to_rho(u: Any, v: Any, mask_rho: Any, mask_u: Any | None = None, mask_v: Any | None = None):
    """Destagger native U/V arrays using finite, wet-aware adjacent values."""
    _, np = _modules()
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    rho_shape = np.asarray(mask_rho).shape
    if u.shape[-2:] != (rho_shape[0], rho_shape[1] - 1):
        raise ValueError(f"u shape {u.shape[-2:]} is incompatible with rho grid {rho_shape}")
    if v.shape[-2:] != (rho_shape[0] - 1, rho_shape[1]):
        raise ValueError(f"v shape {v.shape[-2:]} is incompatible with rho grid {rho_shape}")
    if mask_u is not None:
        u = np.where(np.asarray(mask_u, dtype=bool), u, np.nan)
    if mask_v is not None:
        v = np.where(np.asarray(mask_v, dtype=bool), v, np.nan)
    u_rho = np.full(u.shape[:-2] + rho_shape, np.nan, dtype=float)
    v_rho = np.full(v.shape[:-2] + rho_shape, np.nan, dtype=float)
    u_rho[..., :, 0] = u[..., :, 0]
    u_rho[..., :, -1] = u[..., :, -1]
    v_rho[..., 0, :] = v[..., 0, :]
    v_rho[..., -1, :] = v[..., -1, :]
    for output, source, axis in ((u_rho, u, -1), (v_rho, v, -2)):
        first = np.take(source, indices=range(source.shape[axis] - 1), axis=axis)
        second = np.take(source, indices=range(1, source.shape[axis]), axis=axis)
        ok1, ok2 = np.isfinite(first), np.isfinite(second)
        value = np.divide(
            np.where(ok1, first, 0) + np.where(ok2, second, 0),
            ok1.astype(float) + ok2.astype(float),
            out=np.full_like(first, np.nan),
            where=(ok1 | ok2),
        )
        if axis == -1:
            output[..., :, 1:-1] = value
        else:
            output[..., 1:-1, :] = value
    wet = np.asarray(mask_rho, dtype=bool)
    return np.where(wet, u_rho, np.nan), np.where(wet, v_rho, np.nan)


def rotate_to_earth(u_rho: Any, v_rho: Any, angle: Any):
    _, np = _modules()
    angle = np.asarray(angle, dtype=float)
    if angle.shape != np.asarray(u_rho).shape[-2:] or angle.shape != np.asarray(v_rho).shape[-2:]:
        raise ValueError("angle shape must match the rho grid")
    finite_vectors = np.isfinite(np.asarray(u_rho)) | np.isfinite(np.asarray(v_rho))
    required_angle = np.any(finite_vectors, axis=tuple(range(finite_vectors.ndim - 2))) if finite_vectors.ndim > 2 else finite_vectors
    if not np.isfinite(angle[required_angle]).all():
        raise ValueError("angle must be finite wherever current components are finite")
    if np.any(np.abs(angle[required_angle]) > 2 * np.pi + 1e-6):
        raise ValueError("angle values exceed the plausible radian range +/-2*pi")
    east = u_rho * np.cos(angle) - v_rho * np.sin(angle)
    north = u_rho * np.sin(angle) + v_rho * np.cos(angle)
    return east, north, np.hypot(east, north)


def roms_depths(sigma: Any, stretching: Any, hc: float, h: Any, zeta: Any, vtransform: int):
    """Calculate ROMS depths for transform 1 or 2 with live zeta."""
    _, np = _modules()
    sigma = np.asarray(sigma, dtype=float)
    stretching = np.asarray(stretching, dtype=float)
    h = np.asarray(h, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    if sigma.ndim != 1 or stretching.shape != sigma.shape:
        raise ValueError("sigma and stretching must be same-length 1-D arrays")
    if h.shape != zeta.shape:
        raise ValueError("h and zeta shapes must match")
    s = sigma.reshape((-1,) + (1,) * h.ndim)
    cs = stretching.reshape((-1,) + (1,) * h.ndim)
    hb = h[None, ...]
    zb = zeta[None, ...]
    if vtransform == 1:
        z0 = (s - cs) * hc + cs * hb
        return z0 + zb * (1.0 + np.divide(z0, hb, out=np.zeros_like(z0), where=hb != 0))
    if vtransform == 2:
        z0 = np.divide(hc * s + hb * cs, hc + hb, out=np.full_like(hb * cs, np.nan), where=(hc + hb) != 0)
        return zb + (zb + hb) * z0
    raise ValueError(f"unsupported ROMS Vtransform={vtransform}")


def layer_thickness(s_w: Any, Cs_w: Any, hc: float, h: Any, zeta: Any, vtransform: int):
    _, np = _modules()
    z_w = roms_depths(s_w, Cs_w, hc, h, zeta, vtransform)
    thickness = np.abs(np.diff(z_w, axis=0))
    expected = np.asarray(h, dtype=float) + np.asarray(zeta, dtype=float)
    closure = np.nansum(thickness, axis=0) - expected
    finite = np.isfinite(expected) & (expected > 0)
    max_abs = float(np.nanmax(np.abs(closure[finite]))) if finite.any() else float("nan")
    return thickness, max_abs


def weighted_vertical_average(data: Any, thickness: Any, wet: Any | None = None):
    _, np = _modules()
    values = np.asarray(data, dtype=float)
    weights = np.asarray(thickness, dtype=float)
    if values.shape != weights.shape:
        raise ValueError(f"data/thickness shape mismatch: {values.shape} != {weights.shape}")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    numerator = np.sum(np.where(valid, values * weights, 0.0), axis=0)
    denominator = np.sum(np.where(valid, weights, 0.0), axis=0)
    result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return np.where(np.asarray(wet, dtype=bool), result, np.nan) if wet is not None else result


def view_index(s_rho: Any, view: str | int) -> int | None:
    _, np = _modules()
    sigma = np.asarray(s_rho, dtype=float)
    if isinstance(view, int):
        if view >= sigma.size:
            raise ValueError(f"sigma index {view} exceeds layer count {sigma.size}")
        return view
    order = np.argsort(np.abs(sigma))
    if view == "surface":
        return int(order[0])
    if view == "near_surface":
        return int(order[1] if sigma.size > 1 else order[0])
    if view == "bottom":
        return int(np.argmax(np.abs(sigma)))
    if view == "depth_average":
        return None
    raise ValueError(f"unsupported view {view!r}")


def view_suffix(view: str | int) -> str:
    return f"sigma_{view}" if isinstance(view, int) else view


def _mask(ds: Any, name: str, shape: tuple[int, int]):
    value, _ = _strict_mask(ds, name, shape)
    return value


def _static(ds: Any) -> dict[str, Any]:
    _, np = _modules()
    contract = geometry_snapshot(ds)
    result = {name: _variable_data(ds.variables[name]) for name in ("lon_rho", "lat_rho", "h", "angle", "s_rho", "s_w", "Cs_r", "Cs_w")}
    rho_shape = result["h"].shape
    result["mask_rho"] = _mask(ds, "mask_rho", rho_shape)
    u_shape, v_shape = (rho_shape[0], rho_shape[1] - 1), (rho_shape[0] - 1, rho_shape[1])
    result["mask_u"] = _mask(ds, "mask_u", u_shape)
    result["mask_v"] = _mask(ds, "mask_v", v_shape)
    for name in ("lon_u", "lat_u", "lon_v", "lat_v"):
        if name in ds.variables:
            result[name] = _variable_data(ds.variables[name])
    if "lon_u" not in result:
        result["lon_u"] = average_adjacent(result["lon_rho"], -1, result["mask_u"])
        result["lat_u"] = average_adjacent(result["lat_rho"], -1, result["mask_u"])
    if "lon_v" not in result:
        result["lon_v"] = average_adjacent(result["lon_rho"], -2, result["mask_v"])
        result["lat_v"] = average_adjacent(result["lat_rho"], -2, result["mask_v"])
    result["hc"] = contract["hc"]
    result["Vtransform"] = contract["Vtransform"]
    result["Vstretching"] = contract["Vstretching"]
    result["angle_contract"] = contract["angle_contract"]
    return result


@dataclass(frozen=True)
class SourceRecord:
    path: Path
    time_index: int
    raw_time: datetime
    normalized_time: datetime
    offset_seconds: float
    source_key: str
    source_archive: str
    source_url: str
    archive_role: str
    container: str
    endpoint: str
    listing_endpoint: str
    source_time_units: str
    source_calendar: str
    decoder_calendar: str
    calendar_alias_applied: bool


def _source_key_for_path(path: Path, ds: Any, model: str | None) -> str:
    explicit = getattr(ds, "source_key", None)
    if explicit:
        return str(explicit)
    if model:
        parts = path.resolve().parts
        try:
            raw_index = max(index for index, part in enumerate(parts) if part.lower() == "raw")
        except ValueError:
            pass
        else:
            relative = "/".join(parts[raw_index + 1:])
            if relative:
                return f"{model}/netcdf/{relative}"
    return path.name


def _source_metadata_for_path(path: Path, ds: Any, model: str | None,
                              bound: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Resolve source metadata, preferring an integrity-verified manifest object."""
    if bound is not None:
        source_archive = str(bound.get("source_archive") or bound.get("source_id") or "")
        values = {
            "source_key": str(bound.get("key") or ""),
            "source_archive": source_archive,
            "source_url": str(bound.get("url") or ""),
            "archive_role": str(bound.get("archive_role") or ""),
            "container": str(bound.get("container") or ""),
            "endpoint": str(bound.get("endpoint") or ""),
            "listing_endpoint": str(bound.get("listing_endpoint") or ""),
        }
        if any(not values[name] for name in ("source_key", "source_archive", "source_url", "endpoint")):
            raise ValueError("verified transfer binding has incomplete source archive provenance")
        return values
    sidecar = path.with_name(path.name + ".download.json")
    metadata: Mapping[str, Any] = {}
    if sidecar.is_file():
        try:
            candidate = read_json(sidecar)
            if isinstance(candidate, Mapping):
                metadata = candidate
        except Exception:
            pass
    source_archive = str(metadata.get("source_id") or "unbound")
    key = str(metadata.get("key") or _source_key_for_path(path, ds, model))
    url = str(metadata.get("url") or "")
    if source_archive in {"aws_operational", "ncei_long_term"} and model:
        descriptor = archive_sources.get_source_descriptor(source_archive, model)
    else:
        descriptor = {"archive_role": "unbound", "container": "", "endpoint": "",
                      "listing_endpoint": ""}
    return {
        "source_key": key, "source_archive": source_archive, "source_url": url,
        "archive_role": descriptor["archive_role"], "container": descriptor["container"],
        "endpoint": descriptor["endpoint"],
        "listing_endpoint": descriptor["listing_endpoint"],
    }


def _source_summary(records: Sequence[SourceRecord]) -> dict[str, Any]:
    archives: dict[tuple[str, str], dict[str, str]] = {}
    for record in records:
        identity = (record.source_archive, record.endpoint)
        archives[identity] = {
            "source_archive": record.source_archive,
            "archive_role": record.archive_role,
            "container": record.container,
            "endpoint": record.endpoint,
            "listing_endpoint": record.listing_endpoint,
        }
    values = sorted(archives.values(), key=lambda item: (item["source_archive"], item["endpoint"]))
    return {
        "record_count": len(records), "archive_count": len(values), "archives": values,
        "endpoints": {item["source_archive"]: item["endpoint"] for item in values},
    }


def _source_time_metadata(records: Sequence[SourceRecord]) -> list[dict[str, Any]]:
    """Summarize lossless source calendar metadata used during decoding."""
    unique = {
        (record.source_time_units, record.source_calendar, record.decoder_calendar,
         record.calendar_alias_applied)
        for record in records
    }
    fields = ("source_time_units", "source_calendar", "decoder_calendar",
              "calendar_alias_applied")
    return [dict(zip(fields, values))
            for values in sorted(unique, key=lambda row: tuple(map(str, row)))]


def collect_records(
    paths: Sequence[str | Path],
    start: datetime,
    end: datetime,
    product: str = "fields",
    model: str | None = None,
    requested_variables: Sequence[str] | None = None,
    bound_objects: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[SourceRecord], dict[str, Any]]:
    netCDF4, _ = _modules()
    records: list[SourceRecord] = []
    reference: dict[str, Any] | None = None
    schema_reference: dict[str, Any] | None = None
    for raw_path in paths:
        path = Path(raw_path).resolve()
        bound = None if bound_objects is None else bound_objects.get(str(path))
        with netCDF4.Dataset(path) as ds:
            candidate = geometry_snapshot(ds) if product == "fields" else None
            if candidate is not None:
                if reference is None:
                    reference = candidate
                else:
                    assert_geometry(reference, candidate, path)
            candidate_schema = (
                requested_variable_schema(ds, requested_variables or ())
                if product == "fields" and requested_variables is not None else None
            )
            if candidate_schema is not None:
                if schema_reference is None:
                    schema_reference = candidate_schema
                elif schema_reference != candidate_schema:
                    differing = sorted(
                        name for name in set(schema_reference) | set(candidate_schema)
                        if schema_reference.get(name) != candidate_schema.get(name)
                    )
                    raise ValueError(
                        f"ROMS requested-variable schema/dimension drift in {path}: "
                        f"{', '.join(differing)}"
                    )
            time_metadata = ocean_time_metadata(ds)
            times = decode_ocean_times(ds)
            source = _source_metadata_for_path(path, ds, model, bound)
            for index, raw_time in enumerate(times):
                normalized, offset = normalize_time(raw_time, 360 if product == "stations" else 3600)
                if start <= normalized < end:
                    records.append(SourceRecord(
                        path, index, raw_time, normalized, offset,
                        source["source_key"], source["source_archive"], source["source_url"],
                        source["archive_role"], source["container"], source["endpoint"],
                        source["listing_endpoint"], time_metadata["source_time_units"],
                        time_metadata["source_calendar"], time_metadata["decoder_calendar"],
                        time_metadata["calendar_alias_applied"],
                    ))
    records.sort(key=lambda record: (record.normalized_time, record.raw_time, str(record.path)))
    deduplicated: dict[datetime, SourceRecord] = {}
    duplicates: list[dict[str, Any]] = []
    for record in records:
        previous = deduplicated.get(record.normalized_time)
        if previous is None:
            deduplicated[record.normalized_time] = record
            continue
        # At cycle boundaries retain the record encountered from the preceding file.
        duplicates.append({"time_utc": iso_utc(record.normalized_time), "kept": str(previous.path), "discarded": str(record.path)})
    result = list(deduplicated.values())
    return result, {
        "geometry": reference,
        "dynamic_schema": schema_reference,
        "duplicates": duplicates,
    }


def _copy_attrs(source: Any, destination: Any) -> None:
    for name in source.ncattrs():
        if name == "_FillValue":
            continue
        try:
            destination.setncattr(name, source.getncattr(name))
        except (TypeError, ValueError):
            pass


def _write_static(
    output: Any,
    static: Mapping[str, Any],
    source_attrs: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    _, np = _modules()
    for name, dims in {
        "lon_rho": ("eta_rho", "xi_rho"), "lat_rho": ("eta_rho", "xi_rho"),
        "h": ("eta_rho", "xi_rho"), "angle": ("eta_rho", "xi_rho"),
        "mask_rho": ("eta_rho", "xi_rho"), "lon_u": ("eta_u", "xi_u"),
        "lat_u": ("eta_u", "xi_u"), "mask_u": ("eta_u", "xi_u"),
        "lon_v": ("eta_v", "xi_v"), "lat_v": ("eta_v", "xi_v"),
        "mask_v": ("eta_v", "xi_v"), "s_rho": ("s_rho",), "s_w": ("s_w",),
        "Cs_r": ("s_rho",), "Cs_w": ("s_w",),
    }.items():
        dtype = "i1" if name.startswith("mask_") else "f8"
        variable = output.createVariable(name, dtype, dims, zlib=dtype != "i1", complevel=4)
        value = static[name].astype(np.int8) if name.startswith("mask_") else static[name]
        variable[:] = value
        if source_attrs is not None:
            for attr_name, attr_value in source_attrs.get(name, {}).items():
                if attr_name != "_FillValue":
                    try:
                        variable.setncattr(attr_name, attr_value)
                    except (TypeError, ValueError):
                        pass
    output.createVariable("hc", "f8").assignValue(static["hc"])
    output.createVariable("Vtransform", "i4").assignValue(static["Vtransform"])
    output.createVariable("Vstretching", "i4").assignValue(static["Vstretching"])
    output.variables["angle"].setncattr("angle_convention", static["angle_contract"]["convention"])


def extract_fields(
    request: Mapping[str, Any] | str | Path,
    paths: Sequence[str | Path],
    output_path: str | Path,
    config: ModelConfig,
    *,
    manifest_output: str | Path | None = None,
    transfer_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    netCDF4, np = _modules()
    normalized = normalize_request_input(request, config)
    if normalized["product"] != "fields":
        raise ValueError("field extraction is unavailable for passthrough products")
    start, end = parse_utc(normalized["start_utc"]), parse_utc(normalized["end_utc_exclusive"])
    bound_by_path: dict[str, Mapping[str, Any]] | None = None
    if transfer_provenance is not None:
        if (transfer_provenance.get("status") != "pass"
                or canonical_json_sha256(transfer_provenance.get("request"))
                != canonical_json_sha256(normalized)):
            raise ValueError("extraction provenance is not a verified fetch-manifest binding")
        bound_objects = transfer_provenance.get("objects")
        if not isinstance(bound_objects, list) or not bound_objects:
            raise ValueError("verified fetch-manifest binding has no source objects")
        bound_by_path = {}
        for item in bound_objects:
            if not isinstance(item, Mapping) or item.get("status") != "pass":
                raise ValueError("verified fetch-manifest binding contains an invalid object")
            path = str(Path(str(item.get("local_path", ""))).resolve())
            if path in bound_by_path:
                raise ValueError("verified fetch-manifest binding contains duplicate local paths")
            bound_by_path[path] = item
        supplied = {str(Path(path).resolve()) for path in paths}
        if supplied != set(bound_by_path):
            raise ValueError("extraction inputs do not exactly match the verified fetch-manifest binding")
    records, record_meta = collect_records(
        paths, start, end, "fields", config.model, normalized["variables"], bound_by_path,
    )
    if not records:
        raise RuntimeError("no downloaded field records overlap the request")
    expected_stamps = expected_times(start, end, 3600)
    record_times = {record.normalized_time for record in records}
    missing_times = [stamp for stamp in expected_stamps if stamp not in record_times]
    extra_times = [record.normalized_time for record in records if record.normalized_time not in set(expected_stamps)]
    if extra_times:
        raise RuntimeError("downloaded field selection contains off-contract timestamps")
    if missing_times and normalized["missing_policy"] == "error":
        raise RuntimeError(
            f"downloaded data are missing {len(missing_times)} requested hourly timestamps: "
            + ", ".join(iso_utc(stamp) for stamp in missing_times[:8])
        )
    with netCDF4.Dataset(records[0].path) as first:
        static = _static(first)
        static_source_attrs = {
            name: {attr: first.variables[name].getncattr(attr) for attr in first.variables[name].ncattrs()}
            for name in STATIC_NAMES if name in first.variables
        }
        available = set(first.variables)
        for variable in normalized["variables"]:
            if variable not in available:
                raise ValueError(f"requested variable {variable!r} is unavailable; discovered: {', '.join(sorted(available))}")
    if ("u" in normalized["variables"]) != ("v" in normalized["variables"]):
        raise ValueError("u and v must be paired")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    rho_shape = static["h"].shape
    u_shape, v_shape = static["mask_u"].shape, static["mask_v"].shape
    output_variables: dict[str, Any] = {}
    closure_values: list[float] = []
    finite_coverage: dict[str, list[float]] = {}
    source_summary = _source_summary(records)
    source_time_metadata = _source_time_metadata(records)
    with netCDF4.Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("time", len(records))
        output.createDimension("eta_rho", rho_shape[0]); output.createDimension("xi_rho", rho_shape[1])
        output.createDimension("eta_u", u_shape[0]); output.createDimension("xi_u", u_shape[1])
        output.createDimension("eta_v", v_shape[0]); output.createDimension("xi_v", v_shape[1])
        output.createDimension("s_rho", len(static["s_rho"])); output.createDimension("s_w", len(static["s_w"]))
        output.createDimension("source_key_strlen", max(1, max(len(record.source_key) for record in records)))
        output.createDimension("source_archive_strlen", max(1, max(len(record.source_archive) for record in records)))
        output.createDimension("source_url_strlen", max(1, max(len(record.source_url) for record in records)))
        output.schema_version = COMPACT_SCHEMA_VERSION
        output.model = config.model
        output.vector_reference = "earth_relative_on_rho_grid"
        output.velocity_processing = "vertical view on native C-grid; finite-aware destagger to rho; angle rotation"
        output.angle_convention = static["angle_contract"]["convention"]
        output.created_utc = iso_utc(datetime.now(UTC))
        output.request_schema_version = normalized["schema_version"]
        output.request_sha256 = canonical_json_sha256(normalized)
        output.request_json = __import__("json").dumps(json_clean(normalized), sort_keys=True, separators=(",", ":"))
        output.source_provenance_json = json.dumps(source_summary, sort_keys=True, separators=(",", ":"))
        output.source_archives_json = json.dumps(source_summary["archives"], sort_keys=True, separators=(",", ":"))
        output.source_endpoints_json = json.dumps(source_summary["endpoints"], sort_keys=True, separators=(",", ":"))
        output.source_time_metadata_json = json.dumps(source_time_metadata, sort_keys=True, separators=(",", ":"))
        if len(source_summary["archives"]) == 1:
            output.source_bucket = source_summary["archives"][0]["container"]
            output.source_endpoint = source_summary["archives"][0]["endpoint"]
        else:
            output.source_bucket = "mixed"
            output.source_endpoint = "mixed"
        if transfer_provenance:
            output.fetch_manifest_path = str(transfer_provenance.get("manifest_path", ""))
            output.fetch_manifest_sha256 = str(transfer_provenance.get("manifest_sha256", ""))
        _write_static(output, static, static_source_attrs)
        time_var = output.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
        time_var.standard_name = "time"
        time_var.calendar = "proleptic_gregorian"
        raw_time_var = output.createVariable("source_ocean_time", "f8", ("time",))
        raw_time_var.units = time_var.units
        raw_time_var.calendar = time_var.calendar
        offset_var = output.createVariable("time_normalization_offset_seconds", "f8", ("time",))
        key_var = output.createVariable("source_key", "S1", ("time", "source_key_strlen"))
        key_var.long_name = "NOAA source object key"
        archive_var = output.createVariable("source_archive", "S1", ("time", "source_archive_strlen"))
        archive_var.long_name = "NOAA archive source identifier"
        url_var = output.createVariable("source_url", "S1", ("time", "source_url_strlen"))
        url_var.long_name = "canonical NOAA source object URL"
        time_var[:] = [record.normalized_time.timestamp() for record in records]
        raw_time_var[:] = [record.raw_time.timestamp() for record in records]
        offset_var[:] = [record.offset_seconds for record in records]
        for index, record in enumerate(records):
            for variable, dimension, value in (
                (key_var, "source_key_strlen", record.source_key),
                (archive_var, "source_archive_strlen", record.source_archive),
                (url_var, "source_url_strlen", record.source_url),
            ):
                width = len(output.dimensions[dimension])
                encoded = np.frombuffer(value.encode("utf-8")[:width], dtype="S1")
                variable[index, : encoded.size] = encoded

        def ensure(name: str, dims: tuple[str, ...], source: Any | None = None):
            if name not in output_variables:
                variable = output.createVariable(name, "f4", dims, zlib=True, complevel=4, fill_value=np.float32(9.96921e36))
                if source is not None:
                    _copy_attrs(source, variable)
                if variable.ndim == 3:
                    if name.startswith("u_native_"):
                        coordinates = "lon_u lat_u time"
                    elif name.startswith("v_native_"):
                        coordinates = "lon_v lat_v time"
                    else:
                        coordinates = "lon_rho lat_rho time"
                    variable.setncattr("coordinates", coordinates)
                    variable.setncattr("time", "time")
                    for stale in ("grid",):
                        if stale in variable.ncattrs():
                            variable.delncattr(stale)
                    if name.endswith("_depth_average"):
                        variable.setncattr("cell_methods", "time: point s_rho: mean (weighted by layer thickness)")
                    elif "cell_methods" in variable.ncattrs():
                        variable.setncattr("cell_methods", "time: point")
                output_variables[name] = variable
                finite_coverage[name] = []
            return output_variables[name]

        for time_index, record in enumerate(records):
            with netCDF4.Dataset(record.path) as ds:
                candidate = geometry_snapshot(ds)
                assert_geometry(record_meta["geometry"], candidate, record.path)
                zeta = _variable_data(ds.variables["zeta"], record.time_index)
                zeta = np.where(static["mask_rho"], zeta, np.nan)
                if "zeta" in normalized["variables"]:
                    ensure("zeta", ("time", "eta_rho", "xi_rho"), ds.variables["zeta"])[time_index] = zeta
                    finite_coverage["zeta"].append(float(np.isfinite(zeta[static["mask_rho"]]).mean()))
                h_u = average_adjacent(static["h"], -1, static["mask_u"])
                h_v = average_adjacent(static["h"], -2, static["mask_v"])
                zeta_u = average_adjacent(zeta, -1, static["mask_u"])
                zeta_v = average_adjacent(zeta, -2, static["mask_v"])
                thickness_rho, closure_rho = layer_thickness(static["s_w"], static["Cs_w"], static["hc"], static["h"], zeta, static["Vtransform"])
                thickness_u, closure_u = layer_thickness(static["s_w"], static["Cs_w"], static["hc"], h_u, zeta_u, static["Vtransform"])
                thickness_v, closure_v = layer_thickness(static["s_w"], static["Cs_w"], static["hc"], h_v, zeta_v, static["Vtransform"])
                closure_values.extend([closure_rho, closure_u, closure_v])
                for view_number, view in enumerate(normalized["vertical_views"]):
                    index = view_index(static["s_rho"], view)
                    suffix = view_suffix(view)
                    for source_name in normalized["variables"]:
                        if source_name in {"zeta", "u", "v"}:
                            continue
                        variable = ds.variables[source_name]
                        data = _variable_data(variable, record.time_index)
                        if data.ndim == 2:
                            if view_number:
                                continue
                            field = data
                        elif data.ndim == 3:
                            field = weighted_vertical_average(data, thickness_rho, static["mask_rho"]) if index is None else data[index]
                        else:
                            raise ValueError(f"unsupported dynamic rank for {source_name}: {data.shape}")
                        field = np.where(static["mask_rho"], field, np.nan)
                        output_name = source_name if data.ndim == 2 else ("salinity" if source_name == "salt" else source_name) + f"_{suffix}"
                        ensure(output_name, ("time", "eta_rho", "xi_rho"), variable)[time_index] = field
                        finite_coverage[output_name].append(float(np.isfinite(field[static["mask_rho"]]).mean()))
                    if "u" in normalized["variables"]:
                        u_all = _variable_data(ds.variables["u"], record.time_index)
                        v_all = _variable_data(ds.variables["v"], record.time_index)
                        u_native = weighted_vertical_average(u_all, thickness_u, static["mask_u"]) if index is None else np.where(static["mask_u"], u_all[index], np.nan)
                        v_native = weighted_vertical_average(v_all, thickness_v, static["mask_v"]) if index is None else np.where(static["mask_v"], v_all[index], np.nan)
                        ensure(f"u_native_{suffix}", ("time", "eta_u", "xi_u"), ds.variables["u"])[time_index] = u_native
                        ensure(f"v_native_{suffix}", ("time", "eta_v", "xi_v"), ds.variables["v"])[time_index] = v_native
                        finite_coverage[f"u_native_{suffix}"].append(float(np.isfinite(u_native[static["mask_u"]]).mean()))
                        finite_coverage[f"v_native_{suffix}"].append(float(np.isfinite(v_native[static["mask_v"]]).mean()))
                        u_rho, v_rho = destagger_to_rho(u_native, v_native, static["mask_rho"], static["mask_u"], static["mask_v"])
                        east, north, speed = rotate_to_earth(u_rho, v_rho, static["angle"])
                        for name, field in ((f"eastward_velocity_{suffix}", east), (f"northward_velocity_{suffix}", north), (f"current_speed_{suffix}", speed)):
                            out = ensure(name, ("time", "eta_rho", "xi_rho"))
                            if name.startswith("eastward"):
                                out.standard_name = "eastward_sea_water_velocity"
                            elif name.startswith("northward"):
                                out.standard_name = "northward_sea_water_velocity"
                            else:
                                out.long_name = "earth-relative current speed"
                            out.units = getattr(ds.variables["u"], "units", "m s-1")
                            out[time_index] = field
                            finite_coverage[name].append(float(np.isfinite(field[static["mask_rho"]]).mean()))
    os.replace(temporary, destination)
    manifest = {
        "schema_version": f"{config.model}_extraction_manifest_v2",
        "created_utc": iso_utc(datetime.now(UTC)), "request": normalized,
        "output": {"path": str(destination.resolve()), "size": destination.stat().st_size, "sha256": sha256_file(destination)},
        "records": [
            {"path": str(record.path.resolve()), "source_key": record.source_key,
             "source_archive": record.source_archive, "source_url": record.source_url,
             "raw_time_utc": iso_utc(record.raw_time), "normalized_time_utc": iso_utc(record.normalized_time),
             "normalization_offset_seconds": record.offset_seconds,
             "source_time_units": record.source_time_units,
             "source_calendar": record.source_calendar,
             "decoder_calendar": record.decoder_calendar,
             "calendar_alias_applied": record.calendar_alias_applied}
            for record in records
        ],
        "duplicate_records": record_meta["duplicates"], "grid_hash": record_meta["geometry"]["grid_hash"],
        "Vtransform": static["Vtransform"], "Vstretching": static["Vstretching"],
        "angle_contract": static["angle_contract"],
        "requested_variable_schema": record_meta["dynamic_schema"],
        "maximum_absolute_thickness_closure_m": float(np.nanmax(closure_values)),
        "thickness_closure_tolerance_m": 1e-5,
        "missing_requested_times": [iso_utc(stamp) for stamp in missing_times],
        "finite_wet_coverage": finite_coverage,
        "vector_processing": "native-grid vertical average; finite-aware C-grid destagger; angle rotation; speed after rotation",
        "source_provenance": source_summary,
        "source_time_metadata": source_time_metadata,
        "request_sha256": canonical_json_sha256(normalized),
        "fetch_manifest": None if not transfer_provenance else {
            "path": transfer_provenance.get("manifest_path"),
            "sha256": transfer_provenance.get("manifest_sha256"),
        },
    }
    if manifest["maximum_absolute_thickness_closure_m"] > manifest["thickness_closure_tolerance_m"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError("ROMS W-level thickness closure exceeds tolerance")
    write_json_atomic(manifest_output or destination.parent / "extraction_manifest.json", manifest)
    return manifest


def compact_health(path: str | Path, request: Mapping[str, Any], *, coverage_threshold: float = 0.95) -> dict[str, Any]:
    netCDF4, np = _modules()
    critical: list[str] = []
    warnings: list[str] = []
    source_records: list[dict[str, str]] = []
    source_provenance: Mapping[str, Any] | None = None
    source = Path(path)
    with netCDF4.Dataset(source) as ds:
        if getattr(ds, "schema_version", None) != COMPACT_SCHEMA_VERSION:
            critical.append("compact schema_version is missing or invalid")
        if "time" not in ds.variables:
            return {
                "path": str(source.resolve()), "size": source.stat().st_size,
                "sha256": sha256_file(source), "status": "fail",
                "critical_findings": ["compact output has no time coordinate"],
                "warnings": [], "variables": {},
            }
        if getattr(ds, "request_sha256", None) != canonical_json_sha256(request):
            critical.append("compact request provenance does not match the requested contract")
        try:
            values: dict[str, list[str]] = {}
            for name in ("source_key", "source_archive", "source_url"):
                variable = ds.variables.get(name)
                if variable is None or not variable.dimensions or variable.dimensions[0] != "time":
                    raise ValueError(f"compact output has no time-indexed {name}")
                values[name] = [
                    str(value).rstrip("\x00")
                    for value in netCDF4.chartostring(variable[:]).tolist()
                ]
            if len({len(item) for item in values.values()}) != 1:
                raise ValueError("compact record source provenance lengths are inconsistent")
            model = str(request.get("schema_version", "")).removesuffix("_request_v2")
            for key, source_archive, url in zip(
                    values["source_key"], values["source_archive"], values["source_url"]):
                if source_archive not in {"aws_operational", "ncei_long_term", "unbound"}:
                    raise ValueError(f"unsupported compact source archive: {source_archive!r}")
                if source_archive == "unbound":
                    if url:
                        raise ValueError("unbound compact record unexpectedly claims a source URL")
                else:
                    expected_url = archive_sources.canonical_object_url(source_archive, model, key)
                    if url != expected_url:
                        raise ValueError("compact record source URL is not canonical for its archive/key")
                source_records.append({"source_key": key, "source_archive": source_archive,
                                       "source_url": url})
            source_provenance = json.loads(str(getattr(ds, "source_provenance_json", "")))
            archives = json.loads(str(getattr(ds, "source_archives_json", "")))
            endpoints = json.loads(str(getattr(ds, "source_endpoints_json", "")))
            if (not isinstance(source_provenance, Mapping)
                    or source_provenance.get("archives") != archives
                    or source_provenance.get("endpoints") != endpoints
                    or source_provenance.get("record_count") != len(source_records)):
                raise ValueError("compact global archive/endpoint JSON is inconsistent")
            expected_archive_ids = {item["source_archive"] for item in source_records}
            if (set(endpoints) != expected_archive_ids
                    or {item.get("source_archive") for item in archives} != expected_archive_ids):
                raise ValueError("compact global archives do not cover its record sources exactly")
            for item in archives:
                if item["source_archive"] == "unbound":
                    if any(item.get(field) for field in ("container", "endpoint", "listing_endpoint")):
                        raise ValueError("unbound compact source claims a remote archive endpoint")
                    warnings.append("compact source provenance is unbound to a verified fetch manifest")
                else:
                    descriptor = archive_sources.get_source_descriptor(item["source_archive"], model)
                    for field in ("archive_role", "container", "endpoint", "listing_endpoint"):
                        if item.get(field) != descriptor[field]:
                            raise ValueError(f"compact global source {field} is not canonical")
                    if endpoints.get(item["source_archive"]) != descriptor["endpoint"]:
                        raise ValueError("compact global source endpoint mapping is not canonical")
        except Exception as exc:
            critical.append(f"compact source provenance is invalid: {exc}")
        time = np.asarray(ds.variables["time"][:], dtype=float)
        if time.size == 0 or np.any(np.diff(time) <= 0):
            critical.append("time is empty or not strictly increasing")
        expected_stamps = expected_times(
            parse_utc(request["start_utc"]), parse_utc(request["end_utc_exclusive"]), 3600,
        )
        actual_stamps = [datetime.fromtimestamp(float(value), tz=UTC) for value in time]
        if actual_stamps != expected_stamps:
            missing = [iso_utc(value) for value in expected_stamps if value not in set(actual_stamps)]
            extra = [iso_utc(value) for value in actual_stamps if value not in set(expected_stamps)]
            critical.append(
                f"compact timestamps do not exactly match request (missing={missing[:8]}, extra={extra[:8]})"
            )
        if time.size > 1 and np.any(np.abs(np.diff(time) - 3600) > 1):
            critical.append("compact time cadence is not hourly")
        try:
            if "h" not in ds.variables:
                raise ValueError("compact output has no rho-grid bathymetry")
            rho_shape = tuple(ds.variables["h"].shape)
            mask, _ = _strict_mask(ds, "mask_rho", rho_shape)
            _strict_mask(ds, "mask_u", (rho_shape[0], rho_shape[1] - 1))
            _strict_mask(ds, "mask_v", (rho_shape[0] - 1, rho_shape[1]))
            _vertical_contract(ds)
            angle_contract = _angle_contract(ds, rho_shape, mask)
            if getattr(ds, "angle_convention", None) != ANGLE_CONVENTION:
                raise ValueError("compact output has no canonical angle_convention provenance")
            if getattr(ds.variables["angle"], "angle_convention", None) != ANGLE_CONVENTION:
                raise ValueError("compact angle variable has no canonical convention provenance")
            if angle_contract["convention"] != ANGLE_CONVENTION:
                raise ValueError("compact angle convention is invalid")
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            critical.append(f"compact geometry/metadata is invalid: {exc}")
            mask = np.empty((0, 0), dtype=bool)
        expected_output_names: set[str] = set()
        for source_name in request["variables"]:
            if source_name == "zeta":
                expected_output_names.add("zeta")
            elif source_name in {"u", "v"}:
                for view in request["vertical_views"]:
                    suffix = view_suffix(view)
                    expected_output_names.update({
                        f"u_native_{suffix}", f"v_native_{suffix}",
                        f"eastward_velocity_{suffix}", f"northward_velocity_{suffix}",
                        f"current_speed_{suffix}",
                    })
            else:
                for view in request["vertical_views"]:
                    prefix = "salinity" if source_name == "salt" else source_name
                    expected_output_names.add(f"{prefix}_{view_suffix(view)}")
        absent = sorted(expected_output_names - set(ds.variables))
        if absent:
            critical.append("compact output is missing requested variables: " + ", ".join(absent))
        checks: dict[str, Any] = {}
        for name in sorted(expected_output_names & set(ds.variables)):
            variable = ds.variables[name]
            if not variable.dimensions or variable.dimensions[0] != "time" or variable.ndim != 3:
                critical.append(f"{name} does not have the expected time-first 3-D shape")
                continue
            if name.startswith("u_native_"):
                grid_mask = np.asarray(ds.variables["mask_u"][:] == 1) if "mask_u" in ds.variables else None
            elif name.startswith("v_native_"):
                grid_mask = np.asarray(ds.variables["mask_v"][:] == 1) if "mask_v" in ds.variables else None
            else:
                grid_mask = mask
            if grid_mask is None or variable.shape[-2:] != grid_mask.shape:
                critical.append(f"{name} grid shape is inconsistent with its native wet mask")
                continue
            data = _filled(variable[:], dtype=float)
            frame_coverage = [float(np.isfinite(frame[grid_mask]).mean()) for frame in data]
            if not frame_coverage:
                critical.append(f"{name} has no time frames")
                checks[name] = {"minimum_finite_wet_coverage": None, "frame_coverage": []}
                continue
            checks[name] = {"minimum_finite_wet_coverage": min(frame_coverage), "frame_coverage": frame_coverage}
            if any(value < coverage_threshold for value in frame_coverage):
                critical.append(f"{name} finite wet coverage is below {coverage_threshold:.0%}")
            if any(not np.isfinite(frame[grid_mask]).any() for frame in data):
                critical.append(f"{name} has an all-NaN wet frame")
        for name in sorted(expected_output_names & set(ds.variables)):
            if name.startswith("current_speed_"):
                if name not in checks:
                    continue
                suffix = name[len("current_speed_"):]
                east_name, north_name = f"eastward_velocity_{suffix}", f"northward_velocity_{suffix}"
                if east_name not in ds.variables or north_name not in ds.variables:
                    critical.append(f"{name} is missing paired earth-relative components")
                    continue
                speed = _filled(ds.variables[name][:], dtype=float)
                expected_speed = np.hypot(_filled(ds.variables[east_name][:], dtype=float), _filled(ds.variables[north_name][:], dtype=float))
                error = float(np.nanmax(np.abs(speed - expected_speed)))
                checks[name]["maximum_speed_consistency_error"] = error
                if error > 1e-5:
                    critical.append(f"{name} is inconsistent with paired components")
        for name, limits in (("salinity_", (-2.0, 50.0)), ("current_speed_", (0.0, 10.0))):
            values = [
                _filled(ds.variables[var][:], dtype=float)
                for var in ds.variables if var.startswith(name)
            ]
            if values:
                minimum = min(float(np.nanmin(value)) for value in values)
                maximum = max(float(np.nanmax(value)) for value in values)
                if minimum < limits[0] or maximum > limits[1]:
                    warnings.append(f"{name.rstrip('_')} range [{minimum:.3g}, {maximum:.3g}] exceeds broad plausibility limits {limits}")
    return {
        "path": str(source.resolve()), "size": source.stat().st_size, "sha256": sha256_file(source),
        "status": "pass" if not critical else "fail", "critical_findings": critical,
        "warnings": warnings, "variables": checks,
        "source_records": source_records, "source_provenance": source_provenance,
    }


def raw_consistency(
    paths: Sequence[str | Path],
    request: Mapping[str, Any],
    *,
    selection_output: str | Path | None = None,
    coverage_threshold: float = 0.95,
) -> dict[str, Any]:
    netCDF4, np = _modules()
    critical: list[str] = []
    warnings: list[str] = []
    reference = None
    raw_times: list[datetime] = []
    files: list[dict[str, Any]] = []
    selected_records: dict[datetime, dict[str, Any]] = {}
    duplicate_records: list[dict[str, Any]] = []
    start = parse_utc(request["start_utc"])
    end = parse_utc(request["end_utc_exclusive"])
    cadence = 360 if request["product"] == "stations" else 3600
    for path in paths:
        try:
            with netCDF4.Dataset(path) as ds:
                times = decode_ocean_times(ds)
                raw_times.extend(times)
                for index, raw_time in enumerate(times):
                    stamp, offset = normalize_time(raw_time, cadence)
                    if not start <= stamp < end:
                        continue
                    candidate = {
                        "path": str(Path(path).resolve()), "time_index": index,
                        "raw_time_utc": iso_utc(raw_time), "normalized_time_utc": iso_utc(stamp),
                        "normalization_offset_seconds": offset,
                    }
                    prior = selected_records.get(stamp)
                    if prior is None:
                        selected_records[stamp] = candidate
                    else:
                        duplicate_records.append({
                            "time_utc": iso_utc(stamp), "kept": prior["path"],
                            "discarded": candidate["path"],
                            "rule": "preceding_cycle_terminal_record",
                        })
                geometry = geometry_snapshot(ds) if request["product"] == "fields" else None
                if geometry is not None:
                    if reference is None:
                        reference = geometry
                    else:
                        assert_geometry(reference, geometry, path)
                    s_rho = _filled(ds.variables["s_rho"][:], dtype=float)
                    if np.any(np.diff(s_rho) == 0) or not (np.all(np.diff(s_rho) > 0) or np.all(np.diff(s_rho) < 0)):
                        critical.append(f"{path}: s_rho is not strictly monotonic")
                files.append({"path": str(Path(path).resolve()), "times_utc": [iso_utc(value) for value in times], "grid_hash": geometry and geometry["grid_hash"]})
        except Exception as exc:
            critical.append(f"{path}: {type(exc).__name__}: {exc}")
    normalized = [normalize_time(value, cadence)[0] for value in raw_times]
    unique = sorted(set(normalized))
    if len(unique) < len(normalized):
        warnings.append(f"raw sources contain {len(normalized) - len(unique)} duplicate normalized timestamps")
    expected = expected_times(start, end, cadence)
    selected_times = sorted(selected_records)
    missing = [iso_utc(stamp) for stamp in expected if stamp not in selected_records]
    extra = [iso_utc(stamp) for stamp in selected_times if stamp not in set(expected)]
    if request["missing_policy"] == "error" and missing:
        critical.append("raw sources are missing requested timestamps: " + ", ".join(missing[:8]))
    if extra:
        critical.append("raw selection contains off-contract timestamps: " + ", ".join(extra[:8]))
    selected_coverage: dict[str, list[float]] = {}
    if request["product"] == "stations":
        netCDF4, np = _modules()
        station_variables: set[str] | None = None
        for stamp in selected_times:
            record = selected_records[stamp]
            with netCDF4.Dataset(record["path"]) as ds:
                dynamic = {
                    name for name, variable in ds.variables.items()
                    if variable.dimensions and variable.dimensions[0] in {"ocean_time", "time"}
                    and name not in {"ocean_time", "time"}
                }
                station_variables = dynamic if station_variables is None else station_variables & dynamic
                for name in sorted(dynamic):
                    values = _variable_data(ds.variables[name], int(record["time_index"]))
                    selected_coverage.setdefault(name, []).append(float(np.isfinite(values).mean()))
        if not station_variables:
            critical.append("station passthrough has no dynamic data variables")
        for name, values in selected_coverage.items():
            if any(value < coverage_threshold for value in values):
                critical.append(f"station variable {name} finite coverage is below {coverage_threshold:.0%}")
    selection = {
        "schema_version": f"{request['product']}_cropped_selection_v1",
        "created_utc": iso_utc(datetime.now(UTC)),
        "request": dict(request), "request_sha256": canonical_json_sha256(request),
        "cadence_seconds": cadence,
        "expected_times_utc": [iso_utc(stamp) for stamp in expected],
        "selected_records": [selected_records[stamp] for stamp in selected_times],
        "missing_times": missing, "extra_times": extra,
        "duplicate_records": duplicate_records,
        "finite_coverage": selected_coverage,
    }
    if selection_output:
        write_json_atomic(selection_output, selection)
    return {
        "status": "pass" if not critical else "fail", "critical_findings": critical,
        "warnings": warnings, "files": files, "unique_time_count": len(unique),
        "requested_time_selection": selection,
        "selection_artifact": str(Path(selection_output).resolve()) if selection_output else None,
    }


def delete_raw_cache(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    paths = manifest_paths(run_path)
    raw_root = (run_path / "cache" / "raw").resolve()
    removed: list[dict[str, Any]] = []
    for path in paths:
        path = path.resolve()
        try:
            path.relative_to(raw_root)
        except ValueError as exc:
            raise RuntimeError(f"refusing to delete manifest path outside the run cache: {path}") from exc
        if path.is_file():
            removed.append({"path": str(path), "size": path.stat().st_size})
            path.unlink()
        for suffix in (".download.json", ".part", ".part.json"):
            path.with_name(path.name + suffix).unlink(missing_ok=True)
    record = {
        "schema_version": "dbofs_raw_cache_deletion_v1",
        "created_utc": iso_utc(datetime.now(UTC)),
        "removed": removed, "removed_bytes": sum(item["size"] for item in removed),
        "recovery": "re-download from source using a newly reviewed plan",
    }
    write_json_atomic(run_path / "raw_cache_deletion.json", record)
    return record


def evaluate_health(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    config: ModelConfig,
    *,
    output: str | Path | None = None,
    plots_dir: str | Path | None = None,
) -> dict[str, Any]:
    normalized = normalize_request_input(request, config)
    run_path = Path(run_dir)
    transfer = verify_transfers(run_path, expected_request=normalized)
    legacy_v1 = bool(transfer.get("legacy_read_only"))
    paths = [Path(item["local_path"]) for item in transfer.get("objects", []) if item.get("status") == "pass"]
    selection_path = run_path / f"{normalized['product']}_cropped_selection.json"
    raw = raw_consistency(
        paths, normalized,
        selection_output=selection_path if normalized["product"] != "fields" else None,
    ) if transfer["status"] == "pass" and paths else {
        "status": "fail", "critical_findings": ["no integrity-verified raw files"],
        "warnings": [], "files": [],
    }
    extraction_manifest_path = run_path / "extraction_manifest.json"
    extraction_manifest: dict[str, Any] | None = None
    compact_path: Path | None = None
    manifest_findings: list[str] = []
    if normalized["product"] == "fields":
        if legacy_v1:
            manifest_findings.append(
                "legacy v1 extraction evidence is read-only; create a new v2 plan before transfer or re-extraction")
        else:
            pass
        if not legacy_v1 and not extraction_manifest_path.is_file():
            manifest_findings.append("extraction_manifest.json is missing")
        elif not legacy_v1:
            try:
                extraction_manifest = read_json(extraction_manifest_path)
                if extraction_manifest.get("schema_version") != f"{config.model}_extraction_manifest_v2":
                    manifest_findings.append("extraction manifest schema_version is invalid")
                if canonical_json_sha256(extraction_manifest.get("request")) != canonical_json_sha256(normalized):
                    manifest_findings.append("extraction manifest request does not match health request")
                output_meta = extraction_manifest.get("output")
                if not isinstance(output_meta, Mapping) or not output_meta.get("path"):
                    manifest_findings.append("extraction manifest has no compact output metadata")
                else:
                    compact_path = Path(str(output_meta["path"])).resolve()
                    if not compact_path.is_file():
                        manifest_findings.append("extraction manifest compact output is missing")
                    elif compact_path.stat().st_size != output_meta.get("size"):
                        manifest_findings.append("compact output size does not match extraction manifest")
                    elif sha256_file(compact_path) != output_meta.get("sha256"):
                        manifest_findings.append("compact output SHA-256 does not match extraction manifest")
                closure = extraction_manifest.get("maximum_absolute_thickness_closure_m")
                tolerance = extraction_manifest.get("thickness_closure_tolerance_m", 1e-5)
                if not isinstance(closure, (int, float)) or not isinstance(tolerance, (int, float)) or closure > tolerance:
                    manifest_findings.append("ROMS vertical thickness closure is missing or exceeds tolerance")
                provenance = extraction_manifest.get("fetch_manifest")
                if not isinstance(provenance, Mapping) or provenance.get("sha256") != transfer.get("manifest_sha256"):
                    manifest_findings.append("extraction manifest is not bound to the verified fetch manifest")
                extraction_records = extraction_manifest.get("records")
                transfer_objects = transfer.get("objects")
                if (not isinstance(extraction_records, list) or not extraction_records
                        or not isinstance(transfer_objects, list) or not transfer_objects):
                    manifest_findings.append("extraction/transfer record provenance is missing")
                else:
                    source_by_key = {
                        item.get("key"): item for item in transfer_objects
                        if isinstance(item, Mapping) and item.get("status") == "pass"
                    }
                    for record in extraction_records:
                        source = source_by_key.get(record.get("source_key")) if isinstance(record, Mapping) else None
                        if (source is None
                                or record.get("source_archive") != source.get("source_archive")
                                or record.get("source_url") != source.get("url")
                                or str(Path(str(record.get("path", ""))).resolve())
                                != str(Path(str(source.get("local_path", ""))).resolve())):
                            manifest_findings.append(
                                "extraction record archive/URL/path is outside the verified fetch binding")
                            break
                    archives = {}
                    for source in source_by_key.values():
                        identity = (str(source["source_archive"]), str(source["endpoint"]))
                        archives[identity] = {
                            "source_archive": identity[0],
                            "archive_role": str(source.get("archive_role") or ""),
                            "container": str(source.get("container") or ""),
                            "endpoint": identity[1],
                            "listing_endpoint": str(source.get("listing_endpoint") or ""),
                        }
                    archive_values = sorted(
                        archives.values(), key=lambda item: (item["source_archive"], item["endpoint"]))
                    expected_source_summary = {
                        "record_count": len(extraction_records),
                        "archive_count": len(archive_values),
                        "archives": archive_values,
                        "endpoints": {item["source_archive"]: item["endpoint"] for item in archive_values},
                    }
                    if extraction_manifest.get("source_provenance") != expected_source_summary:
                        manifest_findings.append(
                            "extraction global archive/endpoint provenance is inconsistent")
            except Exception as exc:
                manifest_findings.append(f"extraction manifest is unreadable: {type(exc).__name__}: {exc}")
    compact = (
        compact_health(compact_path, normalized)
        if normalized["product"] == "fields" and compact_path is not None and compact_path.is_file()
        else None
    )
    critical = list(transfer.get("failures", [])) + list(raw.get("critical_findings", []))
    critical.extend(manifest_findings)
    warnings = list(raw.get("warnings", []))
    if compact is not None:
        if extraction_manifest is not None:
            expected_records = [
                {name: str(item.get(name) or "")
                 for name in ("source_key", "source_archive", "source_url")}
                for item in extraction_manifest.get("records", [])
                if isinstance(item, Mapping)
            ]
            if compact.get("source_records") != expected_records:
                critical.append("compact record-level source provenance does not match extraction manifest")
            if compact.get("source_provenance") != extraction_manifest.get("source_provenance"):
                critical.append("compact global archive/endpoint provenance does not match extraction manifest")
        critical.extend(compact["critical_findings"])
        warnings.extend(compact["warnings"])
    elif normalized["product"] == "fields":
        critical.append("compact output could not be verified from extraction_manifest.json")
    plot_paths: list[str] = []
    if plots_dir and compact is not None:
        try:
            import matplotlib.pyplot as plt

            destination = Path(plots_dir) / "finite_wet_coverage.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            for name, values in compact["variables"].items():
                ax.plot(values["frame_coverage"], label=name)
            ax.axhline(0.95, color="red", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1.02); ax.set_xlabel("Frame"); ax.set_ylabel("Finite wet fraction")
            if compact["variables"]:
                ax.legend(fontsize=6, ncol=2)
            fig.tight_layout(); fig.savefig(destination, dpi=140); plt.close(fig)
            plot_paths.append(str(destination.resolve()))
        except Exception as exc:
            warnings.append(f"health diagnostic plot failed: {type(exc).__name__}: {exc}")
    report = {
        "schema_version": f"{config.model}_health_v2", "created_utc": iso_utc(datetime.now(UTC)),
        "request": normalized, "status": "pass" if not critical else "fail",
        "critical_findings": critical, "warnings": warnings, "transfer_integrity": transfer,
        "raw_source_consistency": raw, "compact_output": compact, "diagnostic_plots": plot_paths,
        "extraction_manifest": None if extraction_manifest is None else {
            "path": str(extraction_manifest_path.resolve()),
            "sha256": sha256_file(extraction_manifest_path),
        },
        "raw_cache_deletion": None,
    }
    if not critical and normalized["cache_policy"] == "delete_after_extract" and normalized["product"] == "fields":
        report["raw_cache_deletion"] = delete_raw_cache(run_path)
    write_json_atomic(output or run_path / "health_check.json", report)
    return report
