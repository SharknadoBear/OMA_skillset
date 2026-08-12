#!/usr/bin/env python3
"""Shared ROMS C-grid extraction and health routines for OFS connectors."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import roms_aws_core as aws
except ImportError:  # direct script/module loading
    import roms_aws_core as aws

UTC = timezone.utc
COMPACT_SCHEMA_VERSION = "roms_compact_fields_v1"
ANGLE_CONVENTION = "xi_axis_counterclockwise_from_east_radians"


def _modules():
    try:
        import netCDF4
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("netCDF4 and numpy are required for ROMS extraction") from exc
    return netCDF4, np


def _filled(value: Any):
    _, np = _modules()
    array = np.ma.filled(value, np.nan).astype(np.float64, copy=False)
    array[~np.isfinite(array)] = np.nan
    return array


def _coordinate(variable: Any):
    """Read a ROMS coordinate without netCDF4 valid_min/max auto-masking.

    Live NOAA files may store the nominal zero endpoint of ``Cs_w`` as a
    tiny positive roundoff value while declaring ``valid_max=0``. NetCDF4's
    automatic valid-range mask would otherwise remove the surface W point.
    Explicit fill values are still honored here.
    """
    _, np = _modules()
    variable.set_auto_mask(False)
    try:
        result = np.asarray(variable[:], dtype=np.float64)
    finally:
        variable.set_auto_mask(True)
    for name in ("_FillValue", "missing_value"):
        if hasattr(variable, name):
            marker = np.asarray(getattr(variable, name)).reshape(-1)[0]
            result = np.where(result == marker, np.nan, result)
    result[~np.isfinite(result)] = np.nan
    return result


def _source_values(variable: Any, index: Any):
    """Read source data without applying declared plausibility ranges."""
    _, np = _modules()
    variable.set_auto_mask(False)
    try:
        result = np.asarray(variable[index], dtype=np.float64)
    finally:
        variable.set_auto_mask(True)
    for name in ("_FillValue", "missing_value"):
        if hasattr(variable, name):
            for marker in np.asarray(getattr(variable, name)).reshape(-1):
                result = np.where(result == marker, np.nan, result)
    result[~np.isfinite(result)] = np.nan
    return result


def _finite_pair_mean(first: Any, second: Any):
    _, np = _modules()
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    count = np.isfinite(a).astype(np.int8) + np.isfinite(b).astype(np.int8)
    total = np.where(np.isfinite(a), a, 0.0) + np.where(np.isfinite(b), b, 0.0)
    result = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _scalar(dataset: Any, name: str, default: Any = None) -> Any:
    _, np = _modules()
    if name in dataset.variables:
        value = dataset.variables[name][:]
        return np.asarray(value).reshape(-1)[0].item()
    if hasattr(dataset, name):
        return getattr(dataset, name)
    if default is not None:
        return default
    raise RuntimeError(f"required ROMS scalar {name!r} is missing")


def roms_depths(sigma: Any, stretching: Any, h: Any, zeta: Any,
                hc: float, vtransform: int):
    """Return ROMS depths with shape ``(sigma, ...)`` for transform 1 or 2."""
    _, np = _modules()
    s = np.asarray(sigma, dtype=np.float64)
    cs = np.asarray(stretching, dtype=np.float64)
    bathy = np.asarray(h, dtype=np.float64)
    surface = np.asarray(zeta, dtype=np.float64)
    if s.ndim != 1 or cs.shape != s.shape:
        raise ValueError("sigma and stretching must be equal-length one-dimensional arrays")
    if bathy.shape != surface.shape:
        raise ValueError("h and zeta must have identical horizontal shapes")
    if np.any(np.isfinite(bathy) & (bathy <= 0)):
        raise ValueError("finite bathymetry must be positive")
    expand = (slice(None),) + (None,) * bathy.ndim
    ss, cc = s[expand], cs[expand]
    if int(vtransform) == 1:
        z0 = (ss - cc) * float(hc) + cc * bathy[None, ...]
        depth = z0 + surface[None, ...] * (1.0 + z0 / bathy[None, ...])
    elif int(vtransform) == 2:
        z0 = (float(hc) * ss + bathy[None, ...] * cc) / (float(hc) + bathy[None, ...])
        depth = surface[None, ...] + (surface[None, ...] + bathy[None, ...]) * z0
    else:
        raise ValueError(f"unsupported ROMS Vtransform={vtransform!r}; expected 1 or 2")
    return depth


def layer_thickness(s_w: Any, cs_w: Any, h: Any, zeta: Any,
                    hc: float, vtransform: int):
    _, np = _modules()
    return np.abs(np.diff(roms_depths(s_w, cs_w, h, zeta, hc, vtransform), axis=0))


def weighted_vertical_average(data: Any, weights: Any, wet_mask: Any | None = None):
    """Average over sigma with finite-layer renormalization."""
    _, np = _modules()
    values = np.asarray(data, dtype=np.float64)
    layer_weights = np.asarray(weights, dtype=np.float64)
    if values.shape != layer_weights.shape:
        raise ValueError(f"data and weights must have identical shapes: {values.shape} != {layer_weights.shape}")
    valid = np.isfinite(values) & np.isfinite(layer_weights) & (layer_weights > 0)
    numerator = np.sum(np.where(valid, values * layer_weights, 0.0), axis=0)
    denominator = np.sum(np.where(valid, layer_weights, 0.0), axis=0)
    result = np.full(denominator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    if wet_mask is not None:
        result = np.where(np.asarray(wet_mask) == 1, result, np.nan)
    return result


def rho_to_u(value: Any):
    _, np = _modules()
    array = np.asarray(value, dtype=np.float64)
    return _finite_pair_mean(array[..., :, :-1], array[..., :, 1:])


def rho_to_v(value: Any):
    _, np = _modules()
    array = np.asarray(value, dtype=np.float64)
    return _finite_pair_mean(array[..., :-1, :], array[..., 1:, :])


def destagger_u(u: Any, mask_rho: Any | None = None):
    """Move U-edge values to rho points using finite-aware adjacent means."""
    _, np = _modules()
    edge = np.asarray(u, dtype=np.float64)
    output = np.full(edge.shape[:-1] + (edge.shape[-1] + 1,), np.nan)
    output[..., 0] = edge[..., 0]
    output[..., -1] = edge[..., -1]
    output[..., 1:-1] = _finite_pair_mean(edge[..., :-1], edge[..., 1:])
    if mask_rho is not None:
        output = np.where(np.asarray(mask_rho) == 1, output, np.nan)
    return output


def destagger_v(v: Any, mask_rho: Any | None = None):
    """Move V-edge values to rho points using finite-aware adjacent means."""
    _, np = _modules()
    edge = np.asarray(v, dtype=np.float64)
    output = np.full(edge.shape[:-2] + (edge.shape[-2] + 1, edge.shape[-1]), np.nan)
    output[..., 0, :] = edge[..., 0, :]
    output[..., -1, :] = edge[..., -1, :]
    output[..., 1:-1, :] = _finite_pair_mean(edge[..., :-1, :], edge[..., 1:, :])
    if mask_rho is not None:
        output = np.where(np.asarray(mask_rho) == 1, output, np.nan)
    return output


def rotate_to_earth(u_rho: Any, v_rho: Any, angle: Any):
    _, np = _modules()
    u = np.asarray(u_rho, dtype=np.float64)
    v = np.asarray(v_rho, dtype=np.float64)
    theta = np.asarray(angle, dtype=np.float64)
    if u.shape != v.shape or u.shape != theta.shape:
        raise ValueError("rho-grid u, v, and angle must have identical shapes")
    east = u * np.cos(theta) - v * np.sin(theta)
    north = u * np.sin(theta) + v * np.cos(theta)
    return east, north, np.hypot(east, north)


def _angle_metadata(variable: Any, values: Any, rho_shape: tuple[int, ...]) -> dict[str, str]:
    """Validate the ROMS angle convention before any vector rotation."""
    _, np = _modules()
    theta = np.asarray(values, dtype=np.float64)
    if theta.shape != rho_shape or tuple(variable.dimensions) != ("eta_rho", "xi_rho"):
        raise RuntimeError("angle must be a two-dimensional rho-grid variable")
    if not np.all(np.isfinite(theta)):
        raise RuntimeError("angle must be finite on the complete rho grid")
    units = str(getattr(variable, "units", "")).strip().lower()
    if units not in {"radian", "radians", "rad"}:
        raise RuntimeError("angle units must explicitly be radians; degrees or missing units are rejected")
    standard_name = str(getattr(variable, "standard_name", "")).strip()
    long_name = str(getattr(variable, "long_name", "")).strip()
    normalized_long_name = re.sub(r"[^a-z0-9]+", " ", long_name.lower()).split()
    semantic_ok = standard_name == "grid_angle_of_rotation_from_east_to_y" or (
        "angle" in normalized_long_name and "xi" in normalized_long_name
        and "east" in normalized_long_name
    )
    if not semantic_ok:
        raise RuntimeError("angle metadata does not establish the ROMS XI-axis-from-east convention")
    if theta.size and float(np.max(np.abs(theta))) > 2.0 * np.pi + 1e-10:
        raise RuntimeError("angle values exceed the valid radian range")
    return {
        "angle_units": units,
        "angle_standard_name": standard_name,
        "angle_long_name": long_name,
        "angle_convention": ANGLE_CONVENTION,
    }


def _mask(dataset: Any, name: str, fallback: Any):
    _, np = _modules()
    return np.asarray(dataset.variables[name][:], dtype=np.int8) if name in dataset.variables else np.asarray(fallback, dtype=np.int8)


def read_geometry(dataset: Any) -> dict[str, Any]:
    _, np = _modules()
    required = ("lon_rho", "lat_rho", "h", "angle", "s_rho", "s_w", "Cs_r", "Cs_w")
    missing = [name for name in required if name not in dataset.variables]
    if missing:
        raise RuntimeError(f"ROMS geometry is missing: {', '.join(missing)}")
    lon_rho, lat_rho = _filled(dataset.variables["lon_rho"][:]), _filled(dataset.variables["lat_rho"][:])
    h = _filled(dataset.variables["h"][:])
    angle_variable = dataset.variables["angle"]
    angle = _filled(angle_variable[:])
    if lon_rho.shape != lat_rho.shape or lon_rho.shape != h.shape or h.shape != angle.shape:
        raise RuntimeError("rho-grid lon/lat/h/angle shapes are inconsistent")
    mask_rho = _mask(dataset, "mask_rho", np.isfinite(h))
    mask_u = _mask(dataset, "mask_u", mask_rho[:, :-1] * mask_rho[:, 1:])
    mask_v = _mask(dataset, "mask_v", mask_rho[:-1, :] * mask_rho[1:, :])
    for name, mask in (("mask_rho", mask_rho), ("mask_u", mask_u), ("mask_v", mask_v)):
        if not set(np.unique(mask)).issubset({0, 1}):
            raise RuntimeError(f"{name} must use 0=land and 1=wet semantics")
    wet = mask_rho == 1
    if not np.all(np.isfinite(lon_rho[wet]) & np.isfinite(lat_rho[wet]) &
                  np.isfinite(h[wet]) & (h[wet] > 0) & np.isfinite(angle[wet])):
        raise RuntimeError("rho-grid wet coordinates, h, and angle must be finite with positive h")
    angle_metadata = _angle_metadata(angle_variable, angle, h.shape)
    lon_u = _filled(dataset.variables["lon_u"][:]) if "lon_u" in dataset.variables else rho_to_u(lon_rho)
    lat_u = _filled(dataset.variables["lat_u"][:]) if "lat_u" in dataset.variables else rho_to_u(lat_rho)
    lon_v = _filled(dataset.variables["lon_v"][:]) if "lon_v" in dataset.variables else rho_to_v(lon_rho)
    lat_v = _filled(dataset.variables["lat_v"][:]) if "lat_v" in dataset.variables else rho_to_v(lat_rho)
    s_rho = _coordinate(dataset.variables["s_rho"])
    s_w = _coordinate(dataset.variables["s_w"])
    cs_r = _coordinate(dataset.variables["Cs_r"])
    cs_w = _coordinate(dataset.variables["Cs_w"])
    if len(s_w) != len(s_rho) + 1 or cs_r.shape != s_rho.shape or cs_w.shape != s_w.shape:
        raise RuntimeError("ROMS sigma coordinate dimensions are inconsistent")
    vertical = {"s_rho": s_rho, "s_w": s_w, "Cs_r": cs_r, "Cs_w": cs_w}
    directions: dict[str, int] = {}
    for name, value in vertical.items():
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise RuntimeError(f"{name} must be a finite one-dimensional coordinate")
        difference = np.diff(value)
        if np.all(difference > 0):
            directions[name] = 1
        elif np.all(difference < 0):
            directions[name] = -1
        else:
            raise RuntimeError(f"{name} must be strictly monotonic")
    if len(set(directions.values())) != 1:
        raise RuntimeError("ROMS sigma and stretching coordinates must share one orientation")
    raw_vtransform = _scalar(dataset, "Vtransform")
    raw_vstretching = _scalar(dataset, "Vstretching")
    raw_hc = _scalar(dataset, "hc")
    try:
        vtransform_value = float(raw_vtransform)
        vstretching_value = float(raw_vstretching)
        hc_value = float(raw_hc)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Vtransform, Vstretching, and hc must be numeric scalars") from exc
    if (not np.isfinite(vtransform_value) or vtransform_value != int(vtransform_value)):
        raise RuntimeError("Vtransform must be a finite integer")
    if (not np.isfinite(vstretching_value) or vstretching_value != int(vstretching_value)
            or int(vstretching_value) < 1):
        raise RuntimeError("Vstretching must be a positive finite integer")
    if not np.isfinite(hc_value) or hc_value < 0:
        raise RuntimeError("hc must be finite and non-negative")
    vtransform = int(vtransform_value)
    if vtransform not in {1, 2}:
        raise RuntimeError(f"unsupported Vtransform={vtransform}")
    geometry = {
        "lon_rho": lon_rho, "lat_rho": lat_rho, "h": h, "angle": angle,
        "mask_rho": mask_rho, "lon_u": lon_u, "lat_u": lat_u, "mask_u": mask_u,
        "lon_v": lon_v, "lat_v": lat_v, "mask_v": mask_v,
        "s_rho": s_rho, "s_w": s_w, "Cs_r": cs_r, "Cs_w": cs_w,
        "hc": hc_value, "Vtransform": vtransform,
        "Vstretching": int(vstretching_value),
        **angle_metadata,
    }
    return geometry


def _geometry_hashes(geometry: Mapping[str, Any]) -> dict[str, str]:
    _, np = _modules()
    result = {}
    for name, value in geometry.items():
        if hasattr(value, "shape"):
            result[name] = hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
    return result


def assert_geometry(reference: Mapping[str, Any], candidate: Mapping[str, Any], source: Path) -> None:
    _, np = _modules()
    for name in ("lon_rho", "lat_rho", "h", "angle", "mask_rho", "lon_u", "lat_u",
                 "mask_u", "lon_v", "lat_v", "mask_v", "s_rho", "s_w", "Cs_r", "Cs_w"):
        if np.asarray(reference[name]).shape != np.asarray(candidate[name]).shape or not np.allclose(
                reference[name], candidate[name], rtol=0.0, atol=1e-10, equal_nan=True):
            raise RuntimeError(f"geometry/schema drift in {source}: {name}")
    for name in ("hc", "Vtransform", "Vstretching"):
        if reference[name] != candidate[name]:
            raise RuntimeError(f"geometry/schema drift in {source}: {name}")
    for name in ("angle_units", "angle_standard_name", "angle_long_name", "angle_convention"):
        if reference[name] != candidate[name]:
            raise RuntimeError(f"geometry/schema drift in {source}: {name}")


def _read_record(variable: Any, time_index: int):
    index = [slice(None)] * variable.ndim
    for axis, name in enumerate(variable.dimensions):
        if name in {"ocean_time", "time"}:
            index[axis] = time_index
    return _source_values(variable, tuple(index))


def _view_index(sigma: Any, view: str | int) -> int:
    _, np = _modules()
    values = np.asarray(sigma)
    if isinstance(view, int):
        if view >= len(values):
            raise ValueError(f"sigma index {view} is outside 0..{len(values)-1}")
        return view
    order = np.argsort(np.abs(values))
    if view == "surface":
        return int(order[0])
    if view == "near_surface":
        return int(order[1] if len(order) > 1 else order[0])
    if view == "bottom":
        return int(np.argmax(np.abs(values)))
    raise ValueError(f"{view!r} has no fixed sigma index")


def view_suffix(view: str | int) -> str:
    return f"sigma_{view}" if isinstance(view, int) else str(view)


def _horizontal_grid(variable: Any) -> str:
    dims = set(variable.dimensions)
    if {"eta_u", "xi_u"}.issubset(dims):
        return "u"
    if {"eta_v", "xi_v"}.issubset(dims):
        return "v"
    if {"eta_rho", "xi_rho"}.issubset(dims):
        return "rho"
    raise RuntimeError(f"unsupported ROMS horizontal dimensions for {variable.name}: {variable.dimensions}")


def _grid_arrays(geometry: Mapping[str, Any], grid: str, zeta_rho: Any):
    if grid == "rho":
        return geometry["h"], zeta_rho, geometry["mask_rho"]
    if grid == "u":
        return rho_to_u(geometry["h"]), rho_to_u(zeta_rho), geometry["mask_u"]
    if grid == "v":
        return rho_to_v(geometry["h"]), rho_to_v(zeta_rho), geometry["mask_v"]
    raise ValueError(grid)


def reduce_vertical(data: Any, view: str | int, grid: str,
                    geometry: Mapping[str, Any], zeta_rho: Any):
    _, np = _modules()
    values = np.asarray(data, dtype=np.float64)
    _, _, mask = _grid_arrays(geometry, grid, zeta_rho)
    if view == "depth_average":
        h, zeta, _ = _grid_arrays(geometry, grid, zeta_rho)
        weights = layer_thickness(geometry["s_w"], geometry["Cs_w"], h, zeta,
                                  geometry["hc"], geometry["Vtransform"])
        return weighted_vertical_average(values, weights, mask)
    return np.where(mask == 1, values[_view_index(geometry["s_rho"], view)], np.nan)


def _copy_attributes(source: Any, destination: Any) -> None:
    stale_after_reduction = {
        "_FillValue", "scale_factor", "add_offset", "time", "coordinates",
        "cell_methods", "grid", "location", "field",
    }
    for name in source.ncattrs():
        if name not in stale_after_reduction:
            try:
                destination.setncattr(name, source.getncattr(name))
            except (TypeError, ValueError):
                pass


def _source_key(config: aws.ModelConfig, path: Path) -> str:
    sidecar = path.with_name(path.name + ".download.json")
    if sidecar.is_file():
        try:
            key = aws.read_json(sidecar).get("key")
            if isinstance(key, str) and key.startswith(config.prefix):
                return key
        except Exception:
            pass
    parts = path.resolve().parts
    for index in range(len(parts) - 1):
        if parts[index:index + 2] == ("cache", "raw"):
            relative = "/".join(parts[index + 2:])
            if relative:
                return config.prefix + relative
    return path.name


def _source_metadata(config: aws.ModelConfig, path: Path,
                     bound: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Resolve record provenance, preferring a verified manifest binding."""
    if bound is not None:
        source_id = str(bound.get("source_archive") or bound.get("source_id") or "")
        key = str(bound.get("key") or "")
        url = str(bound.get("url") or "")
        endpoint = str(bound.get("endpoint") or "")
        if not source_id or not key or not url or not endpoint:
            raise ValueError("verified fetch binding has incomplete source archive provenance")
        return {
            "source_key": key, "source_archive": source_id, "source_url": url,
            "archive_role": str(bound.get("archive_role") or ""),
            "container": str(bound.get("container") or ""),
            "endpoint": endpoint,
            "listing_endpoint": str(bound.get("listing_endpoint") or ""),
        }
    sidecar = path.with_name(path.name + ".download.json")
    value: Mapping[str, Any] = {}
    if sidecar.is_file():
        try:
            candidate = aws.read_json(sidecar)
            if isinstance(candidate, Mapping):
                value = candidate
        except Exception:
            pass
    source_id = str(value.get("source_id") or "unbound")
    key = str(value.get("key") or _source_key(config, path))
    url = str(value.get("url") or "")
    if source_id in {"aws_operational", "ncei_long_term"}:
        descriptor = aws.archive_sources.get_source_descriptor(source_id, config.model)
    else:
        descriptor = {"archive_role": "unbound", "container": "", "endpoint": "",
                      "listing_endpoint": ""}
    return {
        "source_key": key, "source_archive": source_id, "source_url": url,
        "archive_role": descriptor["archive_role"], "container": descriptor["container"],
        "endpoint": descriptor["endpoint"],
        "listing_endpoint": descriptor["listing_endpoint"],
    }


def _source_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    archives: dict[tuple[str, str], dict[str, str]] = {}
    for record in records:
        identity = (str(record["source_archive"]), str(record["endpoint"]))
        archives[identity] = {
            "source_archive": identity[0],
            "archive_role": str(record.get("archive_role") or ""),
            "container": str(record.get("container") or ""),
            "endpoint": identity[1],
            "listing_endpoint": str(record.get("listing_endpoint") or ""),
        }
    values = sorted(archives.values(), key=lambda item: (item["source_archive"], item["endpoint"]))
    return {
        "record_count": len(records), "archive_count": len(values), "archives": values,
        "endpoints": {item["source_archive"]: item["endpoint"] for item in values},
    }


def _source_time_metadata(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize lossless source calendar metadata used during decoding."""
    fields = ("source_time_units", "source_calendar", "decoder_calendar",
              "calendar_alias_applied")
    unique = {tuple(item.get(name) for name in fields) for item in records}
    return [dict(zip(fields, values))
            for values in sorted(unique, key=lambda row: tuple(map(str, row)))]


def collect_records(config: aws.ModelConfig, paths: Sequence[str | Path],
                    start: datetime, end: datetime, *,
                    bound_objects: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    netCDF4, _ = _modules()
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        bound = None if bound_objects is None else bound_objects.get(str(path))
        source = _source_metadata(config, path, bound)
        with netCDF4.Dataset(path) as dataset:
            for item in aws.decode_times(dataset):
                stamp = aws.parse_utc(item["normalized_time_utc"])
                if start <= stamp < end:
                    records.append({**item, **source, "time": stamp, "path": path})
    # Stable first-source preference; field files normally contain one unique hour.
    records.sort(key=lambda item: (item["time"], str(item["path"])))
    unique: dict[datetime, dict[str, Any]] = {}
    duplicates = []
    for record in records:
        if record["time"] in unique:
            duplicates.append({"time_utc": aws.iso(record["time"]),
                               "preferred": str(unique[record["time"]]["path"]),
                               "rejected": str(record["path"])})
        else:
            unique[record["time"]] = record
    ordered = list(unique.values())
    for record in ordered:
        record["duplicates"] = duplicates
    return ordered


def _create_grid(output: Any, geometry: Mapping[str, Any]) -> None:
    dims = {
        "eta_rho": geometry["h"].shape[0], "xi_rho": geometry["h"].shape[1],
        "eta_u": geometry["mask_u"].shape[0], "xi_u": geometry["mask_u"].shape[1],
        "eta_v": geometry["mask_v"].shape[0], "xi_v": geometry["mask_v"].shape[1],
        "s_rho": len(geometry["s_rho"]), "s_w": len(geometry["s_w"]),
    }
    for name, size in dims.items():
        output.createDimension(name, size)
    definitions = {
        "lon_rho": (("eta_rho", "xi_rho"), "f8"), "lat_rho": (("eta_rho", "xi_rho"), "f8"),
        "h": (("eta_rho", "xi_rho"), "f8"), "angle": (("eta_rho", "xi_rho"), "f8"),
        "mask_rho": (("eta_rho", "xi_rho"), "i1"),
        "lon_u": (("eta_u", "xi_u"), "f8"), "lat_u": (("eta_u", "xi_u"), "f8"),
        "mask_u": (("eta_u", "xi_u"), "i1"),
        "lon_v": (("eta_v", "xi_v"), "f8"), "lat_v": (("eta_v", "xi_v"), "f8"),
        "mask_v": (("eta_v", "xi_v"), "i1"),
        "s_rho": (("s_rho",), "f8"), "s_w": (("s_w",), "f8"),
        "Cs_r": (("s_rho",), "f8"), "Cs_w": (("s_w",), "f8"),
    }
    for name, (var_dims, dtype) in definitions.items():
        var = output.createVariable(name, dtype, var_dims, zlib=True, complevel=4)
        var[:] = geometry[name]
        if name == "angle":
            var.setncattr("units", geometry["angle_units"])
            if geometry["angle_standard_name"]:
                var.setncattr("standard_name", geometry["angle_standard_name"])
            if geometry["angle_long_name"]:
                var.setncattr("long_name", geometry["angle_long_name"])
            var.setncattr("angle_convention", geometry["angle_convention"])
    output.setncattr("hc", geometry["hc"])
    output.setncattr("Vtransform", geometry["Vtransform"])
    output.setncattr("Vstretching", geometry["Vstretching"])


def _create_field(output: Any, name: str, dims: tuple[str, ...], source: Any | None = None,
                  *, standard_name: str | None = None, units: str | None = None):
    variable = output.createVariable(name, "f4", dims, zlib=True, complevel=4,
                                     fill_value=9.96921e36)
    if source is not None:
        _copy_attributes(source, variable)
        variable.setncattr("source_variable", source.name)
        variable.setncattr("source_dimensions", " ".join(source.dimensions))
    if standard_name:
        variable.setncattr("standard_name", standard_name)
    if units:
        variable.setncattr("units", units)
    return variable


def extract_fields(config: aws.ModelConfig, request: Mapping[str, Any] | str | Path,
                   paths: Sequence[str | Path], output_path: str | Path,
                   *, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    netCDF4, np = _modules()
    normalized = aws.load_request(config, request) if isinstance(request, (str, Path)) else aws.validate_request(config, request)
    if normalized["product"] != "fields":
        raise ValueError(f"{normalized['product']} is passthrough-only and cannot be extracted")
    if not paths:
        raise ValueError("at least one downloaded field file is required")
    fetch_binding: dict[str, Any] | None = None
    if provenance is not None:
        fetch_binding = dict(provenance)
        if (fetch_binding.get("schema_version") != f"{config.model}_verified_fetch_binding_v2"
                or fetch_binding.get("verified") is not True):
            raise ValueError("extraction provenance is not a verified fetch binding")
        if fetch_binding.get("request_sha256") != aws.canonical_json_sha256(normalized):
            raise ValueError("verified fetch binding request does not match extraction request")
        bound_objects = fetch_binding.get("objects")
        if not isinstance(bound_objects, list) or not bound_objects:
            raise ValueError("verified fetch binding has no objects")
        bound_paths = {str(Path(str(item.get("local_path", ""))).resolve())
                       for item in bound_objects if isinstance(item, Mapping)}
        supplied_paths = {str(Path(path).resolve()) for path in paths}
        if len(bound_paths) != len(bound_objects) or supplied_paths != bound_paths:
            raise ValueError("extraction inputs do not exactly match the verified fetch binding")
    start, end = aws.parse_utc(normalized["start_utc"]), aws.parse_utc(normalized["end_utc_exclusive"])
    bound_by_path = None
    if fetch_binding:
        bound_by_path = {
            str(Path(str(item["local_path"])).resolve()): item
            for item in fetch_binding["objects"]
        }
    records = collect_records(config, paths, start, end, bound_objects=bound_by_path)
    expected = aws.expected_times(start, end, 3600)
    missing = [aws.iso(stamp) for stamp in expected if stamp not in {item["time"] for item in records}]
    if missing and normalized["missing_policy"] == "error":
        raise RuntimeError(f"downloaded data are missing {len(missing)} requested hours: {', '.join(missing[:8])}")
    if not records:
        raise RuntimeError("no downloaded records overlap the request window")
    if fetch_binding:
        selected_paths = {str(Path(item["path"]).resolve()) for item in records}
        selected_keys = {str(item["source_key"]) for item in records}
        expected_paths = {str(item["local_path"]) for item in fetch_binding["objects"]}
        expected_keys = {str(item["key"]) for item in fetch_binding["objects"]}
        if selected_paths != expected_paths or selected_keys != expected_keys:
            raise RuntimeError("verified manifest objects do not exactly match selected extraction records")

    with netCDF4.Dataset(records[0]["path"]) as first:
        geometry = read_geometry(first)
        absent = [name for name in normalized["variables"] if name not in first.variables]
        if absent:
            raise RuntimeError(f"requested variables are unavailable: {', '.join(absent)}")
        source_meta = {name: {"dimensions": tuple(first.variables[name].dimensions),
                              "grid": _horizontal_grid(first.variables[name])}
                       for name in normalized["variables"]}
        if "u" in source_meta and source_meta["u"]["grid"] != "u":
            raise RuntimeError("source u is not on the expected ROMS U-edge grid")
        if "v" in source_meta and source_meta["v"]["grid"] != "v":
            raise RuntimeError("source v is not on the expected ROMS V-edge grid")
    for record in records[1:]:
        with netCDF4.Dataset(record["path"]) as candidate:
            assert_geometry(geometry, read_geometry(candidate), record["path"])
            for name, metadata in source_meta.items():
                if name not in candidate.variables or tuple(candidate.variables[name].dimensions) != metadata["dimensions"]:
                    raise RuntimeError(f"variable schema drift in {record['path']}: {name}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    coverage: dict[str, list[float]] = {}
    thickness_errors: list[float] = []
    source_summary = _source_summary(records)
    source_time_metadata = _source_time_metadata(records)
    with netCDF4.Dataset(temporary, "w", format="NETCDF4") as output:
        output.createDimension("time", None)
        output.createDimension("source_key_strlen", max(1, max(len(item["source_key"]) for item in records)))
        output.createDimension("source_archive_strlen", max(1, max(len(item["source_archive"]) for item in records)))
        output.createDimension("source_url_strlen", max(1, max(len(item["source_url"]) for item in records)))
        _create_grid(output, geometry)
        time_var = output.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
        time_var.calendar = "proleptic_gregorian"
        original_var = output.createVariable("original_time", "f8", ("time",))
        original_var.units = time_var.units
        original_var.calendar = time_var.calendar
        adjustment_var = output.createVariable("time_adjustment_seconds", "f8", ("time",))
        key_var = output.createVariable("source_key", "S1", ("time", "source_key_strlen"))
        key_var.long_name = "NOAA source object key"
        archive_var = output.createVariable(
            "source_archive", "S1", ("time", "source_archive_strlen"))
        archive_var.long_name = "NOAA archive source identifier"
        url_var = output.createVariable("source_url", "S1", ("time", "source_url_strlen"))
        url_var.long_name = "canonical NOAA source object URL"
        created: dict[str, Any] = {}

        with netCDF4.Dataset(records[0]["path"]) as sample:
            for name in normalized["variables"]:
                source = sample.variables[name]
                has_sigma = "s_rho" in source.dimensions
                grid = source_meta[name]["grid"]
                if not has_sigma:
                    dims = ("time",) + tuple(dim for dim in source.dimensions
                                             if dim not in {"ocean_time", "time"})
                    created[name] = _create_field(output, name, dims, source)
                    created[name].setncattr("coordinates", f"lon_{grid} lat_{grid}")
                    continue
                for view in normalized["vertical_views"]:
                    suffix = view_suffix(view)
                    out_name = ("salinity_" + suffix) if name == "salt" else f"{name}_{suffix}"
                    horizontal = ("eta_rho", "xi_rho") if grid == "rho" else (("eta_u", "xi_u") if grid == "u" else ("eta_v", "xi_v"))
                    created[out_name] = _create_field(output, out_name, ("time",) + horizontal, source)
                    created[out_name].setncattr("coordinates", f"lon_{grid} lat_{grid}")
                    if name in {"u", "v"}:
                        created[out_name].setncattr("vector_reference", "grid_relative")
            if "u" in normalized["variables"] and "v" in normalized["variables"]:
                for view in normalized["vertical_views"]:
                    suffix = view_suffix(view)
                    created[f"eastward_velocity_{suffix}"] = _create_field(
                        output, f"eastward_velocity_{suffix}", ("time", "eta_rho", "xi_rho"),
                        standard_name="eastward_sea_water_velocity", units="m s-1")
                    created[f"northward_velocity_{suffix}"] = _create_field(
                        output, f"northward_velocity_{suffix}", ("time", "eta_rho", "xi_rho"),
                        standard_name="northward_sea_water_velocity", units="m s-1")
                    created[f"current_speed_{suffix}"] = _create_field(
                        output, f"current_speed_{suffix}", ("time", "eta_rho", "xi_rho"),
                        standard_name="sea_water_speed", units="m s-1")
                    for prefix in ("eastward_velocity", "northward_velocity", "current_speed"):
                        created[f"{prefix}_{suffix}"].setncattr("coordinates", "lon_rho lat_rho")

        output.setncattr("schema_version", COMPACT_SCHEMA_VERSION)
        output.setncattr("source_model", config.model)
        output.setncattr(
            "vector_provenance",
            "native_u_v_grid_relative; derived_east_north_earth_relative_on_rho_grid",
        )
        output.setncattr("native_vector_reference", "grid_relative_on_staggered_c_grid")
        output.setncattr("derived_vector_reference", "earth_relative_on_rho_grid")
        output.setncattr("angle_convention", geometry["angle_convention"])
        output.setncattr("velocity_processing", "vertically reduce native u/v, wet-aware destagger to rho, rotate by angle")
        output.setncattr("request_json", json.dumps(
            aws.json_clean(normalized), sort_keys=True, separators=(",", ":")))
        output.setncattr("source_provenance_json", json.dumps(
            source_summary, sort_keys=True, separators=(",", ":")))
        output.setncattr("source_archives_json", json.dumps(
            source_summary["archives"], sort_keys=True, separators=(",", ":")))
        output.setncattr("source_endpoints_json", json.dumps(
            source_summary["endpoints"], sort_keys=True, separators=(",", ":")))
        output.setncattr("source_time_metadata_json", json.dumps(
            source_time_metadata, sort_keys=True, separators=(",", ":")))
        output.setncattr("input_provenance_mode", "verified_fetch_manifest" if fetch_binding else "explicit_unbound_inputs")
        if fetch_binding:
            output.setncattr("fetch_manifest_sha256", fetch_binding["fetch_manifest_sha256"])
            output.setncattr("fetch_binding_sha256", aws.canonical_json_sha256(fetch_binding))

        for out_index, record in enumerate(records):
            with netCDF4.Dataset(record["path"]) as dataset:
                zeta = _read_record(dataset.variables["zeta"], record["index"])
                zeta = np.where(geometry["mask_rho"] == 1, zeta, np.nan)
                # Thickness closure at rho points.
                thickness = layer_thickness(geometry["s_w"], geometry["Cs_w"], geometry["h"], zeta,
                                            geometry["hc"], geometry["Vtransform"])
                target = geometry["h"] + zeta
                wet = (geometry["mask_rho"] == 1) & np.isfinite(target) & (target > 0)
                rel = np.abs(np.sum(thickness, axis=0) - target) / np.maximum(target, 1e-12)
                finite_error = rel[wet & np.isfinite(rel)]
                thickness_errors.append(float(finite_error.max()) if finite_error.size else math.nan)
                time_var[out_index] = record["time"].timestamp()
                original_var[out_index] = aws.parse_utc(record["original_time_utc"]).timestamp()
                adjustment_var[out_index] = record["adjustment_seconds"]
                for variable, dimension, value in (
                    (key_var, "source_key_strlen", record["source_key"]),
                    (archive_var, "source_archive_strlen", record["source_archive"]),
                    (url_var, "source_url_strlen", record["source_url"]),
                ):
                    width = len(output.dimensions[dimension])
                    text = value.encode("utf-8")[:width]
                    row = np.zeros(width, dtype="S1")
                    row[:len(text)] = np.frombuffer(text, dtype="S1")
                    variable[out_index, :] = row
                reduced: dict[tuple[str, str], Any] = {}
                for name in normalized["variables"]:
                    source = dataset.variables[name]
                    values = _read_record(source, record["index"])
                    if "s_rho" not in source.dimensions:
                        mask = geometry[f"mask_{source_meta[name]['grid']}"]
                        value = np.where(mask == 1, values, np.nan)
                        created[name][out_index] = value
                        coverage.setdefault(name, []).append(float(np.count_nonzero(np.isfinite(value) & (mask == 1)) / max(1, np.count_nonzero(mask == 1))))
                        continue
                    grid = source_meta[name]["grid"]
                    for view in normalized["vertical_views"]:
                        suffix = view_suffix(view)
                        value = reduce_vertical(values, view, grid, geometry, zeta)
                        reduced[(name, suffix)] = value
                        out_name = ("salinity_" + suffix) if name == "salt" else f"{name}_{suffix}"
                        created[out_name][out_index] = value
                        mask = geometry[f"mask_{grid}"]
                        coverage.setdefault(out_name, []).append(float(np.count_nonzero(np.isfinite(value) & (mask == 1)) / max(1, np.count_nonzero(mask == 1))))
                if "u" in normalized["variables"] and "v" in normalized["variables"]:
                    for view in normalized["vertical_views"]:
                        suffix = view_suffix(view)
                        u_rho = destagger_u(reduced[("u", suffix)], geometry["mask_rho"])
                        v_rho = destagger_v(reduced[("v", suffix)], geometry["mask_rho"])
                        east, north, speed = rotate_to_earth(u_rho, v_rho, geometry["angle"])
                        for prefix, value in (("eastward_velocity", east), ("northward_velocity", north), ("current_speed", speed)):
                            name = f"{prefix}_{suffix}"
                            created[name][out_index] = value
                            wet_count = max(1, np.count_nonzero(geometry["mask_rho"] == 1))
                            coverage.setdefault(name, []).append(float(np.count_nonzero(np.isfinite(value) & (geometry["mask_rho"] == 1)) / wet_count))
    os.replace(temporary, destination)
    critical = []
    warnings = []
    if any(math.isfinite(value) and value > 1e-6 for value in thickness_errors):
        critical.append("ROMS W-level thickness does not close to h+zeta")
    low_coverage = {name: min(values) for name, values in coverage.items() if values and min(values) < 0.95}
    if low_coverage:
        critical.append(f"finite wet coverage below 95%: {low_coverage}")
    # Plausibility is warning-only and data are never clipped.
    speed_consistency: dict[str, float] = {}
    with netCDF4.Dataset(destination) as result:
        for name, variable in result.variables.items():
            if name.startswith("salinity_"):
                finite = _filled(variable[:])
                finite = finite[np.isfinite(finite)]
                if finite.size and (float(finite.min()) < -1 or float(finite.max()) > 50):
                    warnings.append(f"broad salinity plausibility excursion in {name}")
            if name.startswith("current_speed_"):
                finite = _filled(variable[:])
                finite = finite[np.isfinite(finite)]
                if finite.size and float(finite.max()) > 10:
                    warnings.append(f"broad current-speed plausibility excursion in {name}")
        for view in normalized["vertical_views"]:
            suffix = view_suffix(view)
            east_name, north_name, speed_name = (
                f"eastward_velocity_{suffix}", f"northward_velocity_{suffix}",
                f"current_speed_{suffix}",
            )
            if speed_name in result.variables:
                east = _filled(result.variables[east_name][:])
                north = _filled(result.variables[north_name][:])
                speed = _filled(result.variables[speed_name][:])
                difference = np.abs(speed - np.hypot(east, north))
                finite_difference = difference[np.isfinite(difference)]
                error = float(finite_difference.max()) if finite_difference.size else math.inf
                speed_consistency[suffix] = error
                if error > 1e-5:
                    critical.append(f"derived current-speed consistency failed for {suffix}")
    selected_input_paths = sorted({str(Path(item["path"]).resolve()) for item in records})
    selected_input_keys = sorted({str(item["source_key"]) for item in records})
    report = {
        "schema_version": f"{config.model}_extraction_report_v2",
        "created_utc": aws.iso(datetime.now(UTC)), "request": normalized,
        "output": str(destination.resolve()), "output_size": destination.stat().st_size,
        "output_sha256": aws.sha256_file(destination), "record_count": len(records),
        "times_utc": [aws.iso(item["time"]) for item in records], "missing_times": missing,
        "duplicate_records": records[0].get("duplicates", []),
        "geometry_hashes": _geometry_hashes(geometry),
        "angle_metadata": {
            "units": geometry["angle_units"],
            "standard_name": geometry["angle_standard_name"],
            "long_name": geometry["angle_long_name"],
            "convention": geometry["angle_convention"],
        },
        "input_provenance_mode": "verified_fetch_manifest" if fetch_binding else "explicit_unbound_inputs",
        "fetch_binding": aws.json_clean(fetch_binding),
        "records": [
            {"path": str(Path(item["path"]).resolve()),
             "source_key": item["source_key"],
             "source_archive": item["source_archive"],
             "source_url": item["source_url"],
             "normalized_time_utc": aws.iso(item["time"]),
             "source_time_units": item["source_time_units"],
             "source_calendar": item["source_calendar"],
             "decoder_calendar": item["decoder_calendar"],
             "calendar_alias_applied": item["calendar_alias_applied"]}
            for item in records
        ],
        "source_provenance": source_summary,
        "source_time_metadata": source_time_metadata,
        "selected_input_paths": selected_input_paths,
        "selected_input_keys": selected_input_keys,
        "max_thickness_relative_error": max((value for value in thickness_errors if math.isfinite(value)), default=None),
        "speed_consistency_max_abs_error": speed_consistency,
        "finite_wet_coverage": coverage, "critical": critical, "warnings": warnings,
        "status": "healthy" if not critical else "critical",
    }
    aws.write_json_atomic(destination.with_suffix(".health.json"), report)
    manifest = {**report, "schema_version": f"{config.model}_extraction_manifest_v2"}
    aws.write_json_atomic(destination.parent / "extraction_manifest.json", manifest)
    return report


def evaluate_health(config: aws.ModelConfig, request: Mapping[str, Any] | str | Path,
                    run_dir: str | Path, compact_paths: Sequence[str | Path] = ()) -> dict[str, Any]:
    normalized = aws.load_request(config, request) if isinstance(request, (str, Path)) else aws.validate_request(config, request)
    run_path = Path(run_dir).resolve()
    critical, warnings, transfers = [], [], []
    expected_fetch_binding: dict[str, Any] | None = None
    manifest_path = run_path / "fetch_manifest.json"
    cleanup_path = run_path / "cache_cleanup.json"
    cleanup_record: Mapping[str, Any] | None = None
    legacy_v1_handled = False
    if cleanup_path.is_file():
        try:
            candidate = aws.read_json(cleanup_path)
            if not isinstance(candidate, Mapping) or candidate.get("schema_version") != f"{config.model}_cache_cleanup_v1":
                raise ValueError("unexpected cleanup schema")
            cleanup_record = candidate
        except Exception as exc:
            critical.append(f"cache cleanup record is invalid: {type(exc).__name__}: {exc}")

    if manifest_path.is_file():
        try:
            legacy_candidate = aws.read_json(manifest_path)
        except Exception:
            legacy_candidate = None
        if (isinstance(legacy_candidate, Mapping)
                and legacy_candidate.get("schema_version") == f"{config.model}_fetch_manifest_v1"):
            legacy = aws.verify_legacy_v1_manifest(
                config, legacy_candidate, manifest_path, normalized)
            transfers = [{
                "key": item.get("key"), "path": item.get("local_path"),
                "integrity_ok": item.get("status") == "pass", "state": "legacy_v1_present",
                "etag": item.get("etag"), "etag_semantics": "opaque_provenance",
                "findings": ([item.get("reason")] if item.get("reason") else []),
            } for item in legacy.get("objects", [])]
            if legacy.get("status") == "pass":
                actual_paths = [item["local_path"] for item in legacy["objects"]
                                if item.get("status") == "pass"]
                try:
                    time_audit = aws.audit_time_records(config, normalized, actual_paths)
                except Exception as exc:
                    time_audit = {"error": f"{type(exc).__name__}: {exc}"}
                    critical.append("legacy v1 raw time-coordinate audit failed")
                warnings.append(
                    "read-only legacy v1 AWS evidence was validated; create a new v2 plan before any transfer")
            else:
                time_audit = {"error": "; ".join(legacy.get("failures", []))}
                critical.extend(f"legacy v1 integrity failure: {item}"
                                for item in legacy.get("failures", []))
            legacy_v1_handled = True

    if manifest_path.is_file() and not legacy_v1_handled:
        try:
            manifest = aws.read_json(manifest_path)
            if manifest.get("schema_version") != f"{config.model}_fetch_manifest_v2":
                raise ValueError("unexpected fetch-manifest schema")
            if aws.validate_request(config, manifest.get("request", {})) != normalized:
                raise ValueError("fetch-manifest request does not match the health request")
            provenance_failures = aws.verify_approved_plan_provenance(config, manifest)
            if provenance_failures:
                critical.extend(f"approved-plan provenance failure: {item}"
                                for item in provenance_failures)
            else:
                expected_fetch_binding = aws.manifest_fetch_binding(config, manifest, manifest_path)
            outcomes = manifest.get("outcomes")
            if not isinstance(outcomes, list) or not outcomes:
                raise ValueError("fetch manifest has no outcomes")
            manifest_hash = aws.sha256_file(manifest_path)
            cleanup_objects = {
                item.get("key"): item for item in (cleanup_record or {}).get("objects", [])
                if isinstance(item, Mapping)
            }
            cleanup_matches_manifest = bool(
                cleanup_record and cleanup_record.get("manifest_sha256") == manifest_hash
                and Path(str(cleanup_record.get("manifest_path", ""))).resolve() == manifest_path
                and normalized["cache_policy"] == "delete_after_extract"
            )
            actual_paths: list[str] = []
            intentional_count = 0
            raw_root = (run_path / "cache" / "raw").resolve()
            for outcome in outcomes:
                key = outcome.get("key")
                path = Path(str(outcome.get("local_path", ""))).resolve()
                try:
                    path.relative_to(raw_root)
                    in_cache = True
                except ValueError:
                    in_cache = False
                size = outcome.get("size")
                digest = outcome.get("sha256")
                exact = (isinstance(size, int) and not isinstance(size, bool) and size > 0
                         and isinstance(digest, str) and len(digest) == 64)
                findings: list[str] = []
                ok = False
                if in_cache and exact and path.is_file():
                    verified_path, findings, _ = aws.verify_cached_outcome(
                        config, outcome, raw_root=raw_root)
                    ok = verified_path is not None and not findings
                state = "present"
                if ok:
                    actual_paths.append(str(path))
                elif in_cache and exact and not path.exists() and cleanup_matches_manifest:
                    prior = cleanup_objects.get(key)
                    ok = bool(prior and Path(str(prior.get("local_path", ""))).resolve() == path
                              and prior.get("size") == size and prior.get("sha256") == digest
                              and prior.get("deleted") is True)
                    if ok:
                        sidecar = path.with_name(path.name + ".download.json")
                        ok = not sidecar.exists()
                    if ok:
                        state = "intentionally_deleted_after_health"
                        intentional_count += 1
                transfers.append({"key": key, "path": str(path), "integrity_ok": ok,
                                  "state": state, "etag": outcome.get("etag"),
                                  "etag_semantics": "opaque_provenance",
                                  "findings": findings})
                if not ok:
                    detail = "; ".join(findings) if findings else "file/cleanup evidence mismatch"
                    critical.append(f"raw integrity failure: {key}: {detail}")
            if actual_paths:
                time_audit = aws.audit_time_records(config, normalized, actual_paths)
            elif intentional_count == len(outcomes) and cleanup_matches_manifest:
                saved_audit = cleanup_record.get("time_audit")
                if not isinstance(saved_audit, Mapping):
                    raise ValueError("cleanup record has no prior time audit")
                time_audit = dict(saved_audit)
            else:
                raise ValueError("no integrity-verified raw files are available for time audit")
            if time_audit.get("missing_times") and normalized["missing_policy"] == "error":
                critical.append(f"raw time coverage has {len(time_audit['missing_times'])} gaps")
            if not time_audit.get("unique") or not time_audit.get("strictly_monotonic"):
                critical.append("raw normalized timestamps are not unique and strictly monotonic")
        except Exception as exc:
            time_audit = {"error": f"{type(exc).__name__}: {exc}"}
            critical.append("raw manifest/time-coordinate audit failed")
    elif not legacy_v1_handled:
        time_audit = None
        critical.append("fetch_manifest.json is absent; raw transfer integrity cannot be verified")

    compact = []
    if normalized["product"] == "fields" and not compact_paths:
        critical.append("fields health requires at least one compact product")
    for raw in compact_paths:
        path = Path(raw).resolve()
        report_path = path.with_suffix(".health.json")
        if not path.is_file() or not report_path.is_file():
            critical.append(f"compact product or health report missing: {path}")
            continue
        try:
            report = aws.read_json(report_path)
            if report.get("schema_version") != f"{config.model}_extraction_report_v2":
                raise ValueError("unexpected compact health schema")
            if aws.validate_request(config, report.get("request", {})) != normalized:
                raise ValueError("compact health request does not match")
            actual_size = path.stat().st_size
            actual_hash = aws.sha256_file(path)
            if report.get("output_size") != actual_size or report.get("output_sha256") != actual_hash:
                raise ValueError("compact size/SHA-256 does not match its health report")
            if Path(str(report.get("output", ""))).resolve() != path:
                raise ValueError("compact health report names a different output")
            extraction_manifest_path = path.parent / "extraction_manifest.json"
            if not extraction_manifest_path.is_file():
                raise ValueError("extraction_manifest.json is missing")
            extraction_manifest = aws.read_json(extraction_manifest_path)
            if extraction_manifest.get("schema_version") != f"{config.model}_extraction_manifest_v2":
                raise ValueError("extraction manifest schema is invalid")
            comparable_manifest = dict(extraction_manifest)
            comparable_manifest["schema_version"] = report["schema_version"]
            if aws.canonical_json_sha256(comparable_manifest) != aws.canonical_json_sha256(report):
                raise ValueError("extraction manifest does not match compact report")
            binding = report.get("fetch_binding")
            if expected_fetch_binding is None:
                raise ValueError("compact extraction cannot be bound to a verified fetch manifest")
            if (not isinstance(binding, Mapping)
                    or aws.canonical_json_sha256(binding) != aws.canonical_json_sha256(expected_fetch_binding)):
                raise ValueError("compact extraction fetch binding does not match verified outcomes")
            expected_paths = sorted(item["local_path"] for item in expected_fetch_binding["objects"])
            expected_keys = sorted(item["key"] for item in expected_fetch_binding["objects"])
            if report.get("selected_input_paths") != expected_paths or report.get("selected_input_keys") != expected_keys:
                raise ValueError("compact selected input keys/paths do not match verified outcomes")
            report_records = report.get("records")
            if not isinstance(report_records, list) or not report_records:
                raise ValueError("compact extraction report has no record-level source provenance")
            binding_by_key = {item["key"]: item for item in expected_fetch_binding["objects"]}
            for record in report_records:
                if not isinstance(record, Mapping) or record.get("source_key") not in binding_by_key:
                    raise ValueError("compact record source key is outside the verified fetch binding")
                source = binding_by_key[record["source_key"]]
                if (record.get("source_archive") != source.get("source_archive")
                        or record.get("source_url") != source.get("url")
                        or str(Path(str(record.get("path", ""))).resolve()) != source.get("local_path")):
                    raise ValueError("compact record archive/URL/path provenance is inconsistent")
            expected_summary = _source_summary([
                {**item, "source_archive": item["source_archive"]}
                for item in expected_fetch_binding["objects"]
            ])
            if report.get("source_provenance") != expected_summary:
                raise ValueError("compact global archive/endpoint provenance is inconsistent")
            angle_metadata = report.get("angle_metadata")
            if (not isinstance(angle_metadata, Mapping)
                    or angle_metadata.get("convention") != ANGLE_CONVENTION
                    or str(angle_metadata.get("units", "")).lower() not in {"rad", "radian", "radians"}):
                raise ValueError("compact extraction angle provenance is invalid")
            netCDF4, _ = _modules()
            with netCDF4.Dataset(path) as dataset:
                if getattr(dataset, "schema_version", None) != COMPACT_SCHEMA_VERSION:
                    raise ValueError("compact NetCDF schema_version is invalid")
                embedded = json.loads(getattr(dataset, "request_json", ""))
                if aws.validate_request(config, embedded) != normalized:
                    raise ValueError("compact NetCDF request_json does not match")
                if getattr(dataset, "fetch_manifest_sha256", None) != expected_fetch_binding["fetch_manifest_sha256"]:
                    raise ValueError("compact NetCDF fetch-manifest digest does not match")
                if getattr(dataset, "fetch_binding_sha256", None) != aws.canonical_json_sha256(expected_fetch_binding):
                    raise ValueError("compact NetCDF fetch-binding digest does not match")
                embedded_sources = {}
                for name in ("source_key", "source_archive", "source_url"):
                    variable = dataset.variables.get(name)
                    if variable is None or tuple(variable.dimensions)[:1] != ("time",):
                        raise ValueError(f"compact NetCDF has no time-indexed {name} provenance")
                    embedded_sources[name] = [str(value).rstrip("\x00") for value in netCDF4.chartostring(variable[:]).tolist()]
                for name in embedded_sources:
                    expected_values = [str(item[name]) for item in report_records]
                    if embedded_sources[name] != expected_values:
                        raise ValueError(f"compact NetCDF {name} does not match extraction provenance")
                for attr, expected_value in (
                    ("source_provenance_json", expected_summary),
                    ("source_archives_json", expected_summary["archives"]),
                    ("source_endpoints_json", expected_summary["endpoints"]),
                ):
                    try:
                        actual_value = json.loads(getattr(dataset, attr, ""))
                    except Exception as exc:
                        raise ValueError(f"compact NetCDF {attr} is invalid") from exc
                    if actual_value != expected_value:
                        raise ValueError(f"compact NetCDF {attr} does not match verified sources")
                if getattr(dataset, "angle_convention", None) != ANGLE_CONVENTION:
                    raise ValueError("compact NetCDF angle convention is missing or invalid")
                angle = dataset.variables.get("angle")
                if angle is None or getattr(angle, "angle_convention", None) != ANGLE_CONVENTION:
                    raise ValueError("compact angle variable convention is missing or invalid")
            compact.append({**report, "integrity_ok": True})
            critical.extend(report.get("critical", []))
            warnings.extend(report.get("warnings", []))
            if report.get("status") != "healthy":
                critical.append(f"compact extraction status is not healthy: {path}")
        except Exception as exc:
            compact.append({"path": str(path), "integrity_ok": False,
                            "error": f"{type(exc).__name__}: {exc}"})
            critical.append(f"compact integrity failure: {path}")

    cleanup = dict(cleanup_record) if cleanup_record else None
    raw_cache_present = any(item.get("state") == "present" and item.get("integrity_ok")
                            for item in transfers)
    if not critical and normalized["cache_policy"] == "delete_after_extract" and raw_cache_present:
        try:
            cleanup = delete_raw_cache(run_path, time_audit=time_audit)
        except Exception as exc:
            critical.append(f"post-health cache cleanup failed: {type(exc).__name__}: {exc}")
    report = {
        "schema_version": f"{config.model}_download_health_v2",
        "created_utc": aws.iso(datetime.now(UTC)), "request": normalized,
        "transfers": transfers, "time_audit": time_audit, "compact_products": compact,
        "cache_cleanup": cleanup,
        "critical": critical, "warnings": warnings,
        "status": "healthy" if not critical else "critical",
    }
    aws.write_json_atomic(run_path / "health_report.json", report)
    return report


def delete_raw_cache(run_dir: str | Path, *, time_audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Delete verified manifest objects after health and record the decision."""
    run_path = Path(run_dir).resolve()
    manifest_path = run_path / "fetch_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("cannot clean raw cache without fetch_manifest.json")
    manifest = aws.read_json(manifest_path)
    manifest_schema = str(manifest.get("schema_version", ""))
    if not manifest_schema.endswith("_fetch_manifest_v2"):
        raise RuntimeError("cannot clean raw cache from an invalid manifest schema")
    model = manifest_schema[:-len("_fetch_manifest_v2")]
    outcomes = manifest.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise RuntimeError("cannot clean raw cache from an empty manifest")
    staged: list[dict[str, Any]] = []
    raw_root = (run_path / "cache" / "raw").resolve()
    for outcome in outcomes:
        path = Path(outcome.get("local_path", "")).resolve()
        try:
            path.relative_to(raw_root)
        except ValueError:
            raise RuntimeError(f"refusing to delete manifest path outside run cache: {path}")
        sidecar = path.with_name(path.name + ".download.json")
        size, digest = outcome.get("size"), outcome.get("sha256")
        if (not isinstance(size, int) or isinstance(size, bool) or size <= 0
                or not isinstance(digest, str) or len(digest) != 64
                or not path.is_file() or path.stat().st_size != size
                or aws.sha256_file(path) != digest or not sidecar.is_file()):
            raise RuntimeError(f"refusing to delete unverified cached object: {path}")
        staged.append({"key": outcome.get("key"), "local_path": str(path),
                       "size": size, "sha256": digest, "sidecar": str(sidecar)})
    deleted, total = [], 0
    for item in staged:
        for target in (Path(item["local_path"]), Path(item["sidecar"])):
            size = target.stat().st_size
            target.unlink()
            deleted.append(str(target))
            total += size
        item["deleted"] = True
    record = {
        "schema_version": f"{model}_cache_cleanup_v1",
        "created_utc": aws.iso(datetime.now(UTC)),
        "manifest_path": str(manifest_path), "manifest_sha256": aws.sha256_file(manifest_path),
        "objects": staged, "deleted_paths": deleted, "deleted_bytes": total,
        "time_audit": aws.json_clean(time_audit),
        "recovery": "deleted NOAA source objects must be re-downloaded",
    }
    aws.write_json_atomic(run_path / "cache_cleanup.json", record)
    return record
