#!/usr/bin/env python3
"""Health-check SSCOFS fetch/extraction evidence and make midpoint maps.

The checker is deliberately independent of the downloader.  It validates the
request, estimate, and fetch manifest; streams hashes instead of loading files;
and reads NetCDF variables one time slice at a time.  Native ``fields`` runs
receive FVCOM-specific mesh, sigma, wet-mask, and derived-field checks.  The
``stations`` and ``regulargrid`` passthrough products receive object/time checks
without being incorrectly required to contain an unstructured mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


UTC = timezone.utc
FIELD_NAMES = (
    "salinity_surface",
    "salinity_near_surface",
    "salinity_bottom",
    "salinity_depth_average",
    "u_surface",
    "v_surface",
    "current_speed_surface",
    "u_near_surface",
    "v_near_surface",
    "current_speed_near_surface",
    "u_bottom",
    "v_bottom",
    "current_speed_bottom",
    "u_depth_average",
    "v_depth_average",
    "current_speed_depth_average",
)
MAP_NAMES = (
    "salinity_surface",
    "salinity_bottom",
    "salinity_depth_average",
    "current_speed_surface",
    "current_speed_bottom",
    "current_speed_depth_average",
)
TIME_KEYS = ("valid_time_utc", "valid_time", "datetime", "time_utc")
PATH_KEYS = ("local_path", "destination", "path", "file")
SIZE_KEYS = ("size", "size_bytes", "expected_bytes", "content_length", "ContentLength")
HASH_KEYS = ("sha256", "sha256_hex", "hash_sha256")
DERIVED_FIELD_RE = re.compile(
    r"^(?:salinity|u|v|current_speed)_(?:surface|near_surface|bottom|depth_average|sigma_\d{3})$"
)


def _read_json(path: str | Path | None) -> Any:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _parse_utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            # FVCOM ``Times`` commonly uses a space or underscore.
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d_%H:%M:%S"):
                try:
                    parsed = datetime.strptime(str(value).strip(), fmt)
                    break
                except ValueError:
                    parsed = None  # type: ignore[assignment]
            if parsed is None:
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _expected_times(request: dict[str, Any], step_seconds: int = 3600) -> list[datetime]:
    start = _parse_utc(request.get("start_utc"))
    end = _parse_utc(request.get("end_utc_exclusive"))
    if start is None or end is None or end <= start:
        return []
    result: list[datetime] = []
    cursor = start
    while cursor < end:
        result.append(cursor)
        cursor += timedelta(seconds=step_seconds)
    return result


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _first(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if mapping.get(key) not in (None, ""):
            return mapping[key]
    return None


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _manifest_records(manifest: Any) -> list[dict[str, Any]]:
    """Return de-duplicated leaf-ish manifest records containing local paths."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in _walk_dicts(manifest):
        path = _first(item, PATH_KEYS)
        if not isinstance(path, (str, os.PathLike)):
            continue
        key = str(item.get("key") or item.get("object_key") or "")
        identity = (str(path), key)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(item)
    return result


def _resolve_record_path(record: dict[str, Any], run_dir: Path) -> Path:
    value = Path(str(_first(record, PATH_KEYS)))
    if value.is_absolute():
        return value
    direct = run_dir / value
    if direct.exists():
        return direct
    cache = run_dir / "cache" / value
    return cache if cache.exists() else direct


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _verify_manifest(
    manifest: Any, run_dir: Path, *, allow_deleted: bool = False
) -> tuple[dict[str, Any], list[datetime], list[str], list[str]]:
    records = _manifest_records(manifest)
    details: list[dict[str, Any]] = []
    critical: list[str] = []
    warnings: list[str] = []
    observed_times: list[datetime] = []
    bytes_checked = 0
    hashes_checked = 0
    cache_hits = 0
    for record in records:
        path = _resolve_record_path(record, run_dir)
        expected_size = _integer(_first(record, SIZE_KEYS))
        expected_hash = _first(record, HASH_KEYS)
        status = str(record.get("status") or record.get("outcome") or "").lower()
        cache_hits += int(bool(record.get("cache_hit")) or status in {"cache_hit", "cached"})
        for key in TIME_KEYS:
            parsed = _parse_utc(record.get(key))
            if parsed is not None:
                observed_times.append(parsed)
                break
        item: dict[str, Any] = {
            "key": record.get("key") or record.get("object_key"),
            "path": str(path),
            "exists": path.is_file(),
            "status": status or None,
            "expected_size": expected_size,
            "etag": record.get("etag") or record.get("ETag"),
        }
        completed = (
            status in {"downloaded", "complete", "completed", "success", "cache_hit", "cached", "resumed", "ok"}
            or bool(record.get("success"))
            or path.is_file()
        )
        if status in {"failed", "error"}:
            critical.append(f"Fetch manifest records a failed object: {item['key'] or path.name}.")
        if completed and expected_size is None:
            critical.append(f"Manifest object lacks an expected byte size: {item['key'] or path.name}.")
        if completed and not item["etag"]:
            critical.append(f"Manifest object lacks its source ETag: {item['key'] or path.name}.")
        if completed and not (isinstance(expected_hash, str) and len(expected_hash.strip()) == 64):
            critical.append(f"Manifest object lacks a valid SHA-256: {item['key'] or path.name}.")
        if not path.is_file():
            # Records for estimate-only/planned objects may have a URL-ish path.
            if allow_deleted and completed:
                item["expected_deleted_after_extract"] = True
            elif status not in {"planned", "skipped", "estimate_only"}:
                critical.append(f"Manifest object is absent locally: {path}.")
            details.append(item)
            continue
        actual_size = path.stat().st_size
        bytes_checked += actual_size
        item["actual_size"] = actual_size
        item["size_ok"] = expected_size is None or actual_size == expected_size
        if not item["size_ok"]:
            critical.append(
                f"Size mismatch for {path.name}: expected {expected_size}, found {actual_size}."
            )
        if isinstance(expected_hash, str) and len(expected_hash.strip()) == 64:
            actual_hash = _sha256(path)
            hashes_checked += 1
            item["expected_sha256"] = expected_hash.lower()
            item["actual_sha256"] = actual_hash
            item["sha256_ok"] = actual_hash == expected_hash.lower()
            if not item["sha256_ok"]:
                critical.append(f"SHA-256 mismatch for {path.name}.")
        elif completed and path.suffix.lower() not in {".json", ".png", ".gif"}:
            # The missing-hash failure above is intentionally not downgraded.
            item["sha256_ok"] = False
        details.append(item)
    return (
        {
            "record_count": len(records),
            "existing_count": sum(bool(item["exists"]) for item in details),
            "size_checked_count": sum(item.get("expected_size") is not None for item in details),
            "hash_checked_count": hashes_checked,
            "bytes_streamed_for_integrity": bytes_checked,
            "cache_hit_count": cache_hits,
            "objects": details,
        },
        observed_times,
        critical,
        warnings,
    )


def _find_json(run_dir: Path, name: str) -> Path | None:
    direct = run_dir / name
    if direct.is_file():
        return direct
    matches = sorted(run_dir.rglob(name))
    return matches[0] if matches else None


def _candidate_netcdfs(run_dir: Path) -> list[Path]:
    result = []
    for suffix in ("*.nc", "*.nc4", "*.cdf"):
        result.extend(run_dir.rglob(suffix))
    return sorted({p for p in result if p.is_file() and not p.name.endswith(".part")})


def _dataset_score(path: Path) -> tuple[int, int]:
    try:
        from netCDF4 import Dataset
        with Dataset(path, "r") as ds:
            names = set(ds.variables)
            score = 10 * sum(bool(DERIVED_FIELD_RE.match(name)) for name in names)
            score += 4 * int("nv" in names) + 2 * int("Times" in names) + int("time" in names)
            score += 5 * int("compact" in path.name.lower() or "extract" in path.name.lower())
            return score, -len(path.parts)
    except Exception:
        return -1, -len(path.parts)


def _select_dataset(run_dir: Path) -> tuple[Path | None, list[Path]]:
    candidates = _candidate_netcdfs(run_dir)
    if not candidates:
        return None, []
    ranked = sorted(candidates, key=_dataset_score, reverse=True)
    return ranked[0], candidates


def _decode_char_rows(values: Any) -> list[str]:
    import numpy as np
    arr = np.asanyarray(values)
    if arr.ndim == 0:
        return [str(arr.item())]
    if arr.dtype.kind in {"S", "U"} and arr.ndim == 1 and arr.dtype.itemsize > 1:
        return [str(item.decode() if isinstance(item, bytes) else item) for item in arr]
    rows = arr if arr.ndim > 1 else arr.reshape(1, -1)
    result = []
    for row in rows:
        pieces = []
        for value in row:
            if isinstance(value, bytes):
                pieces.append(value.decode("ascii", errors="ignore"))
            else:
                pieces.append(str(value))
        result.append("".join(pieces).replace("\x00", "").strip())
    return result


def _netcdf_times(ds: Any) -> list[datetime]:
    if "Times" in ds.variables:
        parsed = [_parse_utc(item) for item in _decode_char_rows(ds.variables["Times"][:])]
        result = [item for item in parsed if item is not None]
        if result:
            return result
    if "time" in ds.variables:
        var = ds.variables["time"]
        values = var[:]
        units = getattr(var, "units", None)
        if units:
            try:
                from netCDF4 import num2date
                decoded = num2date(values, units=units, calendar=getattr(var, "calendar", "standard"))
                result = []
                for item in decoded:
                    result.append(datetime(item.year, item.month, item.day, item.hour, item.minute, item.second, tzinfo=UTC))
                return result
            except Exception:
                pass
    return []


def _time_check(
    request: dict[str, Any], observed: Sequence[datetime], source: str
) -> tuple[dict[str, Any], list[str], list[str]]:
    station_internal = request.get("product") == "stations" and source == "netcdf"
    station_objects = request.get("product") == "stations" and source == "fetch_manifest"
    cadence_seconds = 360 if station_internal else 3600
    expected = [] if station_objects else _expected_times(request, cadence_seconds)
    observed_list = [item.astimezone(UTC) for item in observed]
    critical: list[str] = []
    warnings: list[str] = []
    unique = len(set(observed_list)) == len(observed_list)
    monotonic = all(b > a for a, b in zip(observed_list, observed_list[1:]))
    deltas = [(b - a).total_seconds() for a, b in zip(observed_list, observed_list[1:])]
    if station_objects:
        # Station products are cycle files whose internal data are six-minute;
        # the object cadence need not be hourly.
        cadence_ok = all(delta > 0 and abs((delta / 360.0) - round(delta / 360.0)) < 1.0e-6 for delta in deltas)
    else:
        cadence_ok = all(abs(delta - float(cadence_seconds)) < 1.0 for delta in deltas)
    expected_set, observed_set = set(expected), set(observed_list)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    skip = request.get("missing_policy", "error") == "skip"
    if not observed_list:
        critical.append("No valid timestamps were found in extracted data or the fetch manifest.")
    if not unique:
        critical.append("Observed SSCOFS timestamps are not unique.")
    if not monotonic and len(observed_list) > 1:
        critical.append("Observed SSCOFS timestamps are not strictly monotonic.")
    if missing:
        message = f"The request window is missing {len(missing)} expected hourly timestamp(s)."
        (warnings if skip else critical).append(message)
    if not cadence_ok and len(observed_list) > 1:
        message = f"Observed timestamps do not maintain the expected {cadence_seconds}-second cadence."
        (warnings if skip else critical).append(message)
    if extra:
        warnings.append(f"Found {len(extra)} timestamp(s) outside the requested half-open window.")
    return (
        {
            "source": source,
            "expected_count": len(expected),
            "observed_count": len(observed_list),
            "unique": unique,
            "strictly_monotonic": monotonic,
            "expected_cadence_seconds": None if station_objects else cadence_seconds,
            "cadence_ok": cadence_ok,
            "hourly_cadence": cadence_ok if cadence_seconds == 3600 else None,
            "start": _iso(observed_list[0]) if observed_list else None,
            "end": _iso(observed_list[-1]) if observed_list else None,
            "missing": [_iso(item) for item in missing],
            "extra": [_iso(item) for item in extra],
        },
        critical,
        warnings,
    )


def _as_float(values: Any) -> Any:
    import numpy as np
    if np.ma.isMaskedArray(values):
        return np.ma.filled(values.astype(float), np.nan)
    return np.asarray(values, dtype=float)


def _connectivity(ds: Any, nnode: int) -> tuple[Any | None, dict[str, Any], list[str]]:
    import numpy as np
    critical: list[str] = []
    if "nv" not in ds.variables:
        return None, {"present": False}, ["FVCOM connectivity variable 'nv' is missing."]
    raw = np.asanyarray(ds.variables["nv"][:])
    raw = np.ma.filled(raw, -999999) if np.ma.isMaskedArray(raw) else raw
    if raw.ndim != 2:
        return None, {"present": True, "shape": list(raw.shape)}, ["FVCOM connectivity is not two-dimensional."]
    triangles = raw.T if raw.shape[0] == 3 else raw
    if triangles.shape[1] != 3:
        return None, {"present": True, "shape": list(raw.shape)}, ["FVCOM connectivity does not contain three nodes per element."]
    triangles = triangles.astype(np.int64, copy=False)
    minimum = int(triangles.min()) if triangles.size else -1
    maximum = int(triangles.max()) if triangles.size else -1
    if minimum >= 1 and maximum <= nnode:
        base = 1
        triangles = triangles - 1
    elif minimum >= 0 and maximum < nnode:
        base = 0
    else:
        base = None
        critical.append(f"Connectivity indices [{minimum}, {maximum}] are outside the {nnode}-node mesh.")
    repeated = int(np.sum((triangles[:, 0] == triangles[:, 1]) | (triangles[:, 1] == triangles[:, 2]) | (triangles[:, 0] == triangles[:, 2])))
    if repeated:
        critical.append(f"Connectivity contains {repeated} element(s) with repeated node indices.")
    return triangles, {
        "present": True,
        "shape": list(raw.shape),
        "element_count": int(triangles.shape[0]),
        "index_base": base,
        "index_min": minimum,
        "index_max": maximum,
        "repeated_node_elements": repeated,
    }, critical


def _read_coord(ds: Any, names: Sequence[str]) -> tuple[str | None, Any | None]:
    for name in names:
        if name in ds.variables:
            return name, _as_float(ds.variables[name][:]).reshape(-1)
    return None, None


def _mesh_check(ds: Any) -> tuple[dict[str, Any], Any | None, Any | None, Any | None, list[str], list[str]]:
    import numpy as np
    critical: list[str] = []
    warnings: list[str] = []
    lon_name, lon = _read_coord(ds, ("lon", "longitude", "x"))
    lat_name, lat = _read_coord(ds, ("lat", "latitude", "y"))
    if lon is None or lat is None or lon.size != lat.size:
        return {"valid": False}, lon, lat, None, ["FVCOM node coordinates are missing or inconsistent."], warnings
    nnode = int(lon.size)
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        critical.append("FVCOM node coordinates contain non-finite values.")
    triangles, conn, conn_critical = _connectivity(ds, nnode)
    critical.extend(conn_critical)
    zero_area = None
    if triangles is not None and not conn_critical:
        twice_area = (
            (lon[triangles[:, 1]] - lon[triangles[:, 0]]) * (lat[triangles[:, 2]] - lat[triangles[:, 0]])
            - (lon[triangles[:, 2]] - lon[triangles[:, 0]]) * (lat[triangles[:, 1]] - lat[triangles[:, 0]])
        )
        zero_area = int(np.sum(~np.isfinite(twice_area) | (np.abs(twice_area) <= 1.0e-14)))
        if zero_area:
            critical.append(f"FVCOM mesh contains {zero_area} zero/non-finite-area element(s).")
    report = {
        "valid": not critical,
        "node_count": nnode,
        "lon_variable": lon_name,
        "lat_variable": lat_name,
        "bbox": [float(np.nanmin(lon)), float(np.nanmin(lat)), float(np.nanmax(lon)), float(np.nanmax(lat))],
        "connectivity": conn,
        "zero_or_nonfinite_area_elements": zero_area,
    }
    return report, lon, lat, triangles, critical, warnings


def _sigma_check(ds: Any, triangles: Any | None) -> tuple[dict[str, Any], list[str], list[str]]:
    import numpy as np
    critical: list[str] = []
    warnings: list[str] = []
    if "siglev" not in ds.variables:
        return {"present": False}, ["FVCOM sigma-interface variable 'siglev' is missing."], warnings
    var = ds.variables["siglev"]
    values = _as_float(var[:])
    dims = tuple(var.dimensions)
    axes = [index for index, dim in enumerate(dims) if "siglev" in dim.lower()]
    layer_axis = axes[0] if axes else int(np.argmin(values.shape))
    levels = np.moveaxis(values, layer_axis, 0)
    interface_profile = np.nanmedian(levels.reshape(levels.shape[0], -1), axis=1)
    surface_interface_index = int(np.nanargmin(np.abs(interface_profile)))
    bottom_interface_index = int(np.nanargmax(np.abs(interface_profile)))
    if surface_interface_index == bottom_interface_index:
        critical.append("Sigma interfaces do not resolve distinct surface and bottom indices.")
    weights = np.abs(np.diff(levels, axis=0))
    sums = np.nansum(weights, axis=0)
    finite = np.isfinite(sums)
    normalized_fraction = float(np.mean(np.isclose(sums[finite], 1.0, atol=2.0e-2))) if np.any(finite) else 0.0
    nonpositive = int(np.sum(np.isfinite(weights) & (weights <= 0.0)))
    if normalized_fraction < 0.99:
        critical.append(f"Only {normalized_fraction:.3%} of finite node sigma-weight columns sum to one (tolerance 0.02).")
    if nonpositive:
        warnings.append(f"Sigma interfaces yield {nonpositive} zero-thickness layer entries.")
    element_fraction = None
    if triangles is not None and weights.ndim == 2 and weights.shape[1] > int(triangles.max()):
        element_weights = np.nanmean(weights[:, triangles], axis=2)
        element_sums = np.nansum(element_weights, axis=0)
        valid = np.isfinite(element_sums)
        element_fraction = float(np.mean(np.isclose(element_sums[valid], 1.0, atol=2.0e-2))) if np.any(valid) else 0.0
        if element_fraction < 0.99:
            critical.append(f"Only {element_fraction:.3%} of element sigma-weight columns sum to one.")
    surface_layer_index = None
    bottom_layer_index = None
    layer_profile = None
    if "siglay" in ds.variables:
        siglay_var = ds.variables["siglay"]
        siglay_values = _as_float(siglay_var[:])
        siglay_axes = [index for index, dim in enumerate(siglay_var.dimensions) if "siglay" in dim.lower()]
        siglay_axis = siglay_axes[0] if siglay_axes else int(np.argmin(siglay_values.shape))
        siglay_layers = np.moveaxis(siglay_values, siglay_axis, 0)
        layer_profile = np.nanmedian(siglay_layers.reshape(siglay_layers.shape[0], -1), axis=1)
        surface_layer_index = int(np.nanargmin(np.abs(layer_profile)))
        bottom_layer_index = int(np.nanargmax(np.abs(layer_profile)))
        if surface_layer_index == bottom_layer_index:
            critical.append("Sigma layers do not resolve distinct surface and bottom indices.")
    else:
        critical.append("FVCOM sigma-layer variable 'siglay' is missing; derived surface/bottom indices cannot be verified.")
    return {
        "present": True,
        "dimensions": list(dims),
        "shape": list(values.shape),
        "layer_axis": layer_axis,
        "layer_count": int(weights.shape[0]),
        "node_weight_sum_ok_fraction": normalized_fraction,
        "element_weight_sum_ok_fraction": element_fraction,
        "zero_or_negative_thickness_entries": nonpositive,
        "interface_profile_median": [float(item) for item in interface_profile],
        "surface_interface_index": surface_interface_index,
        "bottom_interface_index": bottom_interface_index,
        "surface_interface_median": float(interface_profile[surface_interface_index]),
        "bottom_interface_median": float(interface_profile[bottom_interface_index]),
        "layer_profile_median": [float(item) for item in layer_profile] if layer_profile is not None else None,
        "surface_layer_index": surface_layer_index,
        "bottom_layer_index": bottom_layer_index,
        "vertical_order": (
            "surface_to_bottom" if surface_layer_index == 0 else
            "bottom_to_surface" if bottom_layer_index == 0 else
            "nonmonotonic"
        ) if layer_profile is not None else None,
    }, critical, warnings


def _time_axis(var: Any) -> int | None:
    for index, dim in enumerate(var.dimensions):
        if dim.lower() in {"time", "times"} or "time" in dim.lower():
            return index
    return None


def _frame(var: Any, index: int) -> Any:
    axis = _time_axis(var)
    if axis is None:
        return _as_float(var[:]).squeeze()
    selection = [slice(None)] * var.ndim
    selection[axis] = index
    return _as_float(var[tuple(selection)]).squeeze()


def _location(var: Any, nnode: int, nele: int) -> str | None:
    dims = [dim.lower() for dim in var.dimensions]
    if any(dim in {"node", "nodes"} or dim.endswith("node") for dim in dims):
        return "node"
    if any(dim in {"nele", "element", "elements", "cell"} or dim.endswith("nele") for dim in dims):
        return "element"
    shape = var.shape
    if nnode in shape:
        return "node"
    if nele in shape:
        return "element"
    return None


def _find_wet_mask(ds: Any, location: str, nnode: int, nele: int) -> Any | None:
    preferred = (
        ("wet_nodes", "wet_nodes_prev_int", "wet_nodes_prev_ext", "wet_node_mask")
        if location == "node"
        else ("wet_cells", "wet_cells_prev_int", "wet_cells_prev_ext", "wet_cell_mask")
    )
    for name in preferred:
        if name in ds.variables and _location(ds.variables[name], nnode, nele) == location:
            return ds.variables[name]
    for name, var in ds.variables.items():
        lower = name.lower()
        if "wet" in lower and _location(var, nnode, nele) == location:
            return var
    return None


def _plausibility(name: str, minimum: float, maximum: float) -> str | None:
    if name.startswith("salinity") and (minimum < -1.0 or maximum > 50.0):
        return f"{name} range [{minimum:.4g}, {maximum:.4g}] is outside the broad -1 to 50 salinity warning range."
    if name.startswith("current_speed") and (minimum < -1.0e-6 or maximum > 15.0):
        return f"{name} range [{minimum:.4g}, {maximum:.4g}] is outside the broad 0 to 15 m/s current-speed warning range."
    if (name.startswith("u_") or name.startswith("v_")) and max(abs(minimum), abs(maximum)) > 15.0:
        return f"{name} range [{minimum:.4g}, {maximum:.4g}] exceeds the broad +/-15 m/s velocity warning range."
    return None


def _vertical_suffixes(raw_views: Any) -> set[str]:
    if not isinstance(raw_views, list):
        raw_views = [raw_views]
    suffixes: set[str] = set()
    for item in raw_views:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            suffixes.add(f"sigma_{item:03d}")
            continue
        if isinstance(item, Mapping):
            index = item.get("sigma_index", item.get("index"))
            try:
                suffixes.add(f"sigma_{int(index):03d}")
            except (TypeError, ValueError):
                pass
            continue
        text = str(item).lower().strip()
        if text in {"surface", "near_surface", "bottom", "depth_average"}:
            suffixes.add(text)
            continue
        match = re.fullmatch(r"(?:sigma(?:_index)?[:=_-]?)?(\d+)", text)
        if match:
            suffixes.add(f"sigma_{int(match.group(1)):03d}")
    return suffixes


def _sigma_layer_profile(ds: Any) -> Any | None:
    import numpy as np
    if "siglay" not in ds.variables:
        return None
    var = ds.variables["siglay"]
    values = _as_float(var[:])
    axes = [index for index, dim in enumerate(var.dimensions) if "siglay" in dim.lower()]
    axis = axes[0] if axes else int(np.argmin(values.shape))
    layers = np.moveaxis(values, axis, 0)
    return np.nanmedian(layers.reshape(layers.shape[0], -1), axis=1)


def _source_layer_frame(var: Any, time_index: int, layer_index: int) -> Any:
    selection = [slice(None)] * var.ndim
    for axis, dim in enumerate(var.dimensions):
        lower = dim.lower()
        if "time" in lower:
            selection[axis] = time_index
        elif "siglay" in lower:
            selection[axis] = layer_index
    return _as_float(var[tuple(selection)]).squeeze()


def _field_checks(ds: Any, request: dict[str, Any], nnode: int, nele: int) -> tuple[dict[str, Any], list[str], list[str]]:
    import numpy as np
    critical: list[str] = []
    warnings: list[str] = []
    available = set(ds.variables)
    raw_requested = request.get("variables", [])
    if isinstance(raw_requested, str):
        raw_requested = [raw_requested]
    requested = {str(item).lower() for item in raw_requested}
    views_raw = request.get("vertical_views", ["surface", "bottom", "depth_average"])
    requested_suffixes = _vertical_suffixes(views_raw)
    need_salinity = bool(requested.intersection({"salinity", "salt"}))
    need_velocity = bool(requested.intersection({"u", "v", "velocity", "current", "current_speed"}))
    expected: set[str] = set()
    if need_salinity:
        expected.update(f"salinity_{view}" for view in requested_suffixes)
    if need_velocity:
        for view in requested_suffixes:
            expected.update((f"u_{view}", f"v_{view}", f"current_speed_{view}"))
    missing = sorted(expected - available)
    if missing:
        critical.append("Requested derived variables are missing: " + ", ".join(missing) + ".")
    paired: dict[str, bool] = {}
    available_suffixes = {
        name.removeprefix("u_") for name in available if name.startswith("u_") and DERIVED_FIELD_RE.match(name)
    }
    for view in sorted(requested_suffixes.union(available_suffixes)):
        u_present, v_present = f"u_{view}" in available, f"v_{view}" in available
        paired[view] = u_present == v_present
        if u_present != v_present:
            critical.append(f"Velocity view '{view}' does not contain a paired u/v component set.")
    time_count = len(_netcdf_times(ds))
    if not time_count:
        for dim in ds.dimensions:
            if "time" in dim.lower():
                time_count = len(ds.dimensions[dim])
                break
    variables: dict[str, Any] = {}
    for name in sorted(name for name in available if DERIVED_FIELD_RE.match(name)):
        if name not in ds.variables:  # defensive for unusual mapping implementations
            continue
        var = ds.variables[name]
        location = _location(var, nnode, nele)
        wet_var = _find_wet_mask(ds, location, nnode, nele) if location else None
        axis = _time_axis(var)
        frames = var.shape[axis] if axis is not None else 1
        expected_location = "node" if name.startswith("salinity_") else "element"
        if location != expected_location:
            critical.append(
                f"{name} is centered at {location or 'an unknown location'}; expected {expected_location}."
            )
        if axis is None:
            critical.append(f"{name} has no time dimension.")
        elif time_count and frames != time_count:
            critical.append(f"{name} has {frames} frame(s), but verified Times contains {time_count}.")
        if var.ndim != 2:
            critical.append(f"{name} has {var.ndim} dimensions; compact derived fields must be time-by-space.")
        if wet_var is None:
            critical.append(f"{name} has no applicable {expected_location} wet mask in the compact product.")
        elif _time_axis(wet_var) is not None and time_count and wet_var.shape[_time_axis(wet_var)] != time_count:
            critical.append(f"Wet mask {wet_var.name} does not contain the verified {time_count} time frame(s).")
        coverage: list[float] = []
        all_nan: list[int] = []
        global_min = math.inf
        global_max = -math.inf
        wet_counts: list[int] = []
        for index in range(frames):
            values = np.asarray(_frame(var, index), dtype=float).reshape(-1)
            wet = np.ones(values.size, dtype=bool)
            if wet_var is not None:
                mask_index = min(index, (wet_var.shape[_time_axis(wet_var)] - 1)) if _time_axis(wet_var) is not None else 0
                mask = np.asarray(_frame(wet_var, mask_index)).reshape(-1)
                if mask.size == values.size:
                    wet = np.isfinite(mask) & (mask > 0)
                else:
                    critical.append(
                        f"Wet mask {wet_var.name} length {mask.size} does not match {name} length {values.size}."
                    )
            finite = np.isfinite(values)
            wet_count = int(np.sum(wet))
            wet_counts.append(wet_count)
            fraction = float(np.sum(finite & wet) / wet_count) if wet_count else 0.0
            coverage.append(fraction)
            if not np.any(finite):
                all_nan.append(index)
            if np.any(finite & wet):
                global_min = min(global_min, float(np.nanmin(values[finite & wet])))
                global_max = max(global_max, float(np.nanmax(values[finite & wet])))
        minimum_coverage = min(coverage) if coverage else 0.0
        if minimum_coverage < 0.95:
            critical.append(f"{name} minimum finite wet coverage is {minimum_coverage:.3%}, below 95%.")
        if all_nan:
            critical.append(f"{name} has all-NaN frame(s): {all_nan}.")
        if math.isfinite(global_min) and math.isfinite(global_max):
            plausibility = _plausibility(name, global_min, global_max)
            if plausibility:
                warnings.append(plausibility)
        variables[name] = {
            "dimensions": list(var.dimensions),
            "shape": list(var.shape),
            "location": location,
            "wet_mask": getattr(wet_var, "name", None),
            "frame_count": frames,
            "minimum_finite_wet_fraction": minimum_coverage,
            "finite_wet_fraction_by_frame": coverage,
            "all_nan_frames": all_nan,
            "wet_count_min": min(wet_counts) if wet_counts else 0,
            "wet_min": global_min if math.isfinite(global_min) else None,
            "wet_max": global_max if math.isfinite(global_max) else None,
        }
    speed_consistency: dict[str, Any] = {}
    for view in sorted(requested_suffixes.union(available_suffixes)):
        names = (f"u_{view}", f"v_{view}", f"current_speed_{view}")
        if not all(name in ds.variables for name in names):
            continue
        u_var, v_var, speed_var = (ds.variables[name] for name in names)
        if u_var.shape != v_var.shape or u_var.shape != speed_var.shape:
            critical.append(f"The u/v/speed shapes for '{view}' are inconsistent.")
            speed_consistency[view] = {
                "compared_value_count": 0,
                "max_absolute_error": None,
                "max_relative_error": None,
                "consistent": False,
            }
            continue
        axis = _time_axis(speed_var)
        frames = speed_var.shape[axis] if axis is not None else 1
        max_abs_error = 0.0
        max_relative_error = 0.0
        compared = 0
        for index in range(frames):
            u_values = np.asarray(_frame(u_var, index), dtype=float).reshape(-1)
            v_values = np.asarray(_frame(v_var, index), dtype=float).reshape(-1)
            speed_values = np.asarray(_frame(speed_var, index), dtype=float).reshape(-1)
            if not (u_values.size == v_values.size == speed_values.size):
                critical.append(f"{names[2]} does not share the u/v component shape.")
                continue
            expected_speed = np.hypot(u_values, v_values)
            finite = np.isfinite(expected_speed) & np.isfinite(speed_values)
            if not np.any(finite):
                continue
            difference = np.abs(speed_values[finite] - expected_speed[finite])
            scale = np.maximum(expected_speed[finite], 1.0e-8)
            max_abs_error = max(max_abs_error, float(np.max(difference)))
            max_relative_error = max(max_relative_error, float(np.max(difference / scale)))
            compared += int(np.sum(finite))
        consistent = max_abs_error <= 1.0e-5 or max_relative_error <= 5.0e-4
        if compared and not consistent:
            critical.append(
                f"{names[2]} is inconsistent with sqrt({names[0]}^2 + {names[1]}^2): "
                f"max absolute error {max_abs_error:.4g}, relative error {max_relative_error:.4g}."
            )
        speed_consistency[view] = {
            "compared_value_count": compared,
            "max_absolute_error": max_abs_error if compared else None,
            "max_relative_error": max_relative_error if compared else None,
            "consistent": consistent if compared else None,
        }

    source_aliases: dict[str, set[str]] = {
        "salt": {"salinity"},
        "salinity": {"salinity"},
        "velocity": {"u", "v"},
        "current": {"u", "v"},
        "current_speed": {"u", "v"},
        "u": {"u"},
        "v": {"v"},
    }
    requested_sources: set[str] = set()
    for name in requested:
        requested_sources.update(source_aliases.get(name, {name}))
    source_health: dict[str, Any] = {}
    for name in sorted(requested_sources):
        if name not in ds.variables:
            critical.append(f"Requested native source variable '{name}' is missing from the compact product.")
            continue
        var = ds.variables[name]
        axis = _time_axis(var)
        location = _location(var, nnode, nele)
        spatial_size = nnode if location == "node" else nele if location == "element" else 0
        wet_var = _find_wet_mask(ds, location, nnode, nele) if location else None
        if axis is None:
            critical.append(f"Requested native source variable '{name}' has no time dimension.")
            frames = 1
        else:
            frames = var.shape[axis]
            if time_count and frames != time_count:
                critical.append(f"Requested native source variable '{name}' has {frames} frames, expected {time_count}.")
        if location is None:
            critical.append(f"Requested native source variable '{name}' has unknown node/element centering.")
        if wet_var is None:
            critical.append(f"Requested native source variable '{name}' has no applicable wet mask.")
        coverages: list[float] = []
        all_nan: list[int] = []
        for index in range(frames):
            values = np.asarray(_frame(var, index), dtype=float)
            if not spatial_size or values.size % spatial_size:
                critical.append(f"Requested native source variable '{name}' cannot be aligned to its spatial dimension.")
                break
            layers = values.reshape(-1, spatial_size)
            wet = np.ones(spatial_size, dtype=bool)
            if wet_var is not None:
                wet_axis = _time_axis(wet_var)
                mask_index = min(index, wet_var.shape[wet_axis] - 1) if wet_axis is not None else 0
                mask = np.asarray(_frame(wet_var, mask_index)).reshape(-1)
                if mask.size != spatial_size:
                    critical.append(f"Wet mask {wet_var.name} cannot be aligned to native source variable '{name}'.")
                else:
                    wet = np.isfinite(mask) & (mask > 0)
            denominator = int(np.sum(wet)) * layers.shape[0]
            finite = np.isfinite(layers) & wet[None, :]
            coverages.append(float(np.sum(finite) / denominator) if denominator else 0.0)
            if not np.any(np.isfinite(layers)):
                all_nan.append(index)
        minimum_coverage = min(coverages) if coverages else 0.0
        if minimum_coverage < 0.95:
            critical.append(
                f"Requested native source variable '{name}' minimum finite wet coverage is {minimum_coverage:.3%}, below 95%."
            )
        if all_nan:
            critical.append(f"Requested native source variable '{name}' has all-NaN frame(s): {all_nan}.")
        source_health[name] = {
            "dimensions": list(var.dimensions),
            "shape": list(var.shape),
            "location": location,
            "wet_mask": getattr(wet_var, "name", None),
            "frame_count": frames,
            "minimum_finite_wet_fraction": minimum_coverage,
            "all_nan_frames": all_nan,
        }
    vertical_consistency: dict[str, Any] = {}
    layer_profile = _sigma_layer_profile(ds)
    layer_indices: dict[str, int] = {}
    if layer_profile is not None and len(layer_profile):
        order = np.argsort(np.abs(layer_profile))
        layer_indices["surface"] = int(order[0])
        layer_indices["near_surface"] = int(order[1] if len(order) > 1 else order[0])
        layer_indices["bottom"] = int(order[-1])
    for name in sorted(available):
        match = re.fullmatch(r"(salinity|u|v)_(surface|near_surface|bottom|sigma_(\d{3}))", name)
        if not match or match.group(1) not in ds.variables:
            continue
        source_name, suffix, explicit_index = match.group(1), match.group(2), match.group(3)
        layer_index = int(explicit_index) if explicit_index is not None else layer_indices.get(suffix)
        source_var, derived_var = ds.variables[source_name], ds.variables[name]
        if layer_index is None or not any("siglay" in dim.lower() for dim in source_var.dimensions):
            critical.append(f"Cannot resolve the native sigma layer used by {name}.")
            continue
        siglay_axis = next(index for index, dim in enumerate(source_var.dimensions) if "siglay" in dim.lower())
        if layer_index < 0 or layer_index >= source_var.shape[siglay_axis]:
            critical.append(f"{name} requests sigma index {layer_index}, outside source variable '{source_name}'.")
            continue
        derived_time_axis = _time_axis(derived_var)
        frames = derived_var.shape[derived_time_axis] if derived_time_axis is not None else 1
        max_abs_error = 0.0
        compared = 0
        for index in range(frames):
            source_values = np.asarray(_source_layer_frame(source_var, index, layer_index), dtype=float).reshape(-1)
            derived_values = np.asarray(_frame(derived_var, index), dtype=float).reshape(-1)
            if source_values.size != derived_values.size:
                critical.append(f"{name} cannot be aligned to source variable '{source_name}'.")
                continue
            finite = np.isfinite(source_values) & np.isfinite(derived_values)
            if np.any(finite):
                max_abs_error = max(
                    max_abs_error,
                    float(np.max(np.abs(source_values[finite] - derived_values[finite]))),
                )
                compared += int(np.sum(finite))
        consistent = bool(compared) and max_abs_error <= 1.0e-5
        if not consistent:
            critical.append(
                f"{name} is inconsistent with dynamically resolved source sigma layer {layer_index} "
                f"(max absolute error {max_abs_error:.4g})."
            )
        vertical_consistency[name] = {
            "source_variable": source_name,
            "resolved_sigma_index": layer_index,
            "compared_value_count": compared,
            "max_absolute_error": max_abs_error if compared else None,
            "consistent": consistent,
        }
    return {
        "available_variables": sorted(available),
        "expected_derived_variables": sorted(expected),
        "missing_derived_variables": missing,
        "paired_uv": paired,
        "speed_consistency": speed_consistency,
        "variables": variables,
        "requested_native_source_variables": sorted(requested_sources),
        "source_variable_health": source_health,
        "vertical_view_source_consistency": vertical_consistency,
    }, critical, warnings


def _plot_map(path: Path, lon: Any, lat: Any, triangles: Any, values: Any, title: str, name: str) -> str:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("no finite values")
    vmin, vmax = [float(item) for item in np.nanpercentile(finite, [1.0, 99.0])]
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    cmap = "viridis" if name.startswith("salinity") else "magma"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 7.0), constrained_layout=True)
    tri = Triangulation(lon, lat, triangles)
    if values.size == len(lon):
        artist = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    elif values.size == len(triangles):
        artist = ax.tripcolor(tri, facecolors=values, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    else:
        raise ValueError(f"field length {values.size} does not match nodes/elements")
    fig.colorbar(artist, ax=ax, shrink=0.82)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _make_maps(ds: Any, plots_dir: Path, lon: Any, lat: Any, triangles: Any, times: Sequence[datetime]) -> tuple[list[dict[str, Any]], list[str]]:
    plots: list[dict[str, Any]] = []
    warnings: list[str] = []
    if lon is None or lat is None or triangles is None:
        return plots, ["Representative maps were not created because usable FVCOM geometry is absent."]
    midpoint = len(times) // 2 if times else 0
    for name in MAP_NAMES:
        if name not in ds.variables:
            continue
        var = ds.variables[name]
        axis = _time_axis(var)
        frame_index = min(midpoint, var.shape[axis] - 1) if axis is not None else 0
        try:
            values = _frame(var, frame_index)
            timestamp = _iso(times[frame_index]) if frame_index < len(times) else f"frame {frame_index}"
            output = plots_dir / f"{name}_midpoint.png"
            saved = _plot_map(output, lon, lat, triangles, values, f"SSCOFS {name} — {timestamp}", name)
            plots.append({"variable": name, "frame_index": frame_index, "time": timestamp, "path": saved})
        except Exception as exc:
            warnings.append(f"Could not create midpoint map for {name}: {exc}")
    return plots, warnings


def _netcdf_check(path: Path, request: dict[str, Any], plots_dir: Path) -> tuple[dict[str, Any], list[datetime], list[str], list[str]]:
    critical: list[str] = []
    warnings: list[str] = []
    try:
        from netCDF4 import Dataset
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}, [], [f"netCDF4 is unavailable: {exc}"], warnings
    try:
        with Dataset(path, "r") as ds:
            times = _netcdf_times(ds)
            attrs = {name: str(ds.getncattr(name)) for name in ds.ncattrs() if name.lower() in {"source", "history", "institution", "title", "source_bucket", "source_prefix"}}
            result: dict[str, Any] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "dimensions": {name: len(dim) for name, dim in ds.dimensions.items()},
                "variables": sorted(ds.variables),
                "source_attributes": attrs,
                "time_count": len(times),
            }
            if request.get("product", "fields") == "fields":
                mesh, lon, lat, triangles, failures, notes = _mesh_check(ds)
                critical.extend(failures)
                warnings.extend(notes)
                sigma, failures, notes = _sigma_check(ds, triangles)
                critical.extend(failures)
                warnings.extend(notes)
                nnode = int(mesh.get("node_count") or 0)
                nele = int(mesh.get("connectivity", {}).get("element_count") or 0)
                fields, failures, notes = _field_checks(ds, request, nnode, nele)
                critical.extend(failures)
                warnings.extend(notes)
                plots, notes = _make_maps(ds, plots_dir, lon, lat, triangles, times)
                warnings.extend(notes)
                result.update({"mesh": mesh, "sigma": sigma, "field_health": fields, "plots": plots})
            else:
                result["plots"] = []
            return result, times, critical, warnings
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}, [], [f"Could not inspect NetCDF {path.name}: {exc}"], warnings


def _aggregate_passthrough_times(paths: Sequence[Path]) -> tuple[list[datetime], dict[str, Any], list[str]]:
    records: list[dict[str, Any]] = []
    all_times: list[datetime] = []
    warnings: list[str] = []
    try:
        from netCDF4 import Dataset
    except Exception as exc:
        return [], {"file_count": 0, "error": str(exc)}, [f"Cannot aggregate passthrough Times: {exc}"]
    for path in paths:
        try:
            with Dataset(path, "r") as ds:
                times = _netcdf_times(ds)
            all_times.extend(times)
            records.append({
                "path": str(path),
                "time_count": len(times),
                "start": _iso(times[0]) if times else None,
                "end": _iso(times[-1]) if times else None,
            })
        except Exception as exc:
            records.append({"path": str(path), "error": str(exc), "time_count": 0})
            warnings.append(f"Could not decode passthrough Times from {path.name}: {exc}")
    sorted_times = sorted(all_times)
    unique_times = sorted(set(sorted_times))
    duplicate_count = len(sorted_times) - len(unique_times)
    if duplicate_count:
        warnings.append(
            f"Passthrough files contain {duplicate_count} overlapping internal timestamp(s); coverage uses their unique union."
        )
    return unique_times, {
        "file_count": len(paths),
        "decoded_time_count": len(sorted_times),
        "unique_time_count": len(unique_times),
        "overlap_count": duplicate_count,
        "files": records,
    }, warnings


def _raw_geometry_signature(path: Path) -> dict[str, Any] | None:
    """Hash static FVCOM geometry/schema one variable at a time."""
    try:
        import numpy as np
        from netCDF4 import Dataset
        with Dataset(path, "r") as ds:
            names = set(ds.variables)
            if not {"salinity", "u", "v", "nv"}.issubset(names):
                return None
            if any(DERIVED_FIELD_RE.match(name) for name in names):
                return None
            digest = hashlib.sha256()
            geometry_variables = [
                name for name in ("lon", "lat", "lonc", "latc", "x", "y", "xc", "yc", "h", "nv", "siglay", "siglev")
                if name in ds.variables
            ]
            shapes: dict[str, list[int]] = {}
            for name in geometry_variables:
                var = ds.variables[name]
                shapes[name] = list(var.shape)
                digest.update(name.encode("utf-8"))
                digest.update(str(var.dtype).encode("ascii"))
                digest.update(json.dumps(list(var.shape)).encode("ascii"))
                values = var[:]
                if np.ma.isMaskedArray(values):
                    values = np.ma.filled(values, np.nan if np.issubdtype(values.dtype, np.floating) else -999999)
                array = np.ascontiguousarray(values)
                digest.update(memoryview(array).cast("B"))
                del values, array
            schema = {
                name: {"dimensions": list(ds.variables[name].dimensions), "dtype": str(ds.variables[name].dtype)}
                for name in ("salinity", "u", "v")
            }
            digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
            return {
                "path": str(path),
                "fingerprint_sha256": digest.hexdigest(),
                "geometry_variables": geometry_variables,
                "geometry_shapes": shapes,
                "field_schema": schema,
            }
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}


def _raw_consistency(candidates: Sequence[Path]) -> tuple[dict[str, Any], list[str], list[str]]:
    signatures = [item for item in (_raw_geometry_signature(path) for path in candidates) if item is not None]
    critical: list[str] = []
    warnings: list[str] = []
    usable = [item for item in signatures if not item.get("error")]
    errors = [item for item in signatures if item.get("error")]
    for item in errors:
        critical.append(f"Could not inspect raw FVCOM schema for {Path(item['path']).name}: {item['error']}")
    groups: dict[str, list[str]] = {}
    for item in usable:
        groups.setdefault(str(item["fingerprint_sha256"]), []).append(str(item["path"]))
    if len(groups) > 1:
        critical.append(f"Raw SSCOFS files contain {len(groups)} distinct mesh/schema fingerprints.")
    return {
        "raw_field_file_count": len(signatures),
        "consistent": len(groups) <= 1 and not errors,
        "fingerprint_groups": groups,
        "files": signatures,
        "method": "SHA-256 over static geometry/topology/sigma variables and source-field schemas; dynamic fields are never loaded.",
    }, critical, warnings


def _source_summary(estimate: Any, manifest: Any) -> dict[str, Any]:
    keys: set[str] = set()
    urls: set[str] = set()
    etags: set[str] = set()
    layouts: set[str] = set()
    for item in list(_walk_dicts(estimate)) + list(_walk_dicts(manifest)):
        key = item.get("key") or item.get("object_key")
        if isinstance(key, str) and key.startswith("sscofs/"):
            keys.add(key)
            layouts.add("nested_daily" if "/netcdf/" in key and len(key.split("/")) > 6 else "legacy_monthly")
        url = item.get("url") or item.get("source_url")
        if isinstance(url, str) and url.startswith("http"):
            urls.add(url)
        etag = item.get("etag") or item.get("ETag")
        if etag:
            etags.add(str(etag).strip('"'))
    return {
        "provider": "NOAA/NOS Operational Forecast Systems",
        "system": "SSCOFS",
        "bucket": "noaa-nos-ofs-pds",
        "access": "anonymous HTTPS and S3 ListObjectsV2",
        "object_count": len(keys),
        "keys": sorted(keys),
        "urls": sorted(urls),
        "etag_count": len(etags),
        "layouts_observed": sorted(layouts),
    }


def _keyed_objects(value: Any) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for item in _walk_dicts(value):
        key = item.get("key") or item.get("object_key")
        if not isinstance(key, str) or not key.startswith("sscofs/"):
            continue
        size = _integer(_first(item, SIZE_KEYS))
        current = objects.get(key)
        # Prefer the richest occurrence when a record contains nested copies.
        if current is None or (current.get("size") is None and size is not None):
            objects[key] = {
                "size": size,
                "etag": item.get("etag") or item.get("ETag"),
                "sha256": _first(item, HASH_KEYS),
                "status": item.get("status") or item.get("outcome"),
            }
    return objects


def _crosscheck_estimate_manifest(estimate: Any, manifest: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    estimated = _keyed_objects(estimate)
    fetched = _keyed_objects(manifest)
    critical: list[str] = []
    warnings: list[str] = []
    missing_from_manifest = sorted(set(estimated) - set(fetched))
    unexpected = sorted(set(fetched) - set(estimated))
    size_mismatches: list[dict[str, Any]] = []
    etag_mismatches: list[dict[str, Any]] = []
    for key in sorted(set(estimated).intersection(fetched)):
        expected_size = estimated[key].get("size")
        fetched_size = fetched[key].get("size")
        if expected_size is not None and fetched_size is not None and expected_size != fetched_size:
            size_mismatches.append({"key": key, "estimated_size": expected_size, "manifest_size": fetched_size})
        estimated_etag = str(estimated[key].get("etag") or "").strip('"')
        fetched_etag = str(fetched[key].get("etag") or "").strip('"')
        if estimated_etag and fetched_etag and estimated_etag != fetched_etag:
            etag_mismatches.append({"key": key, "estimated_etag": estimated_etag, "manifest_etag": fetched_etag})
    if missing_from_manifest:
        critical.append(f"fetch_manifest.json omits {len(missing_from_manifest)} object(s) selected by download_estimate.json.")
    if size_mismatches:
        critical.append(f"Estimate/manifest byte sizes disagree for {len(size_mismatches)} object(s).")
    if etag_mismatches:
        critical.append(f"Estimate/manifest ETags disagree for {len(etag_mismatches)} object(s).")
    if unexpected:
        warnings.append(f"fetch_manifest.json contains {len(unexpected)} object(s) not selected by the estimate.")
    return {
        "estimated_object_count": len(estimated),
        "manifest_object_count": len(fetched),
        "missing_from_manifest": missing_from_manifest,
        "unexpected_in_manifest": unexpected,
        "size_mismatches": size_mismatches,
        "etag_mismatches": etag_mismatches,
        "consistent": not missing_from_manifest and not size_mismatches and not etag_mismatches,
    }, critical, warnings


def _json_clean(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats to strict-JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_clean(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(child) for child in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # Avoid a mandatory numpy import for manifest-only passthrough checks.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_clean(value.item())
        except (TypeError, ValueError):
            pass
    return value


def evaluate_health(request_path: str | Path, run_dir: str | Path, output: str | Path, plots_dir: str | Path) -> dict[str, Any]:
    request = _read_json(request_path)
    if not isinstance(request, dict):
        raise ValueError("request JSON must be an object")
    run_path = Path(run_dir)
    output_path = Path(output)
    plots_path = Path(plots_dir)
    manifest_path = _find_json(run_path, "fetch_manifest.json")
    estimate_path = _find_json(run_path, "download_estimate.json")
    manifest = _read_json(manifest_path) if manifest_path else {}
    estimate = _read_json(estimate_path) if estimate_path else {}
    critical: list[str] = []
    warnings: list[str] = []
    dataset_path, candidates = _select_dataset(run_path)
    allow_deleted = bool(
        request.get("cache_policy") == "delete_after_extract"
        and dataset_path is not None
        and _dataset_score(dataset_path)[0] >= 10
    )
    manifest_report, manifest_times, failures, notes = _verify_manifest(
        manifest, run_path, allow_deleted=allow_deleted
    )
    manifest_failures = list(failures)
    critical.extend(failures)
    warnings.extend(notes)
    crosscheck, failures, notes = _crosscheck_estimate_manifest(estimate, manifest)
    critical.extend(failures)
    warnings.extend(notes)
    if manifest_path and not manifest_report.get("record_count"):
        critical.append("fetch_manifest.json contains no downloadable object records.")
    if estimate_path and not crosscheck.get("estimated_object_count"):
        critical.append("download_estimate.json contains no selected SSCOFS object records.")
    dataset_report: dict[str, Any] = {"path": None}
    dataset_times: list[datetime] = []
    if dataset_path:
        dataset_report, dataset_times, failures, notes = _netcdf_check(dataset_path, request, plots_path)
        critical.extend(failures)
        warnings.extend(notes)
        if request.get("product") in {"stations", "regulargrid"}:
            aggregate_times, aggregate_report, notes = _aggregate_passthrough_times(candidates)
            if aggregate_times:
                dataset_times = aggregate_times
            dataset_report["passthrough_time_aggregate"] = aggregate_report
            warnings.extend(notes)
    elif request.get("product", "fields") == "fields":
        critical.append("No NetCDF evidence was found for the native-fields request.")
    else:
        warnings.append("No NetCDF file was available for passthrough metadata inspection.")
    raw_consistency: dict[str, Any] = {"raw_field_file_count": 0, "consistent": None}
    if request.get("product", "fields") == "fields":
        raw_consistency, failures, notes = _raw_consistency(candidates)
        critical.extend(failures)
        warnings.extend(notes)
        if not raw_consistency.get("raw_field_file_count") and not allow_deleted:
            warnings.append("No cached native-field source files were available for cross-file mesh/schema comparison.")
    use_dataset_times = len(dataset_times) > 1 or not manifest_times
    observed_times = dataset_times if use_dataset_times else manifest_times
    time_source = "netcdf" if use_dataset_times else "fetch_manifest"
    time_report, failures, notes = _time_check(request, observed_times, time_source)
    critical.extend(failures)
    warnings.extend(notes)
    if not manifest_path:
        message = "fetch_manifest.json was not found; raw object sizes and hashes were not independently checked."
        critical.append(message)
    if not estimate_path:
        message = "download_estimate.json was not found; planned-source provenance is incomplete."
        critical.append(message)
    critical = list(dict.fromkeys(critical))
    warnings = [item for item in dict.fromkeys(warnings) if item not in critical]
    status = "fail" if critical else ("warning" if warnings else "pass")
    result = {
        "schema_version": "sscofs_health_v1",
        "generated_utc": _iso(datetime.now(UTC)),
        "status": status,
        "request_path": str(Path(request_path)),
        "run_dir": str(run_path),
        "request": request,
        "source_provenance": _source_summary(estimate, manifest),
        "estimate_path": str(estimate_path) if estimate_path else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_integrity": manifest_report,
        "estimate_manifest_consistency": crosscheck,
        "time_health": time_report,
        "selected_netcdf": dataset_report,
        "raw_source_consistency": raw_consistency,
        "netcdf_candidates": [str(path) for path in candidates],
        "critical_caveats": critical,
        "warnings": warnings,
        "acceptance": {
            "time_complete": not bool(time_report.get("missing")),
            "time_unique_monotonic_cadence": bool(time_report.get("unique") and time_report.get("strictly_monotonic") and time_report.get("cadence_ok")),
            "object_integrity": (
                not manifest_failures
                and bool(manifest_path)
                and bool(estimate_path)
                and bool(manifest_report.get("record_count"))
                and bool(crosscheck.get("consistent"))
            ),
            "finite_wet_threshold": 0.95,
            "passed": not critical,
        },
        "reporting_policy": "Critical caveats fail acceptance. Broad physical-range checks are warnings only; source values are never clipped.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _json_clean(result)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Path to an sscofs_request_v1 JSON request.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing estimate, manifest, raw cache, and/or compact NetCDF.")
    parser.add_argument("--output", required=True, help="Health JSON path to write.")
    parser.add_argument("--plots-dir", required=True, help="Directory for representative midpoint PNG maps.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_health(args.request, args.run_dir, args.output, args.plots_dir)
    except Exception as exc:
        print(f"health check failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "critical_caveat_count": len(result["critical_caveats"]),
        "warning_count": len(result["warnings"]),
        "output": str(args.output),
    }, indent=2))
    return 0 if result["status"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
