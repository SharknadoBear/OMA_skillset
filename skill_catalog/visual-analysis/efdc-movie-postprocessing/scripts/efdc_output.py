"""Load, validate, concatenate, and derive scalar fields from EFDC NetCDF output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from netCDF4 import Dataset, chartostring, num2date


TIME_NAMES = ("time", "ocean_time", "Times")
LON_NAMES = ("lon", "longitude", "xlon", "lon_rho")
LAT_NAMES = ("lat", "latitude", "ylat", "lat_rho")
MASK_NAMES = ("mask", "wet_mask")
DEPTH_NAMES = ("depth", "h", "bathymetry")
SIGMA_NAMES = ("sigma", "siglay", "s_rho", "z")
VERTICAL_DIM_NAMES = {"sigma", "siglay", "siglev", "s_rho", "s_w", "z"}


@dataclass(frozen=True)
class EFDCGrid:
    """Canonical native EFDC curvilinear grid."""

    lon: np.ndarray
    lat: np.ndarray
    mask: np.ndarray
    source_mask: np.ndarray
    depth: np.ndarray | None
    sigma: np.ndarray | None
    geometry_sha256: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.lon.shape


@dataclass(frozen=True)
class ScalarSeries:
    """Concatenated, time-normalized scalar frames and provenance."""

    grid: EFDCGrid
    times: np.ndarray
    original_times: np.ndarray
    time_offsets_seconds: np.ndarray
    values: np.ndarray
    record_sources: tuple[str, ...]
    record_indices: tuple[int, ...]
    sources: tuple[dict[str, Any], ...]
    duplicate_times_removed: int
    resolution: dict[str, Any]


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _find_name(ds: Dataset, candidates: Sequence[str]) -> str | None:
    lookup = {name.lower(): name for name in ds.variables}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _as_float(var) -> np.ndarray:
    # Disable netCDF4's automatic mask so a finite non-fill payload outside the
    # declared EFDC wet cells cannot disappear behind an independently packed
    # coordinate/mask convention.
    raw_value = var[:]
    raw = raw_value.data if np.ma.isMaskedArray(raw_value) else raw_value
    data = np.asarray(raw, dtype=np.float64)
    for attr in ("_FillValue", "missing_value"):
        if hasattr(var, attr):
            values = np.atleast_1d(getattr(var, attr)).astype(float)
            for value in values:
                if np.isfinite(value):
                    data[np.isclose(data, value, rtol=0.0, atol=0.0)] = np.nan
    return data


def _to_2d(var, name: str) -> np.ndarray:
    arr = _as_float(var)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Grid variable {name!r} must be two-dimensional after squeezing; got {arr.shape}.")
    return arr


def _geometry_hash(
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
    source_mask: np.ndarray,
    depth: np.ndarray | None,
    sigma: np.ndarray | None,
) -> str:
    digest = hashlib.sha256()
    for name, array in (("lon", lon), ("lat", lat), ("mask", mask),
                        ("source_mask", source_mask), ("depth", depth), ("sigma", sigma)):
        digest.update(name.encode("ascii"))
        if array is None:
            digest.update(b"none")
            continue
        canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def read_grid(ds: Dataset) -> EFDCGrid:
    """Read a canonical EFDC grid from an open dataset."""

    compact = str(getattr(ds, "schema_version", "")) == "efdc_compact_fields_v1"
    if compact and str(getattr(ds, "vertical_method", "")) != "efdc_layer_top_sigma_with_bed_edge_1":
        raise ValueError("Compact EFDC input has an incompatible or missing vertical_method.")
    lon_name = _find_name(ds, LON_NAMES)
    lat_name = _find_name(ds, LAT_NAMES)
    if lon_name is None or lat_name is None:
        raise KeyError("EFDC input must contain two-dimensional lon and lat variables.")
    lon = _to_2d(ds.variables[lon_name], lon_name)
    lat = _to_2d(ds.variables[lat_name], lat_name)
    if lon.shape != lat.shape:
        raise ValueError(f"Longitude and latitude shapes differ: {lon.shape} versus {lat.shape}.")

    if "mask" not in ds.variables:
        raise KeyError("EFDC input must preserve the source mask; implicit all-wet grids are unsafe.")
    source_mask = _to_2d(ds.variables["mask"], "mask")
    if source_mask.shape != lon.shape:
        raise ValueError(f"Mask shape {source_mask.shape} differs from coordinate shape {lon.shape}.")
    wet = np.isfinite(source_mask) & (source_mask == 5.0)
    recognized = wet | (source_mask == 0.0) | (source_mask < 0.0)
    unexpected = np.isfinite(source_mask) & ~recognized
    if np.any(unexpected):
        codes = np.unique(source_mask[unexpected])
        raise ValueError(f"Ambiguous EFDC mask convention; unexpected finite codes {codes[:10].tolist()}.")
    if not np.any(wet):
        raise ValueError("EFDC grid contains no active cells with source mask == 5.")
    if "wet_mask" in ds.variables:
        declared = _to_2d(ds.variables["wet_mask"], "wet_mask")
        if declared.shape != lon.shape or not np.array_equal(declared == 1.0, wet):
            raise ValueError("Derived wet_mask disagrees with the source mask == 5 contract.")
    if np.any(wet & (~np.isfinite(lon) | ~np.isfinite(lat))):
        raise ValueError("Every active EFDC cell must have finite longitude and latitude.")
    mask = wet.astype(np.int8)

    depth_name = _find_name(ds, DEPTH_NAMES)
    depth = _to_2d(ds.variables[depth_name], depth_name) if depth_name else None
    if depth is not None and depth.shape != lon.shape:
        raise ValueError(f"Depth shape {depth.shape} differs from coordinate shape {lon.shape}.")
    if depth is not None and np.any(wet & (~np.isfinite(depth) | (depth <= 0.0))):
        raise ValueError("EFDC bathymetric depth must be finite and positive at every mask == 5 cell.")

    sigma_name = _find_name(ds, SIGMA_NAMES)
    sigma = None
    if sigma_name:
        sigma = np.squeeze(_as_float(ds.variables[sigma_name]))
        if sigma.ndim != 1 or sigma.size < 1:
            raise ValueError(f"Sigma coordinate {sigma_name!r} must be one-dimensional; got {sigma.shape}.")
        if not np.isfinite(sigma).all():
            raise ValueError("Sigma coordinate contains non-finite values.")
        # EFDC layer tops are resolved scientifically by value, independent of
        # storage ordering.  sigma_weights performs the strict uniqueness and
        # [0,1) convention checks.
        sigma_weights(sigma)

    return EFDCGrid(
        lon=lon,
        lat=lat,
        mask=mask,
        source_mask=source_mask,
        depth=depth,
        sigma=sigma,
        geometry_sha256=_geometry_hash(lon, lat, mask, source_mask, depth, sigma),
    )


def _assert_same_grid(reference: EFDCGrid, candidate: EFDCGrid, path: Path) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(f"EFDC grid shape drift in {path}: {candidate.shape} versus {reference.shape}.")
    checks = (
        ("lon", reference.lon, candidate.lon, 1.0e-10),
        ("lat", reference.lat, candidate.lat, 1.0e-10),
        ("mask", reference.mask, candidate.mask, 0.0),
        ("source_mask", reference.source_mask, candidate.source_mask, 0.0),
        ("depth", reference.depth, candidate.depth, 1.0e-8),
        ("sigma", reference.sigma, candidate.sigma, 1.0e-10),
    )
    for name, left, right, atol in checks:
        if (left is None) != (right is None):
            raise ValueError(f"EFDC {name} schema drift in {path}.")
        if left is not None and not np.allclose(left, right, rtol=0.0, atol=atol, equal_nan=True):
            raise ValueError(f"EFDC {name} geometry drift in {path}.")


def _datetime64_from_parts(value: Any) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(value, "ns")
    return np.datetime64(
        datetime(
            int(value.year),
            int(value.month),
            int(value.day),
            int(getattr(value, "hour", 0)),
            int(getattr(value, "minute", 0)),
            int(getattr(value, "second", 0)),
            int(getattr(value, "microsecond", 0)),
        ),
        "ns",
    )


def _decode_char_times(var) -> np.ndarray:
    raw = var[:]
    if np.asarray(raw).ndim > 1:
        strings = chartostring(raw)
    else:
        strings = raw
    decoded: list[np.datetime64] = []
    for item in np.atleast_1d(strings):
        if isinstance(item, bytes):
            text = item.decode("ascii", errors="strict")
        else:
            text = str(item)
        text = text.strip().replace("_", "T").replace(" ", "T", 1).rstrip("Z")
        decoded.append(np.datetime64(text, "ns"))
    return np.asarray(decoded, dtype="datetime64[ns]")


def decode_times(ds: Dataset) -> tuple[np.ndarray, str]:
    """Decode raw EFDC timestamps without cadence snapping."""

    time_name = _find_name(ds, TIME_NAMES)
    if time_name is None:
        raise KeyError("EFDC input has no recognized time variable.")
    var = ds.variables[time_name]
    if np.dtype(var.dtype).kind in {"S", "U"}:
        return _decode_char_times(var), time_name

    values = np.asarray(np.ma.filled(np.ma.asarray(var[:]), np.nan), dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"Time variable {time_name!r} contains missing values.")
    units = str(getattr(var, "units", "")).strip()
    calendar = str(getattr(var, "calendar", "standard"))
    if "since" in units.lower():
        decoded = num2date(
            values,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
        return np.asarray([_datetime64_from_parts(value) for value in decoded], dtype="datetime64[ns]"), time_name
    if units.lower() in {"seconds since epoch", "unix_seconds", "seconds"} or (
        not units and values.size and np.nanmedian(np.abs(values)) > 1.0e8
    ):
        epoch_ns = np.rint(values * 1.0e9).astype("int64")
        return epoch_ns.astype("datetime64[ns]"), time_name
    raise ValueError(f"Cannot decode time variable {time_name!r} with units {units!r}.")


def normalize_times(
    raw_times: np.ndarray,
    *,
    tolerance_seconds: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Snap fields to HH:30 or stations to six-minute slots within tolerance."""

    times = np.asarray(raw_times).astype("datetime64[ns]")
    if times.size == 0:
        return times.copy(), np.empty(0, dtype=float), None
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative.")
    unique_ns = np.unique(times.astype("int64"))
    cadence: int | None = None
    if unique_ns.size > 1:
        median_seconds = float(np.median(np.diff(unique_ns))) / 1.0e9
        for candidate in (3600, 360):
            if abs(median_seconds - candidate) <= max(120.0, tolerance_seconds * 2.0):
                cadence = candidate
                break
    if cadence is None:
        return times.copy(), np.zeros(times.size, dtype=float), None

    ns = times.astype("int64")
    cadence_ns = int(cadence * 1_000_000_000)
    # SJROFS native fields are timestamped at the middle of each hour.  The
    # six-minute station stream is phase aligned to the Unix epoch.
    phase_ns = 1_800_000_000_000 if cadence == 3600 else 0
    snapped_ns = (
        np.rint((ns.astype(np.float64) - phase_ns) / cadence_ns).astype("int64")
        * cadence_ns + phase_ns
    )
    adjustment_seconds = (snapped_ns - ns) / 1.0e9
    eligible = np.abs(adjustment_seconds) <= tolerance_seconds
    normalized_ns = np.where(eligible, snapped_ns, ns)
    applied = (normalized_ns - ns) / 1.0e9
    return normalized_ns.astype("datetime64[ns]"), applied.astype(float), cadence


def parse_time(value: str | datetime | np.datetime64 | None) -> np.datetime64 | None:
    if value is None:
        return None
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    if isinstance(value, datetime):
        return _datetime64_from_parts(value)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(parsed, "ns")


def sigma_weights(sigma: np.ndarray) -> np.ndarray:
    """Return EFDC layer-top weights in original storage order.

    SJROFS labels each layer by its positive-down top fraction.  Therefore the
    bed edge at one closes the last layer; trapezoidal point quadrature is not
    valid for these files.
    """

    levels = np.asarray(sigma, dtype=float).reshape(-1)
    if levels.size < 1 or not np.isfinite(levels).all():
        raise ValueError("Depth averaging requires finite EFDC layer-top sigma values.")
    order = np.argsort(levels)
    sorted_levels = levels[order]
    tolerance = 1.0e-6
    if abs(float(sorted_levels[0])) > tolerance:
        raise ValueError("EFDC layer-top sigma must begin near zero.")
    if np.any(sorted_levels < -tolerance) or np.any(sorted_levels >= 1.0):
        raise ValueError("EFDC layer-top sigma values must lie in [0, 1).")
    sorted_weights = np.diff(np.concatenate((sorted_levels, [1.0])))
    if np.any(sorted_weights <= 0.0) or not np.isclose(np.sum(sorted_weights), 1.0, atol=1.0e-10):
        raise ValueError("EFDC layer-top sigma must be unique and yield positive bed-closed weights.")
    weights = np.empty_like(sorted_weights)
    weights[order] = sorted_weights
    return weights


def resolve_layer_index(sigma: np.ndarray | None, layer: str) -> int:
    selector = str(layer).lower()
    if selector.startswith("index:"):
        index = int(selector.split(":", 1)[1])
        if sigma is not None and not 0 <= index < len(sigma):
            raise IndexError(f"Sigma index {index} is outside 0..{len(sigma) - 1}.")
        return index
    if sigma is None:
        raise ValueError(f"Layer {layer!r} requires a sigma coordinate; use index:N only if ordering is known.")
    order = np.argsort(np.asarray(sigma, dtype=float))
    if selector == "surface":
        return int(order[0])
    if selector == "near_surface":
        if len(order) < 2:
            raise ValueError("near_surface requires at least two sigma points.")
        return int(order[1])
    if selector == "bottom":
        return int(order[-1])
    raise ValueError(f"Unknown layer selector {layer!r}.")


def layer_suffix(layer: str) -> str:
    selector = str(layer).lower()
    if selector.startswith("index:"):
        return f"sigma_{int(selector.split(':', 1)[1])}"
    aliases = {"depth_mean": "depth_average", "depth-averaged": "depth_average"}
    selector = aliases.get(selector, selector)
    if selector not in {"surface", "near_surface", "bottom", "depth_average"}:
        raise ValueError(f"Unknown layer selector {layer!r}.")
    return selector


def _time_axis(var, time_name: str, time_count: int) -> int | None:
    for index, dim in enumerate(var.dimensions):
        if dim.lower() in {time_name.lower(), "time", "ocean_time"}:
            return index
    candidates = [index for index, size in enumerate(var.shape) if size == time_count]
    return candidates[0] if candidates else None


def _vertical_axis(dims: Sequence[str], shape: Sequence[int], sigma: np.ndarray | None) -> int | None:
    for index, dim in enumerate(dims):
        if dim.lower() in VERTICAL_DIM_NAMES or dim.lower().startswith("sig"):
            return index
    # Shape-only inference is reserved for arrays with a time axis plus at
    # least three non-time axes.  A small spatial y dimension can otherwise
    # equal the sigma count by coincidence.
    if sigma is not None and len(shape) >= 4:
        candidates = [index for index, size in enumerate(shape[1:], start=1) if size == len(sigma)]
        if len(candidates) == 1:
            return candidates[0]
    return None


def _orient_frames(data: np.ndarray, grid_shape: tuple[int, int], variable: str) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    while arr.ndim > 3 and 1 in arr.shape[1:]:
        singleton = next(index for index in range(1, arr.ndim) if arr.shape[index] == 1)
        arr = np.squeeze(arr, axis=singleton)
    if arr.ndim != 3:
        raise ValueError(f"Variable {variable!r} does not reduce to (time,y,x); got {arr.shape}.")
    if arr.shape[1:] == grid_shape:
        return arr
    if arr.shape[1:] == grid_shape[::-1]:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"Variable {variable!r} spatial shape {arr.shape[1:]} differs from grid {grid_shape}.")


def _read_scalar_variable(
    ds: Dataset,
    name: str,
    *,
    grid: EFDCGrid,
    time_name: str,
    time_count: int,
    layer: str,
) -> np.ndarray:
    var = ds.variables[name]
    arr = _as_float(var)
    dims = list(var.dimensions)
    axis = _time_axis(var, time_name, time_count)
    if axis is None:
        raise ValueError(f"Movie variable {name!r} has no recognized time axis.")
    arr = np.moveaxis(arr, axis, 0)
    dim = dims.pop(axis)
    dims.insert(0, dim)

    wet = grid.mask == 1
    if arr.ndim < 3:
        raise ValueError(f"Variable {name!r} has no EFDC spatial grid axes.")
    if arr.shape[-2:] == grid.shape:
        source_order = arr
    elif arr.shape[-2:] == grid.shape[::-1]:
        source_order = np.swapaxes(arr, -2, -1)
    else:
        raise ValueError(f"Variable {name!r} trailing spatial shape {arr.shape[-2:]} differs from grid {grid.shape}.")
    flattened = source_order.reshape((-1,) + grid.shape)
    standard_name = str(getattr(var, "standard_name", "")).lower()
    atmospheric = name.lower() in {"air_u", "air_v", "uwind", "vwind", "eastward_wind", "northward_wind"} or standard_name in {"eastward_wind", "northward_wind", "wind_speed"}
    if not atmospheric and np.any(np.isfinite(flattened[:, ~wet])):
        raise ValueError(
            f"Variable {name!r} contains finite dynamic values outside source mask == 5; "
            "the EFDC wet convention is ambiguous or corrupt."
        )

    vertical_axis = _vertical_axis(dims, arr.shape, grid.sigma)
    if vertical_axis is not None:
        selector = str(layer).lower().replace("depth_mean", "depth_average")
        if selector in {"depth_average", "depth-averaged"}:
            if grid.sigma is None or arr.shape[vertical_axis] != len(grid.sigma):
                raise ValueError(f"Cannot depth-average {name!r} without a matching sigma coordinate.")
            weights = sigma_weights(grid.sigma)
            reshape = [1] * arr.ndim
            reshape[vertical_axis] = len(weights)
            expanded = weights.reshape(reshape)
            finite = np.isfinite(arr)
            numerator = np.sum(np.where(finite, arr * expanded, 0.0), axis=vertical_axis)
            denominator = np.sum(np.where(finite, expanded, 0.0), axis=vertical_axis)
            with np.errstate(invalid="ignore", divide="ignore"):
                arr = np.where(denominator > 0, numerator / denominator, np.nan)
        else:
            index = resolve_layer_index(grid.sigma, layer)
            if not 0 <= index < arr.shape[vertical_axis]:
                raise IndexError(f"Layer index {index} is outside variable {name!r} axis length {arr.shape[vertical_axis]}.")
            arr = np.take(arr, index, axis=vertical_axis)
    frames = _orient_frames(arr, grid.shape, name)
    # Atmospheric fields may cover the full packed grid in source files, but
    # every EFDC scalar/quiver visualization is clipped to active water.
    return np.where(wet[None, :, :], frames, np.nan)


def _has_earth_components(ds: Dataset, east_name: str, north_name: str, *, wind: bool = False) -> None:
    east = ds.variables[east_name]
    north = ds.variables[north_name]
    if east.dimensions != north.dimensions:
        raise ValueError(f"Vector components {east_name!r}/{north_name!r} have mismatched dimensions.")
    if east.shape != north.shape:
        raise ValueError(f"Vector components {east_name!r}/{north_name!r} have mismatched shapes.")
    east_standard = str(getattr(east, "standard_name", "")).lower()
    north_standard = str(getattr(north, "standard_name", "")).lower()
    expected = (
        ("eastward_wind", "northward_wind")
        if wind
        else ("eastward_sea_water_velocity", "northward_sea_water_velocity")
    )
    declared = str(getattr(ds, "vector_components", "")).lower()
    if (east_standard, north_standard) == expected or "earth" in declared:
        return
    east_words = " ".join(
        str(getattr(east, attr, "")).lower() for attr in ("standard_name", "long_name")
    )
    north_words = " ".join(
        str(getattr(north, attr, "")).lower() for attr in ("standard_name", "long_name")
    )
    if "eastward" in east_words and "northward" in north_words:
        return
    combined = " ".join(
        str(getattr(var, attr, "")).lower()
        for var in (east, north)
        for attr in ("long_name", "coordinates")
    )
    if "grid" in combined or "xi" in combined or "eta" in combined:
        raise ValueError("Grid-relative components require rotation metadata; silent rotation is disabled.")
    raise ValueError(
        f"Cannot prove {east_name!r}/{north_name!r} are collocated earth-relative components; "
        "SJROFS requires CF eastward/northward standard_name metadata."
    )


def validate_vector_components(
    input_path: str | Path,
    east_name: str,
    north_name: str,
    *,
    wind: bool = False,
) -> None:
    """Reject unpaired or unproven grid-relative components before quiver use."""

    with Dataset(Path(input_path).expanduser().resolve(), "r") as ds:
        if east_name not in ds.variables or north_name not in ds.variables:
            raise KeyError(f"Vector components {east_name!r}/{north_name!r} are not both present.")
        _has_earth_components(ds, east_name, north_name, wind=wind)


def _variable_lookup(ds: Dataset) -> dict[str, str]:
    return {name.lower(): name for name in ds.variables}


def _resolve_scalar(
    ds: Dataset,
    requested: str,
    layer: str,
) -> tuple[str, tuple[str, ...], bool]:
    lookup = _variable_lookup(ds)
    key = requested.lower()
    suffix = layer_suffix(layer)
    suffixed = f"{key}_{suffix}"

    if key == "salinity":
        if suffixed in lookup:
            return "direct", (lookup[suffixed],), False
        if "salinity" in lookup:
            return "direct", (lookup["salinity"],), False
        if "salt" in lookup:
            return "direct", (lookup["salt"],), False
        raise KeyError("salinity requires salinity_<view>, salinity, or native salt.")

    if key == "current_speed":
        if suffixed in lookup:
            return "direct", (lookup[suffixed],), False
        if key in lookup:
            return "direct", (lookup[key],), False
        compact_pair = (f"eastward_velocity_{suffix}", f"northward_velocity_{suffix}")
        if all(name in lookup for name in compact_pair):
            return "magnitude", (lookup[compact_pair[0]], lookup[compact_pair[1]]), False
        pair = (f"u_{suffix}", f"v_{suffix}")
        if all(name in lookup for name in pair):
            return "magnitude", (lookup[pair[0]], lookup[pair[1]]), False
        if "u" in lookup and "v" in lookup:
            return "magnitude", (lookup["u"], lookup["v"]), False
        raise KeyError("current_speed requires a ready-made field or paired u/v components.")

    if key == "wind_speed":
        if key in lookup:
            return "direct", (lookup[key],), True
        pairs = (("air_u", "air_v"), ("uwind", "vwind"), ("eastward_wind", "northward_wind"))
        for east, north in pairs:
            if east in lookup and north in lookup:
                return "magnitude", (lookup[east], lookup[north]), True
        raise KeyError("wind_speed requires a ready-made field or paired air_u/air_v components.")

    if suffixed in lookup:
        return "direct", (lookup[suffixed],), False
    if key in lookup:
        return "direct", (lookup[key],), False
    raise KeyError(f"Variable {requested!r} was not found. Available examples: {list(ds.variables)[:25]}")


def _read_requested_frames(
    ds: Dataset,
    *,
    requested: str,
    layer: str,
    grid: EFDCGrid,
    time_name: str,
    time_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode, names, wind = _resolve_scalar(ds, requested, layer)
    frames = [
        _read_scalar_variable(
            ds,
            name,
            grid=grid,
            time_name=time_name,
            time_count=time_count,
            layer=layer,
        )
        for name in names
    ]
    if mode == "magnitude":
        _has_earth_components(ds, names[0], names[1], wind=wind)
        values = np.hypot(frames[0], frames[1])
    else:
        values = frames[0]
    return values, {
        "requested_variable": requested,
        "resolved_mode": mode,
        "source_variables": list(names),
        "layer": layer,
        "derived_magnitude": mode == "magnitude",
        "wind_components": wind and mode == "magnitude",
        "vertical_method": str(
            getattr(ds, "vertical_method", "efdc_layer_top_sigma_with_bed_edge_1")
        ),
        "vector_method": "collocated_earth_relative_no_rotation",
        "source_variable_metadata": {
            name: {
                "units": str(getattr(ds.variables[name], "units", "")),
                "standard_name": str(getattr(ds.variables[name], "standard_name", "")),
                "long_name": str(getattr(ds.variables[name], "long_name", name)),
                "dimensions": list(ds.variables[name].dimensions),
            }
            for name in names
        },
    }


def load_scalar_series(
    inputs: Sequence[str | Path],
    *,
    variable: str,
    layer: str = "surface",
    start: str | datetime | np.datetime64 | None = None,
    end_exclusive: str | datetime | np.datetime64 | None = None,
    snap_tolerance_seconds: float = 60.0,
) -> ScalarSeries:
    """Load, grid-check, normalize, deduplicate, and crop scalar EFDC records."""

    paths = [Path(path).expanduser().resolve() for path in inputs]
    if not paths:
        raise ValueError("At least one input file is required.")
    if len(set(paths)) != len(paths):
        raise ValueError("The same input path was supplied more than once.")

    reference: EFDCGrid | None = None
    source_payloads: list[tuple[Path, np.ndarray, np.ndarray, int, dict[str, Any]]] = []
    sources: list[dict[str, Any]] = []
    common_resolution: dict[str, Any] | None = None
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Dataset(path, "r") as ds:
            grid = read_grid(ds)
            if reference is None:
                reference = grid
            else:
                _assert_same_grid(reference, grid, path)
            raw_times, time_name = decode_times(ds)
            frames, resolution = _read_requested_frames(
                ds,
                requested=variable,
                layer=layer,
                grid=grid,
                time_name=time_name,
                time_count=len(raw_times),
            )
            if frames.shape[0] != len(raw_times):
                raise ValueError(f"Time/frame count mismatch in {path}: {len(raw_times)} versus {frames.shape[0]}.")
            if common_resolution is None:
                common_resolution = resolution
            elif common_resolution != resolution:
                raise ValueError(f"Variable resolution drift in {path}: {resolution} versus {common_resolution}.")
        source_order_time = int(np.min(raw_times.astype("int64"))) if len(raw_times) else np.iinfo("int64").max
        source_payloads.append((path, raw_times, frames, source_order_time, resolution))
        sources.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "raw_record_count": len(raw_times),
                "raw_first_time_utc": None
                if not len(raw_times)
                else np.datetime_as_string(raw_times[0], unit="ns") + "Z",
                "raw_last_time_utc": None
                if not len(raw_times)
                else np.datetime_as_string(raw_times[-1], unit="ns") + "Z",
                "geometry_sha256": grid.geometry_sha256,
            }
        )
    assert reference is not None

    all_raw = np.concatenate([payload[1] for payload in source_payloads])
    all_normalized, all_offsets, cadence = normalize_times(
        all_raw, tolerance_seconds=snap_tolerance_seconds
    )
    records: list[tuple[int, int, str, int, np.datetime64, np.datetime64, float, np.ndarray]] = []
    cursor = 0
    for path, raw_times, frames, source_order_time, _ in source_payloads:
        count = len(raw_times)
        normalized = all_normalized[cursor : cursor + count]
        offsets = all_offsets[cursor : cursor + count]
        for local_index in range(count):
            records.append(
                (
                    int(normalized[local_index].astype("int64")),
                    source_order_time,
                    str(path),
                    local_index,
                    normalized[local_index],
                    raw_times[local_index],
                    float(offsets[local_index]),
                    frames[local_index],
                )
            )
        cursor += count
    records.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    deduplicated: list[tuple[int, int, str, int, np.datetime64, np.datetime64, float, np.ndarray]] = []
    seen: set[int] = set()
    for record in records:
        if record[0] in seen:
            continue
        seen.add(record[0])
        deduplicated.append(record)
    duplicate_count = len(records) - len(deduplicated)

    start_time = parse_time(start)
    end_time = parse_time(end_exclusive)
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("start must be earlier than end_exclusive.")
    selected = [
        record
        for record in deduplicated
        if (start_time is None or record[4] >= start_time)
        and (end_time is None or record[4] < end_time)
    ]
    if not selected:
        raise ValueError("No scalar records fall within the requested time window.")

    times = np.asarray([record[4] for record in selected], dtype="datetime64[ns]")
    if len(times) > 1 and np.any(np.diff(times.astype("int64")) <= 0):
        raise ValueError("Internal error: deduplicated timestamps are not strictly increasing.")
    resolution = dict(common_resolution or {})
    resolution.update(
        {
            "normalized_cadence_seconds": cadence,
            "snap_tolerance_seconds": float(snap_tolerance_seconds),
            "surface_sigma_index": None
            if reference.sigma is None
            else resolve_layer_index(reference.sigma, "surface"),
            "bottom_sigma_index": None
            if reference.sigma is None
            else resolve_layer_index(reference.sigma, "bottom"),
        }
    )
    return ScalarSeries(
        grid=reference,
        times=times,
        original_times=np.asarray([record[5] for record in selected], dtype="datetime64[ns]"),
        time_offsets_seconds=np.asarray([record[6] for record in selected], dtype=float),
        values=np.stack([record[7] for record in selected], axis=0),
        record_sources=tuple(record[2] for record in selected),
        record_indices=tuple(record[3] for record in selected),
        sources=tuple(sources),
        duplicate_times_removed=duplicate_count,
        resolution=resolution,
    )


def inspect_inputs(inputs: Sequence[str | Path]) -> dict[str, Any]:
    """Return JSON-ready metadata and geometry checks for EFDC inputs."""

    paths = [Path(path).expanduser().resolve() for path in inputs]
    if not paths:
        raise ValueError("At least one input file is required.")
    reference: EFDCGrid | None = None
    files: list[dict[str, Any]] = []
    all_times: list[np.ndarray] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Dataset(path, "r") as ds:
            grid = read_grid(ds)
            if reference is None:
                reference = grid
            else:
                _assert_same_grid(reference, grid, path)
            raw_times, time_name = decode_times(ds)
            normalized, offsets, cadence = normalize_times(raw_times, tolerance_seconds=60.0)
            all_times.append(normalized)
            variables = {
                name: {
                    "dimensions": list(var.dimensions),
                    "shape": list(var.shape),
                    "dtype": str(var.dtype),
                    "standard_name": getattr(var, "standard_name", None),
                    "units": getattr(var, "units", None),
                }
                for name, var in ds.variables.items()
            }
            files.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "data_model": ds.data_model,
                    "dimensions": {name: len(dim) for name, dim in ds.dimensions.items()},
                    "variables": variables,
                    "time_variable": time_name,
                    "record_count": len(raw_times),
                    "first_time_utc": None
                    if not len(normalized)
                    else np.datetime_as_string(normalized[0], unit="ns") + "Z",
                    "last_time_utc": None
                    if not len(normalized)
                    else np.datetime_as_string(normalized[-1], unit="ns") + "Z",
                    "cadence_seconds": cadence,
                    "maximum_absolute_snap_seconds": float(np.max(np.abs(offsets))) if len(offsets) else 0.0,
                    "geometry_sha256": grid.geometry_sha256,
                    "source_grid": getattr(ds, "source_grid", None),
                    "vector_components": getattr(ds, "vector_components", None),
                }
            )
    assert reference is not None
    combined = np.sort(np.concatenate(all_times)) if all_times else np.empty(0, dtype="datetime64[ns]")
    unique = np.unique(combined)
    return {
        "status": "pass",
        "file_count": len(files),
        "files": files,
        "geometry": {
            "shape": list(reference.shape),
            "geometry_sha256": reference.geometry_sha256,
            "wet_cell_count": int(np.count_nonzero(reference.mask == 1)),
            "sigma": None if reference.sigma is None else reference.sigma.tolist(),
            "surface_sigma_index": None
            if reference.sigma is None
            else resolve_layer_index(reference.sigma, "surface"),
            "bottom_sigma_index": None
            if reference.sigma is None
            else resolve_layer_index(reference.sigma, "bottom"),
            "sigma_weights": None
            if reference.sigma is None or len(reference.sigma) < 2
            else sigma_weights(reference.sigma).tolist(),
        },
        "combined_time": {
            "record_count": len(combined),
            "unique_record_count": len(unique),
            "duplicate_record_count": len(combined) - len(unique),
            "unique_monotonic_after_sort": bool(len(unique) <= 1 or np.all(np.diff(unique.astype("int64")) > 0)),
        },
    }
