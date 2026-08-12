"""Load, validate, concatenate, and derive fields from staggered ROMS NetCDF output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from netCDF4 import Dataset, chartostring, num2date


TIME_NAMES = ("ocean_time", "time", "Times")
VERTICAL_DIM_NAMES = {"s_rho", "sigma", "siglay", "z"}
EARTH_EAST_NAMES = ("eastward_velocity", "eastward_sea_water_velocity")
EARTH_NORTH_NAMES = ("northward_velocity", "northward_sea_water_velocity")
COMPACT_SCHEMA_VERSION = "roms_compact_fields_v1"
ANGLE_STANDARD_NAME = "grid_angle_of_rotation_from_east_to_y"
ANGLE_UNITS = "radians"
ANGLE_CONVENTION = "xi_axis_counterclockwise_from_east"
CURRENT_PROVENANCE_ATTRIBUTES = (
    "schema_version",
    "source_model",
    "model",
    "derived_vector_reference",
    "vector_provenance",
    "vector_reference",
    "velocity_processing",
)


@dataclass(frozen=True)
class ROMSGrid:
    """Canonical rho-grid geometry and ROMS vertical metadata."""

    lon: np.ndarray
    lat: np.ndarray
    mask: np.ndarray
    h: np.ndarray
    angle: np.ndarray
    angle_units: str
    angle_convention: str
    s_rho: np.ndarray | None
    s_w: np.ndarray | None
    cs_r: np.ndarray | None
    cs_w: np.ndarray | None
    hc: float | None
    vtransform: int | None
    vstretching: int | None
    geometry_sha256: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.lon.shape


@dataclass(frozen=True)
class ScalarSeries:
    """One scalar field concatenated on the ROMS rho grid."""

    grid: ROMSGrid
    times: np.ndarray
    original_times: np.ndarray
    time_offsets_seconds: np.ndarray
    values: np.ndarray
    record_sources: tuple[str, ...]
    record_indices: tuple[int, ...]
    sources: tuple[dict[str, Any], ...]
    duplicate_times_removed: int
    resolution: dict[str, Any]


@dataclass(frozen=True)
class VectorSeries:
    """Earth-relative current components concatenated on the rho grid."""

    grid: ROMSGrid
    times: np.ndarray
    original_times: np.ndarray
    time_offsets_seconds: np.ndarray
    east: np.ndarray
    north: np.ndarray
    speed: np.ndarray
    record_sources: tuple[str, ...]
    record_indices: tuple[int, ...]
    sources: tuple[dict[str, Any], ...]
    duplicate_times_removed: int
    resolution: dict[str, Any]


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _find_name(ds: Dataset, candidates: Sequence[str]) -> str | None:
    lookup = {name.lower(): name for name in ds.variables}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _as_float(var, *, ignore_valid_range: bool = False) -> np.ndarray:
    if ignore_valid_range:
        # NOAA ROMS files can advertise valid_max=0 while storing the
        # mathematically equivalent surface stretching coordinate as a tiny
        # positive round-off value.  Disable netCDF4's valid-range auto-mask
        # for coordinate reads only; explicit fill/missing sentinels are still
        # converted to NaN below.
        var.set_auto_mask(False)
        try:
            masked = np.ma.asarray(var[:])
        finally:
            var.set_auto_mask(True)
    else:
        masked = np.ma.asarray(var[:])
    data = np.asarray(np.ma.filled(masked, np.nan), dtype=np.float64)
    for attribute in ("_FillValue", "missing_value"):
        if not hasattr(var, attribute):
            continue
        for value in np.atleast_1d(getattr(var, attribute)).astype(float):
            if np.isfinite(value):
                data[data == value] = np.nan
    return data


def _dimension_kind(name: str) -> str | None:
    key = str(name).lower()
    if key == "eta" or key.startswith("eta_") or key in {"y", "y_rho", "nj"}:
        return "eta"
    if key == "xi" or key.startswith("xi_") or key in {"x", "x_rho", "ni"}:
        return "xi"
    return None


def _horizontal_axes(dimensions: Sequence[str], variable: str) -> tuple[int, int]:
    eta = [index for index, name in enumerate(dimensions) if _dimension_kind(name) == "eta"]
    xi = [index for index, name in enumerate(dimensions) if _dimension_kind(name) == "xi"]
    if len(eta) != 1 or len(xi) != 1 or eta[0] == xi[0]:
        raise ValueError(
            f"ROMS variable {variable!r} must have one named eta and one named xi axis; "
            f"got dimensions {tuple(dimensions)!r}."
        )
    return eta[0], xi[0]


def _read_2d(ds: Dataset, names: Sequence[str], label: str) -> np.ndarray:
    name = _find_name(ds, names)
    if name is None:
        raise KeyError(f"ROMS input has no recognized {label} variable ({', '.join(names)}).")
    variable = ds.variables[name]
    array = _as_float(variable)
    if array.ndim != 2:
        raise ValueError(f"ROMS {label} variable {name!r} must be two-dimensional; got {array.shape}.")
    eta_axis, xi_axis = _horizontal_axes(variable.dimensions, name)
    return np.moveaxis(array, (eta_axis, xi_axis), (0, 1))


def _binary_mask(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"ROMS {label} must contain only finite binary 0/1 values.")
    invalid = (array != 0.0) & (array != 1.0)
    if np.any(invalid):
        examples = np.unique(array[invalid])[:5].tolist()
        raise ValueError(f"ROMS {label} must be binary 0/1; found {examples!r}.")
    return array.astype(np.float64, copy=False)


def _normalized_metadata_text(value: Any) -> str:
    return " ".join(str(_plain_attribute(value)).strip().lower().replace("_", " ").replace("-", " ").split())


def _validate_angle_metadata(variable, angle: np.ndarray, wet: np.ndarray) -> tuple[str, str]:
    units = _normalized_metadata_text(getattr(variable, "units", ""))
    if units not in {"rad", "radian", "radians"}:
        raise ValueError(
            f"ROMS angle variable {variable.name!r} must explicitly use radian units; got {units!r}."
        )
    standard_name = str(_plain_attribute(getattr(variable, "standard_name", ""))).strip().lower()
    long_name = _normalized_metadata_text(getattr(variable, "long_name", ""))
    semantic_match = standard_name == ANGLE_STANDARD_NAME or (
        "angle" in long_name and "xi" in long_name and "east" in long_name
    )
    if not semantic_match:
        raise ValueError(
            f"ROMS angle variable {variable.name!r} has ambiguous semantics; require standard_name "
            f"{ANGLE_STANDARD_NAME!r} or a long_name identifying the XI axis and east."
        )
    wet_angle = np.asarray(angle, dtype=float)[wet]
    if not np.isfinite(wet_angle).all():
        raise ValueError("ROMS angle contains non-finite values on wet rho cells.")
    if np.any(np.abs(wet_angle) > 2.0 * np.pi + 1.0e-6):
        raise ValueError("ROMS angle values exceed the accepted radian range [-2*pi, 2*pi].")
    return ANGLE_UNITS, ANGLE_CONVENTION


def _read_1d_optional(ds: Dataset, names: Sequence[str]) -> np.ndarray | None:
    name = _find_name(ds, names)
    if name is None:
        return None
    array = np.squeeze(_as_float(ds.variables[name], ignore_valid_range=True))
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"ROMS vertical coordinate {name!r} must be a finite one-dimensional vector.")
    if array.size > 1:
        difference = np.diff(array)
        if not (np.all(difference > 0) or np.all(difference < 0)):
            raise ValueError(f"ROMS vertical coordinate {name!r} must be strictly monotonic.")
    return array


def _scalar_attribute(ds: Dataset, variable_names: Sequence[str], attribute_names: Sequence[str]) -> float | None:
    for variable_name in variable_names:
        actual = _find_name(ds, (variable_name,))
        if actual is not None:
            values = np.squeeze(_as_float(ds.variables[actual]))
            if values.size == 1 and np.isfinite(values).all():
                return float(values.reshape(-1)[0])
    for name in attribute_names:
        if hasattr(ds, name):
            value = np.asarray(getattr(ds, name)).reshape(-1)
            if value.size and np.isfinite(float(value[0])):
                return float(value[0])
    return None


def _geometry_hash(grid_values: Sequence[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in grid_values:
        digest.update(name.encode("ascii"))
        if value is None:
            digest.update(b"none")
        elif np.isscalar(value):
            digest.update(repr(value).encode("ascii"))
        else:
            array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def read_grid(ds: Dataset) -> ROMSGrid:
    """Read rho-grid geometry and vertical metadata from raw or compact ROMS files."""

    lon = _read_2d(ds, ("lon_rho", "longitude_rho", "lon", "longitude"), "rho longitude")
    lat = _read_2d(ds, ("lat_rho", "latitude_rho", "lat", "latitude"), "rho latitude")
    if lon.shape != lat.shape:
        raise ValueError(f"ROMS rho longitude/latitude shapes differ: {lon.shape} and {lat.shape}.")
    mask = _read_2d(ds, ("mask_rho",), "rho mask")
    h = _read_2d(ds, ("h", "depth", "bathymetry"), "bathymetry")
    angle_name = _find_name(ds, ("angle", "angle_rho"))
    if angle_name is None:
        raise KeyError("ROMS input has no recognized rho-grid angle variable (angle, angle_rho).")
    angle_variable = ds.variables[angle_name]
    angle = _read_2d(ds, (angle_name,), "rho-grid angle")
    for label, value in (("mask", mask), ("bathymetry", h), ("angle", angle)):
        if value.shape != lon.shape:
            raise ValueError(f"ROMS {label} shape {value.shape} differs from rho grid {lon.shape}.")
    mask = _binary_mask(mask, "mask_rho")
    if not np.any(mask == 1):
        raise ValueError("ROMS grid contains no cells with mask_rho == 1.")
    if not np.any((mask == 1) & np.isfinite(lon) & np.isfinite(lat)):
        raise ValueError("ROMS grid contains no finite lon_rho/lat_rho coordinates on wet cells.")
    angle_units, angle_convention = _validate_angle_metadata(angle_variable, angle, mask == 1)

    s_rho = _read_1d_optional(ds, ("s_rho", "sigma"))
    s_w = _read_1d_optional(ds, ("s_w",))
    cs_r = _read_1d_optional(ds, ("Cs_r", "cs_r"))
    cs_w = _read_1d_optional(ds, ("Cs_w", "cs_w"))
    if (s_rho is None) != (cs_r is None):
        raise ValueError("ROMS s_rho and Cs_r must either both be present or both be absent.")
    if (s_w is None) != (cs_w is None):
        raise ValueError("ROMS s_w and Cs_w must either both be present or both be absent.")
    if s_rho is not None and len(s_rho) != len(cs_r):
        raise ValueError("ROMS s_rho and Cs_r lengths differ.")
    if s_w is not None and len(s_w) != len(cs_w):
        raise ValueError("ROMS s_w and Cs_w lengths differ.")
    if s_rho is not None and s_w is not None and len(s_w) != len(s_rho) + 1:
        raise ValueError("ROMS s_w must contain exactly one more level than s_rho.")

    hc = _scalar_attribute(ds, ("hc",), ("hc",))
    vtransform_raw = _scalar_attribute(ds, ("Vtransform", "vtransform"), ("Vtransform", "vtransform"))
    vstretching_raw = _scalar_attribute(ds, ("Vstretching", "vstretching"), ("Vstretching", "vstretching"))
    vtransform = None if vtransform_raw is None else int(round(vtransform_raw))
    vstretching = None if vstretching_raw is None else int(round(vstretching_raw))
    if s_rho is not None:
        if hc is None or vtransform not in {1, 2}:
            raise ValueError("ROMS sigma metadata requires finite hc and Vtransform equal to 1 or 2.")

    values = (
        ("lon", lon), ("lat", lat), ("mask", mask), ("h", h), ("angle", angle),
        ("angle_units", angle_units), ("angle_convention", angle_convention),
        ("s_rho", s_rho), ("s_w", s_w), ("cs_r", cs_r), ("cs_w", cs_w),
        ("hc", hc), ("vtransform", vtransform), ("vstretching", vstretching),
    )
    return ROMSGrid(
        lon=lon,
        lat=lat,
        mask=mask,
        h=h,
        angle=angle,
        angle_units=angle_units,
        angle_convention=angle_convention,
        s_rho=s_rho,
        s_w=s_w,
        cs_r=cs_r,
        cs_w=cs_w,
        hc=hc,
        vtransform=vtransform,
        vstretching=vstretching,
        geometry_sha256=_geometry_hash(values),
    )


def _assert_same_grid(reference: ROMSGrid, candidate: ROMSGrid, path: Path) -> None:
    if reference.geometry_sha256 != candidate.geometry_sha256:
        raise ValueError(f"ROMS geometry or vertical-schema drift in {path}.")


def _datetime64(value: Any) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(value, "ns")
    return np.datetime64(datetime(
        int(value.year), int(value.month), int(value.day), int(getattr(value, "hour", 0)),
        int(getattr(value, "minute", 0)), int(getattr(value, "second", 0)),
        int(getattr(value, "microsecond", 0)),
    ), "ns")


def decode_times(ds: Dataset) -> tuple[np.ndarray, str]:
    name = _find_name(ds, TIME_NAMES)
    if name is None:
        raise KeyError("ROMS input has no recognized ocean_time, time, or Times variable.")
    var = ds.variables[name]
    if np.dtype(var.dtype).kind in {"S", "U"}:
        raw = var[:]
        strings = chartostring(raw) if np.asarray(raw).ndim > 1 else raw
        result = []
        for item in np.atleast_1d(strings):
            text = (item.decode("ascii") if isinstance(item, bytes) else str(item)).strip()
            result.append(np.datetime64(text.replace("_", "T").replace(" ", "T", 1).rstrip("Z"), "ns"))
        return np.asarray(result, dtype="datetime64[ns]"), name
    values = np.asarray(np.ma.filled(np.ma.asarray(var[:]), np.nan), dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"ROMS time variable {name!r} contains missing values.")
    units = str(getattr(var, "units", "")).strip()
    if "since" in units.lower():
        decoded = num2date(values, units=units, calendar=str(getattr(var, "calendar", "standard")),
                          only_use_cftime_datetimes=False, only_use_python_datetimes=False)
        return np.asarray([_datetime64(value) for value in decoded], dtype="datetime64[ns]"), name
    magnitude = float(np.nanmedian(np.abs(values))) if values.size else 0.0
    if units.lower() in {"nanoseconds since epoch", "unix_nanoseconds", "nanoseconds"} or magnitude > 1.0e15:
        return np.rint(values).astype("int64").astype("datetime64[ns]"), name
    if units.lower() in {"seconds since epoch", "unix_seconds", "seconds"} or (not units and magnitude > 1.0e8):
        return np.rint(values * 1.0e9).astype("int64").astype("datetime64[ns]"), name
    raise ValueError(f"Cannot decode ROMS time variable {name!r} with units {units!r}.")


def normalize_times(raw_times: np.ndarray, tolerance_seconds: float = 60.0) -> tuple[np.ndarray, np.ndarray, int | None]:
    times = np.asarray(raw_times).astype("datetime64[ns]")
    if not len(times):
        return times.copy(), np.empty(0), None
    unique = np.unique(times.astype("int64"))
    cadence = None
    if len(unique) > 1:
        median = float(np.median(np.diff(unique))) / 1.0e9
        for candidate in (3600, 360):
            if abs(median - candidate) <= max(120.0, 2 * tolerance_seconds):
                cadence = candidate
                break
    if cadence is None:
        return times.copy(), np.zeros(len(times)), None
    raw_ns = times.astype("int64")
    step = int(cadence * 1.0e9)
    snapped = np.rint(raw_ns.astype(float) / step).astype("int64") * step
    offsets = (snapped - raw_ns) / 1.0e9
    normalized = np.where(np.abs(offsets) <= tolerance_seconds, snapped, raw_ns)
    return normalized.astype("datetime64[ns]"), (normalized - raw_ns) / 1.0e9, cadence


def parse_time(value: str | datetime | np.datetime64 | None) -> np.datetime64 | None:
    if value is None:
        return None
    if isinstance(value, (datetime, np.datetime64)):
        return _datetime64(value)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(parsed, "ns")


def layer_suffix(layer: str) -> str:
    key = str(layer).lower().replace("depth_mean", "depth_average").replace("depth-averaged", "depth_average")
    if key.startswith("index:"):
        return f"sigma_{int(key.split(':', 1)[1])}"
    if key not in {"surface", "near_surface", "bottom", "depth_average"}:
        raise ValueError(f"Unknown ROMS layer selector {layer!r}.")
    return key


def resolve_layer_index(s_rho: np.ndarray | None, layer: str) -> int:
    key = str(layer).lower()
    if key.startswith("index:"):
        index = int(key.split(":", 1)[1])
        if s_rho is not None and not 0 <= index < len(s_rho):
            raise IndexError(f"Sigma index {index} is outside 0..{len(s_rho) - 1}.")
        return index
    if s_rho is None:
        raise ValueError(f"Layer {layer!r} requires s_rho metadata.")
    order = np.argsort(np.abs(s_rho))
    if key == "surface":
        return int(order[0])
    if key == "near_surface":
        if len(order) < 2:
            raise ValueError("near_surface requires at least two s_rho levels.")
        return int(order[1])
    if key == "bottom":
        return int(order[-1])
    raise ValueError(f"Layer {layer!r} does not select a single sigma level.")


def roms_depths(h: np.ndarray, zeta: np.ndarray, s: np.ndarray, cs: np.ndarray, hc: float, vtransform: int) -> np.ndarray:
    """Compute ROMS depths for Vtransform 1 or 2 with time-varying zeta."""

    h2 = np.asarray(h, dtype=float)
    zeta3 = np.asarray(zeta, dtype=float)
    if zeta3.ndim == 2:
        zeta3 = zeta3[None, ...]
    if h2.ndim != 2 or zeta3.ndim != 3 or zeta3.shape[1:] != h2.shape:
        raise ValueError("h must be (eta,xi) and zeta must be (time,eta,xi).")
    levels = np.asarray(s, dtype=float).reshape(-1)
    curves = np.asarray(cs, dtype=float).reshape(-1)
    if levels.shape != curves.shape:
        raise ValueError("s and Cs vectors must have identical lengths.")
    h4 = h2[None, None, :, :]
    z4 = zeta3[:, None, :, :]
    s4 = levels[None, :, None, None]
    c4 = curves[None, :, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        if int(vtransform) == 1:
            z0 = (s4 - c4) * float(hc) + c4 * h4
            result = z0 + z4 * (1.0 + z0 / h4)
        elif int(vtransform) == 2:
            z0 = (float(hc) * s4 + h4 * c4) / (float(hc) + h4)
            result = z4 + (z4 + h4) * z0
        else:
            raise ValueError(f"Unsupported ROMS Vtransform {vtransform}; expected 1 or 2.")
    return np.where(np.isfinite(h4) & (h4 > 0), result, np.nan)


def _time_axis(ds: Dataset, var, time_name: str) -> int:
    recognized = {time_name.lower(), "time", "ocean_time"}
    time_variable = ds.variables.get(time_name)
    if time_variable is not None and time_variable.dimensions:
        recognized.add(time_variable.dimensions[0].lower())
    candidates = [index for index, dimension in enumerate(var.dimensions)
                  if dimension.lower() in recognized]
    if len(candidates) != 1:
        raise ValueError(
            f"ROMS field {var.name!r} must have exactly one named time axis; "
            f"got dimensions {tuple(var.dimensions)!r}."
        )
    return candidates[0]


def _vertical_axis(dimensions: Sequence[str]) -> int | None:
    candidates = [index for index, dimension in enumerate(dimensions)
                  if dimension.lower() in VERTICAL_DIM_NAMES or dimension.lower().startswith("s_rho")]
    if len(candidates) > 1:
        raise ValueError(f"ROMS field has multiple recognized vertical axes: {tuple(dimensions)!r}.")
    return candidates[0] if candidates else None


def _canonical_time_field(ds: Dataset, name: str, time_name: str, time_count: int) -> tuple[np.ndarray, bool]:
    """Move a ROMS field to (time,[vertical],eta,xi) from any named dimension order."""

    var = ds.variables[name]
    data = _as_float(var)
    dimensions = list(var.dimensions)
    time_axis = _time_axis(ds, var, time_name)
    vertical_axis = _vertical_axis(dimensions)
    eta_axis, xi_axis = _horizontal_axes(dimensions, name)
    canonical_axes = [time_axis]
    if vertical_axis is not None:
        canonical_axes.append(vertical_axis)
    canonical_axes.extend((eta_axis, xi_axis))
    if len(set(canonical_axes)) != len(canonical_axes) or set(canonical_axes) != set(range(data.ndim)):
        unused = [dimensions[index] for index in range(data.ndim) if index not in canonical_axes]
        raise ValueError(
            f"ROMS field {name!r} has unsupported or ambiguous axes {unused!r}; "
            "expected only named time, optional vertical, eta, and xi axes."
        )
    data = np.moveaxis(data, canonical_axes, range(len(canonical_axes)))
    if data.shape[0] != time_count:
        raise ValueError(f"ROMS field {name!r} time axis has {data.shape[0]} records; expected {time_count}.")
    return data, vertical_axis is not None


def _zeta(ds: Dataset, grid: ROMSGrid, time_name: str, time_count: int) -> np.ndarray:
    name = _find_name(ds, ("zeta", "water_surface_elevation"))
    if name is None:
        raise KeyError("Depth averaging requires the time-varying ROMS zeta field.")
    data, has_vertical = _canonical_time_field(ds, name, time_name, time_count)
    if has_vertical:
        raise ValueError(f"ROMS zeta field {name!r} must not have a vertical axis.")
    if data.shape != (time_count, *grid.shape):
        raise ValueError(f"ROMS zeta shape {data.shape} differs from {(time_count, *grid.shape)}.")
    return np.where(grid.mask[None, ...] == 1, data, np.nan)


def _native_mask(ds: Dataset, name: str, grid: ROMSGrid) -> np.ndarray:
    component = name.lower()
    candidates = ("mask_u",) if component == "u" else ("mask_v",)
    actual = _find_name(ds, candidates)
    if actual is not None:
        mask = _read_2d(ds, (actual,), f"{component} mask")
    elif component == "u":
        mask = grid.mask[:, :-1] * grid.mask[:, 1:]
    else:
        mask = grid.mask[:-1, :] * grid.mask[1:, :]
    expected = (grid.shape[0], grid.shape[1] - 1) if component == "u" else (grid.shape[0] - 1, grid.shape[1])
    if mask.shape != expected:
        raise ValueError(f"ROMS {component} mask shape {mask.shape} differs from {expected}.")
    return _binary_mask(mask, f"mask_{component}")


def _finite_pair_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    count = np.isfinite(left).astype(int) + np.isfinite(right).astype(int)
    total = np.where(np.isfinite(left), left, 0.0) + np.where(np.isfinite(right), right, 0.0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def destagger_u_to_rho(u: np.ndarray, rho_shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(u, dtype=float)
    if values.shape[-2:] != (rho_shape[0], rho_shape[1] - 1):
        raise ValueError(f"ROMS u shape {values.shape[-2:]} is not staggered from rho shape {rho_shape}.")
    output = np.full((*values.shape[:-2], *rho_shape), np.nan)
    output[..., :, 0] = values[..., :, 0]
    output[..., :, -1] = values[..., :, -1]
    if rho_shape[1] > 2:
        output[..., :, 1:-1] = _finite_pair_mean(values[..., :, :-1], values[..., :, 1:])
    return output


def destagger_v_to_rho(v: np.ndarray, rho_shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(v, dtype=float)
    if values.shape[-2:] != (rho_shape[0] - 1, rho_shape[1]):
        raise ValueError(f"ROMS v shape {values.shape[-2:]} is not staggered from rho shape {rho_shape}.")
    output = np.full((*values.shape[:-2], *rho_shape), np.nan)
    output[..., 0, :] = values[..., 0, :]
    output[..., -1, :] = values[..., -1, :]
    if rho_shape[0] > 2:
        output[..., 1:-1, :] = _finite_pair_mean(values[..., :-1, :], values[..., 1:, :])
    return output


def rotate_to_earth(u_rho: np.ndarray, v_rho: np.ndarray, angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(np.asarray(angle, dtype=float))
    sine = np.sin(np.asarray(angle, dtype=float))
    return u_rho * cosine - v_rho * sine, u_rho * sine + v_rho * cosine


def _weighted_average(values: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    numerator = np.sum(np.where(finite, values * weights, 0.0), axis=axis)
    denominator = np.sum(np.where(finite, weights, 0.0), axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _rho_thickness(ds: Dataset, grid: ROMSGrid, time_name: str, time_count: int) -> np.ndarray:
    if grid.s_w is None or grid.cs_w is None or grid.hc is None or grid.vtransform is None:
        raise ValueError("Depth averaging requires s_w, Cs_w, hc, and Vtransform metadata.")
    zeta = _zeta(ds, grid, time_name, time_count)
    z_w = roms_depths(grid.h, zeta, grid.s_w, grid.cs_w, grid.hc, grid.vtransform)
    thickness = np.abs(np.diff(z_w, axis=1))
    expected = grid.h[None, :, :] + zeta
    closure = np.nansum(thickness, axis=1)
    wet = grid.mask[None, :, :] == 1
    tolerance = np.maximum(1.0e-5, np.abs(expected) * 1.0e-6)
    if np.any(wet & np.isfinite(expected) & (np.abs(closure - expected) > tolerance)):
        raise ValueError("ROMS W-level thicknesses do not close to h + zeta.")
    return thickness


def _native_component(ds: Dataset, name: str, layer: str, grid: ROMSGrid,
                      time_name: str, time_count: int) -> np.ndarray:
    component = name.lower()
    data, has_vertical = _canonical_time_field(ds, name, time_name, time_count)
    expected = (grid.shape[0], grid.shape[1] - 1) if component == "u" else (grid.shape[0] - 1, grid.shape[1])
    if not has_vertical:
        raise ValueError(f"Raw ROMS {name!r} has no s_rho axis.")
    if data.shape[2:] != expected:
        raise ValueError(f"ROMS {name!r} horizontal shape {data.shape[2:]} differs from {expected}.")
    key = layer_suffix(layer)
    if key == "depth_average":
        if grid.s_rho is None or data.shape[1] != len(grid.s_rho):
            raise ValueError(f"ROMS {name!r} vertical axis does not match s_rho.")
        rho_thickness = _rho_thickness(ds, grid, time_name, time_count)
        if component == "u":
            weights = 0.5 * (rho_thickness[:, :, :, :-1] + rho_thickness[:, :, :, 1:])
        else:
            weights = 0.5 * (rho_thickness[:, :, :-1, :] + rho_thickness[:, :, 1:, :])
        if weights.shape != data.shape:
            raise ValueError(f"ROMS {name!r} field shape {data.shape} differs from weight shape {weights.shape}.")
        data = _weighted_average(data, weights, 1)
    else:
        index = resolve_layer_index(grid.s_rho, layer)
        data = np.take(data, index, axis=1)
    if data.shape != (time_count, *expected):
        raise ValueError(f"ROMS {name!r} field shape {data.shape} differs from {(time_count, *expected)}.")
    native_mask = _native_mask(ds, name, grid)
    return np.where(native_mask[None, ...] == 1, data, np.nan)


def _read_raw_current(ds: Dataset, layer: str, grid: ROMSGrid,
                      time_name: str, time_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    u_name = _find_name(ds, ("u",))
    v_name = _find_name(ds, ("v",))
    if u_name is None or v_name is None:
        raise KeyError("Earth-relative current derivation requires paired raw ROMS u and v fields.")
    u = _native_component(ds, u_name, layer, grid, time_name, time_count)
    v = _native_component(ds, v_name, layer, grid, time_name, time_count)
    u_rho = destagger_u_to_rho(u, grid.shape)
    v_rho = destagger_v_to_rho(v, grid.shape)
    east, north = rotate_to_earth(u_rho, v_rho, grid.angle)
    wet = grid.mask[None, ...] == 1
    return np.where(wet, east, np.nan), np.where(wet, north, np.nan), {
        "source_variables": [u_name, v_name],
        "component_grid": "native_staggered_u_v",
        "destagger_method": "finite_adjacent_mean_with_one_sided_boundaries",
        "rotation_method": "roms_angle_radians_xi_to_east",
        "angle_units": grid.angle_units,
        "angle_convention": grid.angle_convention,
        "earth_relative_on_rho_grid": True,
        "vtransform": grid.vtransform,
        "vertical_weight_method": "abs_diff_z_w" if layer_suffix(layer) == "depth_average" else None,
    }


def _read_rho_scalar(ds: Dataset, name: str, layer: str, grid: ROMSGrid,
                     time_name: str, time_count: int) -> np.ndarray:
    data, has_vertical = _canonical_time_field(ds, name, time_name, time_count)
    if has_vertical:
        if data.shape[2:] != grid.shape:
            raise ValueError(f"ROMS scalar {name!r} spatial shape {data.shape[2:]} differs from rho grid {grid.shape}.")
        if layer_suffix(layer) == "depth_average":
            weights = _rho_thickness(ds, grid, time_name, time_count)
            if data.shape != weights.shape:
                raise ValueError(f"ROMS scalar {name!r} vertical axis does not match W-level thicknesses.")
            data = _weighted_average(data, weights, 1)
        else:
            data = np.take(data, resolve_layer_index(grid.s_rho, layer), axis=1)
    if data.shape != (time_count, *grid.shape):
        raise ValueError(f"ROMS scalar {name!r} shape {data.shape} differs from {(time_count, *grid.shape)}.")
    return np.where(grid.mask[None, ...] == 1, data, np.nan)


def _direct_pair(ds: Dataset, suffix: str) -> tuple[str, str] | None:
    lookup = {name.lower(): name for name in ds.variables}
    candidates = (
        (f"eastward_velocity_{suffix}", f"northward_velocity_{suffix}"),
        (f"eastward_sea_water_velocity_{suffix}", f"northward_sea_water_velocity_{suffix}"),
        ("eastward_velocity", "northward_velocity"),
        ("eastward_sea_water_velocity", "northward_sea_water_velocity"),
    )
    for east, north in candidates:
        if east in lookup and north in lookup:
            return lookup[east], lookup[north]
    return None


def _plain_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    array = np.asarray(value)
    if array.size != 1:
        return [_plain_attribute(item) for item in array.reshape(-1)]
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="replace")
    return scalar.item() if isinstance(scalar, np.generic) else scalar


def _dataset_schema_version(ds: Dataset) -> str | None:
    if not hasattr(ds, "schema_version"):
        return None
    return str(_plain_attribute(getattr(ds, "schema_version"))).strip()


def _require_compact_schema(ds: Dataset, purpose: str) -> str:
    schema = _dataset_schema_version(ds)
    if schema != COMPACT_SCHEMA_VERSION:
        raise ValueError(
            f"{purpose} requires schema_version={COMPACT_SCHEMA_VERSION!r}; got {schema!r}. "
            "Raw or unclassified eastward/northward names are ambiguous."
        )
    return schema


def _recognized_earth_vector_provenance(ds: Dataset) -> dict[str, Any]:
    schema = _require_compact_schema(ds, "Precomputed earth-relative ROMS vectors")
    accepted_name = None
    accepted_value = None
    for name in ("derived_vector_reference", "vector_reference", "vector_provenance"):
        if not hasattr(ds, name):
            continue
        value = str(_plain_attribute(getattr(ds, name))).strip()
        normalized = "_".join(_normalized_metadata_text(value).split())
        if normalized == "earth_relative_on_rho_grid" or (
            all(token in normalized.split("_") for token in ("earth", "relative", "rho", "grid"))
            and "derived" in normalized.split("_")
        ):
            accepted_name, accepted_value = name, value
            break
    if accepted_name is None:
        raise ValueError(
            "Compact ROMS current fields require explicit recognized earth-relative vector provenance "
            "on the rho grid."
        )
    processing = str(_plain_attribute(getattr(ds, "velocity_processing", ""))).strip()
    if not processing:
        raise ValueError("Compact ROMS current fields require non-empty velocity_processing provenance.")
    return {
        "input_schema_version": schema,
        "vector_provenance_attribute": accepted_name,
        "vector_provenance_value": accepted_value,
        "velocity_processing": processing,
    }


def _earth_direct_names(ds: Dataset) -> list[str]:
    prefixes = (*EARTH_EAST_NAMES, *EARTH_NORTH_NAMES, "current_speed", "sea_water_speed")
    return sorted(name for name in ds.variables if any(name.lower() == prefix or name.lower().startswith(prefix + "_")
                                                        for prefix in prefixes))


def _direct_speed_name(ds: Dataset, suffix: str) -> str | None:
    lookup = {name.lower(): name for name in ds.variables}
    for candidate in (f"current_speed_{suffix}", f"sea_water_speed_{suffix}", "current_speed", "sea_water_speed"):
        if candidate in lookup:
            return lookup[candidate]
    return None


def _validate_speed_consistency(speed: np.ndarray, east: np.ndarray, north: np.ndarray, name: str) -> float:
    expected = np.hypot(east, north)
    finite_speed = np.isfinite(speed)
    finite_expected = np.isfinite(expected)
    if not np.array_equal(finite_speed, finite_expected):
        raise ValueError(f"Compact ROMS speed {name!r} finite mask differs from its paired east/north components.")
    if not np.any(finite_expected):
        raise ValueError(f"Compact ROMS speed {name!r} and its paired components contain no finite wet values.")
    difference = np.abs(speed[finite_expected] - expected[finite_expected])
    tolerance = 5.0e-6 + 1.0e-5 * np.abs(expected[finite_expected])
    if np.any(difference > tolerance):
        raise ValueError(
            f"Compact ROMS speed {name!r} is inconsistent with hypot(east,north); "
            f"maximum absolute error is {float(np.max(difference)):.9g}."
        )
    return float(np.max(difference))


def _compact_current_provenance(ds: Dataset) -> dict[str, Any]:
    return {name: _plain_attribute(getattr(ds, name)) for name in CURRENT_PROVENANCE_ATTRIBUTES
            if hasattr(ds, name)}


def _with_current_provenance(ds: Dataset, metadata: dict[str, Any]) -> dict[str, Any]:
    provenance = _compact_current_provenance(ds)
    if not provenance:
        return metadata
    return {**metadata, **provenance, "compact_current_provenance": provenance}


def _read_current(ds: Dataset, layer: str, grid: ROMSGrid,
                   time_name: str, time_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    suffix = layer_suffix(layer)
    pair = _direct_pair(ds, suffix)
    if pair is not None:
        recognized = _recognized_earth_vector_provenance(ds)
        east = _read_rho_scalar(ds, pair[0], layer, grid, time_name, time_count)
        north = _read_rho_scalar(ds, pair[1], layer, grid, time_name, time_count)
        speed_name = _direct_speed_name(ds, suffix)
        speed_error = None
        if speed_name is not None:
            speed = _read_rho_scalar(ds, speed_name, layer, grid, time_name, time_count)
            speed_error = _validate_speed_consistency(speed, east, north, speed_name)
        metadata = {
            "source_variables": list(pair), "component_grid": "rho",
            "destagger_method": "precomputed", "rotation_method": "precomputed_earth_relative",
            "angle_units": grid.angle_units, "angle_convention": grid.angle_convention,
            "earth_relative_on_rho_grid": True, "vtransform": grid.vtransform,
            "vertical_weight_method": "precomputed" if layer_suffix(layer) == "depth_average" else None,
            "verified_speed_variable": speed_name,
            "speed_consistency_max_abs_error": speed_error,
            **recognized,
        }
        return east, north, _with_current_provenance(ds, metadata)
    ambiguous = _earth_direct_names(ds)
    if ambiguous:
        raise ValueError(
            "Raw or unclassified ROMS input contains ambiguous or unpaired earthward current names: "
            f"{ambiguous!r}. Provide paired compact fields with strict provenance or native staggered u/v."
        )
    return _read_raw_current(ds, layer, grid, time_name, time_count)


def _resolve_scalar(ds: Dataset, requested: str, layer: str, grid: ROMSGrid,
                    time_name: str, time_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    key = requested.strip().lower()
    suffix = layer_suffix(layer)
    lookup = {name.lower(): name for name in ds.variables}
    if key in {"u", "v"}:
        raise ValueError("Raw staggered u/v cannot be rendered as rho-grid scalars; request eastward_velocity, northward_velocity, or current_speed.")
    if key in {"current_speed", "eastward_velocity", "northward_velocity",
               "eastward_sea_water_velocity", "northward_sea_water_velocity"}:
        east, north, metadata = _read_current(ds, layer, grid, time_name, time_count)
        if key == "current_speed":
            speed_name = _direct_speed_name(ds, suffix)
            if speed_name is not None:
                values = _read_rho_scalar(ds, speed_name, layer, grid, time_name, time_count)
                _validate_speed_consistency(values, east, north, speed_name)
                mode = "direct_compact_verified_against_components"
            else:
                values = np.hypot(east, north)
                mode = "derived_speed_after_destagger_and_rotation"
        elif key.startswith("eastward"):
            values, mode = east, "derived_east_after_destagger_and_rotation"
        else:
            values, mode = north, "derived_north_after_destagger_and_rotation"
        return values, {"requested_variable": requested, "resolved_mode": mode, "layer": layer, **metadata}

    aliases = [key]
    if key == "salinity":
        aliases = ["salinity", "salt"]
    elif key == "temperature":
        aliases = ["temperature", "temp"]
    names = [f"{alias}_{suffix}" for alias in aliases] + aliases
    for name in names:
        if name in lookup:
            actual = lookup[name]
            compact_schema = _require_compact_schema(ds, f"Precomputed ROMS field {actual!r}") if name not in aliases else None
            values = _read_rho_scalar(ds, actual, layer, grid, time_name, time_count)
            return values, {"requested_variable": requested, "resolved_mode": "direct",
                            "source_variables": [actual], "layer": layer,
                            "input_schema_version": compact_schema,
                            "earth_relative_on_rho_grid": None, "vtransform": grid.vtransform,
                            "vertical_weight_method": "abs_diff_z_w" if suffix == "depth_average" and name in aliases else "precomputed" if suffix == "depth_average" else None}
    raise KeyError(f"ROMS variable {requested!r} was not found. Available examples: {list(ds.variables)[:30]}")


def _load_records(inputs: Sequence[str | Path], reader, *, layer: str,
                  start=None, end_exclusive=None, snap_tolerance_seconds: float = 60.0):
    paths = [Path(value).expanduser().resolve() for value in inputs]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Supply one or more unique ROMS input paths.")
    reference = None
    payloads = []
    sources = []
    common_resolution = None
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Dataset(path) as ds:
            grid = read_grid(ds)
            if reference is None:
                reference = grid
            else:
                _assert_same_grid(reference, grid, path)
            raw_times, time_name = decode_times(ds)
            data, resolution = reader(ds, layer, grid, time_name, len(raw_times))
            input_schema = _dataset_schema_version(ds)
            input_provenance = _compact_current_provenance(ds)
            input_kind = "compact" if input_schema == COMPACT_SCHEMA_VERSION else (
                "raw" if input_schema is None else "unclassified"
            )
            resolution = {
                **resolution,
                "input_kind": input_kind,
                "input_schema_version": input_schema,
                "angle_units": grid.angle_units,
                "angle_convention": grid.angle_convention,
            }
            count = data[0].shape[0] if isinstance(data, tuple) else data.shape[0]
            if count != len(raw_times):
                raise ValueError(f"ROMS time/frame count mismatch in {path}: {len(raw_times)} versus {count}.")
            if common_resolution is None:
                common_resolution = resolution
            elif common_resolution != resolution:
                raise ValueError(f"ROMS variable-resolution drift in {path}.")
        order_time = int(np.min(raw_times.astype("int64"))) if len(raw_times) else np.iinfo("int64").max
        payloads.append((path, raw_times, data, order_time))
        sources.append({
            "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            "raw_record_count": len(raw_times), "geometry_sha256": grid.geometry_sha256,
            "input_kind": input_kind, "input_schema_version": input_schema,
            "current_provenance": input_provenance,
            "raw_first_time_utc": None if not len(raw_times) else np.datetime_as_string(raw_times[0], unit="ns") + "Z",
            "raw_last_time_utc": None if not len(raw_times) else np.datetime_as_string(raw_times[-1], unit="ns") + "Z",
        })
    assert reference is not None
    all_raw = np.concatenate([item[1] for item in payloads])
    normalized, offsets, cadence = normalize_times(all_raw, snap_tolerance_seconds)
    records = []
    cursor = 0
    for path, raw_times, data, order_time in payloads:
        for index in range(len(raw_times)):
            value = tuple(component[index] for component in data) if isinstance(data, tuple) else data[index]
            records.append((int(normalized[cursor + index].astype("int64")), order_time, str(path), index,
                            normalized[cursor + index], raw_times[index], float(offsets[cursor + index]), value))
        cursor += len(raw_times)
    records.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    unique = []
    seen = set()
    for record in records:
        if record[0] not in seen:
            unique.append(record)
            seen.add(record[0])
    start_time, end_time = parse_time(start), parse_time(end_exclusive)
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("start must be earlier than end_exclusive.")
    selected = [record for record in unique if (start_time is None or record[4] >= start_time)
                and (end_time is None or record[4] < end_time)]
    if not selected:
        raise ValueError("No ROMS records fall inside the requested time window.")
    resolution = dict(common_resolution or {})
    resolution.update({
        "normalized_cadence_seconds": cadence,
        "snap_tolerance_seconds": float(snap_tolerance_seconds),
        "surface_sigma_index": None if reference.s_rho is None else resolve_layer_index(reference.s_rho, "surface"),
        "bottom_sigma_index": None if reference.s_rho is None else resolve_layer_index(reference.s_rho, "bottom"),
        "vertical_transform": reference.vtransform,
        "angle_units": reference.angle_units,
        "angle_convention": reference.angle_convention,
    })
    return reference, selected, tuple(sources), len(records) - len(unique), resolution


def load_scalar_series(inputs: Sequence[str | Path], *, variable: str, layer: str = "surface",
                       start=None, end_exclusive=None, snap_tolerance_seconds: float = 60.0) -> ScalarSeries:
    """Load raw/compact ROMS inputs, derive a rho-grid scalar, sort, deduplicate, and crop."""

    def reader(ds, selected_layer, grid, time_name, time_count):
        values, metadata = _resolve_scalar(ds, variable, selected_layer, grid, time_name, time_count)
        return values, metadata

    grid, records, sources, duplicates, resolution = _load_records(
        inputs, reader, layer=layer, start=start, end_exclusive=end_exclusive,
        snap_tolerance_seconds=snap_tolerance_seconds)
    return ScalarSeries(
        grid=grid,
        times=np.asarray([record[4] for record in records], dtype="datetime64[ns]"),
        original_times=np.asarray([record[5] for record in records], dtype="datetime64[ns]"),
        time_offsets_seconds=np.asarray([record[6] for record in records], dtype=float),
        values=np.stack([record[7] for record in records]),
        record_sources=tuple(record[2] for record in records),
        record_indices=tuple(record[3] for record in records), sources=sources,
        duplicate_times_removed=duplicates, resolution=resolution,
    )


def load_current_series(inputs: Sequence[str | Path], *, layer: str = "surface",
                        start=None, end_exclusive=None, snap_tolerance_seconds: float = 60.0) -> VectorSeries:
    """Load earth-relative current components on rho points from raw or compact ROMS inputs."""

    def reader(ds, selected_layer, grid, time_name, time_count):
        east, north, metadata = _read_current(ds, selected_layer, grid, time_name, time_count)
        return (east, north), {"requested_variable": "current", "layer": selected_layer, **metadata}

    grid, records, sources, duplicates, resolution = _load_records(
        inputs, reader, layer=layer, start=start, end_exclusive=end_exclusive,
        snap_tolerance_seconds=snap_tolerance_seconds)
    east = np.stack([record[7][0] for record in records])
    north = np.stack([record[7][1] for record in records])
    return VectorSeries(
        grid=grid, times=np.asarray([record[4] for record in records], dtype="datetime64[ns]"),
        original_times=np.asarray([record[5] for record in records], dtype="datetime64[ns]"),
        time_offsets_seconds=np.asarray([record[6] for record in records], dtype=float),
        east=east, north=north, speed=np.hypot(east, north),
        record_sources=tuple(record[2] for record in records), record_indices=tuple(record[3] for record in records),
        sources=sources, duplicate_times_removed=duplicates, resolution=resolution,
    )


def inspect_inputs(inputs: Sequence[str | Path]) -> dict[str, Any]:
    """Inspect raw/compact ROMS files without deriving requested fields."""

    paths = [Path(value).expanduser().resolve() for value in inputs]
    if not paths:
        raise ValueError("At least one input is required.")
    reference = None
    raw_times = []
    sources = []
    variable_sets = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Dataset(path) as ds:
            grid = read_grid(ds)
            if reference is None:
                reference = grid
            else:
                _assert_same_grid(reference, grid, path)
            times, time_name = decode_times(ds)
            input_schema = _dataset_schema_version(ds)
            raw_times.append(times)
            variable_sets.append(set(ds.variables))
            sources.append({
                "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
                "time_variable": time_name, "record_count": len(times),
                "first_time_utc": None if not len(times) else np.datetime_as_string(times[0], unit="ns") + "Z",
                "last_time_utc": None if not len(times) else np.datetime_as_string(times[-1], unit="ns") + "Z",
                "dimensions": {name: int(len(dimension)) for name, dimension in ds.dimensions.items()},
                "variables": sorted(ds.variables), "geometry_sha256": grid.geometry_sha256,
                "input_kind": "compact" if input_schema == COMPACT_SCHEMA_VERSION else (
                    "raw" if input_schema is None else "unclassified"
                ),
                "input_schema_version": input_schema,
                "current_provenance": _compact_current_provenance(ds),
            })
    assert reference is not None
    combined = np.concatenate(raw_times) if raw_times else np.empty(0, dtype="datetime64[ns]")
    normalized, offsets, cadence = normalize_times(combined)
    unique = np.unique(normalized)
    warnings = []
    if len(unique) and len(unique) != len(normalized):
        warnings.append(f"{len(normalized) - len(unique)} duplicate normalized timestamp(s) will be removed.")
    wet = reference.mask == 1
    return {
        "schema_version": "roms_inspection_v1", "status": "pass_with_warnings" if warnings else "pass",
        "warnings": warnings, "sources": sources,
        "geometry": {
            "shape": list(reference.shape), "geometry_sha256": reference.geometry_sha256,
            "wet_cell_count": int(np.count_nonzero(wet)),
            "longitude_range": [float(np.nanmin(reference.lon[wet])), float(np.nanmax(reference.lon[wet]))],
            "latitude_range": [float(np.nanmin(reference.lat[wet])), float(np.nanmax(reference.lat[wet]))],
            "surface_sigma_index": None if reference.s_rho is None else resolve_layer_index(reference.s_rho, "surface"),
            "bottom_sigma_index": None if reference.s_rho is None else resolve_layer_index(reference.s_rho, "bottom"),
            "s_rho_count": None if reference.s_rho is None else len(reference.s_rho),
            "Vtransform": reference.vtransform, "Vstretching": reference.vstretching, "hc": reference.hc,
            "angle_units": reference.angle_units,
            "angle_convention": reference.angle_convention,
        },
        "combined_time": {
            "raw_record_count": len(combined), "unique_record_count": len(unique),
            "duplicate_count": len(combined) - len(unique), "normalized_cadence_seconds": cadence,
            "maximum_absolute_adjustment_seconds": float(np.max(np.abs(offsets))) if len(offsets) else 0.0,
            "first_time_utc": None if not len(unique) else np.datetime_as_string(unique[0], unit="ns") + "Z",
            "last_time_utc": None if not len(unique) else np.datetime_as_string(unique[-1], unit="ns") + "Z",
        },
        "variables_common_to_all_inputs": sorted(set.intersection(*variable_sets)) if variable_sets else [],
    }
