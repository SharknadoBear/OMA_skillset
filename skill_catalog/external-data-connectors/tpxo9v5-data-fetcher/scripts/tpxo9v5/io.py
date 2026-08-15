"""TPXO9v5 source discovery, inventory, and bounded reads."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4 as nc4
import numpy as np

from .coordinates import select_latitudes, select_longitudes, spatial_span

COORDINATE_ALIASES = {
    "z": (("lon_z", "lonz", "longitude_z"), ("lat_z", "latz", "latitude_z")),
    "u": (("lon_u", "lonu", "longitude_u"), ("lat_u", "latu", "latitude_u")),
    "v": (("lon_v", "lonv", "longitude_v"), ("lat_v", "latv", "latitude_v")),
}
COMPLEX_ALIASES = {
    "elevation": (("hRe", "hre", "h_real"), ("hIm", "him", "h_imag")),
    "u": (("uRe", "ure", "u_real"), ("uIm", "uim", "u_imag")),
    "v": (("vRe", "vre", "v_real"), ("vIm", "vim", "v_imag")),
}
AMPLITUDE_PHASE_ALIASES = {
    "elevation": (("ha", "h_amp"), ("hp", "h_phase")),
    "u": (("ua", "u_amp"), ("up", "u_phase")),
    "v": (("va", "v_amp"), ("vp", "v_phase")),
}
DEPTH_ALIASES = {"z": ("hz", "depth_z"), "u": ("hu", "depth_u"), "v": ("hv", "depth_v")}
MASK_ALIASES = {"z": ("mz", "mask_z"), "u": ("mu", "mask_u"), "v": ("mv", "mask_v")}


@dataclass
class HarmonicField:
    """One bounded harmonic field on one native TPXO staggered grid."""

    name: str
    grid: str
    constituents: list[str]
    longitude: np.ndarray
    latitude: np.ndarray
    coefficient: np.ndarray
    units: str
    depth: np.ndarray | None
    mask: np.ndarray | None
    source_basename: str
    source_variables: tuple[str, str]
    source_span: dict[str, float]
    actual_span: dict[str, float]


def _first_present(ds: nc4.Dataset, names: Iterable[str]) -> str | None:
    lookup = {name.lower(): name for name in ds.variables}
    for candidate in names:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _pair_present(ds: nc4.Dataset, aliases: tuple[tuple[str, ...], tuple[str, ...]]) -> tuple[str, str] | None:
    first = _first_present(ds, aliases[0])
    second = _first_present(ds, aliases[1])
    return (first, second) if first and second else None


def decode_constituents(ds: nc4.Dataset) -> tuple[list[str], str]:
    """Decode a TPXO constituent character variable."""

    name = _first_present(ds, ("con", "constituent", "constituents"))
    if not name:
        raise ValueError("No TPXO constituent variable (con) was found.")
    raw = ds[name][:]
    names: list[str] = []
    if getattr(raw, "ndim", 0) == 1:
        for value in raw:
            if isinstance(value, bytes):
                text = value.decode("ascii", errors="ignore")
            else:
                text = str(value)
            names.append(text.replace("\x00", "").strip().upper())
    else:
        for row in raw:
            try:
                text = row.tobytes().decode("ascii", errors="ignore")
            except AttributeError:
                text = "".join(str(value) for value in row)
            names.append(text.replace("\x00", "").strip().upper())
    return names, ds[name].dimensions[0]


def _finite_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.ptp(finite)) if finite.size else 0.0


def _coordinate_axis(var: nc4.Variable, kind: str) -> tuple[np.ndarray, str]:
    """Extract a one-dimensional axis from a 1-D or rectilinear 2-D coordinate."""

    if var.ndim == 1:
        return np.asarray(var[:], dtype=float), var.dimensions[0]
    if var.ndim != 2:
        raise ValueError(f"Coordinate {var.name} must be 1-D or 2-D, got {var.shape}.")
    first = np.asarray(var[:, 0], dtype=float)
    second = np.asarray(var[0, :], dtype=float)
    first_range = _finite_range(first)
    second_range = _finite_range(second)
    if first_range >= second_range:
        values, dimension = first, var.dimensions[0]
    else:
        values, dimension = second, var.dimensions[1]
    if kind == "longitude" and _finite_range(values) < 1e-9:
        raise ValueError(f"Could not identify a varying longitude axis in {var.name}.")
    if kind == "latitude" and _finite_range(values) < 1e-9:
        raise ValueError(f"Could not identify a varying latitude axis in {var.name}.")
    return values, dimension


def read_grid_axes(ds: nc4.Dataset, grid: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Read longitude/latitude vectors and their NetCDF dimension names."""

    if grid not in COORDINATE_ALIASES:
        raise ValueError(f"Unsupported grid {grid!r}.")
    pair = _pair_present(ds, COORDINATE_ALIASES[grid])
    if not pair:
        raise ValueError(f"No coordinate variables for TPXO {grid}-grid were found.")
    lon, lon_dim = _coordinate_axis(ds[pair[0]], "longitude")
    lat, lat_dim = _coordinate_axis(ds[pair[1]], "latitude")
    if lon_dim == lat_dim:
        raise ValueError(f"Longitude and latitude unexpectedly share dimension {lon_dim!r}.")
    return lon, lat, lon_dim, lat_dim


def classify_file(path: str | Path) -> list[str]:
    """Classify a NetCDF file by variables rather than private file identifiers."""

    roles: list[str] = []
    with nc4.Dataset(path) as ds:
        if _pair_present(ds, COMPLEX_ALIASES["elevation"]) or _pair_present(ds, AMPLITUDE_PHASE_ALIASES["elevation"]):
            roles.append("elevation")
        if (
            _pair_present(ds, COMPLEX_ALIASES["u"])
            or _pair_present(ds, AMPLITUDE_PHASE_ALIASES["u"])
            or _pair_present(ds, COMPLEX_ALIASES["v"])
            or _pair_present(ds, AMPLITUDE_PHASE_ALIASES["v"])
        ):
            roles.append("transport")
        if any(_first_present(ds, DEPTH_ALIASES[grid]) for grid in ("z", "u", "v")):
            roles.append("grid")
    return roles


def discover_source_files(source_dir: str | Path, fields: Iterable[str] = ("elevation",)) -> dict[str, Path]:
    """Discover the grid/elevation/transport files needed for requested fields."""

    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {root}")
    discovered: dict[str, Path] = {}
    for path in sorted(root.glob("*.nc")):
        for role in classify_file(path):
            if role in discovered and discovered[role] != path:
                raise ValueError(f"Multiple candidate {role} files found: {discovered[role].name}, {path.name}")
            discovered[role] = path
    required = {"grid", *[value.strip().lower() for value in fields if value.strip()]}
    missing = sorted(required - discovered.keys())
    if missing:
        raise FileNotFoundError(f"Missing required TPXO source role(s): {', '.join(missing)}")
    return {role: discovered[role] for role in sorted(required)}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def inspect_source(path: str | Path) -> dict[str, Any]:
    """Inspect metadata and spatial spans without loading harmonic arrays."""

    source = Path(path).resolve()
    with nc4.Dataset(source) as ds:
        roles = classify_open_dataset(ds)
        try:
            constituents, _ = decode_constituents(ds)
        except ValueError:
            constituents = []
        grids: dict[str, Any] = {}
        for grid in ("z", "u", "v"):
            try:
                lon, lat, lon_dim, lat_dim = read_grid_axes(ds, grid)
            except ValueError:
                continue
            grids[grid] = {
                "longitude_dimension": lon_dim,
                "latitude_dimension": lat_dim,
                "longitude_count": int(lon.size),
                "latitude_count": int(lat.size),
                "span": spatial_span(lon, lat),
            }
        variables = {}
        for name, variable in ds.variables.items():
            variables[name] = {
                "dimensions": list(variable.dimensions),
                "shape": [int(value) for value in variable.shape],
                "dtype": str(variable.dtype),
                "units": _json_value(getattr(variable, "units", None)),
                "long_name": _json_value(getattr(variable, "long_name", None)),
            }
        return {
            "basename": source.name,
            "size_bytes": source.stat().st_size,
            "format": ds.data_model,
            "roles": roles,
            "constituents": constituents,
            "grids": grids,
            "variables": variables,
        }


def classify_open_dataset(ds: nc4.Dataset) -> list[str]:
    """Classify an already-open dataset."""

    roles: list[str] = []
    if _pair_present(ds, COMPLEX_ALIASES["elevation"]) or _pair_present(ds, AMPLITUDE_PHASE_ALIASES["elevation"]):
        roles.append("elevation")
    if any(
        _pair_present(ds, aliases)
        for aliases in (
            COMPLEX_ALIASES["u"],
            AMPLITUDE_PHASE_ALIASES["u"],
            COMPLEX_ALIASES["v"],
            AMPLITUDE_PHASE_ALIASES["v"],
        )
    ):
        roles.append("transport")
    if any(_first_present(ds, DEPTH_ALIASES[grid]) for grid in ("z", "u", "v")):
        roles.append("grid")
    return roles


def inventory_sources(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Return an inventory for explicit source paths."""

    files = [inspect_source(path) for path in paths]
    return {
        "schema_version": "tpxo9v5_source_inventory_v1",
        "files": files,
        "total_bytes": int(sum(item["size_bytes"] for item in files)),
        "roles": sorted({role for item in files for role in item["roles"]}),
    }


def _masked_float(values: Any) -> np.ndarray:
    array = np.ma.asarray(values, dtype=float)
    return np.asarray(np.ma.filled(array, np.nan), dtype=float)


def _read_subset(
    variable: nc4.Variable,
    constituent_index: int | None,
    constituent_dim: str | None,
    lon_dim: str,
    lat_dim: str,
    lon_indices: np.ndarray,
    lat_indices: np.ndarray,
) -> np.ndarray:
    index: list[Any] = []
    remaining: list[str] = []
    for dimension in variable.dimensions:
        if constituent_dim and dimension == constituent_dim:
            if constituent_index is None:
                raise ValueError(f"A constituent index is required for {variable.name}.")
            index.append(int(constituent_index))
        elif dimension == lon_dim:
            index.append(lon_indices)
            remaining.append(dimension)
        elif dimension == lat_dim:
            index.append(lat_indices)
            remaining.append(dimension)
        else:
            raise ValueError(f"Unsupported extra dimension {dimension!r} in {variable.name}.")
    values = _masked_float(variable[tuple(index)])
    if remaining == [lat_dim, lon_dim]:
        return values
    if remaining == [lon_dim, lat_dim]:
        return values.T
    raise ValueError(f"Could not orient {variable.name}; remaining dimensions are {remaining}.")


def _select_constituents(all_names: list[str], requested: Iterable[str] | None) -> tuple[list[str], list[int]]:
    if requested is None:
        return all_names, list(range(len(all_names)))
    wanted = [name.strip().upper() for name in requested if name.strip()]
    if not wanted:
        return all_names, list(range(len(all_names)))
    lookup = {name: index for index, name in enumerate(all_names)}
    missing = [name for name in wanted if name not in lookup]
    if missing:
        raise ValueError(f"Requested constituent(s) not present: {', '.join(missing)}")
    return wanted, [lookup[name] for name in wanted]


def _harmonic_pair(ds: nc4.Dataset, field: str) -> tuple[str, str, str]:
    pair = _pair_present(ds, COMPLEX_ALIASES[field])
    if pair:
        return pair[0], pair[1], "complex"
    pair = _pair_present(ds, AMPLITUDE_PHASE_ALIASES[field])
    if pair:
        return pair[0], pair[1], "amplitude_phase"
    raise ValueError(f"No supported harmonic variables for {field!r} were found.")


def _read_static_grid(
    grid_path: str | Path,
    grid: str,
    lon_indices: np.ndarray,
    lat_indices: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    with nc4.Dataset(grid_path) as ds:
        _, _, lon_dim, lat_dim = read_grid_axes(ds, grid)
        depth_name = _first_present(ds, DEPTH_ALIASES[grid])
        mask_name = _first_present(ds, MASK_ALIASES[grid])
        depth = (
            _read_subset(ds[depth_name], None, None, lon_dim, lat_dim, lon_indices, lat_indices)
            if depth_name
            else None
        )
        mask = None
        if mask_name:
            mask_variable = ds[mask_name]
            # TPXO9v5a declares mask value 1 as both water (option_1) and
            # _FillValue. Preserve the categorical values instead of allowing
            # netCDF4 to auto-mask every wet cell.
            mask_variable.set_auto_mask(False)
            mask = _read_subset(mask_variable, None, None, lon_dim, lat_dim, lon_indices, lat_indices)
    return depth, mask


def read_harmonic_field(
    harmonic_path: str | Path,
    grid_path: str | Path,
    field: str,
    grid: str,
    bbox: tuple[float, float, float, float],
    constituents: Iterable[str] | None = None,
    padding: float = 0.5,
) -> HarmonicField:
    """Read one bounded elevation or vector component as native complex coefficients."""

    west, south, east, north = bbox
    source = Path(harmonic_path).resolve()
    with nc4.Dataset(source) as ds:
        lon, lat, lon_dim, lat_dim = read_grid_axes(ds, grid)
        lon_selection = select_longitudes(lon, west, east, padding)
        lat_selection = select_latitudes(lat, south, north, padding)
        all_names, constituent_dim = decode_constituents(ds)
        names, indices = _select_constituents(all_names, constituents)
        first_name, second_name, encoding = _harmonic_pair(ds, field)
        first = ds[first_name]
        second = ds[second_name]
        units = str(getattr(first, "units", "")).strip()
        coefficient = np.empty(
            (len(indices), lat_selection.indices.size, lon_selection.indices.size),
            dtype=np.complex128,
        )
        for output_index, source_index in enumerate(indices):
            a = _read_subset(
                first,
                source_index,
                constituent_dim,
                lon_dim,
                lat_dim,
                lon_selection.indices,
                lat_selection.indices,
            )
            b = _read_subset(
                second,
                source_index,
                constituent_dim,
                lon_dim,
                lat_dim,
                lon_selection.indices,
                lat_selection.indices,
            )
            if encoding == "complex":
                coefficient[output_index] = a + 1j * b
            else:
                coefficient[output_index] = a * np.exp(-1j * np.deg2rad(b))

    depth, mask = _read_static_grid(
        grid_path,
        grid,
        lon_selection.indices,
        lat_selection.indices,
    )
    wet = np.ones(coefficient.shape[1:], dtype=bool)
    if mask is not None:
        wet &= np.isfinite(mask) & (mask > 0)
    if depth is not None:
        wet &= np.isfinite(depth) & (depth > 0)
    coefficient[:, ~wet] = np.nan + 1j * np.nan
    return HarmonicField(
        name=field,
        grid=grid,
        constituents=names,
        longitude=lon_selection.values,
        latitude=lat_selection.values,
        coefficient=coefficient,
        units=units,
        depth=depth,
        mask=mask,
        source_basename=source.name,
        source_variables=(first_name, second_name),
        source_span=spatial_span(lon, lat),
        actual_span=spatial_span(lon_selection.values, lat_selection.values),
    )


def transport_to_velocity(field: HarmonicField) -> HarmonicField:
    """Convert depth-integrated transport coefficients to velocity when units are known."""

    if field.name not in {"u", "v"}:
        raise ValueError("Transport-to-velocity conversion applies only to u or v fields.")
    if field.depth is None:
        raise ValueError(f"No positive {field.grid}-grid depth is available for velocity conversion.")
    normalized = field.units.lower().replace(" ", "").replace("**", "^")
    if normalized in {
        "m2/s",
        "m^2/s",
        "meter2/s",
        "meter^2/s",
        "meters2/s",
        "meters^2/s",
        "meter2/second",
        "meters2/second",
        "m2s-1",
    }:
        scale = 1.0
    elif normalized in {
        "cm2/s",
        "cm^2/s",
        "centimeter2/s",
        "centimeter^2/s",
        "centimeters2/s",
        "centimeters^2/s",
        "centimeter2/second",
        "centimeters2/second",
        "cm2s-1",
    }:
        scale = 1.0e-4
    else:
        raise ValueError(f"Cannot convert unrecognized transport units {field.units!r} to velocity.")
    depth = np.asarray(field.depth, dtype=float)
    valid = np.isfinite(depth) & (depth > 0)
    coefficient = np.full_like(field.coefficient, np.nan + 1j * np.nan)
    coefficient[:, valid] = field.coefficient[:, valid] * scale / depth[valid]
    return HarmonicField(
        name=f"velocity_{field.name}",
        grid=field.grid,
        constituents=field.constituents,
        longitude=field.longitude,
        latitude=field.latitude,
        coefficient=coefficient,
        units="m s-1",
        depth=field.depth,
        mask=field.mask,
        source_basename=field.source_basename,
        source_variables=field.source_variables,
        source_span=field.source_span,
        actual_span=field.actual_span,
    )
