#!/usr/bin/env python3
"""Build and validate FVCOM elevation forcing from model-neutral TPXO9v5 harmonics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4 as nc4
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from fvcom_writer import write_elevation_obc
from grid_utils import datetime64_to_mjd, read_obc_nodes_dat


SCHEMA = "fvcom_tpxo_tide_forcing_report_v1"
PRODUCT_SCHEMA = "tpxo9v5_harmonics_v1"
DEFAULT_EXPECTED_CONSTITUENTS = 22
RECONSTRUCTION_RULE_VERSION = "utide_exact_time_nodal_v1"
# UTide FUV flags: NodsatLint, NodsatNone, GwchLint, GwchNone.
# NodsatLint=True freezes f,u at tref; exact-time reconstruction needs all False.
UTIDE_NGFLAGS = (False, False, False, False)


class GateError(RuntimeError):
    """A reproducible scientific or structural gate failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BoundaryPoints:
    node_ids: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray


@dataclass(frozen=True)
class Harmonics:
    names: list[str]
    coefficient_m: np.ndarray
    flags: np.ndarray
    source_mode: str
    source_units: str


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, destination)
    return destination


def load_boundary_points(path: str | Path) -> BoundaryPoints:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        lookup = {name.lower(): name for name in (reader.fieldnames or [])}
        required = {"node_id", "longitude", "latitude"}
        if not required.issubset(lookup):
            raise GateError("obc_points_schema", "OBC CSV must contain node_id, longitude, and latitude.")
        rows = list(reader)
    if not rows:
        raise GateError("obc_points_empty", "OBC point CSV contains no rows.")
    try:
        node_ids = np.asarray([int(row[lookup["node_id"]]) for row in rows], dtype=np.int32)
        longitude = np.asarray([float(row[lookup["longitude"]]) for row in rows], dtype=float)
        latitude = np.asarray([float(row[lookup["latitude"]]) for row in rows], dtype=float)
    except (TypeError, ValueError) as exc:
        raise GateError("obc_points_value", f"Invalid OBC CSV value: {exc}") from exc
    if np.any(node_ids <= 0) or len(set(node_ids.tolist())) != node_ids.size:
        raise GateError("obc_node_ids", "OBC node IDs must be unique positive FVCOM 1-based IDs.")
    if not np.all(np.isfinite(longitude) & np.isfinite(latitude)):
        raise GateError("obc_coordinates", "OBC coordinates must all be finite.")
    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise GateError("obc_latitude", "OBC latitude is outside [-90, 90].")
    return BoundaryPoints(node_ids=node_ids, longitude=longitude, latitude=latitude)


def boundary_order_sha256(points: BoundaryPoints) -> str:
    canonical = "".join(
        f"{int(node)},{lon:.12f},{lat:.12f}\n"
        for node, lon, lat in zip(points.node_ids, points.longitude, points.latitude, strict=True)
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _decode_strings(values: Any) -> list[str]:
    array = np.asarray(values)
    output: list[str] = []
    if array.ndim == 1:
        for value in array:
            if isinstance(value, bytes):
                text = value.decode("ascii", errors="ignore")
            else:
                text = str(value)
            output.append(text.replace("\x00", "").strip())
    else:
        for row in array:
            try:
                text = row.tobytes().decode("ascii", errors="ignore")
            except AttributeError:
                text = "".join(str(value) for value in row)
            output.append(text.replace("\x00", "").strip())
    return output


def _as_float(variable: nc4.Variable) -> np.ndarray:
    return np.asarray(np.ma.filled(np.ma.asarray(variable[:]), np.nan), dtype=float)


def _coefficient_scale_to_metres(units: str) -> float:
    normalized = units.strip().lower().replace("metres", "meter").replace("metre", "meter")
    if normalized in {"m", "meter", "meters"}:
        return 1.0
    if normalized in {"cm", "centimeter", "centimeters"}:
        return 0.01
    if normalized in {"mm", "millimeter", "millimeters"}:
        return 0.001
    raise GateError("elevation_units", f"Unrecognized TPXO elevation coefficient units: {units!r}.")


def _orient(variable: nc4.Variable, dimensions: tuple[str, ...]) -> np.ndarray:
    if set(variable.dimensions) != set(dimensions) or len(variable.dimensions) != len(dimensions):
        raise GateError(
            "harmonic_dimensions",
            f"{variable.name} dimensions {variable.dimensions} do not match required {dimensions}.",
        )
    values = _as_float(variable)
    axes = [variable.dimensions.index(name) for name in dimensions]
    return np.transpose(values, axes)


def _branch_longitudes(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    target_360 = np.mod(np.asarray(target, dtype=float), 360.0)
    source_mid = float(np.nanmedian(source))
    shifts = np.round((source_mid - target_360) / 360.0)
    return target_360 + 360.0 * shifts


def _interpolate_native(
    coefficient: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    target_longitude: np.ndarray,
    target_latitude: np.ndarray,
    nearest_wet_max_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    if nearest_wet_max_degrees < 0:
        raise GateError("nearest_wet_range", "Nearest-wet fallback distance must be non-negative.")
    lon_order = np.argsort(longitude, kind="stable")
    lat_order = np.argsort(latitude, kind="stable")
    lon = np.asarray(longitude, dtype=float)[lon_order]
    lat = np.asarray(latitude, dtype=float)[lat_order]
    values = coefficient[:, lat_order, :][:, :, lon_order]
    if lon.size < 2 or lat.size < 2 or np.any(np.diff(lon) <= 0) or np.any(np.diff(lat) <= 0):
        raise GateError("native_grid", "Native TPXO subset axes must contain at least two unique ascending values.")
    target_lon = _branch_longitudes(target_longitude, lon)
    query = np.column_stack((target_latitude, target_lon))
    output = np.full((values.shape[0], target_lon.size), np.nan + 1j * np.nan, dtype=np.complex128)
    flags = np.full(output.shape, 2, dtype=np.int8)
    source_lon, source_lat = np.meshgrid(lon, lat)
    for index, field in enumerate(values):
        real = RegularGridInterpolator(
            (lat, lon), field.real, method="linear", bounds_error=False, fill_value=np.nan
        )(query)
        imag = RegularGridInterpolator(
            (lat, lon), field.imag, method="linear", bounds_error=False, fill_value=np.nan
        )(query)
        valid = np.isfinite(real) & np.isfinite(imag)
        output[index, valid] = real[valid] + 1j * imag[valid]
        flags[index, valid] = 0
        unresolved = ~valid
        wet = np.isfinite(field.real) & np.isfinite(field.imag)
        if unresolved.any() and wet.any() and nearest_wet_max_degrees > 0:
            latitude_scale = max(0.05, float(np.cos(np.deg2rad(np.mean(target_latitude)))))
            tree = cKDTree(np.column_stack((source_lat[wet], source_lon[wet] * latitude_scale)))
            distance, nearest = tree.query(
                np.column_stack((target_latitude[unresolved], target_lon[unresolved] * latitude_scale)), k=1
            )
            bounded = distance <= nearest_wet_max_degrees
            unresolved_indices = np.where(unresolved)[0]
            chosen = unresolved_indices[bounded]
            output[index, chosen] = field[wet][nearest[bounded]]
            flags[index, chosen] = 1
    return output, flags


def load_harmonics(
    path: str | Path,
    points: BoundaryPoints,
    nearest_wet_max_degrees: float,
) -> Harmonics:
    source = Path(path).expanduser().resolve()
    with nc4.Dataset(source) as ds:
        if getattr(ds, "product_schema", "") != PRODUCT_SCHEMA:
            raise GateError("product_schema", f"Expected {PRODUCT_SCHEMA}, found {getattr(ds, 'product_schema', '')!r}.")
        if "constituent_name" not in ds.variables:
            raise GateError("constituents_missing", "Harmonic product has no constituent_name variable.")
        names = [name.upper() for name in _decode_strings(ds["constituent_name"][:])]
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise GateError("constituents_invalid", "Constituent names must be non-empty and unique.")
        for variable in ("elevation_real", "elevation_imaginary"):
            if variable not in ds.variables:
                raise GateError("elevation_coefficients", f"Harmonic product is missing {variable}.")
        real_var = ds["elevation_real"]
        imag_var = ds["elevation_imaginary"]
        units = str(getattr(real_var, "units", ""))
        if str(getattr(imag_var, "units", "")) != units:
            raise GateError("elevation_units", "Real and imaginary elevation coefficients have different units.")
        scale = _coefficient_scale_to_metres(units)
        if "point" in real_var.dimensions:
            real = _orient(real_var, ("constituent", "point"))
            imag = _orient(imag_var, ("constituent", "point"))
            if real.shape != (len(names), points.node_ids.size):
                raise GateError("point_count", "Harmonic point count does not match ordered OBC point count.")
            if "target_id" not in ds.variables:
                raise GateError("target_id_missing", "Point product lacks target_id; exact OBC node-order proof is required.")
            ids = _decode_strings(ds["target_id"][:])
            if ids != [str(int(value)) for value in points.node_ids]:
                raise GateError("node_order_mismatch", "Harmonic target_id order differs from the OBC node order.")
            if "longitude" not in ds.variables or "latitude" not in ds.variables:
                raise GateError("point_coordinates", "Point product lacks longitude/latitude coordinates.")
            if not np.allclose(_as_float(ds["longitude"]), points.longitude, atol=1e-10, rtol=0.0):
                raise GateError("longitude_mismatch", "Harmonic point longitudes differ from OBC point longitudes.")
            if not np.allclose(_as_float(ds["latitude"]), points.latitude, atol=1e-10, rtol=0.0):
                raise GateError("latitude_mismatch", "Harmonic point latitudes differ from OBC point latitudes.")
            coefficient = (real + 1j * imag) * scale
            if "elevation_interpolation_flag" in ds.variables:
                flags = np.asarray(ds["elevation_interpolation_flag"][:], dtype=np.int8)
                if ds["elevation_interpolation_flag"].dimensions != ("constituent", "point"):
                    flags = _orient(ds["elevation_interpolation_flag"], ("constituent", "point")).astype(np.int8)
            else:
                flags = np.zeros(coefficient.shape, dtype=np.int8)
            mode = "point_exact_order"
        else:
            dimensions = real_var.dimensions
            if len(dimensions) != 3 or dimensions[0] != "constituent":
                raise GateError("native_dimensions", f"Unsupported native coefficient dimensions: {dimensions}.")
            lat_dim, lon_dim = dimensions[1], dimensions[2]
            if lat_dim not in ds.variables or lon_dim not in ds.variables:
                raise GateError("native_coordinates", "Native subset dimension-coordinate variables are missing.")
            real = _orient(real_var, ("constituent", lat_dim, lon_dim))
            imag = _orient(imag_var, ("constituent", lat_dim, lon_dim))
            coefficient, flags = _interpolate_native(
                (real + 1j * imag) * scale,
                _as_float(ds[lon_dim]),
                _as_float(ds[lat_dim]),
                points.longitude,
                points.latitude,
                nearest_wet_max_degrees,
            )
            mode = "native_subset_interpolated_in_builder"
    if coefficient.shape != (len(names), points.node_ids.size):
        raise GateError("coefficient_shape", "Coefficient array does not match constituents by OBC nodes.")
    if np.any(flags == 2) or not np.all(np.isfinite(coefficient.real) & np.isfinite(coefficient.imag)):
        raise GateError(
            "source_water_coverage",
            "At least one constituent/OBC pair is unresolved, dry beyond fallback range, or non-finite.",
        )
    return Harmonics(names=names, coefficient_m=coefficient, flags=flags, source_mode=mode, source_units=units)


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO-8601 time {value!r}.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("Times must include an explicit UTC offset or Z.")
    if parsed.utcoffset().total_seconds() != 0:
        raise argparse.ArgumentTypeError("Forcing times must be expressed in UTC.")
    return parsed.astimezone(timezone.utc)


def build_time_axis(start: datetime, end: datetime, interval_minutes: int) -> tuple[np.ndarray, np.ndarray]:
    if interval_minutes <= 0:
        raise GateError("time_interval", "Time interval must be positive.")
    duration_seconds = int((end - start).total_seconds())
    interval_seconds = interval_minutes * 60
    if duration_seconds <= 0:
        raise GateError("time_range", "End time must be later than start time.")
    if duration_seconds % interval_seconds:
        raise GateError("time_alignment", "Requested period is not an integer number of forcing intervals.")
    start64 = np.datetime64(start.replace(tzinfo=None), "s")
    times = start64 + np.arange(duration_seconds // interval_seconds + 1) * np.timedelta64(interval_seconds, "s")
    mjd = datetime64_to_mjd(times).astype(np.float64)
    if times[0] != start64 or times[-1] != np.datetime64(end.replace(tzinfo=None), "s"):
        raise GateError("time_endpoints", "Constructed time axis does not preserve requested endpoints.")
    if not np.all(np.diff(times).astype("timedelta64[s]").astype(np.int64) == interval_seconds):
        raise GateError("time_monotonic", "Constructed UTC time axis is not strictly monotonic and uniform.")
    return times, mjd


def reconstruct_utide(
    coefficient_m: np.ndarray,
    names: list[str],
    times: np.ndarray,
    latitude: np.ndarray,
) -> np.ndarray:
    try:
        from utide._time_conversion import _python_gregorian_datenum
        from utide._ut_constants import constit_index_dict
        from utide.constituent_selection import linearized_freqs
        from utide.harmonics import ut_E
    except ImportError as exc:
        raise GateError("utide_dependency", "UTide is required for astronomical and nodal corrections.") from exc
    missing = [name for name in names if name not in constit_index_dict]
    if missing:
        raise GateError("utide_constituents", f"UTide does not recognize: {', '.join(missing)}")
    t_utide = _python_gregorian_datenum(times)
    reference_time = float(np.mean(t_utide))
    indices = np.asarray([constit_index_dict[name] for name in names], dtype=int)
    frequencies = linearized_freqs(reference_time)[indices]
    output = np.empty((coefficient_m.shape[1], times.size), dtype=np.float32)
    for node_index, node_latitude in enumerate(latitude):
        # UTide E carries equilibrium astronomical argument V, nodal phase u,
        # and nodal amplitude f. TPXO's native coefficient is A*exp(-i*g),
        # where g is Greenwich phase lag, so the real tide is
        # E*(A*exp(-i*g)/2) plus its complex conjugate.
        basis = ut_E(
            t_utide,
            reference_time,
            frequencies,
            indices,
            float(node_latitude),
            list(UTIDE_NGFLAGS),
            [],
        )
        half_coefficient = 0.5 * coefficient_m[:, node_index]
        signal = basis @ half_coefficient + np.conj(basis) @ np.conj(half_coefficient)
        output[node_index] = np.real(signal).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise GateError("reconstruction_nan", "Astronomical reconstruction produced non-finite elevation.")
    return output


def write_diagnostics(
    path: str | Path,
    points: BoundaryPoints,
    harmonics: Harmonics,
    source_sha256: str,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        partial.unlink()
    amplitude = np.abs(harmonics.coefficient_m)
    phase = np.mod(np.rad2deg(np.arctan2(-harmonics.coefficient_m.imag, harmonics.coefficient_m.real)), 360.0)
    try:
        with nc4.Dataset(partial, "w", format="NETCDF4") as ds:
            ds.Conventions = "CF-1.10"
            ds.product_schema = "fvcom_tpxo_interpolation_diagnostics_v1"
            ds.source_harmonics_sha256 = source_sha256
            ds.coefficient_convention = "coefficient = amplitude * exp(-i * Greenwich_phase_lag)"
            ds.createDimension("constituent", len(harmonics.names))
            ds.createDimension("nobc", points.node_ids.size)
            ds.createVariable("constituent_name", str, ("constituent",))[:] = np.asarray(harmonics.names, dtype=object)
            ds.createVariable("obc_nodes", "i4", ("nobc",))[:] = points.node_ids
            ds.createVariable("longitude", "f8", ("nobc",))[:] = points.longitude
            ds.createVariable("latitude", "f8", ("nobc",))[:] = points.latitude
            for name, values, units in (
                ("elevation_real", harmonics.coefficient_m.real, "m"),
                ("elevation_imaginary", harmonics.coefficient_m.imag, "m"),
                ("elevation_amplitude", amplitude, "m"),
                ("elevation_phase_lag", phase, "degree"),
            ):
                variable = ds.createVariable(name, "f4", ("constituent", "nobc"), zlib=True, complevel=2)
                variable.units = units
                variable[:] = np.asarray(values, dtype=np.float32)
            flag = ds.createVariable("interpolation_flag", "i1", ("constituent", "nobc"))
            flag.flag_values = np.asarray([0, 1, 2], dtype=np.int8)
            flag.flag_meanings = "linear_or_exact nearest_wet unresolved"
            flag[:] = harmonics.flags
        os.replace(partial, destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return destination


def validate_forcing(
    path: str | Path,
    points: BoundaryPoints,
    mjd: np.ndarray,
    expected_interval_seconds: int,
) -> dict[str, Any]:
    problems: list[str] = []
    with nc4.Dataset(path) as ds:
        if getattr(ds, "reconstruction_rule_version", None) != RECONSTRUCTION_RULE_VERSION:
            problems.append("Forcing lacks the exact-time nodal reconstruction rule.")
        if getattr(ds, "utide_ngflags_json", None) != json.dumps(list(UTIDE_NGFLAGS)):
            problems.append("Forcing does not declare exact-time UTide nodal/astronomical flags.")
        if getattr(ds, "builder_sha256", None) != sha256_file(__file__):
            problems.append("Forcing builder hash differs from the validating builder.")
        nodes = np.asarray(ds["obc_nodes"][:], dtype=np.int32)
        time_dtype = np.dtype(ds["time"].dtype)
        time = np.asarray(ds["time"][:], dtype=float)
        elevation = _as_float(ds["elevation"])
        if not np.array_equal(nodes, points.node_ids):
            problems.append("FVCOM obc_nodes order differs from the requested order.")
        if time_dtype.itemsize < 8:
            problems.append("FVCOM time variable is not double precision.")
        if time.shape != mjd.shape or not np.allclose(time, mjd, atol=1e-10, rtol=0.0):
            problems.append("FVCOM time values differ from the constructed UTC axis.")
        if time.size < 2 or not np.all(np.diff(time) > 0):
            problems.append("FVCOM time is not strictly monotonic.")
        elif not np.allclose(np.diff(time) * 86400.0, expected_interval_seconds, atol=1e-5, rtol=0.0):
            problems.append("FVCOM time spacing differs from the requested interval.")
        if elevation.shape != (mjd.size, points.node_ids.size):
            problems.append(f"Unexpected elevation shape {elevation.shape}.")
        if not np.all(np.isfinite(elevation)):
            problems.append("Elevation contains non-finite values.")
    return {
        "status": "pass" if not problems else "fail",
        "problems": problems,
        "time_count": int(mjd.size),
        "obc_node_count": int(points.node_ids.size),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    harmonics_path = Path(args.harmonics).expanduser().resolve()
    points_path = Path(args.obc_points).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    diagnostics_path = (
        Path(args.diagnostics).expanduser().resolve()
        if args.diagnostics
        else output.with_name(f"{output.stem}_interpolation_diagnostics.nc")
    )
    points = load_boundary_points(points_path)
    if args.expected_constituents <= 0:
        raise GateError("expected_constituents", "Expected constituent count must be positive.")
    if args.obc_dat:
        dat_nodes = read_obc_nodes_dat(args.obc_dat)
        if not np.array_equal(dat_nodes, points.node_ids):
            raise GateError("obc_dat_order_mismatch", "OBC DAT node order differs from the OBC point CSV.")
    harmonics = load_harmonics(harmonics_path, points, args.nearest_wet_max_deg)
    if not np.all(np.isin(harmonics.flags, [0, 1, 2])):
        raise GateError("interpolation_flags", "Interpolation flags contain values outside 0, 1, and 2.")
    warnings: list[str] = []
    if len(harmonics.names) != args.expected_constituents:
        if not args.constituent_count_explanation:
            raise GateError(
                "constituent_count",
                f"Expected {args.expected_constituents} constituents but found {len(harmonics.names)}; an explicit explanation is required.",
            )
        warnings.append(
            f"Constituent count differs from expected {args.expected_constituents}: "
            f"{args.constituent_count_explanation}"
        )
    times, mjd = build_time_axis(args.start, args.end, args.interval_minutes)
    elevation = reconstruct_utide(harmonics.coefficient_m, harmonics.names, times, points.latitude)
    import utide
    builder_sha = sha256_file(__file__)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.partial")
    if temporary_output.exists():
        temporary_output.unlink()
    write_elevation_obc(temporary_output, points.node_ids, mjd, elevation, casename=args.case_name)
    with nc4.Dataset(temporary_output, "a") as ds:
        ds.source_product_schema = PRODUCT_SCHEMA
        ds.source_harmonics_sha256 = sha256_file(harmonics_path)
        ds.obc_points_sha256 = sha256_file(points_path)
        ds.obc_order_sha256 = boundary_order_sha256(points)
        ds.astronomical_reconstruction = "UTide astronomical argument V with time-varying nodal f,u"
        ds.reconstruction_rule_version = RECONSTRUCTION_RULE_VERSION
        ds.utide_ngflags_json = json.dumps(list(UTIDE_NGFLAGS))
        ds.utide_version = utide.__version__
        ds.builder_sha256 = builder_sha
        ds.phase_convention = "TPXO Greenwich phase lag; coefficient=A*exp(-i*g)"
        ds.time_coverage_start = args.start.isoformat().replace("+00:00", "Z")
        ds.time_coverage_end = args.end.isoformat().replace("+00:00", "Z")
    health = validate_forcing(temporary_output, points, mjd, args.interval_minutes * 60)
    if health["status"] != "pass":
        temporary_output.unlink(missing_ok=True)
        raise GateError("forcing_health", "; ".join(health["problems"]))
    os.replace(temporary_output, output)
    source_hash = sha256_file(harmonics_path)
    diagnostics = write_diagnostics(diagnostics_path, points, harmonics, source_hash)
    nearest_count = int(np.count_nonzero(harmonics.flags == 1))
    if nearest_count:
        warnings.append(f"Used bounded nearest-wet fallback for {nearest_count} constituent/node values.")
    amplitude = np.abs(harmonics.coefficient_m)
    phase = np.mod(np.rad2deg(np.arctan2(-harmonics.coefficient_m.imag, harmonics.coefficient_m.real)), 360.0)
    payload = {
        "schema_version": SCHEMA,
        "status": "ready",
        "artifact_root": str(output.parent),
        "forcing_file": str(output),
        "diagnostics_file": str(diagnostics),
        "hashes": {
            "source_harmonics_sha256": source_hash,
            "obc_points_sha256": sha256_file(points_path),
            "obc_order_sha256": boundary_order_sha256(points),
            "forcing_sha256": sha256_file(output),
            "diagnostics_sha256": sha256_file(diagnostics),
            "builder_sha256": builder_sha,
        },
        "provenance": {
            "source_product_schema": PRODUCT_SCHEMA,
            "source_mode": harmonics.source_mode,
            "source_units": harmonics.source_units,
            "coefficient_convention": "A*exp(-i*Greenwich_phase_lag)",
            "astronomical_argument": "UTide V",
            "nodal_handling": "UTide time-varying f and u, evaluated at each OBC-node latitude",
            "reconstruction_rule_version": RECONSTRUCTION_RULE_VERSION,
            "utide_ngflags": list(UTIDE_NGFLAGS),
            "utide_version": utide.__version__,
            "time_standard": "UTC",
            "time_coverage_start": args.start.isoformat().replace("+00:00", "Z"),
            "time_coverage_end": args.end.isoformat().replace("+00:00", "Z"),
            "interval_minutes": args.interval_minutes,
        },
        "constituents": harmonics.names,
        "constituent_count": len(harmonics.names),
        "expected_constituent_count": args.expected_constituents,
        "obc_node_count": int(points.node_ids.size),
        "time_count": int(times.size),
        "interpolation": {
            "exact_or_linear_count": int(np.count_nonzero(harmonics.flags == 0)),
            "nearest_wet_count": nearest_count,
            "unresolved_count": int(np.count_nonzero(harmonics.flags == 2)),
        },
        "amplitude_m_range": [float(np.min(amplitude)), float(np.max(amplitude))],
        "phase_lag_degree_range": [float(np.min(phase)), float(np.max(phase))],
        "health": health,
        "warnings": warnings,
        "blocking_reasons": [],
        "resume_token": {
            "state": "forcing_ready",
            "source_harmonics_sha256": source_hash,
            "obc_order_sha256": boundary_order_sha256(points),
        },
    }
    atomic_json(report_path, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--harmonics", required=True, help="Model-neutral tpxo9v5_harmonics_v1 point or native-subset NetCDF.")
    result.add_argument("--obc-points", required=True, help="Ordered CSV: node_id,longitude,latitude.")
    result.add_argument("--obc-dat", help="Optional FVCOM OBC DAT for an additional exact-order gate.")
    result.add_argument("--start", required=True, type=parse_utc, help="Inclusive UTC start, ISO-8601 with Z or +00:00.")
    result.add_argument("--end", required=True, type=parse_utc, help="Inclusive UTC end, ISO-8601 with Z or +00:00.")
    result.add_argument("--interval-minutes", type=int, default=6)
    result.add_argument("--expected-constituents", type=int, default=DEFAULT_EXPECTED_CONSTITUENTS)
    result.add_argument("--constituent-count-explanation", help="Required scientific explanation when actual count differs.")
    result.add_argument("--nearest-wet-max-deg", type=float, default=1.0)
    result.add_argument("--case-name", default="fvcom_tpxo")
    result.add_argument("--output", required=True)
    result.add_argument("--diagnostics", help="Interpolation-diagnostics NetCDF; defaults beside output.")
    result.add_argument("--report", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    report_path = Path(args.report).expanduser().resolve()
    try:
        payload = build(args)
    except GateError as exc:
        payload = {
            "schema_version": SCHEMA,
            "status": "blocked",
            "artifact_root": str(report_path.parent),
            "hashes": {},
            "provenance": {},
            "warnings": [],
            "blocking_reasons": [{"code": exc.code, "message": str(exc)}],
            "resume_token": {"state": "forcing_blocked", "gate": exc.code},
        }
        atomic_json(report_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA,
            "status": "failed",
            "artifact_root": str(report_path.parent),
            "hashes": {},
            "provenance": {},
            "warnings": [],
            "blocking_reasons": [{"code": "unexpected_error", "message": str(exc)}],
            "resume_token": {"state": "forcing_failed", "gate": "unexpected_error"},
        }
        atomic_json(report_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
