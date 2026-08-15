#!/usr/bin/env python3
"""Create native subsets or arbitrary-target TPXO9v5 harmonic products."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

import netCDF4 as nc4
import numpy as np
from tpxo9v5.coordinates import minimal_longitude_interval
from tpxo9v5.interpolation import interpolate_complex_field
from tpxo9v5.io import (
    HarmonicField,
    discover_source_files,
    read_harmonic_field,
    transport_to_velocity,
)
from tpxo9v5.outputs import (
    sha256_file,
    validate_product,
    write_json,
    write_native_product,
    write_point_product,
)


def parse_fields(value: str) -> list[str]:
    fields = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(fields) - {"elevation", "transport"})
    if invalid or not fields:
        raise argparse.ArgumentTypeError("fields must contain elevation and/or transport")
    return list(dict.fromkeys(fields))


def parse_constituents(value: str | None) -> list[str] | None:
    if not value:
        return None
    parsed = [item.strip().upper() for item in value.split(",") if item.strip()]
    return list(dict.fromkeys(parsed)) or None


def load_points_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        lookup = {name.lower(): name for name in (reader.fieldnames or [])}
        if "longitude" not in lookup or "latitude" not in lookup:
            raise ValueError("Point CSV must contain longitude and latitude columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("Point CSV contains no rows.")
    lon = np.asarray([float(row[lookup["longitude"]]) for row in rows], dtype=float)
    lat = np.asarray([float(row[lookup["latitude"]]) for row in rows], dtype=float)
    return lon, lat, (lon.size,)


def load_target_grid(
    path: str | Path,
    longitude_name: str,
    latitude_name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    with nc4.Dataset(path) as ds:
        if longitude_name not in ds.variables or latitude_name not in ds.variables:
            raise ValueError("Target longitude or latitude variable was not found.")
        lon_var = ds[longitude_name]
        lat_var = ds[latitude_name]
        lon = np.asarray(np.ma.filled(lon_var[:], np.nan), dtype=float)
        lat = np.asarray(np.ma.filled(lat_var[:], np.nan), dtype=float)
        if lon.ndim == lat.ndim == 1 and lon_var.dimensions != lat_var.dimensions:
            lon, lat = np.meshgrid(lon, lat)
            shape = lon.shape
        elif lon.shape == lat.shape:
            shape = lon.shape
        else:
            raise ValueError("Target coordinates must be matching arrays or independent 1-D grid axes.")
    return lon.ravel(), lat.ravel(), tuple(int(value) for value in shape)


def target_bbox(longitude: np.ndarray, latitude: np.ndarray) -> tuple[float, float, float, float]:
    if not np.all(np.isfinite(longitude) & np.isfinite(latitude)):
        raise ValueError("Target coordinates must be finite.")
    west, east = minimal_longitude_interval(longitude, padding=0.0)
    return west, float(np.min(latitude)), east, float(np.max(latitude))


def load_fields(
    sources: dict[str, Path],
    requested_fields: Iterable[str],
    bbox: tuple[float, float, float, float],
    constituents: list[str] | None,
    padding: float,
    velocity: bool,
) -> list[HarmonicField]:
    fields: list[HarmonicField] = []
    if "elevation" in requested_fields:
        fields.append(
            read_harmonic_field(
                sources["elevation"], sources["grid"], "elevation", "z", bbox, constituents, padding
            )
        )
    if "transport" in requested_fields:
        u = read_harmonic_field(sources["transport"], sources["grid"], "u", "u", bbox, constituents, padding)
        v = read_harmonic_field(sources["transport"], sources["grid"], "v", "v", bbox, constituents, padding)
        fields.extend((u, v))
        if velocity:
            fields.extend((transport_to_velocity(u), transport_to_velocity(v)))
    return fields


def safe_cleanup(source_paths: Iterable[Path], staging_dir: str | Path | None) -> list[str]:
    """Delete exact staged NetCDF copies only; never recurse or delete outside staging."""

    if staging_dir is None:
        return []
    staging = Path(staging_dir).expanduser().resolve()
    if not staging.is_dir():
        raise FileNotFoundError(f"Staging directory does not exist: {staging}")
    if staging == Path(staging.anchor) or staging == Path.home().resolve():
        raise ValueError(f"Refusing unsafe staging directory: {staging}")
    candidates = sorted({Path(path).resolve() for path in source_paths})
    for source in candidates:
        if not source.is_relative_to(staging):
            raise ValueError(f"Refusing to delete source outside staging directory: {source}")
        if source.suffix.lower() != ".nc":
            raise ValueError(f"Refusing to delete non-NetCDF staged file: {source}")
    removed: list[str] = []
    for source in candidates:
        if source.exists():
            source.unlink()
            removed.append(source.name)
    return removed


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", required=True, help="Directory containing staged or caller-owned TPXO files.")
    parser.add_argument("--staging-dir", help="Explicit managed staging root eligible for cleanup.")
    parser.add_argument("--fields", default="elevation", type=parse_fields, help="Comma-separated elevation and/or transport.")
    parser.add_argument("--constituents", help="Comma-separated names; default is every discovered constituent.")
    parser.add_argument("--padding-deg", type=float, default=0.5, help="Source padding around requested coverage.")
    parser.add_argument("--velocity", action="store_true", help="Also divide recognized transports by matching wet depth.")
    parser.add_argument("--cleanup", choices=("success", "never"), default="success")
    parser.add_argument("--keep-raw", action="store_true", help="Alias for --cleanup never.")
    parser.add_argument("--output", required=True, help="Output NetCDF path.")
    parser.add_argument("--report", required=True, help="Extraction/cleanup JSON report path.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subset = subparsers.add_parser("subset", help="Preserve bounded native staggered grids.")
    add_common_arguments(subset)
    subset.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"), required=True)

    interpolate = subparsers.add_parser("interpolate", help="Interpolate to arbitrary points or a target grid.")
    add_common_arguments(interpolate)
    target = interpolate.add_mutually_exclusive_group(required=True)
    target.add_argument("--points", help="CSV with longitude and latitude columns.")
    target.add_argument("--target-grid", help="NetCDF containing target coordinates.")
    interpolate.add_argument("--target-lon-var", default="lon")
    interpolate.add_argument("--target-lat-var", default="lat")
    interpolate.add_argument("--nearest-wet-max-deg", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    constituents = parse_constituents(args.constituents)
    cleanup_policy = "never" if args.keep_raw else args.cleanup
    sources = discover_source_files(args.source_dir, args.fields)

    target_lon: np.ndarray | None = None
    target_lat: np.ndarray | None = None
    target_shape: tuple[int, ...] | None = None
    if args.mode == "subset":
        bbox = tuple(float(value) for value in args.bbox)
    elif args.points:
        target_lon, target_lat, target_shape = load_points_csv(args.points)
        bbox = target_bbox(target_lon, target_lat)
    else:
        target_lon, target_lat, target_shape = load_target_grid(
            args.target_grid, args.target_lon_var, args.target_lat_var
        )
        bbox = target_bbox(target_lon, target_lat)

    fields = load_fields(
        sources,
        args.fields,
        bbox,
        constituents,
        args.padding_deg,
        args.velocity,
    )
    unique_sources = sorted(set(sources.values()))
    provenance = [
        {
            "role": role,
            "basename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for role, path in sorted(sources.items())
    ]
    metadata = {
        "mode": args.mode,
        "requested_fields": args.fields,
        "requested_bbox": list(bbox),
        "actual_spans": {field.name: field.actual_span for field in fields},
        "source_spans": {field.name: field.source_span for field in fields},
        "source_provenance": provenance,
        "interpolation_method": "complex linear with bounded nearest-wet fallback" if args.mode == "interpolate" else "none; native subset",
        "raw_cleanup_policy": cleanup_policy,
        "raw_cleanup_status": "validated_pending_cleanup",
    }

    if args.mode == "subset":
        destination = write_native_product(args.output, fields, metadata)
    else:
        assert target_lon is not None and target_lat is not None and target_shape is not None
        interpolated = []
        for field in fields:
            values, flags = interpolate_complex_field(
                field, target_lon, target_lat, args.nearest_wet_max_deg
            )
            interpolated.append((field, values, flags))
        destination = write_point_product(
            args.output,
            interpolated,
            target_lon,
            target_lat,
            target_shape,
            metadata,
        )

    health = validate_product(destination)
    if health["status"] != "pass":
        write_json(args.report, {"status": "fail", "health": health, "staged_files_removed": []})
        raise RuntimeError("Generated product failed health validation; staged sources were retained.")
    removed = safe_cleanup(unique_sources, args.staging_dir) if cleanup_policy == "success" else []
    if removed:
        cleanup_status = "removed"
    elif cleanup_policy == "never":
        cleanup_status = "retained_by_request"
    else:
        cleanup_status = "not_applicable_no_staging_directory"
    with nc4.Dataset(destination, "a") as ds:
        ds.raw_cleanup_status = cleanup_status
    report = {
        "schema_version": "tpxo9v5_extract_report_v1",
        "status": "pass",
        "output_basename": destination.name,
        "health": health,
        "requested_bbox": list(bbox),
        "actual_spans": metadata["actual_spans"],
        "source_provenance": provenance,
        "cleanup_policy": cleanup_policy,
        "cleanup_status": cleanup_status,
        "staged_files_removed": removed,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
