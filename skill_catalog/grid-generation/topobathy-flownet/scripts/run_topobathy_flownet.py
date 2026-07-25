"""Build a DHSVM-compatible SegOrder channel/thalweg network with GRASS GIS 8."""

from __future__ import annotations

import argparse
import functools
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from topobathy_flownet_core import (
    NODATA,
    SCHEMA_VERSION,
    affine_cell_area_m2,
    automatic_utm_epsg,
    build_arc_records,
    network_dat_rows,
    sha256_file,
    source_area_to_cells,
)

DEFAULT_OSGEO_SHELL = Path("C:/OSGeo4W/OSGeo4W.bat")
OUTPUT_NAMES = {
    "projected_surface": "projected_surface_positive_up.tif",
    "projected_mask": "analysis_mask_projected.gpkg",
    "flow_direction": "flow_direction.tif",
    "accumulation_cells": "accumulation_cells.tif",
    "drainage_area_m2": "drainage_area_m2.tif",
    "stream_raster": "stream_raster.tif",
    "stream_direction": "stream_direction.tif",
    "raw_stream_vector": "stream_vector_raw.gpkg",
    "dhsvm_gpkg": "topobathy_flownet.gpkg",
    "dhsvm_geojson": "topobathy_flownet.geojson",
    "stream_network_dat": "stream.network.dat",
    "topology_qa": "topology_qa.json",
    "health_check": "health_check.json",
    "segorder_map": "segorder_map.png",
    "accumulation_map": "accumulation_map.png",
    "manifest": "run_manifest.json",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def absolute_paths(out_dir: Path) -> dict[str, Path]:
    root = out_dir.resolve()
    return {name: (root / filename).resolve() for name, filename in OUTPUT_NAMES.items()}


def _environment_value(environment: dict[str, str], name: str) -> str | None:
    expected = name.casefold()
    return next((value for key, value in environment.items() if key.casefold() == expected), None)


@functools.lru_cache(maxsize=8)
def _batch_environment(batch_paths: tuple[str, ...]) -> dict[str, str]:
    """Capture environment changes from trusted OSGeo4W batch scripts."""
    if not batch_paths:
        raise ValueError("At least one OSGeo4W environment batch file is required")
    command_argv = ["cmd", "/d", "/v:off", "/c"]
    for raw_path in batch_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OSGeo4W environment script not found: {path}")
        if any(character in str(path) for character in {'"', "\r", "\n"}):
            raise ValueError(f"Unsafe OSGeo4W environment-script path: {path}")
        command_argv.extend(["call", str(path), "&&"])
    command_argv.append("set")
    result = subprocess.run(
        command_argv,
        text=True,
        capture_output=True,
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not load OSGeo4W environment from {batch_paths}: "
            f"{result.stderr[-2000:] or result.stdout[-2000:]}"
        )
    environment: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and not key.startswith("="):
            environment[key] = value
    if not _environment_value(environment, "PATH"):
        raise RuntimeError("OSGeo4W environment did not define PATH")
    return environment


def build_grass_direct_argv(
    logical_args: Sequence[str],
    *,
    grass_python: str | Path,
    grass_script: str | Path,
) -> list[str]:
    """Preserve every GRASS logical argument outside a batch/cmd boundary."""
    if not logical_args:
        raise ValueError("GRASS logical argument list must not be empty")
    return [str(grass_python), str(grass_script), *[str(value) for value in logical_args[1:]]]


def osgeo_process(
    osgeo_shell: Path,
    args: Sequence[str],
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Resolve OSGeo4W tools to direct executable argv.

    User and run paths are never interpolated into a cmd.exe command string.
    cmd.exe is used only to read trusted installation environment scripts.
    """
    if not args:
        raise ValueError("OSGeo command must contain an executable name")
    shell = osgeo_shell.resolve()
    osgeo_root = shell.parent
    base_environment_script = osgeo_root / "bin" / "o4w_env.bat"
    base_environment = _batch_environment((str(base_environment_script.resolve()),))
    search_path = _environment_value(base_environment, "PATH")
    resolved_launcher = shutil.which(str(args[0]), path=search_path)
    if not resolved_launcher:
        raise FileNotFoundError(f"OSGeo4W command not found in configured PATH: {args[0]}")
    launcher_path = Path(resolved_launcher).resolve()
    logical_args = [str(value) for value in args]

    if launcher_path.suffix.casefold() in {".bat", ".cmd"}:
        grass_name = launcher_path.stem
        grass_environment_script = osgeo_root / "apps" / "grass" / grass_name / "etc" / "env.bat"
        if not grass_name.casefold().startswith("grass") or not grass_environment_script.is_file():
            raise RuntimeError(
                f"Refusing unsupported OSGeo4W batch launcher {launcher_path}; "
                "only a resolved GRASS launcher is translated to direct Python argv"
            )
        environment_scripts = (
            str(base_environment_script.resolve()),
            str(grass_environment_script.resolve()),
        )
        process_environment = _batch_environment(environment_scripts)
        grass_python = _environment_value(process_environment, "GRASS_PYTHON")
        gisbase = _environment_value(process_environment, "GISBASE")
        if not grass_python or not gisbase:
            raise RuntimeError(f"GRASS environment for {grass_name} lacks GRASS_PYTHON or GISBASE")
        grass_script = Path(gisbase) / "etc" / f"{grass_name}.py"
        if not Path(grass_python).is_file() or not grass_script.is_file():
            raise FileNotFoundError(
                f"Resolved GRASS direct launcher is incomplete: python={grass_python}, script={grass_script}"
            )
        process_argv = build_grass_direct_argv(
            logical_args,
            grass_python=grass_python,
            grass_script=grass_script,
        )
        launcher = {
            "mode": "osgeo4w_direct_grass_python_v1",
            "osgeo_shell": str(shell),
            "resolved_batch_launcher": str(launcher_path),
            "resolved_executable": process_argv[0],
            "resolved_script": process_argv[1],
            "environment_scripts": list(environment_scripts),
        }
        return process_argv, dict(process_environment), launcher

    process_argv = [str(launcher_path), *logical_args[1:]]
    launcher = {
        "mode": "osgeo4w_direct_executable_v1",
        "osgeo_shell": str(shell),
        "resolved_executable": str(launcher_path),
        "environment_scripts": [str(base_environment_script.resolve())],
    }
    return process_argv, dict(base_environment), launcher


def run_osgeo(
    osgeo_shell: Path,
    args: Sequence[str],
    *,
    cwd: Path,
    purpose: str,
    check: bool = True,
) -> dict[str, Any]:
    started = now_utc()
    start_clock = time.perf_counter()
    process_argv, process_environment, launcher = osgeo_process(osgeo_shell, args)
    result = subprocess.run(
        process_argv,
        cwd=cwd,
        env=process_environment,
        text=True,
        capture_output=True,
        errors="replace",
    )
    record = {
        "purpose": purpose,
        "started_utc": started,
        "finished_utc": now_utc(),
        "duration_seconds": round(time.perf_counter() - start_clock, 6),
        "args": [str(value) for value in args],
        "executed_argv": process_argv,
        "launcher": launcher,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if check and result.returncode != 0:
        raise RuntimeError(
            f"{purpose} failed with code {result.returncode}: {' '.join(map(str, args))}\n"
            f"{result.stderr[-2000:] or result.stdout[-2000:]}"
        )
    return record


def probe_version(osgeo_shell: Path, args: Sequence[str], cwd: Path) -> str | None:
    try:
        process_argv, process_environment, _ = osgeo_process(osgeo_shell, args)
        result = subprocess.run(
            process_argv,
            cwd=cwd,
            env=process_environment,
            text=True,
            capture_output=True,
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if result.returncode == 0 and text else None


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def resolve_surface_dataset(surface: Path, variable: str | None) -> str:
    import rasterio

    with rasterio.open(surface) as dataset:
        subdatasets = list(dataset.subdatasets)
        if dataset.count >= 1 and dataset.width > 0 and dataset.height > 0 and not subdatasets:
            if variable:
                if surface.suffix.casefold() not in {".nc", ".nc4", ".cdf"}:
                    raise ValueError("--surface-variable is only valid for a NetCDF/HDF input")
            return str(surface)
    if not subdatasets:
        raise ValueError(f"No readable raster band found in {surface}")
    if variable:
        matches = [
            candidate
            for candidate in subdatasets
            if candidate.rsplit(":", 1)[-1].strip('"').casefold() == variable.casefold()
        ]
        if len(matches) != 1:
            available = [candidate.rsplit(":", 1)[-1].strip('"') for candidate in subdatasets]
            raise ValueError(f"NetCDF variable {variable!r} did not select one raster; available={available}")
        return matches[0]
    if len(subdatasets) == 1:
        return subdatasets[0]
    available = [candidate.rsplit(":", 1)[-1].strip('"') for candidate in subdatasets]
    raise ValueError(f"Surface contains multiple raster variables; pass --surface-variable from {available}")


def load_regular_lonlat_netcdf(surface: Path, variable: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load a strict regular lon/lat NetCDF variable as a north-up raster.

    This adapter is deliberately narrow. It only assigns EPSG:4326 after
    validating explicit one-dimensional longitude and latitude coordinates.
    """
    import numpy as np
    import xarray as xr
    from pyproj import CRS
    from rasterio.transform import from_origin

    def coordinate_role(name: str, coordinate) -> str | None:
        lowered_name = name.casefold()
        standard_name = str(coordinate.attrs.get("standard_name", "")).casefold()
        axis = str(coordinate.attrs.get("axis", "")).casefold()
        units = str(coordinate.attrs.get("units", "")).casefold().replace(" ", "_")
        longitude = (
            lowered_name in {"lon", "longitude"}
            or standard_name == "longitude"
            or axis == "x" and units in {"degree_east", "degrees_east"}
        )
        latitude = (
            lowered_name in {"lat", "latitude"}
            or standard_name == "latitude"
            or axis == "y" and units in {"degree_north", "degrees_north"}
        )
        if longitude and latitude:
            raise ValueError(f"Coordinate {name!r} is ambiguously marked as both longitude and latitude")
        return "longitude" if longitude else "latitude" if latitude else None

    def regular_axis(values, name: str) -> tuple[np.ndarray, str, float]:
        axis_values = np.asarray(values, dtype="float64")
        if axis_values.ndim != 1 or axis_values.size < 2 or not np.all(np.isfinite(axis_values)):
            raise ValueError(f"NetCDF {name} coordinate must be a finite one-dimensional axis with at least two cells")
        differences = np.diff(axis_values)
        if np.all(differences > 0):
            direction = "ascending"
        elif np.all(differences < 0):
            direction = "descending"
        else:
            raise ValueError(f"NetCDF {name} coordinate must be strictly monotonic")
        spacing = float(np.median(np.abs(differences)))
        tolerance = max(1.0e-12, spacing * 1.0e-6)
        if not np.allclose(np.abs(differences), spacing, rtol=1.0e-6, atol=tolerance):
            raise ValueError(f"NetCDF {name} coordinate is not regularly spaced")
        return axis_values, direction, spacing

    with xr.open_dataset(surface, decode_coords="all", mask_and_scale=True) as dataset:
        if variable not in dataset.data_vars:
            raise ValueError(
                f"NetCDF variable {variable!r} not found; available data variables={list(dataset.data_vars)}"
            )
        data_array = dataset[variable]
        if data_array.ndim != 2:
            raise ValueError(f"NetCDF variable {variable!r} must be two-dimensional, found dims={data_array.dims}")
        candidates: dict[str, list[tuple[str, Any]]] = {"longitude": [], "latitude": []}
        for name, coordinate in data_array.coords.items():
            role = coordinate_role(name, coordinate)
            if role and coordinate.ndim == 1 and coordinate.dims[0] in data_array.dims:
                candidates[role].append((name, coordinate))
        for role in ("longitude", "latitude"):
            if len(candidates[role]) != 1:
                names = [name for name, _ in candidates[role]]
                raise ValueError(
                    f"NetCDF fallback requires exactly one validated 1-D {role} coordinate; found={names}"
                )
        lon_name, lon_coordinate = candidates["longitude"][0]
        lat_name, lat_coordinate = candidates["latitude"][0]
        lon_dimension = lon_coordinate.dims[0]
        lat_dimension = lat_coordinate.dims[0]
        if lon_dimension == lat_dimension or set(data_array.dims) != {lat_dimension, lon_dimension}:
            raise ValueError(
                f"NetCDF lon/lat dimensions must exactly match the 2-D variable; "
                f"variable_dims={data_array.dims}, lat_dim={lat_dimension}, lon_dim={lon_dimension}"
            )
        longitudes, longitude_order, longitude_spacing = regular_axis(lon_coordinate.values, "longitude")
        latitudes, latitude_order, latitude_spacing = regular_axis(lat_coordinate.values, "latitude")
        if float(longitudes.min()) < -180.0 or float(longitudes.max()) > 180.0:
            raise ValueError("NetCDF longitude coordinates must lie within [-180, 180] before assigning EPSG:4326")
        if float(latitudes.min()) < -90.0 or float(latitudes.max()) > 90.0:
            raise ValueError("NetCDF latitude coordinates must lie within [-90, 90] before assigning EPSG:4326")

        values = np.asarray(data_array.transpose(lat_dimension, lon_dimension).values, dtype="float64")
        if values.shape != (latitudes.size, longitudes.size):
            raise ValueError(
                f"NetCDF coordinate lengths do not match variable shape: "
                f"shape={values.shape}, lat={latitudes.size}, lon={longitudes.size}"
            )
        longitude_reversed = longitude_order == "descending"
        latitude_reversed = latitude_order == "ascending"
        if longitude_reversed:
            longitudes = longitudes[::-1]
            values = values[:, ::-1]
        if latitude_reversed:
            latitudes = latitudes[::-1]
            values = values[::-1, :]
        transform = from_origin(
            float(longitudes[0] - longitude_spacing / 2.0),
            float(latitudes[0] + latitude_spacing / 2.0),
            longitude_spacing,
            latitude_spacing,
        )
        nonfinite_count = int((~np.isfinite(values)).sum())
        source_fill_value = data_array.encoding.get("_FillValue")
        if hasattr(source_fill_value, "item"):
            source_fill_value = source_fill_value.item()
        if isinstance(source_fill_value, float) and not math.isfinite(source_fill_value):
            source_fill_value = None
        adapter = {
            "name": "xarray_regular_lonlat_netcdf_v1",
            "reason": "selected NetCDF GDAL subdataset lacked usable CRS/georeferencing",
            "source_path": str(surface.resolve()),
            "selected_variable": variable,
            "variable_dims_original": list(data_array.dims),
            "variable_shape_original": [int(size) for size in data_array.shape],
            "longitude_coordinate": lon_name,
            "latitude_coordinate": lat_name,
            "longitude_order_original": longitude_order,
            "latitude_order_original": latitude_order,
            "axis_normalization": {
                "longitude_reversed": longitude_reversed,
                "latitude_reversed": latitude_reversed,
                "output_order": "north-to-south rows, west-to-east columns",
            },
            "longitude_spacing_degrees": longitude_spacing,
            "latitude_spacing_degrees": latitude_spacing,
            "assigned_crs": "EPSG:4326",
            "crs_assignment_basis": "validated regular 1-D lon/lat pixel-center coordinates",
            "affine_gdal": list(transform.to_gdal()),
            "finite_value_count": int(np.isfinite(values).sum()),
            "nonfinite_nodata_count": nonfinite_count,
            "source_fill_value": source_fill_value,
        }
    return values, transform, CRS.from_epsg(4326), adapter


def load_mask(mask_path: Path, layer: str | None):
    import geopandas as gpd

    mask = gpd.read_file(mask_path, layer=layer)
    if mask.empty:
        raise ValueError("Analysis mask contains no features")
    if mask.crs is None:
        raise ValueError("Analysis mask has no CRS")
    mask = mask[mask.geometry.notna() & ~mask.geometry.is_empty].copy()
    mask = mask[mask.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if mask.empty:
        raise ValueError("Analysis mask must contain Polygon or MultiPolygon geometry")
    geometry = mask.geometry.union_all() if hasattr(mask.geometry, "union_all") else mask.geometry.unary_union
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Analysis mask could not be normalized to a valid polygon")
    return gpd.GeoDataFrame({"mask_id": [1]}, geometry=[geometry], crs=mask.crs)


def choose_target_crs(mask, target_crs: str | None):
    from pyproj import CRS

    if target_crs:
        crs = CRS.from_user_input(target_crs)
    else:
        centroid = mask.to_crs("EPSG:4326").geometry.iloc[0].centroid
        crs = CRS.from_epsg(automatic_utm_epsg(float(centroid.x), float(centroid.y)))
    if not crs.is_projected:
        raise ValueError("Target CRS must be projected")
    axis_units = {axis.unit_name.casefold() for axis in crs.axis_info if axis.unit_name}
    if not axis_units or not all(unit in {"metre", "meter"} for unit in axis_units):
        raise ValueError(f"Target CRS must use metre units, found {sorted(axis_units)}")
    return crs


def prepare_projected_inputs(
    *,
    surface_path: Path,
    surface_variable: str | None,
    surface_positive: str,
    mask,
    target_crs,
    target_resolution_m: float | None,
    projected_surface: Path,
    projected_mask: Path,
) -> dict[str, Any]:
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from rasterio.io import MemoryFile
    from rasterio.mask import mask as raster_mask
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    dataset_name = resolve_surface_dataset(surface_path, surface_variable)

    def crop_source(source):
        source_mask = mask.to_crs(source.crs)
        source_geometry = source_mask.geometry.iloc[0]
        clipped, clipped_transform = raster_mask(
            source,
            [source_geometry.__geo_interface__],
            indexes=1,
            crop=True,
            filled=False,
        )
        return clipped.astype("float64").filled(np.nan), clipped_transform, source.crs

    with rasterio.open(dataset_name) as detected_source:
        transform_values = tuple(detected_source.transform)
        transform_determinant = (
            detected_source.transform.a * detected_source.transform.e
            - detected_source.transform.b * detected_source.transform.d
        )
        usable_georeferencing = (
            detected_source.crs is not None
            and all(math.isfinite(value) for value in transform_values)
            and math.isfinite(transform_determinant)
            and not math.isclose(transform_determinant, 0.0)
        )
        if usable_georeferencing:
            source_values, clipped_transform, source_crs = crop_source(detected_source)
            input_adapter = {
                "name": "rasterio_georeferenced_raster_v1",
                "source_dataset": dataset_name,
                "driver": detected_source.driver,
                "source_crs": detected_source.crs.to_string(),
                "source_affine_gdal": list(detected_source.transform.to_gdal()),
                "source_width": int(detected_source.width),
                "source_height": int(detected_source.height),
                "source_nodata": detected_source.nodata,
            }
        else:
            netcdf_suffixes = {".nc", ".nc4", ".cdf"}
            if surface_path.suffix.casefold() not in netcdf_suffixes or not surface_variable:
                raise ValueError(
                    "Input surface lacks usable CRS/georeferencing. The only fallback is an explicitly selected "
                    "regular lon/lat NetCDF variable; GeoTIFF and non-lon/lat inputs are never assigned a CRS."
                )
            netcdf_values, netcdf_transform, netcdf_crs, input_adapter = load_regular_lonlat_netcdf(
                surface_path, surface_variable
            )
            memory_values = np.where(np.isfinite(netcdf_values), netcdf_values, NODATA)
            with MemoryFile() as memory_file:
                with memory_file.open(
                    driver="GTiff",
                    width=memory_values.shape[1],
                    height=memory_values.shape[0],
                    count=1,
                    dtype="float64",
                    crs=netcdf_crs,
                    transform=netcdf_transform,
                    nodata=NODATA,
                ) as memory_writer:
                    memory_writer.write(memory_values, 1)
                with memory_file.open() as memory_source:
                    source_values, clipped_transform, source_crs = crop_source(memory_source)

    if surface_positive == "down":
        source_values = -source_values
    source_height, source_width = source_values.shape
    left, bottom, right, top = array_bounds(source_height, source_width, clipped_transform)
    target_transform, target_width, target_height = calculate_default_transform(
        source_crs,
        target_crs,
        source_width,
        source_height,
        left,
        bottom,
        right,
        top,
        resolution=target_resolution_m,
    )
    target = np.full((target_height, target_width), np.nan, dtype="float64")
    reproject(
        source_values,
        target,
        src_transform=clipped_transform,
        src_crs=source_crs,
        src_nodata=np.nan,
        dst_transform=target_transform,
        dst_crs=target_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    projected_mask_frame = gpd.GeoDataFrame(
        {"mask_id": [1]},
        geometry=[mask.to_crs(target_crs).geometry.iloc[0]],
        crs=target_crs,
    )
    inside = rasterize(
        [(projected_mask_frame.geometry.iloc[0], 1)],
        out_shape=target.shape,
        transform=target_transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    finite_inside = inside & np.isfinite(target)
    inside_count = int(inside.sum())
    valid_count = int(finite_inside.sum())
    if inside_count == 0:
        raise ValueError("Projected analysis mask does not cover any raster cells")
    target[~finite_inside] = NODATA
    projected_surface.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        projected_surface,
        "w",
        driver="GTiff",
        width=target.shape[1],
        height=target.shape[0],
        count=1,
        dtype="float32",
        crs=target_crs,
        transform=target_transform,
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    ) as output:
        output.write(target.astype("float32"), 1)
        output.update_tags(
            vertical_positive="up",
            source_surface=str(surface_path.resolve()),
            source_variable=surface_variable or "",
            input_adapter=input_adapter["name"],
        )
    projected_mask.unlink(missing_ok=True)
    projected_mask_frame.to_file(projected_mask, layer="analysis_mask", driver="GPKG")
    cell_area = affine_cell_area_m2(
        target_transform.a,
        target_transform.b,
        target_transform.d,
        target_transform.e,
    )
    finite_values = target[finite_inside]
    return {
        "surface_dataset": dataset_name,
        "input_adapter": input_adapter,
        "target_crs": target_crs.to_string(),
        "target_crs_wkt": target_crs.to_wkt(),
        "width": int(target.shape[1]),
        "height": int(target.shape[0]),
        "cell_area_m2": cell_area,
        "nominal_cell_size_m": math.sqrt(cell_area),
        "mask_cell_count": inside_count,
        "finite_mask_cell_count": valid_count,
        "finite_mask_coverage_fraction": valid_count / inside_count,
        "elevation_min_m": float(np.nanmin(finite_values)) if finite_values.size else None,
        "elevation_max_m": float(np.nanmax(finite_values)) if finite_values.size else None,
    }


def run_grass(
    *,
    osgeo_shell: Path,
    grass_command: str,
    grass_location: Path,
    paths: dict[str, Path],
    threshold_cells: int,
    cell_area_m2: float,
    memory_mb: int,
    cwd: Path,
    command_records: list[dict[str, Any]],
) -> None:
    def execute_raw(args: Sequence[str], purpose: str) -> None:
        record = run_osgeo(
            osgeo_shell,
            args,
            cwd=cwd,
            purpose=purpose,
            check=False,
        )
        command_records.append(record)
        if record["returncode"] != 0:
            raise RuntimeError(
                f"{purpose} failed with code {record['returncode']}: {' '.join(map(str, args))}\n"
                f"{record['stderr_tail'][-2000:] or record['stdout_tail'][-2000:]}"
            )

    execute_raw(
        [grass_command, "-c", str(paths["projected_surface"]), str(grass_location), "-e"],
        "create fresh projected GRASS location",
    )
    prefix = [grass_command, str(grass_location / "PERMANENT"), "--exec"]

    def execute(args: Sequence[str], purpose: str) -> None:
        execute_raw([*prefix, *args], purpose)

    execute(
        ["r.in.gdal", f"input={paths['projected_surface']}", "output=surface_positive_up", "--overwrite"],
        "import positive-up surface",
    )
    execute(
        ["v.in.ogr", f"input={paths['projected_mask']}", "layer=analysis_mask", "output=analysis_mask", "--overwrite"],
        "import analysis mask",
    )
    execute(["g.region", "raster=surface_positive_up", "-a"], "set computational region")
    execute(["r.mask", "vector=analysis_mask", "--overwrite"], "apply polygon analysis mask")
    execute(
        [
            "r.watershed",
            "-s",
            "-a",
            "elevation=surface_positive_up",
            f"threshold={threshold_cells}",
            "accumulation=accumulation_cells",
            "drainage=flow_direction",
            "stream=watershed_stream",
            f"memory={memory_mb}",
            "--overwrite",
        ],
        "compute D8/SFD flow direction and positive accumulation",
    )
    execute(
        ["r.mapcalc", f"expression=drainage_area_m2=accumulation_cells*{cell_area_m2:.17g}", "--overwrite"],
        "convert accumulation cells to physical drainage area",
    )
    execute(
        [
            "r.stream.extract",
            "elevation=surface_positive_up",
            "accumulation=accumulation_cells",
            f"threshold={threshold_cells}",
            "mexp=0",
            "stream_length=0",
            f"memory={memory_mb}",
            "stream_raster=stream_raster",
            "stream_vector=stream_vector",
            "direction=stream_direction",
            "--overwrite",
        ],
        "extract stream network with the fixed DHSVM-compatible method",
    )
    raster_exports = [
        ("flow_direction", paths["flow_direction"], "Int32"),
        ("accumulation_cells", paths["accumulation_cells"], "Float64"),
        ("drainage_area_m2", paths["drainage_area_m2"], "Float64"),
        ("stream_raster", paths["stream_raster"], "Int32"),
        ("stream_direction", paths["stream_direction"], "Int32"),
    ]
    for source_name, output_path, output_type in raster_exports:
        execute(
            [
                "r.out.gdal",
                f"input={source_name}",
                f"output={output_path}",
                "format=GTiff",
                f"type={output_type}",
                f"nodata={NODATA:g}",
                "createopt=COMPRESS=DEFLATE",
                "--overwrite",
            ],
            f"export {source_name}",
        )
    execute(
        [
            "v.out.ogr",
            "input=stream_vector",
            f"output={paths['raw_stream_vector']}",
            "format=GPKG",
            "output_layer=stream_vector_raw",
            "--overwrite",
        ],
        "export raw GRASS stream vector",
    )
def raster_sampler(path: Path):
    import numpy as np
    import rasterio

    def sample(points):
        values: list[float] = []
        with rasterio.open(path) as dataset:
            nodata = dataset.nodata
            for item in dataset.sample(points):
                value = float(item[0]) if len(item) else NODATA
                if not np.isfinite(value) or (nodata is not None and math.isclose(value, float(nodata))):
                    value = NODATA
                values.append(value)
        return values

    return sample


def read_raw_lines(path: Path):
    import geopandas as gpd

    frame = gpd.read_file(path, layer="stream_vector_raw")
    lines: list[list[tuple[float, float]]] = []
    for geometry in frame.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "LineString":
            parts = [geometry]
        elif geometry.geom_type == "MultiLineString":
            parts = list(geometry.geoms)
        else:
            # GRASS stream vectors can include point primitives in the same
            # layer; only line primitives define DHSVM arcs.
            continue
        for part in parts:
            coordinates = [(float(x), float(y)) for x, y, *_ in part.coords]
            if len(coordinates) >= 2:
                lines.append(coordinates)
    return lines, frame.crs


def write_network_products(records: list[dict[str, Any]], crs, paths: dict[str, Path]) -> None:
    import geopandas as gpd
    from shapely.geometry import LineString

    fields = [
        "arcid",
        "from_node",
        "to_node",
        "local",
        "downarc",
        "uparc",
        "SELEV",
        "EELEV",
        "MAXGRID",
        "dz",
        "slope",
        "meanmsq",
        "segorder",
        "drainage_area_m2",
        "chanclass",
        "hyddepth",
        "hydwidth",
        "effwidth",
        "effdepth",
        "segdepth",
        "Shape_Leng",
    ]
    rows = [{field: record[field] for field in fields} for record in records]
    frame = gpd.GeoDataFrame(rows, geometry=[LineString(record["points"]) for record in records], crs=crs)
    paths["dhsvm_gpkg"].unlink(missing_ok=True)
    frame.to_file(paths["dhsvm_gpkg"], layer="topobathy_flownet", driver="GPKG")
    paths["dhsvm_geojson"].unlink(missing_ok=True)
    frame.to_file(paths["dhsvm_geojson"], driver="GeoJSON")
    paths["stream_network_dat"].write_text(
        "\n".join(network_dat_rows(records)) + ("\n" if records else ""),
        encoding="utf-8",
    )


def create_qa_maps(records: list[dict[str, Any]], paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from matplotlib.colors import LogNorm

    with rasterio.open(paths["projected_surface"]) as dataset:
        surface = dataset.read(1, masked=True).astype("float64")
        bounds = dataset.bounds
        transform = dataset.transform
        dx = abs(transform.a)
        dy = abs(transform.e)
    values = surface.filled(np.nan)
    gy, gx = np.gradient(values, dy, dx)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = math.radians(315.0)
    altitude = math.radians(45.0)
    shade = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    shade = np.clip((shade + 1.0) / 2.0, 0.0, 1.0)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    axis.imshow(shade, extent=extent, origin="upper", cmap="gray", alpha=0.8)
    orders = sorted({record["segorder"] for record in records})
    cmap = plt.get_cmap("viridis", max(1, len(orders)))
    for index, order in enumerate(orders):
        for record in records:
            if record["segorder"] != order:
                continue
            coordinates = np.asarray(record["points"])
            axis.plot(coordinates[:, 0], coordinates[:, 1], color=cmap(index), linewidth=0.7 + 0.35 * order)
        axis.plot([], [], color=cmap(index), linewidth=2, label=f"SegOrder {order}")
    axis.set_title("DHSVM longest-upstream-path SegOrder")
    axis.set_aspect("equal")
    if orders:
        axis.legend(loc="best", fontsize=8)
    fig.savefig(paths["segorder_map"], dpi=180)
    plt.close(fig)

    with rasterio.open(paths["drainage_area_m2"]) as dataset:
        drainage = dataset.read(1, masked=True).astype("float64")
    positive = drainage.compressed()
    positive = positive[np.isfinite(positive) & (positive > 0)]
    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    axis.imshow(shade, extent=extent, origin="upper", cmap="gray", alpha=0.45)
    if positive.size:
        lower = max(float(np.nanpercentile(positive, 2)), float(np.nanmin(positive)))
        upper = max(lower * 1.0001, float(np.nanmax(positive)))
        image = axis.imshow(
            drainage,
            extent=extent,
            origin="upper",
            cmap="magma",
            norm=LogNorm(vmin=lower, vmax=upper),
            alpha=0.72,
        )
        fig.colorbar(image, ax=axis, label="Drainage area (m²)")
    for record in records:
        coordinates = np.asarray(record["points"])
        axis.plot(coordinates[:, 0], coordinates[:, 1], color="#33d6ff", linewidth=0.8)
    axis.set_title("D8/SFD accumulation and extracted network")
    axis.set_aspect("equal")
    fig.savefig(paths["accumulation_map"], dpi=180)
    plt.close(fig)


def validate_outputs(
    *,
    paths: dict[str, Path],
    prep: dict[str, Any],
    topology_qa: dict[str, Any],
    records: list[dict[str, Any]],
    min_finite_coverage: float,
) -> dict[str, Any]:
    import geopandas as gpd
    import rasterio
    from pyproj import CRS

    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"pass": bool(passed), "detail": detail}

    coverage = prep["finite_mask_coverage_fraction"]
    add("finite_mask_coverage", coverage >= min_finite_coverage, coverage)
    crs = CRS.from_user_input(prep["target_crs_wkt"])
    units = {axis.unit_name.casefold() for axis in crs.axis_info if axis.unit_name}
    add("projected_metric_crs", crs.is_projected and all(unit in {"metre", "meter"} for unit in units), crs.to_string())
    add("nonempty_arcs", len(records) > 0, len(records))
    ids = [record["arcid"] for record in records]
    add("unique_arc_ids", len(ids) == len(set(ids)), len(ids))
    add("valid_topology_references", not topology_qa["invalid_references"] and not topology_qa["ambiguous_downstream_nodes"], {
        "invalid_references": topology_qa["invalid_references"],
        "ambiguous_downstream_nodes": topology_qa["ambiguous_downstream_nodes"],
    })
    add("acyclic_assigned_segorder", not topology_qa["has_cycle_or_unresolved_topology"], topology_qa["unassigned_segorder_arcids"])
    add("downstream_order_increase", not topology_qa["segorder_errors"], topology_qa["segorder_errors"])
    network_rows = [line for line in paths["stream_network_dat"].read_text(encoding="utf-8").splitlines() if line.strip()]
    add("network_row_count", len(network_rows) == len(records), {"rows": len(network_rows), "arcs": len(records)})

    readable_errors: list[str] = []
    for name in (
        "projected_surface",
        "flow_direction",
        "accumulation_cells",
        "drainage_area_m2",
        "stream_raster",
        "stream_direction",
    ):
        try:
            with rasterio.open(paths[name]) as dataset:
                if dataset.width <= 0 or dataset.height <= 0 or dataset.count < 1:
                    readable_errors.append(name)
        except Exception as error:
            readable_errors.append(f"{name}: {error}")
    try:
        raw_frame = gpd.read_file(paths["raw_stream_vector"], layer="stream_vector_raw")
        raw_line_count = int(raw_frame.geom_type.isin(["LineString", "MultiLineString"]).sum())
        if raw_line_count < 1:
            readable_errors.append("raw_stream_vector_empty")
        if len(gpd.read_file(paths["dhsvm_gpkg"], layer="topobathy_flownet")) != len(records):
            readable_errors.append("dhsvm_gpkg_row_count")
    except Exception as error:
        readable_errors.append(f"vector: {error}")
    try:
        payload = json.loads(paths["dhsvm_geojson"].read_text(encoding="utf-8"))
        if len(payload.get("features", [])) != len(records):
            readable_errors.append("dhsvm_geojson_row_count")
    except Exception as error:
        readable_errors.append(f"geojson: {error}")
    for name in ("segorder_map", "accumulation_map"):
        if not paths[name].exists() or paths[name].stat().st_size == 0:
            readable_errors.append(name)
    add("readable_outputs", not readable_errors, readable_errors)

    failed = [name for name, result in checks.items() if not result["pass"]]
    return {
        "schema": SCHEMA_VERSION,
        "timestamp_utc": now_utc(),
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "diagnostics": {
            "terminal_segments": topology_qa["terminal_segments"],
            "multiple_terminal_segments": topology_qa["multiple_terminal_segments_diagnostic"],
            "note": "Multiple terminals/sinks are diagnostic and do not fail closed-domain networks.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True, help="Local GeoTIFF or NetCDF topography/topobathymetry surface.")
    parser.add_argument("--surface-variable", help="Raster variable name when a NetCDF contains multiple gridded variables.")
    parser.add_argument(
        "--surface-positive",
        choices=("up", "down"),
        required=True,
        help="Declare the input vertical sign; the projected working surface is always normalized positive-up.",
    )
    parser.add_argument("--mask", type=Path, required=True, help="Local polygon analysis mask.")
    parser.add_argument("--mask-layer", help="Layer name for a multilayer mask dataset.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New or empty run-local output directory.")
    parser.add_argument("--source-area-km2", type=float, required=True, help="Physical source area used to initiate streams.")
    parser.add_argument("--target-crs", help="Optional projected metre CRS; default is local WGS84 UTM from mask centroid.")
    parser.add_argument(
        "--target-resolution-m",
        type=float,
        help="Optional square projected cell size in metres; default preserves source-derived resolution.",
    )
    parser.add_argument("--osgeo-shell", type=Path, default=DEFAULT_OSGEO_SHELL)
    parser.add_argument("--grass-command", default="grass85", help="GRASS 8 executable visible inside OSGeo4W.")
    parser.add_argument("--grass-memory-mb", type=int, default=4096)
    parser.add_argument("--node-snap-m", type=float, help="Endpoint snap tolerance; default is min(0.1 m, 1%% of nominal cell size).")
    parser.add_argument("--min-finite-coverage", type=float, default=0.99)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = now_utc()
    start_clock = time.perf_counter()
    surface = args.surface.expanduser().resolve()
    mask_path = args.mask.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    paths = absolute_paths(out_dir)
    run_dir_claimed = False
    manifest: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "started_utc": started,
        "status": "running",
        "method": {
            "routing": "GRASS 8 D8/SFD r.watershed -s -a",
            "extraction": "r.stream.extract",
            "mexp": 0,
            "stream_length": 0,
            "segorder": "headwaters=1; downstream=1+max(direct upstream)",
        },
        "inputs": {},
        "parameters": {},
        "outputs": {name: str(path) for name, path in paths.items()},
        "commands": [],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": {
                name: package_version(name)
                for name in ("geopandas", "matplotlib", "numpy", "pyproj", "rasterio", "shapely", "xarray")
            },
        },
    }
    try:
        if not surface.is_file():
            raise FileNotFoundError(f"Surface not found: {surface}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        if not args.osgeo_shell.is_file():
            raise FileNotFoundError(f"OSGeo4W shell not found: {args.osgeo_shell}")
        if not (0 < args.min_finite_coverage <= 1):
            raise ValueError("--min-finite-coverage must be in (0, 1]")
        if args.target_resolution_m is not None and args.target_resolution_m <= 0:
            raise ValueError("--target-resolution-m must be positive")
        if args.grass_memory_mb <= 0:
            raise ValueError("--grass-memory-mb must be positive")
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"Refusing implicit reuse: output directory is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir_claimed = True

        manifest["inputs"] = {
            "surface": {"path": str(surface), "sha256": sha256_file(surface), "variable": args.surface_variable},
            "mask": {
                "path": str(mask_path),
                "sha256": sha256_file(mask_path),
                "layer": args.mask_layer,
            },
        }
        input_mask = load_mask(mask_path, args.mask_layer)
        target_crs = choose_target_crs(input_mask, args.target_crs)
        prep = prepare_projected_inputs(
            surface_path=surface,
            surface_variable=args.surface_variable,
            surface_positive=args.surface_positive,
            mask=input_mask,
            target_crs=target_crs,
            target_resolution_m=args.target_resolution_m,
            projected_surface=paths["projected_surface"],
            projected_mask=paths["projected_mask"],
        )
        threshold_cells = source_area_to_cells(args.source_area_km2, prep["cell_area_m2"])
        realized_area_km2 = threshold_cells * prep["cell_area_m2"] / 1_000_000.0
        node_snap_m = args.node_snap_m
        if node_snap_m is None:
            node_snap_m = max(0.001, min(0.1, prep["nominal_cell_size_m"] * 0.01))
        if node_snap_m <= 0:
            raise ValueError("--node-snap-m must be positive")
        manifest["parameters"] = {
            "source_area_km2": args.source_area_km2,
            "threshold_cells": threshold_cells,
            "realized_source_area_km2": realized_area_km2,
            "cell_area_m2": prep["cell_area_m2"],
            "target_crs": prep["target_crs"],
            "target_resolution_m": args.target_resolution_m,
            "surface_positive_input": args.surface_positive,
            "surface_positive_working": "up",
            "node_snap_m": node_snap_m,
            "min_finite_coverage": args.min_finite_coverage,
            "grass_memory_mb": args.grass_memory_mb,
        }
        manifest["preprocessing"] = prep
        if prep["finite_mask_coverage_fraction"] < args.min_finite_coverage:
            raise RuntimeError(
                f"Finite mask coverage {prep['finite_mask_coverage_fraction']:.6f} is below "
                f"{args.min_finite_coverage:.6f}"
            )

        grass_location = out_dir / "grassdata" / "topobathy_flownet"
        manifest["runtime"]["gdal"] = probe_version(args.osgeo_shell, ["gdalinfo", "--version"], out_dir)
        manifest["runtime"]["grass"] = probe_version(args.osgeo_shell, [args.grass_command, "--version"], out_dir)
        run_grass(
            osgeo_shell=args.osgeo_shell.resolve(),
            grass_command=args.grass_command,
            grass_location=grass_location,
            paths=paths,
            threshold_cells=threshold_cells,
            cell_area_m2=prep["cell_area_m2"],
            memory_mb=args.grass_memory_mb,
            cwd=out_dir,
            command_records=manifest["commands"],
        )

        lines, raw_crs = read_raw_lines(paths["raw_stream_vector"])
        records, topology_qa = build_arc_records(
            lines,
            raster_sampler(paths["projected_surface"]),
            raster_sampler(paths["accumulation_cells"]),
            cell_area_m2=prep["cell_area_m2"],
            node_tolerance_m=node_snap_m,
        )
        topology_qa.update(
            {
                "schema": SCHEMA_VERSION,
                "timestamp_utc": now_utc(),
                "required_fields": [
                    "arcid",
                    "from_node",
                    "to_node",
                    "local",
                    "downarc",
                    "uparc",
                    "SELEV",
                    "EELEV",
                    "MAXGRID",
                    "dz",
                    "slope",
                    "meanmsq",
                    "segorder",
                    "drainage_area_m2",
                ],
            }
        )
        write_network_products(records, raw_crs or target_crs, paths)
        topology_qa["stream_network_row_count_matches_arc_count"] = (
            len([line for line in paths["stream_network_dat"].read_text(encoding="utf-8").splitlines() if line.strip()])
            == len(records)
        )
        write_json(paths["topology_qa"], topology_qa)
        create_qa_maps(records, paths)
        health = validate_outputs(
            paths=paths,
            prep=prep,
            topology_qa=topology_qa,
            records=records,
            min_finite_coverage=args.min_finite_coverage,
        )
        write_json(paths["health_check"], health)
        manifest["structural_status"] = health["status"]
        manifest["status"] = "complete" if health["status"] == "pass" else "failed_health_check"
        manifest["topology_summary"] = {
            key: topology_qa[key]
            for key in (
                "arc_count",
                "node_count",
                "headwater_segments",
                "terminal_segments",
                "multiple_terminal_segments_diagnostic",
            )
        }
        return_code = 0 if health["status"] == "pass" else 2
    except Exception as error:
        manifest["status"] = "failed"
        manifest["structural_status"] = "not_evaluated"
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        if run_dir_claimed and not paths["health_check"].exists():
            write_json(
                paths["health_check"],
                {
                    "schema": SCHEMA_VERSION,
                    "timestamp_utc": now_utc(),
                    "status": "fail",
                    "checks": {
                        "process_completion": {
                            "pass": False,
                            "detail": f"{type(error).__name__}: {error}",
                        }
                    },
                    "failed_checks": ["process_completion"],
                },
            )
        return_code = 1
    finally:
        manifest["finished_utc"] = now_utc()
        manifest["duration_seconds"] = round(time.perf_counter() - start_clock, 6)
        if run_dir_claimed:
            write_json(paths["manifest"], manifest)
    if return_code:
        print(json.dumps({"status": manifest["status"], "manifest": str(paths["manifest"])}, indent=2), file=sys.stderr)
    else:
        print(json.dumps({"status": manifest["status"], "manifest": str(paths["manifest"])}, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
