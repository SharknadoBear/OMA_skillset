"""Build a DHSVM-compatible SegOrder channel/thalweg network with GRASS GIS 8."""

from __future__ import annotations

import argparse
import json
import math
import platform
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


def osgeo_command(osgeo_shell: Path, args: Sequence[str]) -> list[str]:
    command = "call " + subprocess.list2cmdline([str(osgeo_shell), *map(str, args)])
    return ["cmd", "/d", "/s", "/c", command]


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
    result = subprocess.run(
        osgeo_command(osgeo_shell, args),
        cwd=cwd,
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
        result = subprocess.run(
            osgeo_command(osgeo_shell, args),
            cwd=cwd,
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
                raise ValueError("--surface-variable is only valid for a NetCDF/HDF subdataset container")
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
    from rasterio.mask import mask as raster_mask
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    dataset_name = resolve_surface_dataset(surface_path, surface_variable)
    with rasterio.open(dataset_name) as source:
        if source.crs is None:
            raise ValueError("Input surface has no CRS")
        source_mask = mask.to_crs(source.crs)
        source_geometry = source_mask.geometry.iloc[0]
        clipped, clipped_transform = raster_mask(
            source,
            [source_geometry.__geo_interface__],
            indexes=1,
            crop=True,
            filled=False,
        )
        source_values = clipped.astype("float64").filled(np.nan)
        if surface_positive == "down":
            source_values = -source_values
        source_height, source_width = source_values.shape
        left, bottom, right, top = array_bounds(source_height, source_width, clipped_transform)
        target_transform, target_width, target_height = calculate_default_transform(
            source.crs,
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
            src_crs=source.crs,
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
                for name in ("geopandas", "matplotlib", "numpy", "pyproj", "rasterio", "shapely")
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
