#!/usr/bin/env python3
"""Validate immutable Lake Superior boundary and bathymetry preparation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely import contains_xy, make_valid
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import split, unary_union
import xarray as xr


EXPECTED_CHART_DATUM_M = 183.2
SUPERIOR_REFERENCE = Point(-87.20, 47.50)
DOWNSTREAM_REFERENCE = Point(-82.50, 44.80)
PROJECTED_CRS = "EPSG:32616"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(workspace_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def _polygon_parts(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    valid = make_valid(geometry)
    if isinstance(valid, Polygon):
        return [valid]
    if isinstance(valid, MultiPolygon):
        return [value for value in valid.geoms if not value.is_empty]
    if hasattr(valid, "geoms"):
        return [
            polygon
            for value in valid.geoms
            for polygon in _polygon_parts(value)
        ]
    return []


def _projected_area_m2(geometry: Any) -> float:
    return float(
        gpd.GeoSeries([geometry], crs="EPSG:4326")
        .to_crs(PROJECTED_CRS)
        .iloc[0]
        .area
    )


def _require_fresh_output_file(output_path: Path) -> None:
    """Reject an existing readiness report before reading preparation inputs."""
    if output_path.exists():
        raise FileExistsError(
            "Lake Superior readiness output must not already exist: "
            f"{output_path}"
        )


def _conversion_report_path(
    bathy_path: Path,
    bathy: xr.Dataset,
) -> Path:
    """Resolve fresh declared metadata with a frozen-v2 compatibility fallback."""

    declared = str(
        bathy.attrs.get("lake_depth_conversion_metadata_json", "")
    ).strip()
    if declared:
        path = Path(declared)
        if not path.is_absolute():
            path = bathy_path.parent / path
        return path.resolve()
    return (
        bathy_path.parent / "lake_superior_depth_conversion_v2.json"
    ).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--bound-case-manifest-output",
        help=(
            "Optional fresh manifest copy bound to the new readiness artifact; "
            "written only when every validation check passes."
        ),
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    case_path = _resolve(workspace_root, args.case_manifest)
    output_path = _resolve(workspace_root, args.output)
    _require_fresh_output_file(output_path)
    bound_case_path = (
        _resolve(workspace_root, args.bound_case_manifest_output)
        if args.bound_case_manifest_output
        else None
    )
    if bound_case_path is not None:
        _require_fresh_output_file(bound_case_path)
        if bound_case_path in {case_path, output_path}:
            raise ValueError(
                "Bound case manifest must be a fresh file distinct from "
                "the source case manifest and readiness output."
            )
    case = _load_json(case_path)
    boundary_manifest_path = _resolve(
        workspace_root,
        case["boundary"]["resolution_manifest"],
    )
    bathy_path = _resolve(workspace_root, case["bathymetry"]["netcdf"])
    boundary_manifest = _load_json(boundary_manifest_path)
    gpkg_path = Path(
        boundary_manifest["outputs"]["boundary_resolution_gpkg"]
    ).resolve()
    topology_path = Path(
        boundary_manifest["outputs"]["boundary_resolution_diagnostics_json"]
    ).resolve()
    topology = _load_json(topology_path)

    domain_frame = gpd.read_file(gpkg_path, layer="resolved_domain_polygon")
    island_frame = gpd.read_file(gpkg_path, layer="resolved_island_polygons")
    boundary_nodes = gpd.read_file(gpkg_path, layer="boundary_nodes")
    open_boundaries = gpd.read_file(gpkg_path, layer="resolved_open_boundary")
    gate_frame = gpd.read_file(gpkg_path, layer="numerical_land_gate")
    selected_shell_frame = gpd.read_file(
        gpkg_path,
        layer="source_l2_lake_polygon",
    )
    if len(domain_frame) != 1:
        raise RuntimeError(f"Expected one resolved domain, found {len(domain_frame)}.")
    domain = domain_frame.geometry.iloc[0]
    if len(gate_frame) != 1 or len(selected_shell_frame) != 1:
        raise RuntimeError(
            "Expected exactly one numerical gate and one retained L2 shell."
        )
    gate = gate_frame.geometry.iloc[0]
    selected_shell = selected_shell_frame.geometry.iloc[0]
    if not isinstance(gate, LineString):
        raise RuntimeError("Numerical land gate must be one LineString.")

    source_manifest_path = Path(
        boundary_manifest["inputs"]["gshhg_source_cache_manifest"]
    ).resolve()
    source_manifest = _load_json(source_manifest_path)
    source_files = [
        Path(record["path"]).resolve()
        for record in source_manifest["source_files"]
    ]
    level2_path = next(
        path
        for path in source_files
        if path.suffix.lower() == ".shp" and path.stem.endswith("_L2")
    )
    level3_path = next(
        path
        for path in source_files
        if path.suffix.lower() == ".shp" and path.stem.endswith("_L3")
    )
    level2 = gpd.read_file(level2_path).to_crs("EPSG:4326")
    level3 = gpd.read_file(level3_path).to_crs("EPSG:4326")
    joined_source_candidates = [
        polygon
        for geometry in level2.geometry
        for polygon in _polygon_parts(geometry)
        if polygon.covers(SUPERIOR_REFERENCE)
        and polygon.covers(DOWNSTREAM_REFERENCE)
    ]
    if len(joined_source_candidates) != 1:
        raise RuntimeError(
            "Frozen GSHHG L2 must yield exactly one joined upper-lakes source."
        )
    joined_source = joined_source_candidates[0]

    gate_coords = list(gate.coords)
    start = np.asarray(gate_coords[0], dtype=float)
    end = np.asarray(gate_coords[-1], dtype=float)
    direction = end - start
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0.0:
        raise RuntimeError("Numerical land gate has zero length.")
    direction /= direction_norm
    splitter = LineString(
        [
            tuple(start - 0.002 * direction),
            tuple(end + 0.002 * direction),
        ]
    )
    split_parts = [
        value
        for value in split(joined_source, splitter).geoms
        if isinstance(value, Polygon) and value.area > 1.0e-10
    ]
    reconstructed_shell = next(
        (
            value
            for value in split_parts
            if value.covers(SUPERIOR_REFERENCE)
        ),
        None,
    )
    downstream_shell = next(
        (
            value
            for value in split_parts
            if value.covers(DOWNSTREAM_REFERENCE)
        ),
        None,
    )
    references_separated = bool(
        reconstructed_shell is not None
        and downstream_shell is not None
        and reconstructed_shell is not downstream_shell
    )
    if reconstructed_shell is None:
        raise RuntimeError("Frozen-source gate split did not retain Superior.")

    selected_l3_records: list[dict[str, Any]] = []
    selected_l3_polygons: list[Polygon] = []
    gate_l3_intersections: list[str] = []
    all_joined_l3: list[Polygon] = []
    for row_index, row in level3.iterrows():
        source_id = str(row.get("id", row.get("ID", row_index)))
        for part_index, polygon in enumerate(_polygon_parts(row.geometry)):
            if joined_source.intersects(polygon):
                all_joined_l3.append(polygon)
            if gate.intersects(polygon):
                gate_l3_intersections.append(source_id)
            if not reconstructed_shell.covers(
                polygon.representative_point()
            ):
                continue
            outside_fraction = (
                float(
                    polygon.difference(reconstructed_shell).area
                    / polygon.area
                )
                if polygon.area > 0.0
                else 1.0
            )
            if outside_fraction > 1.0e-8:
                continue
            selected_l3_polygons.append(polygon)
            selected_l3_records.append(
                {
                    "source_row": int(row_index),
                    "source_id": source_id,
                    "source_part": int(part_index),
                    "bounds": list(map(float, polygon.bounds)),
                }
            )
    expected_domain = make_valid(
        reconstructed_shell.difference(
            unary_union(selected_l3_polygons)
            if selected_l3_polygons
            else Polygon()
        )
    )
    expected_parts = _polygon_parts(expected_domain)
    expected_domain = next(
        (
            value
            for value in expected_parts
            if value.covers(SUPERIOR_REFERENCE)
        ),
        None,
    )
    if expected_domain is None:
        raise RuntimeError("Frozen-source reconstruction lost Lake Superior.")
    expected_island_holes = len(expected_domain.interiors)
    delivered_inventory = boundary_manifest.get(
        "gshhg_l3_island_source_inventory",
        {},
    )
    delivered_inventory_ids = sorted(
        str(value)
        for value in delivered_inventory.get("source_ids", [])
    )
    reconstructed_inventory_ids = sorted(
        {value["source_id"] for value in selected_l3_records}
    )
    shell_difference_m2 = _projected_area_m2(
        selected_shell.symmetric_difference(reconstructed_shell)
    )
    domain_difference_m2 = _projected_area_m2(
        domain.symmetric_difference(expected_domain)
    )
    joined_l3_union = (
        unary_union(all_joined_l3) if all_joined_l3 else Polygon()
    )
    gate_l3_intersection_length_m = float(
        gpd.GeoSeries(
            [gate.intersection(joined_l3_union)],
            crs="EPSG:4326",
        )
        .to_crs(PROJECTED_CRS)
        .iloc[0]
        .length
    )

    with xr.open_dataset(bathy_path, decode_times=False) as opened:
        bathy = opened.load()
    if "depth_m" not in bathy or "elevation_m" not in bathy:
        raise RuntimeError("Bathymetry must contain both elevation_m and depth_m.")
    depth = np.asarray(bathy["depth_m"].values, dtype=float)
    elevation = np.asarray(bathy["elevation_m"].values, dtype=float)
    if "fvcom_wet_domain_mask" not in bathy:
        raise RuntimeError("Bathymetry lacks fvcom_wet_domain_mask.")

    longitude_name = next(
        (name for name in ("longitude", "lon", "x") if name in bathy.coords),
        None,
    )
    latitude_name = next(
        (name for name in ("latitude", "lat", "y") if name in bathy.coords),
        None,
    )
    if longitude_name is None or latitude_name is None:
        raise RuntimeError("Bathymetry lacks recognizable longitude/latitude coordinates.")
    longitude = np.asarray(bathy[longitude_name].values, dtype=float)
    latitude = np.asarray(bathy[latitude_name].values, dtype=float)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    calculated_wet_mask = np.asarray(
        contains_xy(domain, lon_grid, lat_grid),
        dtype=bool,
    )
    delivered_wet_mask = (
        np.asarray(bathy["fvcom_wet_domain_mask"].values, dtype=int) == 1
    )
    wet_mask_exact = bool(
        np.array_equal(calculated_wet_mask, delivered_wet_mask)
    )
    finite = np.isfinite(depth)
    wet_finite = finite[calculated_wet_mask]
    wet_positive = depth[calculated_wet_mask] > 0.0
    raster_bounds = [
        float(np.nanmin(longitude)),
        float(np.nanmin(latitude)),
        float(np.nanmax(longitude)),
        float(np.nanmax(latitude)),
    ]
    domain_bounds = list(map(float, domain.bounds))
    bounds_cover = bool(
        raster_bounds[0] <= domain_bounds[0]
        and raster_bounds[1] <= domain_bounds[1]
        and raster_bounds[2] >= domain_bounds[2]
        and raster_bounds[3] >= domain_bounds[3]
    )
    if not bounds_cover:
        raise RuntimeError(
            f"Bathymetry bounds {raster_bounds} do not cover domain {domain_bounds}."
        )

    expected_depth = np.maximum(EXPECTED_CHART_DATUM_M - elevation, 0.1)
    conversion_error = float(
        np.nanmax(
            np.abs(
                depth[calculated_wet_mask]
                - expected_depth[calculated_wet_mask]
            )
        )
    )
    if conversion_error > 1.0e-4:
        raise RuntimeError(
            f"Positive-down conversion mismatch: maximum error {conversion_error} m."
        )
    caveat = str(bathy.attrs.get("vertical_datum_caveat", ""))
    if not all(token in caveat for token in ("EGM2008", "IGLD 1985", "no geodetic")):
        raise RuntimeError("Bathymetry is missing the explicit vertical-datum caveat.")

    hard_anchor_count = int(boundary_nodes["is_hard_anchor"].astype(bool).sum())
    artificial_vertex_count = int(
        (~boundary_nodes["is_source_vertex"].astype(bool)).sum()
    )
    artificial_nodes = boundary_nodes[
        ~boundary_nodes["is_source_vertex"].astype(bool)
    ]
    artificial_points = list(artificial_nodes.geometry)
    gate_endpoints_recovered = bool(
        len(artificial_points) == 2
        and all(
            min(point.distance(Point(endpoint)) for point in artificial_points)
            <= 1.0e-7
            for endpoint in (start, end)
        )
    )
    checks = {
        "case_id_is_lake_superior": case["case_id"] == "lake_superior",
        "domain_is_valid": bool(domain.is_valid),
        "wet_component_count_is_one": (
            topology["topology"]["wet_component_count"] == 1
        ),
        "gate_is_inside_joined_l2_water": (
            gate.difference(joined_source).length <= 1.0e-10
        ),
        "gate_endpoints_are_shoreline_snapped": (
            Point(start).distance(joined_source.boundary) <= 1.0e-7
            and Point(end).distance(joined_source.boundary) <= 1.0e-7
        ),
        "gate_has_zero_l3_land_intersection": (
            not gate_l3_intersections
            and gate_l3_intersection_length_m <= 1.0e-6
        ),
        "gate_separates_superior_and_downstream_references": (
            references_separated
        ),
        "retained_l2_shell_matches_frozen_source_split": (
            shell_difference_m2 <= 1.0
        ),
        "domain_matches_frozen_l2_l3_reconstruction": (
            domain_difference_m2 <= 1.0
        ),
        "domain_hole_count_matches_frozen_l3": (
            len(domain.interiors) == expected_island_holes
        ),
        "island_layer_count_matches_frozen_l3": (
            len(island_frame) == expected_island_holes
        ),
        "manifest_island_count_matches_frozen_l3": (
            boundary_manifest["qa"]["resolved_island_count"]
            == expected_island_holes
        ),
        "case_expected_island_count_matches_frozen_l3": (
            case["boundary"]["expected_island_holes"]
            == expected_island_holes
        ),
        "manifest_l3_inventory_count_matches_frozen_source": (
            delivered_inventory.get("record_count")
            == len(selected_l3_records)
        ),
        "manifest_l3_inventory_ids_match_frozen_source": (
            delivered_inventory_ids == reconstructed_inventory_ids
        ),
        "open_boundary_layer_is_empty": len(open_boundaries) == 0,
        "manifest_open_boundary_count_is_zero": (
            boundary_manifest["qa"]["open_boundary_chain_count"] == 0
        ),
        "manifest_open_boundaries_is_empty": (
            boundary_manifest["open_boundaries"] == []
        ),
        "case_open_boundaries_is_empty": case["boundary"]["open_boundaries"] == [],
        "closure_has_two_hard_anchors": hard_anchor_count == 2,
        "closure_has_two_artificial_vertices": artificial_vertex_count == 2,
        "closure_artificial_vertices_match_gate_endpoints": (
            gate_endpoints_recovered
        ),
        "bathymetry_wet_mask_matches_domain": wet_mask_exact,
        "bathymetry_all_wet_cells_finite": bool(wet_finite.all()),
        "bathymetry_all_wet_cells_positive": bool(wet_positive.all()),
        "bathymetry_covers_domain_bounds": bounds_cover,
        "chart_datum_is_183_2_m": (
            float(bathy.attrs.get("lake_chart_datum_m", np.nan))
            == EXPECTED_CHART_DATUM_M
        ),
        "depth_conversion_matches_declared_policy": conversion_error <= 1.0e-4,
        "vertical_datum_caveat_is_explicit": all(
            token in caveat for token in ("EGM2008", "IGLD 1985", "no geodetic")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    artifacts = {
        "case_manifest": case_path,
        "boundary_manifest": boundary_manifest_path,
        "boundary_gpkg": gpkg_path,
        "boundary_topology_report": topology_path,
        "boundary_nodes_geojson": Path(
            boundary_manifest["outputs"]["boundary_resolution_nodes_geojson"]
        ).resolve(),
        "boundary_review_map": Path(
            boundary_manifest["outputs"]["boundary_resolution_review_map"]
        ).resolve(),
        "boundary_gate_evidence_map": Path(
            boundary_manifest["outputs"]["outlet_gate_evidence_map"]
        ).resolve(),
        "gshhg_source_manifest": source_manifest_path,
        "gshhg_level2_shapefile": level2_path,
        "gshhg_level3_shapefile": level3_path,
        "bathymetry_netcdf": bathy_path,
        "bathymetry_conversion_report": _conversion_report_path(
            bathy_path,
            bathy,
        ),
        "bathymetry_fetch_metadata": bathy_path.parent
        / "lake_superior_etopo_metadata.json",
        "bathymetry_request": bathy_path.parent / "request.json",
        "bathymetry_download_estimate": bathy_path.parent / "download_estimate.json",
    }
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        failed.extend(f"missing_artifact:{name}" for name in missing)
    hashes = {
        name: {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for name, path in artifacts.items()
        if path.exists()
    }
    payload = {
        "schema_version": "lake_superior_preparation_readiness_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if not failed else "needs_review",
        "case_id": "lake_superior",
        "checks": checks,
        "failed_checks": failed,
        "topology": {
            "wet_component_count": topology["topology"]["wet_component_count"],
            "island_hole_count": len(domain.interiors),
            "frozen_l3_expected_hole_count": expected_island_holes,
            "frozen_l3_source_ids": reconstructed_inventory_ids,
            "open_boundary_chain_count": len(open_boundaries),
            "hard_anchor_count": hard_anchor_count,
            "artificial_closure_vertex_count": artificial_vertex_count,
            "wet_area_km2": topology["topology"]["wet_area_km2"],
            "st_marys_closure_length_m": topology["topology"][
                "st_marys_closure_length_m"
            ],
            "st_marys_gate_l3_intersection_length_m": (
                gate_l3_intersection_length_m
            ),
            "retained_shell_symmetric_difference_m2": (
                shell_difference_m2
            ),
            "wet_domain_symmetric_difference_m2": domain_difference_m2,
        },
        "bathymetry": {
            "raster_bounds_wsen": raster_bounds,
            "domain_bounds_wsen": domain_bounds,
            "wet_domain_cell_count": int(calculated_wet_mask.sum()),
            "wet_finite_cell_count": int(wet_finite.sum()),
            "wet_depth_min_m": float(
                np.nanmin(depth[calculated_wet_mask])
            ),
            "wet_depth_max_m": float(
                np.nanmax(depth[calculated_wet_mask])
            ),
            "maximum_conversion_error_m": conversion_error,
            "chart_datum_m": EXPECTED_CHART_DATUM_M,
            "source_vertical_reference": "EGM2008",
            "target_vertical_reference": "IGLD 1985",
            "vertical_transform_applied": False,
        },
        "artifact_hashes": hashes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if bound_case_path is not None and not failed:
        bound_case = json.loads(json.dumps(case))
        try:
            readiness_path_value = str(
                output_path.relative_to(workspace_root)
            )
        except ValueError:
            readiness_path_value = str(output_path)
        bound_case.setdefault("readiness", {})[
            "validation_artifact"
        ] = {
            "path": readiness_path_value,
            "sha256": _sha256(output_path),
            "schema_version": payload["schema_version"],
            "required_status": "ready",
            "required_checks": sorted(checks),
            "required_input_hashes": [
                "boundary_manifest",
                "boundary_gpkg",
                "bathymetry_netcdf",
            ],
        }
        bound_case_path.parent.mkdir(parents=True, exist_ok=True)
        bound_case_path.write_text(
            json.dumps(bound_case, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
