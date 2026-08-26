"""Build continuous model-boundary loops from an FVCOM boundary-arc package."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, mapping
from shapely.ops import unary_union

from .projection import local_utm_projection, project_geometry, unproject_geometry


def build_model_boundary_loops(
    bdry_arc_gpkg: str | Path,
    manifest_json: str | Path,
    run_dir: str | Path,
    name: str,
    target_resolution_m: float | None = None,
    min_island_area_m2: float = 0.0,
    mode: str = "execute",
) -> dict[str, Any]:
    """Build a continuous exterior loop and island rings from a boundary-arc package."""
    if mode not in {"execute", "test"}:
        raise ValueError("--mode must be execute or test")
    bdry_arc_gpkg = Path(bdry_arc_gpkg)
    manifest_json = Path(manifest_json)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = _read_json(manifest_json) if manifest_json.exists() else {}
    target_resolution_m = float(
        target_resolution_m
        if target_resolution_m is not None
        else source_manifest.get("settings", {}).get("target_resolution_m", 250.0)
    )
    min_island_area_m2 = float(min_island_area_m2)

    wet_gdf = _read_layer(bdry_arc_gpkg, "wet_domain")
    if wet_gdf.empty:
        raise ValueError("bdry_arc_package.gpkg does not contain a non-empty wet_domain layer")
    domain_lonlat = _largest_polygon(wet_gdf.geometry)
    bbox = tuple(float(v) for v in domain_lonlat.bounds)
    projection = local_utm_projection(bbox)
    domain_xy = project_geometry(domain_lonlat, projection).buffer(0)
    if not isinstance(domain_xy, Polygon) or domain_xy.is_empty:
        raise ValueError("wet_domain layer does not contain a valid Polygon")

    open_gdf = _read_layer(bdry_arc_gpkg, "open_boundary_arc")
    land_gdf = _read_layer(bdry_arc_gpkg, "land_boundary_arcs")
    frame_gdf = _read_layer(bdry_arc_gpkg, "frame_clip_boundary_arcs")
    land_patch_gdf = _read_layer(bdry_arc_gpkg, "land_patch_boundary_arcs")
    coast_gdf = _read_layer(bdry_arc_gpkg, "coastline_repaired")
    if coast_gdf.empty:
        coast_gdf = _read_layer(bdry_arc_gpkg, "coastline_raw")

    open_xy = _line_union_xy(open_gdf, projection)
    land_xy = _line_union_xy(land_gdf, projection)
    frame_xy = _line_union_xy(frame_gdf, projection)
    land_patch_xy = _line_union_xy(land_patch_gdf, projection)
    exterior_xy = LineString(domain_xy.exterior.coords)
    tolerance_m = max(2.0 * target_resolution_m, 50.0)
    segments_xy = _classify_exterior_segments(
        exterior_xy,
        open_xy,
        land_xy,
        frame_xy,
        land_patch_xy,
        tolerance_m,
    )
    island_polygons_xy, island_lines_xy = _extract_islands(domain_xy, min_island_area_m2)

    outer_closed = bool(Point(exterior_xy.coords[0]).distance(Point(exterior_xy.coords[-1])) <= max(1.0, 0.01 * target_resolution_m))
    open_overlap_fraction = _line_fraction_near(open_xy, exterior_xy, tolerance_m)
    class_lengths = _class_lengths(segments_xy)
    unclassified_threshold_m = max(2.0 * target_resolution_m, 0.001 * max(exterior_xy.length, 1.0))
    frame_clip_policy = str(source_manifest.get("settings", {}).get("frame_clip_policy", "reject-unintended"))
    residual_boundary_policy = str(
        source_manifest.get("settings", {}).get("residual_boundary_policy", "strict-reject")
    )
    coastline_source = str(source_manifest.get("inputs", {}).get("coastline_source", ""))
    configured_frame_tolerance = source_manifest.get("settings", {}).get("frame_clip_tolerance_m")
    frame_clip_tolerance_m = float(
        configured_frame_tolerance
        if configured_frame_tolerance is not None
        else max(250.0, 0.05 * target_resolution_m)
    )
    source_frame_clip_length_m = float(
        source_manifest.get("wet_domain", {}).get("frame_clip_boundary_length_m", 0.0) or 0.0
    )
    classified_frame_clip_length_m = float(class_lengths.get("frame_clip_boundary", 0.0))
    gate_frame_clip_length_m = max(source_frame_clip_length_m, classified_frame_clip_length_m)
    landward_length_m = max(float(exterior_xy.length) - float(getattr(open_xy, "length", 0.0)), 1.0)
    unintended_frame_clip_fraction = float(gate_frame_clip_length_m / landward_length_m)
    intended_exterior_coverage_fraction = float(
        max(0.0, min(1.0, 1.0 - gate_frame_clip_length_m / max(float(exterior_xy.length), 1.0)))
    )
    failures: list[str] = []
    if source_manifest.get("final_status") == "needs_review":
        failures.append("upstream_bdry_arc_needs_review")
    if not outer_closed:
        failures.append("model_outer_boundary_not_closed")
    closure_method = source_manifest.get("wet_domain", {}).get("closure_method")
    lake_no_open_boundary = closure_method == "lake_closed_boundary_no_open_arc" or bool(source_manifest.get("wet_domain", {}).get("no_ocean_open_boundary"))
    land_patch_fraction = float(source_manifest.get("wet_domain", {}).get("land_patch_boundary_fraction", 0.0) or 0.0)
    open_overlap_threshold = 0.0 if lake_no_open_boundary else (0.90 if closure_method == "island_archipelago_offshore_loop" and land_patch_fraction > 0.0 else 0.98)
    if not lake_no_open_boundary and open_overlap_fraction < open_overlap_threshold:
        failures.append("open_boundary_not_sufficiently_on_model_exterior")
    if class_lengths.get("unclassified_outer_boundary", 0.0) > unclassified_threshold_m:
        failures.append("unclassified_outer_boundary_length_nontrivial")
    frame_gate_enabled = frame_clip_policy == "reject-unintended" and coastline_source == "gshhs"
    if frame_clip_policy == "report-only":
        failures.append("diagnostic_only_report_only_policy")
    if frame_gate_enabled and residual_boundary_policy == "solid-default" and gate_frame_clip_length_m > 0.0:
        failures.append("residual_boundary_role_pending")
    elif frame_gate_enabled:
        if gate_frame_clip_length_m > frame_clip_tolerance_m:
            failures.append("residual_open_exterior_length_nontrivial")
        if (
            unintended_frame_clip_fraction > 0.001
            or intended_exterior_coverage_fraction < 0.999
        ):
            failures.append("residual_open_exterior_fraction_or_coverage_failed")
    final_status = "pass" if not failures else "needs_review"

    layers = _build_layers(
        domain_xy,
        exterior_xy,
        segments_xy,
        island_polygons_xy,
        island_lines_xy,
        open_xy,
        projection,
        final_status,
    )
    outputs = _write_outputs(run_dir, layers, name)
    map_path = run_dir / "model_boundary_colored_map.png"
    _plot_boundary_map(map_path, layers, coast_gdf, domain_lonlat, final_status, len(island_polygons_xy))

    manifest = {
        "schema_version": "fvcom_model_boundary_loops_v1",
        "name": name,
        "created_by": "fvcom-bdry-arc build_model_boundary_loops.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "final_status": final_status,
        "failure_taxonomy": failures,
        "inputs": {
            "bdry_arc_gpkg": str(bdry_arc_gpkg),
            "manifest_json": str(manifest_json),
            "upstream_final_status": source_manifest.get("final_status"),
            "upstream_failure_taxonomy": source_manifest.get("failure_taxonomy", []),
        },
        "settings": {
            "target_resolution_m": float(target_resolution_m),
            "min_island_area_m2": float(min_island_area_m2),
            "mode": mode,
            "classification_tolerance_m": float(tolerance_m),
            "unclassified_length_threshold_m": float(unclassified_threshold_m),
            "open_boundary_overlap_threshold": float(open_overlap_threshold),
            "lake_no_open_boundary": bool(lake_no_open_boundary),
            "frame_clip_policy": frame_clip_policy,
            "residual_boundary_policy": residual_boundary_policy,
            "frame_clip_tolerance_m": float(frame_clip_tolerance_m),
            "frame_clip_fraction_threshold": 0.001,
            "intended_exterior_coverage_threshold": 0.999,
            "frame_clip_gate_enabled": bool(frame_gate_enabled),
        },
        "qa": {
            "outer_boundary_closed": outer_closed,
            "outer_boundary_length_m": float(exterior_xy.length),
            "open_boundary_exterior_overlap_fraction": float(open_overlap_fraction),
            "land_outer_boundary_length_m": float(class_lengths.get("land_outer_boundary", 0.0)),
            "land_patch_boundary_length_m": float(class_lengths.get("land_patch_boundary", 0.0)),
            "frame_clip_boundary_length_m": float(class_lengths.get("frame_clip_boundary", 0.0)),
            "source_frame_clip_boundary_length_m": float(source_frame_clip_length_m),
            "gate_frame_clip_boundary_length_m": float(gate_frame_clip_length_m),
            "unintended_frame_clip_fraction": float(unintended_frame_clip_fraction),
            "intended_exterior_coverage_fraction": float(intended_exterior_coverage_fraction),
            "open_boundary_length_m": float(class_lengths.get("open_boundary", 0.0)),
            "unclassified_outer_boundary_length_m": float(class_lengths.get("unclassified_outer_boundary", 0.0)),
            "island_count": int(len(island_polygons_xy)),
            "largest_island_area_m2": float(max((poly.area for poly in island_polygons_xy), default=0.0)),
            "model_domain_area_m2": float(domain_xy.area),
        },
        "open_boundary_lineage": {
            "adaptive_source_layer": "delivered_open_boundary_arc",
            "compatibility_alias_layer": "source_open_boundary_arc",
            "consumption_policy": "exact_delivered_geometry_not_proximity_classification",
            "delivered_open_boundary_length_m": float(getattr(open_xy, "length", 0.0)),
            "proximity_classified_open_boundary_length_m": float(
                class_lengths.get("open_boundary", 0.0)
            ),
            "proximity_classified_excess_length_m": float(
                max(
                    0.0,
                    class_lengths.get("open_boundary", 0.0)
                    - float(getattr(open_xy, "length", 0.0)),
                )
            ),
            "delivered_source_start_position_m": source_manifest.get("wet_domain", {}).get(
                "delivered_source_start_position_m"
            ),
            "delivered_source_end_position_m": source_manifest.get("wet_domain", {}).get(
                "delivered_source_end_position_m"
            ),
            "discarded_source_open_arc_length_m": source_manifest.get("wet_domain", {}).get(
                "discarded_source_open_arc_length_m"
            ),
        },
        "outputs": {
            **outputs,
            "model_boundary_colored_map": str(map_path),
        },
    }
    manifest_path = run_dir / "model_boundary_loop_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(path, layer=layer)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs or "EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def _largest_polygon(geometries) -> Polygon:
    polygons: list[Polygon] = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polygons.append(geom.buffer(0))
        elif hasattr(geom, "geoms"):
            polygons.extend(part.buffer(0) for part in geom.geoms if isinstance(part, Polygon) and not part.is_empty)
    if not polygons:
        raise ValueError("No Polygon geometry found")
    return max(polygons, key=lambda poly: poly.area)


def _line_union_xy(gdf: gpd.GeoDataFrame, projection):
    if gdf.empty:
        return GeometryCollection()
    lines = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        xy = project_geometry(geom, projection)
        if xy.is_empty:
            continue
        lines.append(xy)
    return unary_union(lines) if lines else GeometryCollection()


def _classify_exterior_segments(
    exterior_xy: LineString,
    open_xy,
    land_xy,
    frame_xy,
    land_patch_xy,
    tolerance_m: float,
) -> list[dict[str, Any]]:
    coords = list(exterior_xy.coords)
    records: list[dict[str, Any]] = []
    for idx in range(len(coords) - 1):
        segment = LineString([coords[idx], coords[idx + 1]])
        if segment.is_empty or segment.length <= 0.0:
            continue
        midpoint = segment.interpolate(0.5, normalized=True)
        segment_class = "unclassified_outer_boundary"
        if _near(midpoint, land_patch_xy, tolerance_m):
            segment_class = "land_patch_boundary"
        elif _near(midpoint, open_xy, tolerance_m):
            segment_class = "open_boundary"
        elif _near(midpoint, land_xy, tolerance_m):
            segment_class = "land_outer_boundary"
        elif _near(midpoint, frame_xy, tolerance_m):
            segment_class = "frame_clip_boundary"
        records.append(
            {
                "sequence_id": idx,
                "segment_class": segment_class,
                "length_m": float(segment.length),
                "geometry": segment,
            }
        )
    return records


def _near(point: Point, geom, tolerance_m: float) -> bool:
    return geom is not None and not geom.is_empty and float(point.distance(geom)) <= tolerance_m


def _extract_islands(domain_xy: Polygon, min_area_m2: float) -> tuple[list[Polygon], list[LineString]]:
    polygons: list[Polygon] = []
    lines: list[LineString] = []
    for ring in domain_xy.interiors:
        poly = Polygon(ring).buffer(0)
        if isinstance(poly, Polygon) and not poly.is_empty and poly.area >= min_area_m2:
            polygons.append(poly)
            lines.append(LineString(ring.coords))
    return polygons, lines


def _line_fraction_near(line, reference: LineString, tolerance_m: float) -> float:
    if line is None or line.is_empty or reference is None or reference.is_empty:
        return 0.0
    length = float(getattr(line, "length", 0.0))
    if length <= 0.0:
        return 0.0
    try:
        near = line.intersection(reference.buffer(tolerance_m))
        return float(min(1.0, max(0.0, getattr(near, "length", 0.0) / length)))
    except Exception:
        return 0.0


def _class_lengths(records: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for record in records:
        key = str(record["segment_class"])
        out[key] = out.get(key, 0.0) + float(record["length_m"])
    return out


def _build_layers(
    domain_xy: Polygon,
    exterior_xy: LineString,
    segments_xy: list[dict[str, Any]],
    island_polygons_xy: list[Polygon],
    island_lines_xy: list[LineString],
    open_xy,
    projection,
    final_status: str,
) -> dict[str, gpd.GeoDataFrame]:
    domain_lonlat = unproject_geometry(domain_xy, projection)
    exterior_lonlat = unproject_geometry(exterior_xy, projection)
    model_domain = gpd.GeoDataFrame(
        [{"segment_class": "model_domain", "final_status": final_status, "geometry": domain_lonlat}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    outer = gpd.GeoDataFrame(
        [{"segment_class": "model_outer_boundary", "closed": True, "geometry": exterior_lonlat}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    segment_records = [
        {
            "sequence_id": int(item["sequence_id"]),
            "segment_class": item["segment_class"],
            "length_m": float(item["length_m"]),
            "geometry": unproject_geometry(item["geometry"], projection),
        }
        for item in segments_xy
    ]
    segments = gpd.GeoDataFrame(segment_records, geometry="geometry", crs="EPSG:4326")
    island_poly_records = [
        {
            "island_id": idx,
            "area_m2": float(poly.area),
            "geometry": unproject_geometry(poly, projection),
        }
        for idx, poly in enumerate(island_polygons_xy)
    ]
    island_line_records = [
        {
            "island_id": idx,
            "length_m": float(line.length),
            "geometry": unproject_geometry(line, projection),
        }
        for idx, line in enumerate(island_lines_xy)
    ]
    delivered_open = _geometry_gdf(open_xy, projection, "delivered_open_boundary_arc")
    source_open = _geometry_gdf(open_xy, projection, "source_open_boundary_arc")
    if island_poly_records:
        island_polygons = gpd.GeoDataFrame(island_poly_records, geometry="geometry", crs="EPSG:4326")
    else:
        island_polygons = gpd.GeoDataFrame(
            {"island_id": [], "area_m2": []},
            geometry=[],
            crs="EPSG:4326",
        )
    if island_line_records:
        island_lines = gpd.GeoDataFrame(island_line_records, geometry="geometry", crs="EPSG:4326")
    else:
        island_lines = gpd.GeoDataFrame(
            {"island_id": [], "length_m": []},
            geometry=[],
            crs="EPSG:4326",
        )
    return {
        "model_domain_polygon": model_domain,
        "model_outer_boundary": outer,
        "model_outer_boundary_segments": segments,
        "island_boundary_polygons": island_polygons,
        "island_boundary_lines": island_lines,
        "delivered_open_boundary_arc": delivered_open,
        "source_open_boundary_arc": source_open,
    }


def _geometry_gdf(geom, projection, segment_class: str) -> gpd.GeoDataFrame:
    if geom is None or geom.is_empty:
        return gpd.GeoDataFrame({"segment_class": []}, geometry=[], crs="EPSG:4326")
    parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    records = [
        {"segment_class": segment_class, "geometry": unproject_geometry(part, projection)}
        for part in parts
        if part is not None and not part.is_empty
    ]
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _write_outputs(run_dir: Path, layers: dict[str, gpd.GeoDataFrame], name: str) -> dict[str, str]:
    gpkg = run_dir / "model_boundary_loops.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    for layer_name, gdf in layers.items():
        _write_layer(gpkg, layer_name, gdf)
    features = []
    for layer_name in ("model_outer_boundary_segments", "island_boundary_lines"):
        gdf = layers[layer_name]
        if gdf.empty:
            continue
        for _, row in gdf.iterrows():
            props = {key: _json_safe(value) for key, value in row.items() if key != "geometry"}
            props["layer"] = layer_name
            features.append({"type": "Feature", "properties": props, "geometry": mapping(row.geometry)})
    segments_path = run_dir / "model_boundary_segments.geojson"
    segments_path.write_text(
        json.dumps({"type": "FeatureCollection", "name": name, "features": features}, indent=2),
        encoding="utf-8",
    )
    return {
        "model_boundary_loops_gpkg": str(gpkg),
        "model_boundary_segments_geojson": str(segments_path),
    }


def _write_layer(gpkg: Path, layer_name: str, gdf: gpd.GeoDataFrame) -> None:
    if gdf.empty:
        gdf = gpd.GeoDataFrame(
            [{"empty_layer": True, "geometry": GeometryCollection()}],
            geometry="geometry",
            crs="EPSG:4326",
        )
    gdf.to_file(gpkg, layer=layer_name, driver="GPKG")


def _plot_boundary_map(
    path: Path,
    layers: dict[str, gpd.GeoDataFrame],
    coast_gdf: gpd.GeoDataFrame,
    domain_lonlat: Polygon,
    final_status: str,
    island_count: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    layers["model_domain_polygon"].plot(ax=ax, facecolor="#7cc6fe", edgecolor="none", alpha=0.25)
    if not coast_gdf.empty:
        sample = coast_gdf.iloc[:: max(1, int(math.ceil(len(coast_gdf) / 12_000)))]
        sample.plot(ax=ax, color="#8a8f98", linewidth=0.25, alpha=0.45)
    segments = layers["model_outer_boundary_segments"]
    _plot_class(segments, ax, "land_outer_boundary", "#005ea8", linewidth=1.6)
    _plot_class(segments, ax, "land_patch_boundary", "#005ea8", linewidth=1.8, linestyle="--")
    _plot_class(segments, ax, "frame_clip_boundary", "#005ea8", linewidth=1.6, linestyle="--")
    _plot_class(segments, ax, "unclassified_outer_boundary", "#f28e2b", linewidth=1.2, linestyle=":")
    _plot_class(segments, ax, "open_boundary", "#d00000", linewidth=2.4)
    if not layers["island_boundary_lines"].empty:
        layers["island_boundary_lines"].plot(ax=ax, color="#1a9850", linewidth=0.8)
    if not layers["source_open_boundary_arc"].empty:
        layers["source_open_boundary_arc"].plot(ax=ax, color="#d00000", linewidth=0.9, alpha=0.35, linestyle="--")
    gpd.GeoSeries([domain_lonlat], crs="EPSG:4326").boundary.plot(ax=ax, color="#0b4f6c", linewidth=0.45, alpha=0.35)
    ax.set_title(f"Model boundary loops - {final_status} - islands: {island_count}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_class(gdf: gpd.GeoDataFrame, ax, segment_class: str, color: str, **kwargs) -> None:
    if gdf.empty or "segment_class" not in gdf.columns:
        return
    subset = gdf[gdf["segment_class"] == segment_class]
    if not subset.empty:
        subset.plot(ax=ax, color=color, **kwargs)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__geo_interface__"):
        return mapping(value)
    return str(value)
