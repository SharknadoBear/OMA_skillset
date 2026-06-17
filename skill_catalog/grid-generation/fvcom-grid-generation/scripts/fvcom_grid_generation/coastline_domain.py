"""Coastline-aware FVCOM domain preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, box, shape
from shapely.ops import polygonize, unary_union

from .bathymetry import BathymetryGrid, load_bathymetry
from .domain import infer_offshore_side
from .open_boundary_designer import OPEN_BOUNDARY_MODES, design_open_boundary
from .projection import local_utm_projection, project_geometry, unproject_geometry
from .size_field import SizeFieldConfig, build_size_field
from .sms_2dm import read_2dm


@dataclass(frozen=True)
class DomainPrepareConfig:
    """Controls for coastline-aware domain preparation."""

    target_resolution_m: float | None = None
    max_nodes: int = 500_000
    min_depth_m: float = 0.05
    offshore_side: str | None = None
    open_boundary_spacing_factor: float = 75.0
    open_boundary_spacing_min_factor: float = 50.0
    open_boundary_spacing_max_factor: float = 100.0
    unresolved_width_elements: float = 3.0
    max_mask_cells: int = 500_000
    gradation: float = 0.15
    gradation_iterations: int = 40
    open_boundary_mode: str = "auto"
    manual_open_boundary: str | Path | None = None
    ocean_direction: tuple[float, float] | None = None
    anchor_seeds: tuple[float, float, float, float] | None = None
    anchor_seed_json: str | Path | None = None
    anchor_max_iterations: int = 40
    anchor_step_factor: float = 1.0
    anchor_min_step_factor: float = 0.1
    anchor_bbox_touch_fraction: float = 0.02
    open_boundary_candidate_rounds: int = 3


@dataclass(frozen=True)
class PreparedDomain:
    """Prepared domain artifacts and metadata."""

    domain_polygon: Polygon
    open_boundary: LineString
    coastline: gpd.GeoDataFrame
    metadata: dict
    open_boundary_candidates: gpd.GeoDataFrame | None = None


def target_resolution_from_bathymetry(bathy: BathymetryGrid) -> float:
    """Estimate finest native bathymetry spacing in meters."""
    lon = np.asarray(bathy.lon, dtype=float)
    lat = np.asarray(bathy.lat, dtype=float)
    lat0 = float(np.nanmean(lat))
    dx = abs(float(np.nanmedian(np.diff(lon)))) * 111_320.0 * max(np.cos(np.radians(lat0)), 0.2)
    dy = abs(float(np.nanmedian(np.diff(lat)))) * 110_540.0
    finite = [value for value in (dx, dy) if np.isfinite(value) and value > 0.0]
    if not finite:
        raise ValueError("Could not infer bathymetry resolution from lon/lat coordinates.")
    return float(min(finite))


def bbox_from_bathymetry_or_arg(bathy: BathymetryGrid, bbox: tuple[float, float, float, float] | None) -> tuple[float, float, float, float]:
    return bbox if bbox is not None else bathy.bbox


def load_coastline(path: str | Path, bbox_wsen: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Read and clip a CUSP coastline product to a bbox."""
    path = Path(path)
    try:
        gdf = gpd.read_file(path, layer="coastline", bbox=bbox_wsen, engine="pyogrio")
    except Exception:
        gdf = gpd.read_file(path, bbox=bbox_wsen)
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    clipped = gpd.clip(gdf, gpd.GeoSeries([box(*bbox_wsen)], crs="EPSG:4326"))
    return clipped.reset_index(drop=True)


def wet_polygon_from_bathymetry(
    bathy: BathymetryGrid,
    bbox_wsen: tuple[float, float, float, float],
    min_depth_m: float = 0.05,
    max_mask_cells: int = 500_000,
) -> tuple[Polygon, dict]:
    """Vectorize the connected wet-mask component in lon/lat."""
    try:
        from affine import Affine
        from rasterio import features
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("rasterio and affine are required for coastline-aware domain preparation.") from exc

    lon = np.asarray(bathy.lon, dtype=float)
    lat = np.asarray(bathy.lat, dtype=float)
    depth = np.asarray(bathy.depth, dtype=float)
    west, south, east, north = bbox_wsen
    lon_mask = (lon >= west) & (lon <= east)
    lat_mask = (lat >= south) & (lat <= north)
    if not np.any(lon_mask) or not np.any(lat_mask):
        raise ValueError("Bathymetry grid does not intersect requested bbox.")

    lon_sub = lon[lon_mask]
    lat_sub = lat[lat_mask]
    depth_sub = depth[np.ix_(lat_mask, lon_mask)]
    cell_count = int(depth_sub.size)
    stride = max(1, int(np.ceil(np.sqrt(cell_count / max(max_mask_cells, 1)))))
    lon_s = lon_sub[::stride]
    lat_s = lat_sub[::stride]
    depth_s = depth_sub[::stride, ::stride]

    wet = (np.isfinite(depth_s) & (depth_s > min_depth_m)).astype("uint8")
    if not np.any(wet):
        raise ValueError("No wet bathymetry cells found in requested bbox.")

    dx = float(np.nanmedian(np.diff(lon_s))) if len(lon_s) > 1 else 0.001
    dy = float(np.nanmedian(np.diff(lat_s))) if len(lat_s) > 1 else 0.001
    transform = Affine.translation(float(lon_s[0] - 0.5 * dx), float(lat_s[0] - 0.5 * dy)) * Affine.scale(dx, dy)
    polygons = []
    for geom, value in features.shapes(wet, mask=wet.astype(bool), transform=transform):
        if int(value) == 1:
            polygons.append(shape(geom))
    if not polygons:
        raise ValueError("Wet-mask vectorization produced no polygons.")

    wet_union = unary_union(polygons).intersection(box(*bbox_wsen)).buffer(0)
    if isinstance(wet_union, MultiPolygon):
        largest = max(wet_union.geoms, key=lambda geom: geom.area)
    elif isinstance(wet_union, Polygon):
        largest = wet_union
    else:
        raise ValueError("Wet mask did not produce a polygonal water domain.")

    metadata = {
        "wet_mask_input_cells": cell_count,
        "wet_mask_stride": int(stride),
        "wet_mask_vectorized_polygons": int(len(polygons)),
        "wet_component_policy": "largest_connected_component",
    }
    return largest.buffer(0), metadata


def filter_unresolved_domain(
    polygon_lonlat: Polygon,
    bbox_wsen: tuple[float, float, float, float],
    target_resolution_m: float,
    unresolved_width_elements: float,
) -> tuple[Polygon, dict]:
    """Remove unresolved holes and narrow wet branches using projected geometry."""
    projection = local_utm_projection(bbox_wsen)
    poly_xy = project_geometry(polygon_lonlat, projection).buffer(0)
    min_width = float(unresolved_width_elements * target_resolution_m)

    cleaned = poly_xy
    eroded = cleaned.buffer(-0.5 * min_width)
    if not eroded.is_empty:
        restored = eroded.buffer(0.5 * min_width).buffer(0)
        if not restored.is_empty:
            cleaned = restored
    if isinstance(cleaned, MultiPolygon):
        cleaned = max(cleaned.geoms, key=lambda geom: geom.area)

    kept_holes = []
    dropped_holes = 0
    for ring in cleaned.interiors:
        hole = Polygon(ring)
        perimeter = max(hole.length, 1.0)
        effective_width = 4.0 * hole.area / perimeter
        if hole.area >= min_width * min_width and effective_width >= min_width:
            kept_holes.append(ring.coords)
        else:
            dropped_holes += 1
    filtered_xy = Polygon(cleaned.exterior.coords, kept_holes).buffer(0)
    filtered = unproject_geometry(filtered_xy, projection).buffer(0)

    metadata = {
        "target_resolution_m": float(target_resolution_m),
        "unresolved_width_m": float(min_width),
        "dropped_unresolved_holes": int(dropped_holes),
        "kept_island_holes": int(len(kept_holes)),
        "thin_waterway_filter": "buffer(-width/2).buffer(width/2)",
        "projected_crs_epsg": projection.epsg,
    }
    return filtered, metadata


def filter_unresolved_holes_only(
    polygon_lonlat: Polygon,
    bbox_wsen: tuple[float, float, float, float],
    target_resolution_m: float,
    unresolved_width_elements: float,
) -> tuple[Polygon, dict]:
    """Drop unresolved island holes while preserving the reference exterior."""
    projection = local_utm_projection(bbox_wsen)
    poly_xy = project_geometry(polygon_lonlat, projection).buffer(0)
    min_width = float(unresolved_width_elements * target_resolution_m)
    kept_holes = []
    dropped_holes = 0
    for ring in poly_xy.interiors:
        hole = Polygon(ring)
        perimeter = max(hole.length, 1.0)
        effective_width = 4.0 * hole.area / perimeter
        if hole.area >= min_width * min_width and effective_width >= min_width:
            kept_holes.append(ring.coords)
        else:
            dropped_holes += 1
    filtered_xy = Polygon(poly_xy.exterior.coords, kept_holes).buffer(0)
    filtered = unproject_geometry(filtered_xy, projection).buffer(0)
    metadata = {
        "target_resolution_m": float(target_resolution_m),
        "unresolved_width_m": float(min_width),
        "dropped_unresolved_holes": int(dropped_holes),
        "kept_island_holes": int(len(kept_holes)),
        "thin_waterway_filter": "not_applied_to_reference_exterior",
        "projected_crs_epsg": projection.epsg,
    }
    return filtered, metadata


def open_boundary_from_reference(path: str | Path) -> LineString:
    """Extract the first ordered ``NS`` open boundary from a reference 2DM."""
    mesh = read_2dm(path)
    if not mesh.open_boundaries:
        raise ValueError(f"No NS open boundary found in reference mesh {path}")
    indices = mesh.open_boundaries[0] - 1
    pts = mesh.nodes[indices]
    return LineString([(float(lon), float(lat)) for lon, lat in pts if np.isfinite(lon) and np.isfinite(lat)])


def domain_polygon_from_reference(path: str | Path) -> Polygon:
    """Reconstruct a domain polygon from the exterior edges of a reference 2DM mesh."""
    mesh = read_2dm(path)
    tri0 = np.asarray(mesh.triangles, dtype=int) - 1
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in tri0:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    lines = []
    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        pa = mesh.nodes[a]
        pb = mesh.nodes[b]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            lines.append(LineString([(float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1]))]))
    if not lines:
        raise ValueError(f"Could not reconstruct exterior boundary from {path}")
    polygons = list(polygonize(unary_union(lines)))
    if not polygons:
        raise ValueError(f"Reference boundary edges did not polygonize: {path}")
    return max(polygons, key=lambda geom: geom.area).buffer(0)


def open_boundary_from_bbox(
    bbox_wsen: tuple[float, float, float, float],
    offshore_side: str,
    spacing_m: float,
) -> LineString:
    """Create a smooth offshore arc along one bbox side."""
    west, south, east, north = bbox_wsen
    projection = local_utm_projection(bbox_wsen)
    bbox_xy = project_geometry(box(*bbox_wsen), projection)
    minx, miny, maxx, maxy = bbox_xy.bounds
    width = maxx - minx
    height = maxy - miny
    length = height if offshore_side in {"east", "west"} else width
    n = max(8, int(np.ceil(length / max(spacing_m, 1.0))) + 1)
    t = np.linspace(0.0, 1.0, n)
    bow = 0.08 * max(width, height) * np.sin(np.pi * t)
    if offshore_side == "east":
        xy = np.column_stack([np.full(n, maxx) + bow, miny + t * height])
    elif offshore_side == "west":
        xy = np.column_stack([np.full(n, minx) - bow, miny + t * height])
    elif offshore_side == "north":
        xy = np.column_stack([minx + t * width, np.full(n, maxy) + bow])
    elif offshore_side == "south":
        xy = np.column_stack([minx + t * width, np.full(n, miny) - bow])
    else:
        raise ValueError("offshore_side must be east, west, north, or south")
    line_xy = LineString(xy)
    return unproject_geometry(line_xy, projection)


def boundary_to_geodataframes(
    domain_polygon: Polygon,
    open_boundary: LineString,
    coastline: gpd.GeoDataFrame,
    open_boundary_candidates: gpd.GeoDataFrame | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """Create standard boundary GeoDataFrame layers."""
    domain = gpd.GeoDataFrame([{"segment_class": "domain", "geometry": domain_polygon}], crs="EPSG:4326")
    open_gdf = gpd.GeoDataFrame([{"segment_class": "open_boundary", "geometry": open_boundary}], crs="EPSG:4326")
    land = gpd.GeoDataFrame([{"segment_class": "land_boundary", "geometry": LineString(domain_polygon.exterior.coords)}], crs="EPSG:4326")
    island_records = [{"segment_class": "island_boundary", "geometry": LineString(ring.coords)} for ring in domain_polygon.interiors]
    islands = (
        gpd.GeoDataFrame(island_records, geometry="geometry", crs="EPSG:4326")
        if island_records
        else gpd.GeoDataFrame({"segment_class": []}, geometry=[], crs="EPSG:4326")
    )
    if coastline.empty:
        coast = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        coast = coastline.copy()
        coast["segment_class"] = "cusp_candidate"
    return {
        "domain": domain,
        "open_boundary": open_gdf,
        "land_boundary": land,
        "island_boundary": islands,
        "coastline_candidates": coast,
        "open_boundary_candidates": open_boundary_candidates
        if open_boundary_candidates is not None
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
    }


def write_domain_package(
    prepared: PreparedDomain,
    run_dir: str | Path,
    name: str,
) -> dict[str, str]:
    """Write GPKG, metadata, and visual-review manifest for a prepared domain."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    gpkg = run_dir / f"{name}_coastline_domain.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    for layer, gdf in boundary_to_geodataframes(
        prepared.domain_polygon,
        prepared.open_boundary,
        prepared.coastline,
        prepared.open_boundary_candidates,
    ).items():
        if gdf.empty:
            continue
        gdf.to_file(gpkg, layer=layer, driver="GPKG")

    metadata_path = run_dir / f"{name}_domain_metadata.json"
    review_path = run_dir / f"{name}_domain_visual_review.json"
    metadata = dict(prepared.metadata)
    metadata["outputs"] = {
        "domain_gpkg": str(gpkg),
        "metadata_json": str(metadata_path),
        "visual_review_json": str(review_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    review = {
        "name": name,
        "status": "needs_review",
        "decision": "needs_review",
        "reviewer": None,
        "notes": "Open the domain review PNG/GPKG before running coastline meshing.",
        "domain_metadata_json": str(metadata_path),
    }
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    return metadata["outputs"]


def prepare_coastline_domain(
    bathymetry: str | Path,
    coastline_gpkg: str | Path,
    run_dir: str | Path,
    name: str,
    bbox_wsen: tuple[float, float, float, float] | None = None,
    reference_2dm: str | Path | None = None,
    config: DomainPrepareConfig | None = None,
) -> PreparedDomain:
    """Prepare a coastline-aware domain and write its standard package."""
    config = config or DomainPrepareConfig()
    if config.open_boundary_mode not in OPEN_BOUNDARY_MODES:
        raise ValueError(f"open_boundary_mode must be one of {OPEN_BOUNDARY_MODES}")
    bathy = load_bathymetry(bathymetry)
    bbox_wsen = bbox_from_bathymetry_or_arg(bathy, bbox_wsen)
    target_resolution = float(config.target_resolution_m or target_resolution_from_bathymetry(bathy))
    resolution_source = "user" if config.target_resolution_m is not None else "bathymetry_native"
    user_offshore_side = config.offshore_side
    offshore_side = user_offshore_side or infer_offshore_side(bathy)
    open_factor = float(np.clip(config.open_boundary_spacing_factor, config.open_boundary_spacing_min_factor, config.open_boundary_spacing_max_factor))
    open_spacing = open_factor * target_resolution

    if reference_2dm:
        reference_polygon = domain_polygon_from_reference(reference_2dm)
        domain_polygon, filter_meta = filter_unresolved_holes_only(
            reference_polygon,
            bbox_wsen,
            target_resolution,
            config.unresolved_width_elements,
        )
        wet_meta = {
            "domain_source": "reference_2dm_exterior_edges",
            "reference_2dm": str(reference_2dm),
        }
        open_boundary = open_boundary_from_reference(reference_2dm)
        open_boundary_candidates = None
        open_design_meta = {
            "open_boundary_mode": "reference-2dm",
            "selected_candidate_id": None,
            "note": "Open boundary was reconstructed from the first NS nodestring in the reference mesh.",
        }
    else:
        wet_polygon, wet_meta = wet_polygon_from_bathymetry(
            bathy,
            bbox_wsen,
            min_depth_m=config.min_depth_m,
            max_mask_cells=config.max_mask_cells,
        )
        domain_polygon, filter_meta = filter_unresolved_domain(
            wet_polygon,
            bbox_wsen,
            target_resolution,
            config.unresolved_width_elements,
        )
    coastline = load_coastline(coastline_gpkg, bbox_wsen)
    if not reference_2dm:
        open_design = design_open_boundary(
            domain_polygon,
            bathy,
            bbox_wsen,
            coastline,
            target_resolution,
            open_spacing,
            offshore_side=user_offshore_side,
            mode=config.open_boundary_mode,
            manual_open_boundary=config.manual_open_boundary,
            ocean_direction=config.ocean_direction,
            anchor_seeds=config.anchor_seeds,
            anchor_seed_json=config.anchor_seed_json,
            anchor_max_iterations=config.anchor_max_iterations,
            anchor_step_factor=config.anchor_step_factor,
            anchor_min_step_factor=config.anchor_min_step_factor,
            anchor_bbox_touch_fraction=config.anchor_bbox_touch_fraction,
            max_rounds=config.open_boundary_candidate_rounds,
            min_depth_m=config.min_depth_m,
        )
        domain_polygon = open_design.domain_polygon
        open_boundary = open_design.open_boundary
        open_boundary_candidates = open_design.candidates
        open_design_meta = open_design.metadata
        offshore_side = str(open_design_meta.get("offshore_side", offshore_side))

    projection = local_utm_projection(bbox_wsen)
    area_m2 = float(project_geometry(domain_polygon, projection).area)
    nominal_tri_area = max((np.sqrt(3.0) / 4.0) * target_resolution**2, 1.0)
    estimated_nodes = int(np.ceil(area_m2 / nominal_tri_area))
    if estimated_nodes > config.max_nodes:
        raise ValueError(
            f"Estimated {estimated_nodes} nodes exceeds --max-nodes {config.max_nodes}; "
            "choose a coarser target resolution or raise the limit."
        )

    size_config = SizeFieldConfig(
        min_size=target_resolution,
        gradation=config.gradation,
        gradation_iterations=config.gradation_iterations,
    )
    size_field = build_size_field(bathy, size_config)
    metadata = {
        "name": name,
        "bbox_wsen": list(map(float, bbox_wsen)),
        "bathymetry": str(bathymetry),
        "coastline_gpkg": str(coastline_gpkg),
        "reference_2dm": str(reference_2dm) if reference_2dm else None,
        "target_resolution_m": target_resolution,
        "resolution_source": resolution_source,
        "offshore_side": offshore_side,
        "open_boundary_mode": config.open_boundary_mode if not reference_2dm else "reference-2dm",
        "open_boundary_spacing_m": open_spacing,
        "open_boundary_spacing_factor": open_factor,
        "area_m2": area_m2,
        "estimated_nodes": estimated_nodes,
        "max_nodes": int(config.max_nodes),
        "coastline_feature_count": int(len(coastline)),
        "projected_crs_epsg": projection.epsg,
        "wet_mask": wet_meta,
        "filtering": filter_meta,
        "open_boundary_design": open_design_meta,
        "gradation_report": size_field.gradation_report,
    }
    prepared = PreparedDomain(
        domain_polygon=domain_polygon,
        open_boundary=open_boundary,
        coastline=coastline,
        metadata=metadata,
        open_boundary_candidates=open_boundary_candidates,
    )
    outputs = write_domain_package(prepared, run_dir, name)
    metadata["outputs"] = outputs
    Path(outputs["metadata_json"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return prepared


def assert_domain_review_passed(review_json: str | Path) -> dict:
    """Raise unless the domain visual review manifest is marked as pass."""
    review_json = Path(review_json)
    if not review_json.exists():
        raise FileNotFoundError(f"Missing domain visual review manifest: {review_json}")
    review = json.loads(review_json.read_text(encoding="utf-8"))
    decision = str(review.get("decision") or review.get("status") or "").lower()
    if decision != "pass":
        raise PermissionError(
            f"Domain visual review is {decision!r}; open the review figure and mark the manifest pass before meshing."
        )
    return review
