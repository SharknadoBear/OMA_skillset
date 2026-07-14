"""End-to-end FVCOM grid-generation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
from shapely import contains_xy

from .bathymetry import coarsen_for_size_field, load_bathymetry
from .boundary import (
    BoundaryConfig,
    boundary_nodes_geojson,
    evaluate_boundary_contract_v2,
    load_boundary_package,
    load_boundary_resolution,
    prepare_boundary_nodes,
)
from .mesh import MeshConfig, generate_mesh
from .metrics import triangle_geometry
from .plotting import write_mesh_gpkg, write_mesh_quality_gpkg, write_mesh_review_map
from .progress import ProgressTracker
from .projection import project_points, unproject_points
from .quality import evaluate_mesh_quality
from .size_field import (
    SizeFieldConfig,
    SizeFieldSemantics,
    boundary_front_seed_points,
    build_size_field,
    write_size_field,
)
from .sms_2dm import read_2dm, write_2dm


@dataclass(frozen=True)
class GridConfig:
    """Runtime controls for FVCOM grid generation."""

    mode: str = "execute"
    land_spacing_m: float = 50.0
    open_spacing_m: float = 3000.0
    coarse_smoke: bool = False
    gradation: float = 0.15
    target_timestep_s: str = "auto"
    max_interior_points: int = 80_000
    max_total_nodes: int = 120_000
    node_budget_stop_fraction: float = 0.90
    refine_iterations: int = 3
    smooth_iterations: int = 8
    fetch_bathymetry: bool = True
    bathy_fallback_policy: str = "cudem-nbs-crm-etopo"
    bathy_resolution_policy: str = "source-priority"
    bathy_target_spacing_arcsec: float = 1.0
    bathy_max_sources: int = 256
    progress_interval_s: float = 10.0
    size_field_max_cells: int = 1_500_000
    boundary_resolution_profile: str = "legacy"
    bathymetry_gradient_policy: str = "auto"
    coastal_gradient_distance_m: float = 25_000.0
    regional_spring_relaxation: bool = True
    spring_relax_iterations: int = 20
    spring_relax_quality_threshold: float = 0.40
    spring_relax_min_angle_deg: float = 28.0
    spring_relax_ring_layers: int = 3
    spring_relax_shape_weight: float = 0.20
    thin_triangle_repair: bool = True
    thin_triangle_quality_threshold: float = 0.25
    thin_triangle_min_angle_deg: float = 20.0
    thin_triangle_max_passes: int = 2
    thin_triangle_max_flips: int = 200
    thin_triangle_max_insertions: int = 50
    area_transition_relaxation: bool = True
    area_transition_max_patches: int = 12
    area_transition_area_change_threshold: float = 0.50
    area_transition_target_gradient_threshold: float = 0.10
    conditioning_profile: str = "auto"
    aggressive_conditioning_rounds: int = 4
    aggressive_boundary_edit_policy: str = "kind-aware-envelope"
    aggressive_max_prunes_per_round: int = 500
    aggressive_max_valence_repairs_per_round: int = 500
    postprocess_profile: str = "none"
    postprocess_boundary_policy: str = "protect-all"
    postprocess_max_passes: int = 8
    postprocess_connectivity_limit: int | None = None


def run_fvcom_grid(
    run_dir: str | Path,
    name: str,
    config: GridConfig | None = None,
    request_text: str | None = None,
    region_bpoly_json: str | Path | None = None,
    offshore_artifacts_json: str | Path | None = None,
    bdry_arc_manifest: str | Path | None = None,
    boundary_loops_gpkg: str | Path | None = None,
    boundary_resolution_manifest: str | Path | None = None,
    bathy_nc: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete mesh workflow and write artifacts."""
    config = config or GridConfig()
    if str(config.postprocess_profile) != "none":
        raise ValueError("Integrated post-generation cleanup is disabled; run postprocess_fvcom_mesh.py explicitly on the finished .2dm")
    if config.boundary_resolution_profile not in {"legacy", "adaptive-coastal-v1", "adaptive-coastal-v2"}:
        raise ValueError("boundary_resolution_profile must be legacy, adaptive-coastal-v1, or adaptive-coastal-v2")
    if config.bathymetry_gradient_policy not in {"auto", "global", "coastal", "off"}:
        raise ValueError("bathymetry_gradient_policy must be auto, global, coastal, or off")
    if config.conditioning_profile not in {"auto", "guarded-v1", "aggressive-local-v2", "none"}:
        raise ValueError("conditioning_profile must be auto, guarded-v1, aggressive-local-v2, or none")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(run_dir=run_dir, name=name, interval_s=float(config.progress_interval_s))
    progress.update("initialize", 0.0, message="Starting FVCOM grid workflow.")
    land_spacing = 250.0 if config.coarse_smoke else float(config.land_spacing_m)
    open_spacing = 5000.0 if config.coarse_smoke else float(config.open_spacing_m)

    upstream = _resolve_upstream_artifacts(
        run_dir,
        name,
        request_text,
        region_bpoly_json,
        offshore_artifacts_json,
        bdry_arc_manifest,
        boundary_loops_gpkg,
        boundary_resolution_manifest,
        bathy_nc,
        land_spacing,
        config,
        progress,
    )
    boundary_loops_gpkg = upstream["boundary_loops_gpkg"]
    boundary_resolution_manifest = upstream.get("boundary_resolution_manifest")
    bathy_nc = upstream["bathy_nc"]
    bdry_arc_manifest = upstream.get("bdry_arc_manifest")
    if config.boundary_resolution_profile in {"adaptive-coastal-v1", "adaptive-coastal-v2"} and not boundary_resolution_manifest:
        raise ValueError(
            f"{config.boundary_resolution_profile} requires --boundary-resolution-manifest or an upstream adaptive boundary run"
        )

    if boundary_resolution_manifest:
        progress.update("load_boundary_resolution", 18.0, message="Loading explicit adaptive boundary chains.", artifact=boundary_resolution_manifest)
        boundary_package, boundary_nodes, resolution_doc = load_boundary_resolution(boundary_resolution_manifest)
        if config.boundary_resolution_profile != "legacy" and resolution_doc.get("profile") != config.boundary_resolution_profile:
            raise ValueError(
                "Requested boundary-resolution profile does not match the supplied manifest: "
                f"{config.boundary_resolution_profile} != {resolution_doc.get('profile')}"
            )
        if (
            config.boundary_resolution_profile == "adaptive-coastal-v2"
            and str(resolution_doc.get("final_status", "needs_review")) != "pass"
        ):
            failures = list(resolution_doc.get("failure_taxonomy", []))
            raise ValueError(
                "adaptive-coastal-v2 boundary package requires scientific review before gridding: "
                + (", ".join(map(str, failures)) if failures else "upstream final_status is not pass")
            )
    else:
        progress.update("load_boundary_loops", 18.0, message="Loading model boundary-loop package.", artifact=boundary_loops_gpkg)
        boundary_package = load_boundary_package(boundary_loops_gpkg)
        progress.update("prepare_boundary_nodes", 22.0, message="Densifying classified boundary nodes.")
        boundary_nodes = prepare_boundary_nodes(
            boundary_package,
            BoundaryConfig(land_spacing_m=land_spacing, open_spacing_m=open_spacing, island_spacing_m=land_spacing),
        )
        resolution_doc = None
    boundary_nodes_path = run_dir / "boundary_nodes.geojson"
    boundary_nodes_path.write_text(json.dumps(boundary_nodes_geojson(boundary_nodes), indent=2), encoding="utf-8")
    boundary_contract_path = None
    if config.boundary_resolution_profile == "adaptive-coastal-v2":
        boundary_contract = evaluate_boundary_contract_v2(boundary_nodes, gradation=float(config.gradation))
        boundary_contract_path = run_dir / "boundary_contract_v2.json"
        boundary_contract_path.write_text(json.dumps(_json_safe(boundary_contract), indent=2), encoding="utf-8")
        progress.update(
            "boundary_contract_v2",
            24.0,
            message="Auditing adaptive-v2 anchors, spacing, and gradation before bathymetry work.",
            artifact=boundary_contract_path,
            extra={"passed": bool(boundary_contract["passed"]), "failures": boundary_contract["failure_taxonomy"]},
        )
        if not boundary_contract["passed"]:
            raise ValueError(
                "adaptive-coastal-v2 boundary contract failed: "
                + ", ".join(boundary_contract["failure_taxonomy"])
            )

    progress.update("load_bathymetry", 28.0, message="Loading positive-down bathymetry grid.", artifact=bathy_nc)
    bathy = load_bathymetry(bathy_nc)
    progress.update(
        "prepare_size_field_bathymetry",
        31.0,
        message="Preparing bounded bathymetry grid for size-field operations.",
        extra={"source_cells": int(bathy.depth.size), "max_cells": int(config.size_field_max_cells)},
    )
    size_bathy = coarsen_for_size_field(bathy, max_cells=int(config.size_field_max_cells))
    size_config = SizeFieldConfig(
        land_spacing_m=land_spacing,
        open_spacing_m=open_spacing,
        max_size_m=8000.0 if boundary_nodes.adaptive_resolution else 20_000.0,
        gradation=float(config.gradation),
        target_timestep_s=str(config.target_timestep_s),
        adaptive_boundary=bool(boundary_nodes.adaptive_resolution),
        bathymetry_gradient_policy=str(config.bathymetry_gradient_policy),
        coastal_gradient_distance_m=float(config.coastal_gradient_distance_m),
        size_field_profile=(
            "adaptive-coastal-v2" if config.boundary_resolution_profile == "adaptive-coastal-v2" else "v1"
        ),
        coverage_policy=("raise" if config.boundary_resolution_profile == "adaptive-coastal-v2" else "auto"),
    )
    progress.update("build_size_field", 34.0, message="Building bathymetry and shoreline-based mesh-size field.")
    size_semantics = (
        _size_field_semantics_v2(size_bathy, boundary_nodes)
        if config.boundary_resolution_profile == "adaptive-coastal-v2"
        else None
    )
    size_field = build_size_field(size_bathy, boundary_nodes, size_config, semantics=size_semantics)
    node_budget_path = None
    if config.boundary_resolution_profile == "adaptive-coastal-v2":
        _, boundary_front_report = boundary_front_seed_points(boundary_nodes)
        interior_estimate = int(
            size_field.report.get("node_budget_estimate", {}).get("estimated_interior_node_count", 0)
        )
        boundary_front_count = int(boundary_front_report.get("accepted_count", 0))
        estimated_total = int(interior_estimate + len(boundary_nodes.xy) + boundary_front_count)
        node_budget = {
            "schema_version": "fvcom_node_budget_preflight_v2",
            "estimated_interior_node_count": interior_estimate,
            "explicit_boundary_node_count": int(len(boundary_nodes.xy)),
            "boundary_front_seed_count": boundary_front_count,
            "estimated_total_node_count": estimated_total,
            "maximum_total_nodes": int(config.max_total_nodes),
            "stop_fraction": float(config.node_budget_stop_fraction),
            "stop_threshold": int(float(config.node_budget_stop_fraction) * int(config.max_total_nodes)),
            "passed": bool(
                estimated_total <= float(config.node_budget_stop_fraction) * int(config.max_total_nodes)
            ),
            "boundary_front": boundary_front_report,
        }
        node_budget_path = run_dir / "node_budget_preflight_v2.json"
        node_budget_path.write_text(json.dumps(_json_safe(node_budget), indent=2), encoding="utf-8")
        progress.update(
            "node_budget_preflight_v2",
            42.0,
            message="Estimating adaptive-v2 boundary, front, and interior node demand.",
            artifact=node_budget_path,
            extra={"passed": node_budget["passed"], "estimated_total": estimated_total},
        )
        if not node_budget["passed"]:
            raise ValueError(
                "adaptive-coastal-v2 node-budget preflight failed: "
                f"estimated {estimated_total} nodes exceeds the stop threshold "
                f"{node_budget['stop_threshold']}"
            )
    progress.update("write_size_field", 46.0, message="Writing size-field artifacts.")
    size_nc, size_png = write_size_field(size_field, run_dir / "size_field.nc", run_dir / "size_field.png")

    mesh_config = MeshConfig(
        refine_iterations=int(config.refine_iterations),
        smooth_iterations=int(config.smooth_iterations),
        max_interior_points=int(config.max_interior_points),
        adaptive_seed=bool(boundary_nodes.adaptive_resolution),
        regional_spring_relaxation=bool(config.regional_spring_relaxation),
        spring_relax_iterations=int(config.spring_relax_iterations),
        spring_relax_quality_threshold=float(config.spring_relax_quality_threshold),
        spring_relax_min_angle_deg=float(config.spring_relax_min_angle_deg),
        spring_relax_ring_layers=int(config.spring_relax_ring_layers),
        spring_relax_shape_weight=float(config.spring_relax_shape_weight),
        thin_triangle_repair=bool(config.thin_triangle_repair),
        thin_triangle_quality_threshold=float(config.thin_triangle_quality_threshold),
        thin_triangle_min_angle_deg=float(config.thin_triangle_min_angle_deg),
        thin_triangle_max_passes=int(config.thin_triangle_max_passes),
        thin_triangle_max_flips=int(config.thin_triangle_max_flips),
        thin_triangle_max_insertions=int(config.thin_triangle_max_insertions),
        area_transition_relaxation=bool(config.area_transition_relaxation),
        area_transition_max_patches=int(config.area_transition_max_patches),
        area_transition_area_change_threshold=float(config.area_transition_area_change_threshold),
        area_transition_target_gradient_threshold=float(config.area_transition_target_gradient_threshold),
        conditioning_profile=str(config.conditioning_profile),
        aggressive_conditioning_rounds=int(config.aggressive_conditioning_rounds),
        aggressive_boundary_edit_policy=str(config.aggressive_boundary_edit_policy),
        aggressive_max_prunes_per_round=int(config.aggressive_max_prunes_per_round),
        aggressive_max_valence_repairs_per_round=int(config.aggressive_max_valence_repairs_per_round),
    )
    def _mesh_progress(message: str, fraction: float, extra: dict[str, Any] | None = None) -> None:
        progress.update(
            "mesh_generation",
            50.0 + 28.0 * max(0.0, min(1.0, float(fraction))),
            message=message,
            extra=extra,
        )

    mesh = generate_mesh(boundary_nodes, size_field, mesh_config, progress_callback=_mesh_progress)
    conditioning_json = run_dir / "mesh_conditioning.json"
    conditioning_json.write_text(json.dumps(_json_safe(mesh.report.get("conditioning", {})), indent=2), encoding="utf-8")
    edit_ledger_json = run_dir / "mesh_edit_ledger.json"
    edit_ledger_json.write_text(
        json.dumps(_json_safe(mesh.report.get("conditioning", {}).get("mesh_edit_ledger", [])), indent=2),
        encoding="utf-8",
    )
    delivered_boundary_nodes_path = run_dir / "delivered_boundary_nodes.geojson"
    delivered_boundary_nodes_path.write_text(
        json.dumps(
            _postclean_boundary_nodes_geojson(
                mesh.nodes_lonlat,
                mesh.constraint_chains,
                mesh.open_boundary_nodes,
                boundary_kinds=mesh.boundary_kinds,
                hard_anchor_mask=mesh.hard_anchor_mask,
                node_lineage=mesh.node_lineage,
                target_spacing_m=mesh.target_spacing_m,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    progress.update("sample_bathymetry_to_mesh", 80.0, message="Sampling bathymetry to the generation-time smoothed mesh.")
    depths = bathy.sample(mesh.nodes_lonlat[:, 0], mesh.nodes_lonlat[:, 1], fill_value=float(np.nanmedian(bathy.depth)))
    depths = np.maximum(np.where(np.isfinite(depths), depths, 2.0), 0.5)
    target_sizes = _element_target_sizes(mesh.nodes_xy, mesh.triangles, boundary_nodes, size_field)
    progress.update("mesh_quality", 86.0, message="Evaluating final FVCOM, topology, and size-compatibility gates.")
    quality = evaluate_mesh_quality(
        mesh.nodes_xy,
        depths,
        mesh.triangles,
        mesh.open_boundary_nodes,
        mesh.report.get("constraint_recovery", {}),
        constraint_chains=mesh.constraint_chains,
        target_size_by_triangle=target_sizes,
    )
    quality["open_boundary_size_error"] = _open_boundary_size_error(mesh.nodes_xy, mesh.open_boundary_nodes, mesh.target_spacing_m)
    if quality["open_boundary_size_error"]["p95_l_over_h"] > 1.55 or quality["open_boundary_size_error"]["maximum_l_over_h"] > 2.0:
        if "open_boundary_size_mismatch" not in quality["failure_taxonomy"]:
            quality["failure_taxonomy"].append("open_boundary_size_mismatch")
        quality["accepted"] = False

    progress.update("write_outputs", 90.0, message="Writing the generation-time smoothed FVCOM mesh and QA artifacts.")
    output_2dm = write_2dm(
        run_dir / "fvcom_grid.2dm",
        mesh.nodes_lonlat,
        depths,
        mesh.triangles,
        mesh.open_boundary_nodes,
        mesh_name=name,
    )
    roundtrip = read_2dm(output_2dm)
    node_count_match = bool(len(roundtrip.nodes_lonlat) == len(mesh.nodes_lonlat))
    triangle_count_match = bool(len(roundtrip.triangles) == len(mesh.triangles))
    open_boundary_count_match = bool(len(roundtrip.open_boundary_nodes) == len(mesh.open_boundary_nodes))
    triangle_connectivity_match = bool(
        triangle_count_match and np.array_equal(roundtrip.triangles, mesh.triangles)
    )
    open_boundary_order_match = bool(
        open_boundary_count_match
        and np.array_equal(roundtrip.open_boundary_nodes, mesh.open_boundary_nodes)
    )
    coordinate_tolerance_m = 0.01
    if node_count_match:
        roundtrip_xy = project_points(roundtrip.nodes_lonlat, boundary_nodes.projection)
        coordinate_shift_m = np.linalg.norm(roundtrip_xy - mesh.nodes_xy, axis=1)
        coordinate_max_shift_m: float | None = float(np.max(coordinate_shift_m, initial=0.0))
        coordinate_rms_shift_m: float | None = float(np.sqrt(np.mean(coordinate_shift_m**2)))
    else:
        coordinate_max_shift_m = None
        coordinate_rms_shift_m = None
    coordinate_within_tolerance = bool(
        coordinate_max_shift_m is not None and coordinate_max_shift_m <= coordinate_tolerance_m
    )
    if node_count_match and triangle_count_match:
        roundtrip_signed_areas = triangle_geometry(roundtrip_xy, roundtrip.triangles - 1)["signed_area"]
        roundtrip_nonpositive_signed_area_count: int | None = int(np.count_nonzero(roundtrip_signed_areas <= 0.0))
        roundtrip_minimum_signed_area_m2: float | None = float(np.min(roundtrip_signed_areas, initial=np.inf))
        roundtrip_positive_signed_areas = bool(roundtrip_nonpositive_signed_area_count == 0)
    else:
        roundtrip_nonpositive_signed_area_count = None
        roundtrip_minimum_signed_area_m2 = None
        roundtrip_positive_signed_areas = False
    finite_positive_depths = bool(
        len(roundtrip.depths) == len(mesh.nodes_lonlat)
        and np.all(np.isfinite(roundtrip.depths))
        and np.all(roundtrip.depths > 0.0)
    )
    quality["roundtrip"] = {
        "node_count": int(len(roundtrip.nodes_lonlat)),
        "triangle_count": int(len(roundtrip.triangles)),
        "open_boundary_node_count": int(len(roundtrip.open_boundary_nodes)),
        "node_count_match": node_count_match,
        "triangle_count_match": triangle_count_match,
        "open_boundary_count_match": open_boundary_count_match,
        "triangle_connectivity_match": triangle_connectivity_match,
        "open_boundary_order_match": open_boundary_order_match,
        "finite_positive_depths": finite_positive_depths,
        "coordinate_tolerance_m": coordinate_tolerance_m,
        "coordinate_max_shift_m": coordinate_max_shift_m,
        "coordinate_rms_shift_m": coordinate_rms_shift_m,
        "coordinate_within_tolerance": coordinate_within_tolerance,
        "nonpositive_signed_area_count": roundtrip_nonpositive_signed_area_count,
        "minimum_signed_area_m2": roundtrip_minimum_signed_area_m2,
        "positive_signed_areas": roundtrip_positive_signed_areas,
        "ok": bool(
            node_count_match
            and triangle_count_match
            and open_boundary_count_match
            and triangle_connectivity_match
            and open_boundary_order_match
            and finite_positive_depths
            and coordinate_within_tolerance
            and roundtrip_positive_signed_areas
        ),
    }
    if not roundtrip_positive_signed_areas:
        quality["failure_taxonomy"].append("2dm_roundtrip_nonpositive_area")
    if not quality["roundtrip"]["ok"]:
        quality["failure_taxonomy"].append("2dm_roundtrip_failed")
        quality["accepted"] = False

    quality_json = run_dir / "mesh_quality.json"
    quality_json.write_text(json.dumps(_json_safe(quality), indent=2), encoding="utf-8")
    mesh_gpkg = write_mesh_gpkg(run_dir / "mesh_nodes_elements.gpkg", mesh.nodes_lonlat, mesh.triangles, depths)
    mesh_quality_gpkg = write_mesh_quality_gpkg(
        run_dir / "mesh_quality_elements.gpkg",
        mesh.nodes_lonlat,
        mesh.triangles,
        mesh.nodes_lonlat,
        mesh.triangles,
    )
    review_map = write_mesh_review_map(
        run_dir / "mesh_review_map.png",
        mesh.nodes_lonlat,
        mesh.triangles,
        depths,
        mesh.open_boundary_nodes,
        boundary_package.domain_polygon_lonlat,
        f"{name} FVCOM grid ({'pass' if quality.get('accepted') else 'needs review'})",
    )
    final_status = "pass" if quality.get("accepted") else "needs_review"
    manifest = {
        "schema_version": "fvcom_grid_generation_manifest_v6",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_by": "fvcom-grid-generation run_fvcom_grid.py",
        "final_status": final_status,
        "failure_taxonomy": list(quality.get("failure_taxonomy", [])),
        "settings": {
            "mode": config.mode,
            "backend": "pure_python_oceanmesh_like",
            "land_spacing_m": float(land_spacing),
            "open_spacing_m": float(open_spacing),
            "coarse_smoke": bool(config.coarse_smoke),
            "gradation": float(config.gradation),
            "target_timestep_s": str(config.target_timestep_s),
            "max_interior_points": int(config.max_interior_points),
            "bathy_fallback_policy": config.bathy_fallback_policy,
            "bathy_resolution_policy": config.bathy_resolution_policy,
            "bathy_target_spacing_arcsec": float(config.bathy_target_spacing_arcsec),
            "bathy_max_sources": int(config.bathy_max_sources),
            "size_field_max_cells": int(config.size_field_max_cells),
            "boundary_resolution_profile": config.boundary_resolution_profile,
            "bathymetry_gradient_policy": str(config.bathymetry_gradient_policy),
            "coastal_gradient_distance_m": float(config.coastal_gradient_distance_m),
            "regional_spring_relaxation": bool(config.regional_spring_relaxation),
            "spring_relax_iterations": int(config.spring_relax_iterations),
            "thin_triangle_repair": bool(config.thin_triangle_repair),
            "thin_triangle_min_angle_deg": float(config.thin_triangle_min_angle_deg),
            "thin_triangle_quality_threshold": float(config.thin_triangle_quality_threshold),
            "area_transition_relaxation": bool(config.area_transition_relaxation),
            "area_transition_max_patches": int(config.area_transition_max_patches),
            "area_transition_area_change_threshold": float(config.area_transition_area_change_threshold),
            "area_transition_target_gradient_threshold": float(config.area_transition_target_gradient_threshold),
            "conditioning_profile_requested": str(config.conditioning_profile),
            "conditioning_profile_effective": str(mesh.report.get("conditioning", {}).get("profile", "guarded-v1")),
            "aggressive_conditioning_rounds": int(config.aggressive_conditioning_rounds),
            "aggressive_boundary_edit_policy": str(config.aggressive_boundary_edit_policy),
            "aggressive_max_prunes_per_round": int(config.aggressive_max_prunes_per_round),
            "aggressive_max_valence_repairs_per_round": int(config.aggressive_max_valence_repairs_per_round),
            "postprocess_profile": "none",
        },
        "inputs": {
            "request_text": request_text,
            "bdry_arc_manifest": str(bdry_arc_manifest) if bdry_arc_manifest else None,
            "boundary_loops_gpkg": str(boundary_loops_gpkg),
            "boundary_resolution_manifest": str(boundary_resolution_manifest) if boundary_resolution_manifest else None,
            "bathy_nc": str(bathy_nc),
            "upstream": upstream,
        },
        "bathymetry": {
            "path": str(bathy_nc),
            "loader_metadata": bathy.metadata,
            "source_cell_count": int(bathy.depth.size),
            "size_field_cell_count": int(size_bathy.depth.size),
            "size_field_bathy_metadata": size_bathy.metadata,
            "fetch_metadata": upstream.get("bathy_metadata"),
            "fetch_metadata_json": upstream.get("bathy_metadata_json"),
            "source_id_map": upstream.get("bathy_source_png"),
            "health_check_json": upstream.get("bathy_health_check_json"),
        },
        "mesh": mesh.report,
        "boundary_resolution": resolution_doc,
        "postprocess": {
            "enabled": False,
            "profile": "none",
            "reason": "broad_legacy_cleanup_remains_standalone; normal_workflow_uses_guarded_generation_time_conditioning",
            "standalone_tool": "scripts/postprocess_fvcom_mesh.py",
        },
        "conditioning": mesh.report.get("conditioning", {}),
        "size_field": size_field.report,
        "quality": quality,
        "outputs": {
            "fvcom_grid_2dm": str(output_2dm),
            "fvcom_grid_manifest": str(run_dir / "fvcom_grid_manifest.json"),
            "mesh_quality_json": str(quality_json),
            "mesh_conditioning_json": str(conditioning_json),
            "mesh_edit_ledger_json": str(edit_ledger_json),
            "mesh_review_map": str(review_map),
            "size_field_nc": str(size_nc),
            "size_field_png": str(size_png),
            "boundary_nodes_geojson": str(boundary_nodes_path),
            "boundary_contract_v2_json": str(boundary_contract_path) if boundary_contract_path else None,
            "node_budget_preflight_v2_json": str(node_budget_path) if node_budget_path else None,
            "delivered_boundary_nodes_geojson": str(delivered_boundary_nodes_path),
            "mesh_nodes_elements_gpkg": str(mesh_gpkg),
            "mesh_quality_elements_gpkg": str(mesh_quality_gpkg),
            "progress_json": str(progress.json_path),
            "progress_jsonl": str(progress.jsonl_path),
        },
    }
    manifest_path = run_dir / "fvcom_grid_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    progress.update("complete", 100.0, message=f"FVCOM grid workflow complete with status {final_status}.", artifact=manifest_path)
    return manifest


def _size_field_semantics_v2(bathy: Any, boundary: Any) -> SizeFieldSemantics:
    """Rasterize adaptive-v2 junction and retained-channel semantics."""
    lon2, lat2 = np.meshgrid(np.asarray(bathy.lon, dtype=float), np.asarray(bathy.lat, dtype=float))
    query_xy = project_points(np.column_stack([lon2.ravel(), lat2.ravel()]), boundary.projection)
    shape = lon2.shape
    domain = boundary.domain_polygon_xy
    domain_mask = np.asarray(
        contains_xy(domain, query_xy[:, 0], query_xy[:, 1]),
        dtype=bool,
    ).reshape(shape)
    coverage_mask = np.isfinite(np.asarray(bathy.depth, dtype=float))

    channel_size_flat = np.full(len(query_xy), np.inf, dtype=float)
    for record in boundary.passage_diagnostics or []:
        action = str(record.get("action", "")).strip().lower()
        resolvable = bool(record.get("resolvable_at_minimum_spacing", False))
        geometry = record.get("geometry_xy")
        target = _finite_positive(record.get("required_target_spacing_m"))
        width = _finite_positive(record.get("width_m"))
        if geometry is None or geometry.is_empty or target is None or width is None:
            continue
        if not resolvable or action not in {"harmonize_paired_spacing", "retain", "keep", "resolved"}:
            continue
        distance = _point_to_polyline_distance(query_xy, np.asarray(geometry.coords, dtype=float))
        active = distance <= max(float(width), 1_000.0)
        channel_size_flat[active] = np.minimum(channel_size_flat[active], float(target))
    channel_size = channel_size_flat.reshape(shape)
    channel_size[~np.isfinite(channel_size)] = np.nan
    channel_size[~domain_mask] = np.nan

    return SizeFieldSemantics(
        channel_size_m=channel_size,
        coverage_mask=coverage_mask,
        domain_mask=domain_mask,
    )


def _finite_positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) and result > 0.0 else None


def _point_to_polyline_distance(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Return vectorized Euclidean distance from points to a short polyline."""
    values = np.asarray(points, dtype=float)
    vertices = np.asarray(line, dtype=float)
    if len(vertices) < 2:
        return np.full(len(values), np.inf, dtype=float)
    best = np.full(len(values), np.inf, dtype=float)
    for start, end in zip(vertices[:-1], vertices[1:]):
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1.0e-20:
            candidate = np.linalg.norm(values - start, axis=1)
        else:
            fraction = np.clip(((values - start) @ segment) / denominator, 0.0, 1.0)
            closest = start + fraction[:, None] * segment
            candidate = np.linalg.norm(values - closest, axis=1)
        best = np.minimum(best, candidate)
    return best


def _resolve_upstream_artifacts(
    run_dir: Path,
    name: str,
    request_text: str | None,
    region_bpoly_json: str | Path | None,
    offshore_artifacts_json: str | Path | None,
    bdry_arc_manifest: str | Path | None,
    boundary_loops_gpkg: str | Path | None,
    boundary_resolution_manifest: str | Path | None,
    bathy_nc: str | Path | None,
    land_spacing: float,
    config: GridConfig,
    progress: ProgressTracker,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if boundary_resolution_manifest and not boundary_loops_gpkg:
        resolution_doc = json.loads(Path(boundary_resolution_manifest).read_text(encoding="utf-8-sig"))
        boundary_loops_gpkg = resolution_doc.get("inputs", {}).get("model_boundary_loops_gpkg")
    if boundary_loops_gpkg and bathy_nc:
        result.update(
            {
                "region_bpoly_json": str(region_bpoly_json) if region_bpoly_json else None,
                "offshore_artifacts_json": str(offshore_artifacts_json) if offshore_artifacts_json else None,
                "bdry_arc_manifest": str(bdry_arc_manifest) if bdry_arc_manifest else None,
                "boundary_loops_gpkg": str(boundary_loops_gpkg),
                "boundary_resolution_manifest": str(boundary_resolution_manifest) if boundary_resolution_manifest else None,
                "bathy_nc": str(bathy_nc),
                "source": "supplied_artifacts",
            }
        )
        return result

    upstream_dir = run_dir / "upstream"
    upstream_dir.mkdir(parents=True, exist_ok=True)
    bpoly_dir = upstream_dir / "region_bpoly"
    bdry_dir = upstream_dir / "bdry_arc"
    bathy_dir = upstream_dir / "cudem_bathy"
    bpoly_skill = _find_skill("fvcom-region-bpoly", catalog_relative=("grid-generation", "fvcom-region-bpoly"))
    bdry_skill = _find_skill("fvcom-bdry-arc", catalog_relative=("grid-generation", "fvcom-bdry-arc"))
    cudem_skill = _find_skill("cudem-bathy", catalog_relative=("external-data-connectors", "cudem-bathy"))

    if boundary_loops_gpkg and bathy_nc is None:
        if not region_bpoly_json:
            raise ValueError("Provide --region-bpoly-json when fetching bathymetry for supplied --boundary-loops-gpkg")
        bathy_info = _fetch_bathy_sources(
            cudem_skill,
            Path(region_bpoly_json),
            bathy_dir,
            name,
            config,
            progress,
        )
        result.update(
            {
                "region_bpoly_json": str(region_bpoly_json),
                "offshore_artifacts_json": str(offshore_artifacts_json) if offshore_artifacts_json else None,
                "bdry_arc_manifest": str(bdry_arc_manifest) if bdry_arc_manifest else None,
                "boundary_loops_gpkg": str(boundary_loops_gpkg),
                "source": "supplied_boundary_generated_bathy",
                **bathy_info,
            }
        )
        return result

    if not request_text:
        raise ValueError("Provide --request-text, or provide --boundary-loops-gpkg with --region-bpoly-json so bathymetry can be fetched.")

    region_bpoly_json = Path(region_bpoly_json) if region_bpoly_json else bpoly_dir / "region_bpoly.json"
    offshore_artifacts_json = Path(offshore_artifacts_json) if offshore_artifacts_json else bpoly_dir / "offshore_boundary_artifacts.json"
    if not region_bpoly_json.exists() or not offshore_artifacts_json.exists():
        progress.update("run_region_bpoly", 3.0, message="Running fvcom-region-bpoly.", artifact=bpoly_dir)
        _run(
            [
                sys.executable,
                str(bpoly_skill / "scripts" / "run_region_bpoly.py"),
                "--request-text",
                request_text,
                "--run-dir",
                str(bpoly_dir),
                "--name",
                f"{name}_bpoly",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
            ],
            progress=progress,
            stage="run_region_bpoly",
            percent=5.0,
        )

    bdry_arc_manifest = Path(bdry_arc_manifest) if bdry_arc_manifest else bdry_dir / "bdry_arc_manifest.json"
    if not bdry_arc_manifest.exists():
        progress.update("run_bdry_arc", 7.0, message="Running fvcom-bdry-arc.", artifact=bdry_dir)
        _run(
            [
                sys.executable,
                str(bdry_skill / "scripts" / "run_bdry_arc.py"),
                "--region-bpoly-json",
                str(region_bpoly_json),
                "--offshore-artifacts-json",
                str(offshore_artifacts_json),
                "--fetch-coastline",
                "--run-dir",
                str(bdry_dir),
                "--name",
                f"{name}_bdry",
                "--mode",
                "test",
                "--target-resolution-m",
                str(max(land_spacing, 250.0)),
                "--gshhs-resolution",
                "f",
                "--boundary-resolution-profile",
                str(config.boundary_resolution_profile),
            ],
            progress=progress,
            stage="run_bdry_arc",
            percent=11.0,
        )
    bdry_doc = json.loads(Path(bdry_arc_manifest).read_text(encoding="utf-8-sig"))
    boundary_loops_gpkg = Path(boundary_loops_gpkg or bdry_doc["outputs"]["model_boundary_loops_gpkg"])
    boundary_resolution_manifest = boundary_resolution_manifest or bdry_doc.get("outputs", {}).get("boundary_resolution_manifest")
    if config.boundary_resolution_profile in {"adaptive-coastal-v1", "adaptive-coastal-v2"} and not boundary_resolution_manifest:
        raise ValueError(f"{config.boundary_resolution_profile} requires a passing boundary_resolution_manifest")

    if bathy_nc is None:
        bathy_info = _fetch_bathy_sources(cudem_skill, Path(region_bpoly_json), bathy_dir, name, config, progress)
        bathy_nc = bathy_info["bathy_nc"]
    else:
        bathy_info = {"bathy_nc": str(bathy_nc)}
    result.update(
        {
            "region_bpoly_json": str(region_bpoly_json),
            "offshore_artifacts_json": str(offshore_artifacts_json),
            "bdry_arc_manifest": str(bdry_arc_manifest),
            "boundary_loops_gpkg": str(boundary_loops_gpkg),
            "boundary_resolution_manifest": str(boundary_resolution_manifest) if boundary_resolution_manifest else None,
            "source": "generated_upstream_chain",
            **bathy_info,
        }
    )
    return result


def _fetch_bathy_sources(
    cudem_skill: Path,
    region_bpoly_json: Path,
    bathy_dir: Path,
    name: str,
    config: GridConfig,
    progress: ProgressTracker,
) -> dict[str, Any]:
    bbox = _region_bbox(region_bpoly_json)
    bathy_dir.mkdir(parents=True, exist_ok=True)
    fetch_name = f"{name}_bathy"
    index_path = bathy_dir / "bathy_source_index.json"
    request_path = bathy_dir / "bathy_request.json"
    nc_path = bathy_dir / f"{fetch_name}_bathy_sources.nc"
    metadata_path = bathy_dir / f"{fetch_name}_metadata.json"
    source_png_path = bathy_dir / f"{fetch_name}_bathy_source_id.png"
    health_path = bathy_dir / "health_check.json"
    request = {
        "name": fetch_name,
        "bbox_wsen": bbox,
        "fallback_policy": config.bathy_fallback_policy,
        "resolution_policy": config.bathy_resolution_policy,
        "target_spacing_arcsec": float(config.bathy_target_spacing_arcsec),
        "max_sources": int(config.bathy_max_sources),
    }
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    if nc_path.exists() and metadata_path.exists():
        progress.update("fetch_bathymetry", 17.0, message="Reusing existing fallback bathymetry product.", artifact=nc_path)
    else:
        progress.update("fetch_bathymetry", 12.0, message="Fetching CUDEM/NBS/CRM/ETOPO fallback bathymetry.", artifact=bathy_dir)
        try:
            _run(
                _bathy_fetch_command(cudem_skill, bbox, bathy_dir, fetch_name, config, index_path),
                progress=progress,
                stage="fetch_bathymetry",
                percent=14.0,
            )
        except RuntimeError as exc:
            required_sources = _parse_required_source_count(str(exc))
            if required_sources is None or required_sources <= int(config.bathy_max_sources):
                raise
            retry_max_sources = required_sources
            request["max_sources"] = retry_max_sources
            request["max_sources_retry_reason"] = "connector_reported_more_intersecting_sources_than_initial_cap"
            request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
            progress.update(
                "fetch_bathymetry",
                14.0,
                message="Retrying fallback bathymetry fetch with raised source cap.",
                artifact=bathy_dir,
                extra={"initial_max_sources": int(config.bathy_max_sources), "retry_max_sources": retry_max_sources},
            )
            _run(
                _bathy_fetch_command(cudem_skill, bbox, bathy_dir, fetch_name, config, index_path, max_sources=retry_max_sources),
                progress=progress,
                stage="fetch_bathymetry_retry",
                percent=15.0,
            )
    if not health_path.exists():
        progress.update("check_bathymetry_health", 16.0, message="Running cudem-bathy health check.", artifact=metadata_path)
        _run(
            [
                sys.executable,
                str(cudem_skill / "scripts" / "check_download_health.py"),
                "--request",
                str(request_path),
                "--run-dir",
                str(bathy_dir),
                "--output",
                str(health_path),
                "--plots-dir",
                str(bathy_dir / "health_plots"),
            ],
            progress=progress,
            stage="check_bathymetry_health",
            percent=17.0,
        )
    metadata = _read_json_if_exists(metadata_path)
    return {
        "bathy_nc": str(nc_path),
        "bathy_metadata_json": str(metadata_path),
        "bathy_metadata": metadata,
        "bathy_source_png": str(source_png_path),
        "bathy_health_check_json": str(health_path),
        "bathy_request_json": str(request_path),
        "bathy_source_index_json": str(index_path),
    }


def _bathy_fetch_command(
    cudem_skill: Path,
    bbox: list[float],
    bathy_dir: Path,
    fetch_name: str,
    config: GridConfig,
    index_path: Path,
    max_sources: int | None = None,
) -> list[str]:
    return [
        sys.executable,
        str(cudem_skill / "scripts" / "fetch_bathy_sources.py"),
        "--bbox",
        *[str(float(v)) for v in bbox],
        "--run-dir",
        str(bathy_dir),
        "--name",
        fetch_name,
        "--index",
        str(index_path),
        "--fallback-policy",
        str(config.bathy_fallback_policy),
        "--resolution-policy",
        str(config.bathy_resolution_policy),
        "--target-spacing-arcsec",
        str(float(config.bathy_target_spacing_arcsec)),
        "--max-sources",
        str(int(max_sources if max_sources is not None else config.bathy_max_sources)),
    ]


def _region_bbox(region_bpoly_json: Path) -> list[float]:
    region_doc = json.loads(region_bpoly_json.read_text(encoding="utf-8-sig"))
    bbox = region_doc.get("envelope_bbox")
    if not bbox:
        bbox = region_doc.get("region_bpoly", {}).get("envelope_bbox")
    if not bbox:
        raise ValueError("Cannot fetch bathymetry without region_bpoly envelope_bbox")
    return [float(value) for value in bbox]


def _parse_required_source_count(text: str) -> int | None:
    match = re.search(r"(\d+)\s+sources\s+intersect\s+bbox,\s+exceeding\s+max_sources", text)
    return int(match.group(1)) if match else None


def _find_skill(skill_name: str, catalog_relative: tuple[str, str]) -> Path:
    installed = Path.home() / ".codex" / "skills" / skill_name
    if installed.exists():
        return installed
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "skill_catalog" / catalog_relative[0] / catalog_relative[1]
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate skill {skill_name}")


def _run(cmd: list[str], *, progress: ProgressTracker | None = None, stage: str = "subprocess", percent: float = 0.0) -> None:
    log_dir = (progress.run_dir if progress else Path.cwd()) / "subprocess_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stage)
    stdout_path = log_dir / f"{safe_stage}.stdout.txt"
    stderr_path = log_dir / f"{safe_stage}.stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(cmd, text=True, stdout=stdout, stderr=stderr)
        if progress:
            progress.update(
                stage,
                percent,
                message=f"Started subprocess: {Path(cmd[1]).name if len(cmd) > 1 else cmd[0]}",
                pid=proc.pid,
                artifact=stdout_path,
                extra={"cmd": cmd, "stderr_log": stderr_path},
            )
        interval = max(float(getattr(progress, "interval_s", 10.0)), 1.0)
        while proc.poll() is None:
            time.sleep(interval)
            if progress:
                progress.update(
                    stage,
                    percent,
                    message="Subprocess still running.",
                    pid=proc.pid,
                    artifact=stdout_path,
                    extra={"stderr_log": stderr_path},
                )
        code = proc.returncode
    if code != 0:
        stdout_tail = _tail_text(stdout_path)
        stderr_tail = _tail_text(stderr_path)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT={stdout_tail}\nSTDERR={stderr_tail}")
    if progress:
        progress.update(stage, percent + 1.0, message="Subprocess finished.", artifact=stdout_path, extra={"stderr_log": stderr_path})

def _tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _element_target_sizes(
    nodes_xy: np.ndarray,
    triangles_1based: np.ndarray,
    boundary_nodes: Any,
    size_field: Any,
) -> np.ndarray:
    triangles = np.asarray(triangles_1based, dtype=int) - 1
    if not len(triangles):
        return np.empty(0, dtype=float)
    centroids_xy = np.mean(np.asarray(nodes_xy, dtype=float)[triangles], axis=1)
    centroids_lonlat = unproject_points(centroids_xy, boundary_nodes.projection)
    return size_field.sample(centroids_lonlat[:, 0], centroids_lonlat[:, 1])


def _open_boundary_size_error(nodes_xy: np.ndarray, open_nodes_one_based: np.ndarray, node_targets: np.ndarray) -> dict[str, Any]:
    indices = np.asarray(open_nodes_one_based, dtype=int) - 1
    indices = indices[(indices >= 0) & (indices < len(nodes_xy))]
    if len(indices) < 2:
        return {
            "definition": "consecutive_open_boundary_edge_length_over_mean_endpoint_target",
            "edge_count": 0,
            "p95_l_over_h": 0.0,
            "maximum_l_over_h": 0.0,
            "count_above_1_55": 0,
            "count_above_2_0": 0,
        }
    lengths = np.linalg.norm(np.diff(np.asarray(nodes_xy, dtype=float)[indices], axis=0), axis=1)
    targets = np.asarray(node_targets, dtype=float)
    endpoint_h = 0.5 * (targets[indices[:-1]] + targets[indices[1:]])
    fallback = float(np.nanmedian(endpoint_h[np.isfinite(endpoint_h)])) if np.isfinite(endpoint_h).any() else 1.0
    endpoint_h = np.where(np.isfinite(endpoint_h) & (endpoint_h > 0.0), endpoint_h, fallback)
    ratio = lengths / np.maximum(endpoint_h, 1.0)
    return {
        "definition": "consecutive_open_boundary_edge_length_over_mean_endpoint_target",
        "edge_count": int(len(ratio)),
        "p95_l_over_h": float(np.quantile(ratio, 0.95)),
        "maximum_l_over_h": float(np.max(ratio)),
        "median_l_over_h": float(np.median(ratio)),
        "count_above_1_55": int(np.count_nonzero(ratio > 1.55)),
        "count_above_2_0": int(np.count_nonzero(ratio > 2.0)),
    }


def _postclean_boundary_nodes_geojson(
    nodes_lonlat: np.ndarray,
    constraint_chains: list[list[int]],
    open_boundary_nodes_1based: np.ndarray,
    *,
    boundary_kinds: list[str] | None = None,
    hard_anchor_mask: np.ndarray | None = None,
    node_lineage: np.ndarray | None = None,
    target_spacing_m: np.ndarray | None = None,
) -> dict[str, Any]:
    open_set = set((np.asarray(open_boundary_nodes_1based, dtype=int) - 1).tolist())
    features: list[dict[str, Any]] = []
    seen: set[int] = set()
    kinds = list(boundary_kinds or ["boundary"] * len(nodes_lonlat))
    hard = np.asarray(hard_anchor_mask if hard_anchor_mask is not None else np.zeros(len(nodes_lonlat), dtype=bool), dtype=bool)
    lineage = np.asarray(node_lineage if node_lineage is not None else np.arange(len(nodes_lonlat), dtype=int), dtype=int)
    targets = np.asarray(target_spacing_m if target_spacing_m is not None else np.full(len(nodes_lonlat), np.nan), dtype=float)
    for chain_id, chain in enumerate(constraint_chains):
        for position, node in enumerate(chain):
            if int(node) in seen:
                continue
            seen.add(int(node))
            lon, lat = nodes_lonlat[int(node)]
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "node_index_zero_based": int(node),
                        "node_id_1based": int(node) + 1,
                        "constraint_chain_id": int(chain_id),
                        "constraint_chain_position": int(position),
                        "is_open_boundary": bool(int(node) in open_set),
                        "boundary_kind": str(kinds[int(node)]),
                        "is_hard_anchor": bool(hard[int(node)]),
                        "source_node_index_zero_based": int(lineage[int(node)]) if lineage[int(node)] >= 0 else None,
                        "is_inserted_by_conditioning": bool(lineage[int(node)] < 0),
                        "target_spacing_m": float(targets[int(node)]) if np.isfinite(targets[int(node)]) else None,
                    },
                    "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                }
            )
    return {"type": "FeatureCollection", "features": features}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
