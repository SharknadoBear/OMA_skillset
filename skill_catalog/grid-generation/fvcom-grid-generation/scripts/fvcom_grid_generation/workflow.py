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

from .bathymetry import coarsen_for_size_field, load_bathymetry
from .boundary import BoundaryConfig, boundary_nodes_geojson, load_boundary_package, prepare_boundary_nodes
from .mesh import MeshConfig, generate_mesh
from .plotting import write_mesh_gpkg, write_mesh_review_map
from .progress import ProgressTracker
from .quality import evaluate_mesh_quality
from .size_field import SizeFieldConfig, build_size_field, write_size_field
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
    refine_iterations: int = 3
    smooth_iterations: int = 8
    fetch_bathymetry: bool = True
    bathy_fallback_policy: str = "cudem-nbs-crm-etopo"
    bathy_resolution_policy: str = "source-priority"
    bathy_target_spacing_arcsec: float = 1.0
    bathy_max_sources: int = 256
    progress_interval_s: float = 10.0
    size_field_max_cells: int = 1_500_000


def run_fvcom_grid(
    run_dir: str | Path,
    name: str,
    config: GridConfig | None = None,
    request_text: str | None = None,
    region_bpoly_json: str | Path | None = None,
    offshore_artifacts_json: str | Path | None = None,
    bdry_arc_manifest: str | Path | None = None,
    boundary_loops_gpkg: str | Path | None = None,
    bathy_nc: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete mesh workflow and write artifacts."""
    config = config or GridConfig()
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
        bathy_nc,
        land_spacing,
        config,
        progress,
    )
    boundary_loops_gpkg = upstream["boundary_loops_gpkg"]
    bathy_nc = upstream["bathy_nc"]
    bdry_arc_manifest = upstream.get("bdry_arc_manifest")

    progress.update("load_boundary_loops", 18.0, message="Loading model boundary-loop package.", artifact=boundary_loops_gpkg)
    boundary_package = load_boundary_package(boundary_loops_gpkg)
    progress.update("prepare_boundary_nodes", 22.0, message="Densifying classified boundary nodes.")
    boundary_nodes = prepare_boundary_nodes(
        boundary_package,
        BoundaryConfig(land_spacing_m=land_spacing, open_spacing_m=open_spacing, island_spacing_m=land_spacing),
    )
    boundary_nodes_path = run_dir / "boundary_nodes.geojson"
    boundary_nodes_path.write_text(json.dumps(boundary_nodes_geojson(boundary_nodes), indent=2), encoding="utf-8")

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
        gradation=float(config.gradation),
        target_timestep_s=str(config.target_timestep_s),
    )
    progress.update("build_size_field", 34.0, message="Building bathymetry and shoreline-based mesh-size field.")
    size_field = build_size_field(size_bathy, boundary_nodes, size_config)
    progress.update("write_size_field", 46.0, message="Writing size-field artifacts.")
    size_nc, size_png = write_size_field(size_field, run_dir / "size_field.nc", run_dir / "size_field.png")

    mesh_config = MeshConfig(
        refine_iterations=int(config.refine_iterations),
        smooth_iterations=int(config.smooth_iterations),
        max_interior_points=int(config.max_interior_points),
    )
    def _mesh_progress(message: str, fraction: float, extra: dict[str, Any] | None = None) -> None:
        progress.update(
            "mesh_generation",
            50.0 + 28.0 * max(0.0, min(1.0, float(fraction))),
            message=message,
            extra=extra,
        )

    mesh = generate_mesh(boundary_nodes, size_field, mesh_config, progress_callback=_mesh_progress)
    progress.update("sample_bathymetry_to_mesh", 80.0, message="Sampling bathymetry to mesh nodes.")
    depths = bathy.sample(mesh.nodes_lonlat[:, 0], mesh.nodes_lonlat[:, 1], fill_value=float(np.nanmedian(bathy.depth)))
    depths = np.maximum(np.where(np.isfinite(depths), depths, 2.0), 0.5)
    progress.update("mesh_quality", 84.0, message="Evaluating FVCOM mesh quality gates.")
    quality = evaluate_mesh_quality(
        mesh.nodes_xy,
        depths,
        mesh.triangles,
        mesh.open_boundary_nodes,
        mesh.report.get("constraint_recovery", {}),
    )

    progress.update("write_outputs", 88.0, message="Writing FVCOM .2dm and QA artifacts.")
    output_2dm = write_2dm(run_dir / "fvcom_grid.2dm", mesh.nodes_lonlat, depths, mesh.triangles, mesh.open_boundary_nodes, mesh_name=name)
    roundtrip = read_2dm(output_2dm)
    quality["roundtrip"] = {
        "node_count": int(len(roundtrip.nodes_lonlat)),
        "triangle_count": int(len(roundtrip.triangles)),
        "open_boundary_node_count": int(len(roundtrip.open_boundary_nodes)),
        "ok": bool(
            len(roundtrip.nodes_lonlat) == len(mesh.nodes_lonlat)
            and len(roundtrip.triangles) == len(mesh.triangles)
            and len(roundtrip.open_boundary_nodes) == len(mesh.open_boundary_nodes)
        ),
    }
    if not quality["roundtrip"]["ok"]:
        quality["failure_taxonomy"].append("2dm_roundtrip_failed")
        quality["accepted"] = False

    quality_json = run_dir / "mesh_quality.json"
    quality_json.write_text(json.dumps(_json_safe(quality), indent=2), encoding="utf-8")
    mesh_gpkg = write_mesh_gpkg(run_dir / "mesh_nodes_elements.gpkg", mesh.nodes_lonlat, mesh.triangles, depths)
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
        "schema_version": "fvcom_grid_generation_manifest_v1",
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
        },
        "inputs": {
            "request_text": request_text,
            "bdry_arc_manifest": str(bdry_arc_manifest) if bdry_arc_manifest else None,
            "boundary_loops_gpkg": str(boundary_loops_gpkg),
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
        "size_field": size_field.report,
        "quality": quality,
        "outputs": {
            "fvcom_grid_2dm": str(output_2dm),
            "fvcom_grid_manifest": str(run_dir / "fvcom_grid_manifest.json"),
            "mesh_quality_json": str(quality_json),
            "mesh_review_map": str(review_map),
            "size_field_nc": str(size_nc),
            "size_field_png": str(size_png),
            "boundary_nodes_geojson": str(boundary_nodes_path),
            "mesh_nodes_elements_gpkg": str(mesh_gpkg),
            "progress_json": str(progress.json_path),
            "progress_jsonl": str(progress.jsonl_path),
        },
    }
    manifest_path = run_dir / "fvcom_grid_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    progress.update("complete", 100.0, message=f"FVCOM grid workflow complete with status {final_status}.", artifact=manifest_path)
    return manifest


def _resolve_upstream_artifacts(
    run_dir: Path,
    name: str,
    request_text: str | None,
    region_bpoly_json: str | Path | None,
    offshore_artifacts_json: str | Path | None,
    bdry_arc_manifest: str | Path | None,
    boundary_loops_gpkg: str | Path | None,
    bathy_nc: str | Path | None,
    land_spacing: float,
    config: GridConfig,
    progress: ProgressTracker,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if boundary_loops_gpkg and bathy_nc:
        result.update(
            {
                "region_bpoly_json": str(region_bpoly_json) if region_bpoly_json else None,
                "offshore_artifacts_json": str(offshore_artifacts_json) if offshore_artifacts_json else None,
                "bdry_arc_manifest": str(bdry_arc_manifest) if bdry_arc_manifest else None,
                "boundary_loops_gpkg": str(boundary_loops_gpkg),
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
            ],
            progress=progress,
            stage="run_bdry_arc",
            percent=11.0,
        )
    bdry_doc = json.loads(Path(bdry_arc_manifest).read_text(encoding="utf-8-sig"))
    boundary_loops_gpkg = Path(boundary_loops_gpkg or bdry_doc["outputs"]["model_boundary_loops_gpkg"])

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
