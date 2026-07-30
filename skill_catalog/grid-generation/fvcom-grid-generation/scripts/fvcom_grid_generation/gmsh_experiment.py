"""Research-only Gmsh experiment orchestration for FVCOM grids.

This module is intentionally not imported by the production package entry
point.  It consumes immutable boundary-loop and bathymetry artifacts, applies
the six-case experiment contract, and delegates triangulation to the optional
Gmsh 4.15.2 backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .bathymetry import BathymetryGrid, load_bathymetry
from .projection import (
    LocalProjection,
    local_utm_projection,
    project_geometry,
    project_points,
    unproject_points,
)
from .quality import evaluate_mesh_quality
from .sms_2dm import read_2dm, write_2dm


SCHEMA_VERSION = "gmsh_fvcom_experiment_v1"
GMSH_VERSION = "4.15.2"
DEFAULT_NODE_LIMIT = 150_000
DEFAULT_PREFLIGHT_LIMIT = 135_000
DEFAULT_NEAR_SIZE_M = 8_000.0
DEFAULT_NEAR_DISTANCE_M = 10_000.0
DEFAULT_FAR_DISTANCE_M = 70_000.0
DEFAULT_STEP_M = 25.0


@dataclass(frozen=True)
class SourceOpenBoundary:
    chain_id: str
    kind: str
    cyclic: bool
    orientation: str
    exterior_segment_indices: tuple[int, ...]


@dataclass(frozen=True)
class PreparedCase:
    manifest: dict[str, Any]
    manifest_path: Path
    workspace_root: Path
    projection: LocalProjection
    exterior_xy: np.ndarray
    holes_xy: tuple[np.ndarray, ...]
    exterior_segment_kinds: tuple[str, ...]
    hard_anchor_vertex_indices: tuple[int, ...]
    open_boundaries: tuple[SourceOpenBoundary, ...]
    source_domain_lonlat: Polygon
    bathymetry: BathymetryGrid
    input_paths: dict[str, Path]
    boundary_revalidation: dict[str, Any]
    manifest_sha256: str


@dataclass(frozen=True)
class BudgetConfig:
    max_nodes: int = DEFAULT_NODE_LIMIT
    preflight_nodes: int = DEFAULT_PREFLIGHT_LIMIT
    near_size_m: float = DEFAULT_NEAR_SIZE_M
    near_distance_m: float = DEFAULT_NEAR_DISTANCE_M
    far_distance_m: float = DEFAULT_FAR_DISTANCE_M
    step_m: float = DEFAULT_STEP_M
    integration_max_cells: int = 250_000


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_case_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a research case manifest."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != "gmsh_fvcom_case_v1":
        raise ValueError(f"Unsupported case schema in {path}")
    if not payload.get("case_id") or not isinstance(payload.get("boundary"), dict):
        raise ValueError(f"Case manifest is missing case_id or boundary: {path}")
    input_kind = payload["boundary"].get("input_kind")
    if input_kind not in {"adaptive_v2", "model_loops_v1"}:
        raise ValueError(f"Unsupported boundary.input_kind={input_kind!r}")
    return payload


def resolve_input_path(value: str | None, workspace_root: str | Path) -> Path | None:
    """Resolve a case path without trusting stale absolute paths in provenance."""
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(workspace_root) / path
    return path.resolve()


def _single_wet_polygon(
    frame: gpd.GeoDataFrame,
    *,
    label: str,
) -> Polygon:
    """Require one valid Polygon rather than silently selecting one component."""
    geometries = [
        geometry
        for geometry in frame.geometry
        if geometry is not None and not geometry.is_empty
    ]
    if len(geometries) != 1:
        raise ValueError(
            f"{label} must contain exactly one nonempty wet-domain feature; "
            f"found {len(geometries)}"
        )
    geometry = geometries[0]
    if isinstance(geometry, MultiPolygon):
        raise ValueError(
            f"{label} contains {len(geometry.geoms)} wet components; expected one"
        )
    if not isinstance(geometry, Polygon):
        raise ValueError(f"{label} must contain a Polygon, found {geometry.geom_type}")
    if not geometry.is_valid or geometry.area <= 0.0:
        raise ValueError(f"{label} wet-domain polygon is invalid or empty")
    return geometry


def _path_is_within(candidate: Path, container: Path) -> bool:
    candidate = candidate.resolve()
    container = container.resolve()
    return bool(candidate == container or candidate.is_relative_to(container))


def _reject_negative_fixture_selection(
    manifest: dict[str, Any],
    workspace_root: Path,
) -> None:
    """Reject active inputs located anywhere inside a declared negative fixture."""
    boundary = manifest["boundary"]
    active_values = (
        boundary.get("resolution_manifest"),
        boundary.get("model_boundary_loops_gpkg"),
        boundary.get("model_boundary_loop_manifest"),
    )
    active_paths = [
        resolved
        for value in active_values
        if value
        and (resolved := resolve_input_path(str(value), workspace_root)) is not None
    ]
    for record in boundary.get("negative_rejection_fixtures", []):
        value = record.get("path") if isinstance(record, dict) else record
        fixture = resolve_input_path(str(value), workspace_root) if value else None
        if fixture is None:
            continue
        if any(_path_is_within(active, fixture) for active in active_paths):
            fixture_id = (
                str(record.get("id", fixture.name))
                if isinstance(record, dict)
                else fixture.name
            )
            raise ValueError(
                f"negative rejection fixture {fixture_id!r} was selected as "
                "an active boundary input"
            )


def _exterior_overlap_fraction(
    open_geometry: LineString | MultiLineString,
    exterior: LineString,
    *,
    tolerance_m: float = 1.0,
) -> float:
    if open_geometry.is_empty or open_geometry.length <= 0.0:
        return 1.0 if exterior.is_empty else 0.0
    covered = open_geometry.intersection(exterior.buffer(tolerance_m)).length
    return float(min(1.0, max(0.0, covered / open_geometry.length)))


def _require_metric(
    failures: list[str],
    condition: bool,
    name: str,
) -> None:
    if not condition:
        failures.append(name)


def file_sha256(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def target_size_m(
    distance_to_obc_m: np.ndarray | float,
    h_uniform_m: float,
    *,
    near_size_m: float = DEFAULT_NEAR_SIZE_M,
    near_distance_m: float = DEFAULT_NEAR_DISTANCE_M,
    far_distance_m: float = DEFAULT_FAR_DISTANCE_M,
    has_open_boundary: bool = True,
) -> np.ndarray:
    """Evaluate the experiment's linear Distance/Threshold size contract."""
    values = np.asarray(distance_to_obc_m, dtype=float)
    if not has_open_boundary:
        return np.full(values.shape, float(h_uniform_m), dtype=float)
    fraction = np.clip(
        (values - float(near_distance_m))
        / max(float(far_distance_m) - float(near_distance_m), 1.0e-12),
        0.0,
        1.0,
    )
    return float(near_size_m) + fraction * (float(h_uniform_m) - float(near_size_m))


def validate_reversed_threshold_direction(
    h_uniform_m: float,
    *,
    near_size_m: float = DEFAULT_NEAR_SIZE_M,
    near_distance_m: float = DEFAULT_NEAR_DISTANCE_M,
    far_distance_m: float = DEFAULT_FAR_DISTANCE_M,
) -> dict[str, Any]:
    """Synthetic guard for Gmsh's semantically named SizeMin/SizeMax fields."""
    distances = np.asarray([0.0, near_distance_m, far_distance_m], dtype=float)
    values = target_size_m(
        distances,
        h_uniform_m,
        near_size_m=near_size_m,
        near_distance_m=near_distance_m,
        far_distance_m=far_distance_m,
    )
    expected = np.asarray([near_size_m, near_size_m, h_uniform_m], dtype=float)
    passed = bool(np.allclose(values, expected, rtol=0.0, atol=1.0e-9))
    return {
        "passed": passed,
        "distances_m": distances.tolist(),
        "evaluated_sizes_m": values.tolist(),
        "expected_sizes_m": expected.tolist(),
        "gmsh_threshold_assignment": {
            "SizeMin": float(near_size_m),
            "SizeMax": float(h_uniform_m),
            "DistMin": float(near_distance_m),
            "DistMax": float(far_distance_m),
        },
        "note": (
            "SizeMin and SizeMax are semantic near/far field names; "
            "their numerical values are intentionally reversed when h_u < 8000 m."
        ),
    }


def estimate_node_count(
    boundary_node_count: int,
    integration_area_m2: np.ndarray,
    distance_to_obc_m: np.ndarray,
    h_uniform_m: float,
    *,
    has_open_boundary: bool,
    config: BudgetConfig | None = None,
) -> float:
    """Evaluate N_boundary + integral 2/(sqrt(3) h(x)^2) dA."""
    config = config or BudgetConfig()
    weights = np.asarray(integration_area_m2, dtype=float)
    distance = np.asarray(distance_to_obc_m, dtype=float)
    if weights.shape != distance.shape:
        raise ValueError("integration_area_m2 and distance_to_obc_m must have equal shape")
    sizes = target_size_m(
        distance,
        h_uniform_m,
        near_size_m=config.near_size_m,
        near_distance_m=config.near_distance_m,
        far_distance_m=config.far_distance_m,
        has_open_boundary=has_open_boundary,
    )
    density = 2.0 / (math.sqrt(3.0) * np.square(np.maximum(sizes, 1.0e-12)))
    return float(boundary_node_count) + float(np.sum(weights * density))


def select_uniform_target_m(
    bathymetry_floor_m: float,
    boundary_node_count: int,
    integration_area_m2: np.ndarray,
    distance_to_obc_m: np.ndarray,
    *,
    has_open_boundary: bool,
    config: BudgetConfig | None = None,
) -> tuple[float, float]:
    """Select the smallest 25 m multiple satisfying the preflight threshold."""
    config = config or BudgetConfig()
    step = float(config.step_m)
    lower_index = max(1, int(math.ceil(float(bathymetry_floor_m) / step)))

    def estimate(index: int) -> float:
        return estimate_node_count(
            boundary_node_count,
            integration_area_m2,
            distance_to_obc_m,
            index * step,
            has_open_boundary=has_open_boundary,
            config=config,
        )

    if estimate(lower_index) <= int(config.preflight_nodes):
        value = lower_index * step
        return float(value), float(estimate(lower_index))
    upper_index = lower_index
    while estimate(upper_index) > int(config.preflight_nodes):
        upper_index *= 2
        if upper_index * step > 2_000_000.0:
            raise RuntimeError("No uniform target can satisfy the preflight node budget")
    low = lower_index
    high = upper_index
    while low < high:
        middle = (low + high) // 2
        if estimate(middle) <= int(config.preflight_nodes):
            high = middle
        else:
            low = middle + 1
    selected = low * step
    return float(selected), float(estimate(low))


def _projection_for_case(manifest: dict[str, Any], domain_lonlat: Polygon) -> LocalProjection:
    value = manifest.get("projected_crs")
    if not value:
        return local_utm_projection(tuple(float(item) for item in domain_lonlat.bounds))
    from pyproj import CRS, Transformer

    crs = CRS.from_user_input(str(value))
    epsg = crs.to_epsg()
    if epsg is None:
        raise ValueError(f"projected_crs must resolve to an EPSG code, got {value!r}")
    west, south, east, north = domain_lonlat.bounds
    return LocalProjection(
        crs=crs,
        to_xy=Transformer.from_crs("EPSG:4326", crs, always_xy=True),
        to_lonlat=Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
        epsg=int(epsg),
        lon0=float((west + east) * 0.5),
        lat0=float((south + north) * 0.5),
    )


def _ring_xy(coords: Iterable[Iterable[float]]) -> np.ndarray:
    values = np.asarray([[float(item[0]), float(item[1])] for item in coords], dtype=float)
    if len(values) > 1 and np.linalg.norm(values[0] - values[-1]) <= 1.0e-8:
        values = values[:-1]
    if len(values) < 3:
        raise ValueError("Boundary ring contains fewer than three distinct vertices")
    if np.any(np.linalg.norm(np.roll(values, -1, axis=0) - values, axis=1) <= 1.0e-8):
        raise ValueError("Boundary ring contains a zero-length source segment")
    return values


def _segment_kinds_from_open_geometry(
    exterior_xy: np.ndarray,
    open_geometry_xy: LineString | MultiLineString,
) -> tuple[str, ...]:
    if open_geometry_xy.is_empty:
        return tuple("land" for _ in range(len(exterior_xy)))
    kinds: list[str] = []
    for index, start in enumerate(exterior_xy):
        end = exterior_xy[(index + 1) % len(exterior_xy)]
        segment = LineString([start, end])
        tolerance = max(0.05, min(5.0, float(segment.length) * 1.0e-5))
        overlap = float(segment.intersection(open_geometry_xy.buffer(tolerance)).length)
        kinds.append("open" if overlap >= 0.90 * float(segment.length) else "land")
    return tuple(kinds)


def _contiguous_segment_runs(kinds: Iterable[str]) -> list[tuple[int, ...]]:
    values = [str(value).lower() == "open" for value in kinds]
    if not values or not any(values):
        return []
    if all(values):
        return [tuple(range(len(values)))]
    starts = [
        index
        for index, value in enumerate(values)
        if value and not values[(index - 1) % len(values)]
    ]
    runs: list[tuple[int, ...]] = []
    for start in starts:
        run: list[int] = []
        index = start
        while values[index]:
            run.append(index)
            index = (index + 1) % len(values)
        runs.append(tuple(run))
    return runs


def _open_boundaries_from_runs(
    manifest: dict[str, Any],
    runs: list[tuple[int, ...]],
    exterior_xy: np.ndarray,
) -> tuple[SourceOpenBoundary, ...]:
    boundary = manifest["boundary"]
    expected = int(boundary.get("expected_open_boundary_count", 1))
    if len(runs) != expected:
        raise ValueError(
            f"Detected {len(runs)} OBC runs, expected {expected} for "
            f"{manifest['case_id']}"
        )
    declared = list(boundary.get("open_boundaries") or [])
    if len(declared) != expected:
        raise ValueError("boundary.open_boundaries does not match expected count")
    if expected == 2:
        run_centers = [
            float(np.mean(exterior_xy[np.asarray(run, dtype=int), 0]))
            for run in runs
        ]
        west, east = sorted(range(2), key=lambda index: run_centers[index])
        assigned: dict[str, tuple[int, ...]] = {}
        for record in declared:
            identifier = str(record["id"])
            token = identifier.lower()
            assigned[identifier] = runs[west] if ("west" in token or "river" in token) else runs[east]
        ordered_runs = [assigned[str(record["id"])] for record in declared]
    else:
        ordered_runs = runs
    output = []
    for record, run in zip(declared, ordered_runs):
        orientation = str(record.get("orientation", "source")).lower()
        if orientation not in {"source", "reverse"}:
            orientation = "source"
        output.append(
            SourceOpenBoundary(
                chain_id=str(record["id"]),
                kind=str(record.get("kind", "ocean_exchange")),
                cyclic=bool(record.get("cyclic", False)),
                orientation=orientation,
                exterior_segment_indices=tuple(int(value) for value in run),
            )
        )
    return tuple(output)


def _validate_adaptive_boundary_evidence(
    manifest: dict[str, Any],
    workspace_root: Path,
    resolution_manifest: Path,
    resolution_payload: dict[str, Any],
    domain_xy: Polygon,
    open_geometry: LineString | MultiLineString,
    holes_xy: tuple[np.ndarray, ...],
    open_boundaries: tuple[SourceOpenBoundary, ...],
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Enforce adaptive-package topology and case-specific scientific gates."""
    boundary = manifest["boundary"]
    expected_holes = boundary.get("expected_island_holes")
    expected_open = int(boundary.get("expected_open_boundary_count", 1))
    qa = dict(resolution_payload.get("qa") or {})
    overlap = _exterior_overlap_fraction(open_geometry, domain_xy.exterior)
    failures: list[str] = []
    _require_metric(failures, bool(domain_xy.is_valid and domain_xy.area > 0.0), "wet_domain_invalid")
    _require_metric(failures, len(open_boundaries) == expected_open, "open_boundary_count_mismatch")
    if expected_holes is not None:
        _require_metric(failures, len(holes_xy) == int(expected_holes), "island_hole_count_mismatch")
    if "wet_component_count" in qa:
        _require_metric(failures, int(qa["wet_component_count"]) == 1, "wet_component_count_not_one")
    if "resolved_domain_valid" in qa:
        _require_metric(failures, bool(qa["resolved_domain_valid"]), "upstream_domain_invalid")
    policy_text = " ".join(
        str(value).lower()
        for key in ("required_revalidation", "build_policy")
        for value in boundary.get(key, [])
    )
    requires_overlap_gate = bool(
        str(manifest["case_id"]) == "long_island_sound"
        or "exterior overlap" in policy_text
    )
    if expected_open and requires_overlap_gate:
        _require_metric(failures, overlap >= 0.98, "open_boundary_exterior_overlap_below_0_98")

    evidence_paths: dict[str, Path] = {}
    case_specific: dict[str, Any] = {}
    if str(manifest["case_id"]) == "long_island_sound":
        diagnostics_value = (resolution_payload.get("outputs") or {}).get(
            "boundary_resolution_diagnostics_json"
        )
        diagnostics_path = resolve_input_path(diagnostics_value, workspace_root)
        if diagnostics_path is None or not diagnostics_path.exists():
            diagnostics_path = resolution_manifest.with_name(
                "boundary_resolution_diagnostics.json"
            )
        if not diagnostics_path.exists():
            failures.append("lis_gate_diagnostics_missing")
        else:
            evidence_paths["boundary_revalidation_evidence"] = diagnostics_path
            diagnostics = json.loads(
                diagnostics_path.read_text(encoding="utf-8-sig")
            )
            validation = dict(diagnostics.get("validation") or {})
            gate_validation = dict(validation.get("gate_validation") or {})
            hard_gates = dict(validation.get("hard_gates") or {})
            required_features = dict(validation.get("required_features") or {})
            gates = dict(gate_validation.get("gates") or {})
            _require_metric(failures, validation.get("status") == "pass", "lis_validation_not_pass")
            _require_metric(
                failures,
                all(
                    bool(hard_gates.get(name))
                    for name in (
                        "endpoint_snap",
                        "exterior_overlap",
                        "one_wet_component",
                        "required_feature_containment",
                        "zero_land_crossing",
                    )
                ),
                "lis_hard_gate_failed",
            )
            _require_metric(failures, len(gates) == 2, "lis_gate_count_not_two")
            for gate_id, gate in gates.items():
                candidate_search = dict(gate.get("candidate_search") or {})
                _require_metric(
                    failures,
                    float(gate.get("maximum_endpoint_snap_m", math.inf)) <= 0.05,
                    f"lis_{gate_id}_endpoint_not_shoreline_snapped",
                )
                _require_metric(
                    failures,
                    float(gate.get("land_crossing_m", math.inf)) <= 0.05,
                    f"lis_{gate_id}_land_crossing",
                )
                _require_metric(
                    failures,
                    float(gate.get("exterior_overlap_fraction", 0.0)) >= 0.98,
                    f"lis_{gate_id}_exterior_overlap_below_0_98",
                )
                _require_metric(
                    failures,
                    int(candidate_search.get("selected_rank", 0)) == 1,
                    f"lis_{gate_id}_not_shortest_valid_candidate",
                )
            _require_metric(
                failures,
                float(required_features.get("containment_fraction", 0.0)) >= 1.0,
                "lis_required_feature_containment_failed",
            )
            wet_domain = dict(validation.get("wet_domain") or {})
            _require_metric(
                failures,
                int(wet_domain.get("wet_component_count", 0)) == 1,
                "lis_wet_component_count_not_one",
            )
            case_specific = {
                "diagnostics_path": str(diagnostics_path),
                "validation_status": validation.get("status"),
                "hard_gates": hard_gates,
                "gate_count": len(gates),
                "gate_metrics": {
                    gate_id: {
                        "length_m": gate.get("length_m"),
                        "maximum_endpoint_snap_m": gate.get(
                            "maximum_endpoint_snap_m"
                        ),
                        "land_crossing_m": gate.get("land_crossing_m"),
                        "exterior_overlap_fraction": gate.get(
                            "exterior_overlap_fraction"
                        ),
                        "selected_rank": (
                            gate.get("candidate_search") or {}
                        ).get("selected_rank"),
                    }
                    for gate_id, gate in gates.items()
                },
                "required_feature_containment_fraction": required_features.get(
                    "containment_fraction"
                ),
                "wet_component_count": wet_domain.get("wet_component_count"),
            }
    elif str(manifest["case_id"]) == "lake_ontario":
        _require_metric(failures, expected_open == 0, "lake_expected_open_count_not_zero")
        _require_metric(
            failures,
            int(qa.get("open_boundary_chain_count", -1)) == 0,
            "lake_open_boundary_chain_present",
        )
        _require_metric(
            failures,
            bool(qa.get("outlet_context_preserved")),
            "lake_outlet_context_not_preserved",
        )
        case_specific = {
            "closed_lake": True,
            "outlet_context_preserved": qa.get("outlet_context_preserved"),
        }

    report = {
        "schema_version": "gmsh_boundary_revalidation_v1",
        "case_id": manifest["case_id"],
        "input_kind": "adaptive_v2",
        "checked_at_utc": utc_now(),
        "passed": not failures,
        "failure_taxonomy": failures,
        "source_manifest": str(resolution_manifest),
        "wet_component_count": 1,
        "island_hole_count": len(holes_xy),
        "open_boundary_count": len(open_boundaries),
        "independent_open_boundary_exterior_overlap_fraction": overlap,
        "upstream_qa": qa,
        "case_specific": case_specific,
    }
    if failures:
        raise ValueError(
            "Automatic adaptive boundary revalidation failed: "
            + ", ".join(failures)
        )
    return report, evidence_paths


def _validate_model_loop_boundary_evidence(
    manifest: dict[str, Any],
    workspace_root: Path,
    loop_manifest: Path,
    loop_payload: dict[str, Any],
    gpkg: Path,
    domain_xy: Polygon,
    open_geometry: LineString | MultiLineString,
    holes_xy: tuple[np.ndarray, ...],
    segment_kinds: tuple[str, ...],
    open_boundaries: tuple[SourceOpenBoundary, ...],
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Recompute geometry gates and verify the upstream land-crossing audit."""
    boundary = manifest["boundary"]
    expected_holes = boundary.get("expected_island_holes")
    expected_open = int(boundary.get("expected_open_boundary_count", 1))
    overlap = _exterior_overlap_fraction(open_geometry, domain_xy.exterior)
    qa = dict(loop_payload.get("qa") or {})
    failures: list[str] = []
    _require_metric(failures, loop_payload.get("final_status") == "pass", "upstream_loop_not_pass")
    _require_metric(failures, bool(qa.get("outer_boundary_closed")), "outer_boundary_not_closed")
    _require_metric(failures, bool(domain_xy.is_valid and domain_xy.area > 0.0), "wet_domain_invalid")
    _require_metric(failures, len(open_boundaries) == expected_open, "open_boundary_count_mismatch")
    _require_metric(failures, overlap >= 0.98, "open_boundary_exterior_overlap_below_0_98")
    if expected_holes is not None:
        _require_metric(failures, len(holes_xy) == int(expected_holes), "island_hole_count_mismatch")

    declared_gpkg = resolve_input_path(
        (loop_payload.get("outputs") or {}).get("model_boundary_loops_gpkg"),
        workspace_root,
    )
    _require_metric(
        failures,
        declared_gpkg is not None and declared_gpkg.resolve() == gpkg.resolve(),
        "loop_manifest_gpkg_mismatch",
    )
    evidence_paths: dict[str, Path] = {}
    source_manifest_value = (loop_payload.get("inputs") or {}).get("manifest_json")
    source_manifest = resolve_input_path(source_manifest_value, workspace_root)
    source_wet: dict[str, Any] = {}
    if source_manifest is None or not source_manifest.exists():
        failures.append("boundary_source_manifest_missing")
    else:
        evidence_paths["boundary_revalidation_evidence"] = source_manifest
        source_payload = json.loads(
            source_manifest.read_text(encoding="utf-8-sig")
        )
        source_wet = dict(source_payload.get("wet_domain") or {})
        _require_metric(failures, source_payload.get("final_status") == "pass", "source_boundary_not_pass")
        _require_metric(
            failures,
            int(source_wet.get("face_count", 0)) == 1,
            "source_wet_component_count_not_one",
        )
        _require_metric(
            failures,
            float(source_wet.get("arc_land_intersection_length_m", math.inf))
            <= 0.05,
            "open_arc_land_crossing",
        )
        _require_metric(
            failures,
            float(source_wet.get("open_arc_boundary_overlap_fraction", 0.0))
            >= 0.98,
            "source_open_arc_exterior_overlap_below_0_98",
        )
        if expected_holes is not None:
            _require_metric(
                failures,
                int(source_wet.get("hole_count", -1)) == int(expected_holes),
                "source_island_hole_count_mismatch",
            )

    cyclic = any(value.cyclic for value in open_boundaries)
    if cyclic:
        _require_metric(
            failures,
            bool(segment_kinds) and all(value == "open" for value in segment_kinds),
            "cyclic_exterior_not_fully_open",
        )
        _require_metric(failures, len(open_boundaries) == 1, "cyclic_open_boundary_count_not_one")

    report = {
        "schema_version": "gmsh_boundary_revalidation_v1",
        "case_id": manifest["case_id"],
        "input_kind": "model_loops_v1",
        "checked_at_utc": utc_now(),
        "passed": not failures,
        "failure_taxonomy": failures,
        "loop_manifest": str(loop_manifest),
        "source_manifest": str(source_manifest) if source_manifest else None,
        "wet_component_count": 1,
        "island_hole_count": len(holes_xy),
        "open_boundary_count": len(open_boundaries),
        "independent_open_boundary_exterior_overlap_fraction": overlap,
        "source_arc_land_intersection_m": source_wet.get(
            "arc_land_intersection_length_m"
        ),
        "source_open_arc_boundary_overlap_fraction": source_wet.get(
            "open_arc_boundary_overlap_fraction"
        ),
        "cyclic_exterior_fully_open": (
            bool(segment_kinds) and all(value == "open" for value in segment_kinds)
            if cyclic
            else None
        ),
        "upstream_loop_qa": qa,
    }
    if failures:
        raise ValueError(
            "Automatic model-loop boundary revalidation failed: "
            + ", ".join(failures)
        )
    return report, evidence_paths


def _load_adaptive_geometry(
    manifest: dict[str, Any],
    workspace_root: Path,
) -> tuple[
    LocalProjection,
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[SourceOpenBoundary, ...],
    Polygon,
    dict[str, Path],
    dict[str, Any],
]:
    boundary = manifest["boundary"]
    resolution_manifest = resolve_input_path(boundary.get("resolution_manifest"), workspace_root)
    if resolution_manifest is None or not resolution_manifest.exists():
        raise FileNotFoundError("Adaptive boundary-resolution manifest is unavailable")
    resolution_payload = json.loads(resolution_manifest.read_text(encoding="utf-8-sig"))
    if resolution_payload.get("final_status") != "pass":
        raise ValueError("Adaptive boundary-resolution manifest is not pass")
    gpkg = resolution_manifest.with_name("boundary_resolution.gpkg")
    if not gpkg.exists():
        candidate = resolve_input_path(
            (resolution_payload.get("outputs") or {}).get("boundary_resolution_gpkg"),
            workspace_root,
        )
        if candidate is not None:
            gpkg = candidate
    if not gpkg.exists():
        raise FileNotFoundError(f"Adaptive boundary GeoPackage is unavailable: {gpkg}")
    layers = set(gpd.list_layers(gpkg)["name"])
    required = {"resolved_domain_polygon", "resolved_open_boundary", "boundary_nodes"}
    if missing := required - layers:
        raise ValueError(f"Adaptive GeoPackage is missing layers: {sorted(missing)}")

    domain_gdf = gpd.read_file(
        gpkg, layer="resolved_domain_polygon"
    ).to_crs("EPSG:4326")
    domain_lonlat = _single_wet_polygon(
        domain_gdf,
        label="adaptive resolved_domain_polygon",
    )
    projection = _projection_for_case(manifest, domain_lonlat)
    nodes = gpd.read_file(gpkg, layer="boundary_nodes").to_crs(projection.crs)
    if not {"chain_id", "chain_position"}.issubset(nodes.columns):
        raise ValueError("Adaptive boundary_nodes lacks chain_id/chain_position")
    nodes = nodes.sort_values(["chain_id", "chain_position"]).reset_index(drop=True)
    grouped = list(nodes.groupby("chain_id", sort=True))
    if not grouped:
        raise ValueError("Adaptive boundary package contains no source vertices")
    loop_arrays: list[np.ndarray] = []
    hard_exterior: list[int] = []
    exterior_vertex_kinds: list[str] = []
    for group_index, (_chain_id, group) in enumerate(grouped):
        array = _ring_xy([[point.x, point.y] for point in group.geometry])
        loop_arrays.append(array)
        if group_index == 0:
            exterior_vertex_kinds = [
                str(value).lower() for value in group["boundary_kind"].tolist()
            ]
            if len(exterior_vertex_kinds) != len(array):
                raise ValueError(
                    "Adaptive exterior boundary_kind count differs from source vertices"
                )
            if "is_hard_anchor" in group:
                hard_exterior = [
                    position
                    for position, value in enumerate(group["is_hard_anchor"].tolist())
                    if bool(value)
                ]
    exterior_xy = loop_arrays[0]
    holes_xy = tuple(loop_arrays[1:])
    # boundary_kind is the authoritative adaptive-v2 classification.  Requiring
    # both endpoints to be open yields exactly the ordered nodestring edges and
    # avoids splitting one arc because of sub-meter geometry-overlay gaps.
    segment_kinds = tuple(
        "open"
        if exterior_vertex_kinds[index] == "open"
        and exterior_vertex_kinds[(index + 1) % len(exterior_vertex_kinds)] == "open"
        else "land"
        for index in range(len(exterior_vertex_kinds))
    )
    open_boundaries = _open_boundaries_from_runs(
        manifest,
        _contiguous_segment_runs(segment_kinds),
        exterior_xy,
    )
    resolved_open = gpd.read_file(
        gpkg, layer="resolved_open_boundary"
    ).to_crs(projection.crs)
    open_geometries = [
        geometry
        for geometry in resolved_open.geometry
        if geometry is not None and not geometry.is_empty
    ]
    open_geometry = (
        unary_union(open_geometries) if open_geometries else LineString()
    )
    domain_xy = Polygon(
        exterior_xy,
        holes=[values.tolist() for values in holes_xy],
    )
    revalidation, evidence_paths = _validate_adaptive_boundary_evidence(
        manifest,
        workspace_root,
        resolution_manifest,
        resolution_payload,
        domain_xy,
        open_geometry,
        holes_xy,
        open_boundaries,
    )
    return (
        projection,
        exterior_xy,
        holes_xy,
        segment_kinds,
        tuple(hard_exterior),
        open_boundaries,
        domain_lonlat,
        {
            "boundary_manifest": resolution_manifest,
            "boundary_gpkg": gpkg,
            **evidence_paths,
        },
        revalidation,
    )


def _load_model_loops_geometry(
    manifest: dict[str, Any],
    workspace_root: Path,
) -> tuple[
    LocalProjection,
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[SourceOpenBoundary, ...],
    Polygon,
    dict[str, Path],
    dict[str, Any],
]:
    boundary = manifest["boundary"]
    loop_manifest = resolve_input_path(boundary.get("model_boundary_loop_manifest"), workspace_root)
    gpkg = resolve_input_path(boundary.get("model_boundary_loops_gpkg"), workspace_root)
    if loop_manifest is None or gpkg is None or not loop_manifest.exists() or not gpkg.exists():
        raise FileNotFoundError("Model-loop manifest or GeoPackage is unavailable")
    loop_payload = json.loads(loop_manifest.read_text(encoding="utf-8-sig"))
    if loop_payload.get("final_status") != "pass":
        raise ValueError("Model-loop manifest is not pass")
    layers = set(gpd.list_layers(gpkg)["name"])
    required = {"model_domain_polygon", "model_outer_boundary_segments"}
    if missing := required - layers:
        raise ValueError(f"Model-loop GeoPackage is missing layers: {sorted(missing)}")
    domain_gdf = gpd.read_file(
        gpkg, layer="model_domain_polygon"
    ).to_crs("EPSG:4326")
    domain_lonlat = _single_wet_polygon(
        domain_gdf,
        label="model_domain_polygon",
    )
    projection = _projection_for_case(manifest, domain_lonlat)
    domain_xy = project_geometry(domain_lonlat, projection)
    exterior_xy = _ring_xy(domain_xy.exterior.coords)
    holes_xy = tuple(_ring_xy(ring.coords) for ring in domain_xy.interiors)
    segments = gpd.read_file(gpkg, layer="model_outer_boundary_segments").to_crs(projection.crs)
    open_geometries = [
        row.geometry
        for _, row in segments.iterrows()
        if str(row.get("segment_class", "")).lower() == "open_boundary"
        and row.geometry is not None
        and not row.geometry.is_empty
    ]
    open_geometry = unary_union(open_geometries) if open_geometries else LineString()
    segment_kinds = _segment_kinds_from_open_geometry(exterior_xy, open_geometry)
    transitions = tuple(
        index
        for index in range(len(segment_kinds))
        if segment_kinds[index] != segment_kinds[(index - 1) % len(segment_kinds)]
    )
    open_boundaries = _open_boundaries_from_runs(
        manifest,
        _contiguous_segment_runs(segment_kinds),
        exterior_xy,
    )
    revalidation, evidence_paths = _validate_model_loop_boundary_evidence(
        manifest,
        workspace_root,
        loop_manifest,
        loop_payload,
        gpkg,
        domain_xy,
        open_geometry,
        holes_xy,
        segment_kinds,
        open_boundaries,
    )
    return (
        projection,
        exterior_xy,
        holes_xy,
        segment_kinds,
        transitions,
        open_boundaries,
        domain_lonlat,
        {
            "boundary_manifest": loop_manifest,
            "boundary_gpkg": gpkg,
            **evidence_paths,
        },
        revalidation,
    )


def prepare_case(
    case_manifest_path: str | Path,
    workspace_root: str | Path,
    *,
    bathymetry_override: str | Path | None = None,
    boundary_loop_override: str | Path | None = None,
    adaptive_resolution_override: str | Path | None = None,
) -> PreparedCase:
    """Load source geometry and bathymetry without altering either artifact."""
    manifest_path = Path(case_manifest_path).resolve()
    manifest_sha256 = file_sha256(manifest_path)
    manifest = load_case_manifest(manifest_path)
    manifest = json.loads(json.dumps(manifest))
    if boundary_loop_override is not None and adaptive_resolution_override is not None:
        raise ValueError("Only one boundary input override can be supplied")
    if boundary_loop_override is not None:
        manifest["boundary"]["input_kind"] = "model_loops_v1"
        manifest["boundary"]["model_boundary_loops_gpkg"] = str(boundary_loop_override)
        override_path = Path(boundary_loop_override)
        manifest["boundary"]["model_boundary_loop_manifest"] = str(
            override_path.with_name("model_boundary_loop_manifest.json")
        )
    if adaptive_resolution_override is not None:
        manifest["boundary"]["input_kind"] = "adaptive_v2"
        manifest["boundary"]["resolution_manifest"] = str(adaptive_resolution_override)
    workspace = Path(workspace_root).resolve()
    _reject_negative_fixture_selection(manifest, workspace)
    if manifest["boundary"]["input_kind"] == "adaptive_v2":
        geometry_values = _load_adaptive_geometry(manifest, workspace)
    else:
        geometry_values = _load_model_loops_geometry(manifest, workspace)
    (
        projection,
        exterior_xy,
        holes_xy,
        segment_kinds,
        hard_anchors,
        open_boundaries,
        domain_lonlat,
        paths,
        boundary_revalidation,
    ) = geometry_values
    expected_holes = manifest["boundary"].get("expected_island_holes")
    if expected_holes is not None and len(holes_xy) != int(expected_holes):
        raise ValueError(
            f"Detected {len(holes_xy)} island holes; expected {expected_holes}"
        )
    bathy_value = (
        str(bathymetry_override)
        if bathymetry_override is not None
        else (manifest.get("bathymetry") or {}).get("netcdf")
    )
    bathy_path = resolve_input_path(bathy_value, workspace)
    if bathy_path is None or not bathy_path.exists():
        raise FileNotFoundError("Full-footprint bathymetry NetCDF is unavailable")
    bathymetry = load_bathymetry(bathy_path)
    paths["bathymetry_netcdf"] = bathy_path
    return PreparedCase(
        manifest=manifest,
        manifest_path=manifest_path,
        workspace_root=workspace,
        projection=projection,
        exterior_xy=exterior_xy,
        holes_xy=holes_xy,
        exterior_segment_kinds=segment_kinds,
        hard_anchor_vertex_indices=hard_anchors,
        open_boundaries=open_boundaries,
        source_domain_lonlat=domain_lonlat,
        bathymetry=bathymetry,
        input_paths=paths,
        boundary_revalidation=boundary_revalidation,
        manifest_sha256=manifest_sha256,
    )


def bathymetry_resolution_floor_m(
    bathymetry: BathymetryGrid,
    projection: LocalProjection,
    *,
    multiplier: float = 3.0,
    roundup_m: float = DEFAULT_STEP_M,
) -> tuple[float, dict[str, float]]:
    """Return 3*sqrt(projected p95 dx*p95 dy), rounded upward to 25 m."""
    lon = np.asarray(bathymetry.lon, dtype=float)
    lat = np.asarray(bathymetry.lat, dtype=float)
    if lon.ndim != 1 or lat.ndim != 1 or len(lon) < 2 or len(lat) < 2:
        raise ValueError("Bathymetry must have at least two 1-D lon/lat coordinates")
    sample_lats = lat[
        np.unique(np.linspace(0, len(lat) - 1, min(9, len(lat))).astype(int))
    ]
    sample_lons = lon[
        np.unique(np.linspace(0, len(lon) - 1, min(9, len(lon))).astype(int))
    ]
    dx_values: list[float] = []
    for latitude in sample_lats:
        points = project_points(
            np.column_stack([lon, np.full(len(lon), float(latitude))]),
            projection,
        )
        dx_values.extend(np.linalg.norm(np.diff(points, axis=0), axis=1).tolist())
    dy_values: list[float] = []
    for longitude in sample_lons:
        points = project_points(
            np.column_stack([np.full(len(lat), float(longitude)), lat]),
            projection,
        )
        dy_values.extend(np.linalg.norm(np.diff(points, axis=0), axis=1).tolist())
    dx_p95 = float(np.quantile(np.asarray(dx_values, dtype=float), 0.95))
    dy_p95 = float(np.quantile(np.asarray(dy_values, dtype=float), 0.95))
    raw = float(multiplier) * math.sqrt(dx_p95 * dy_p95)
    rounded = float(roundup_m) * math.ceil(raw / float(roundup_m))
    return rounded, {
        "projected_cell_dx_p95_m": dx_p95,
        "projected_cell_dy_p95_m": dy_p95,
        "raw_floor_m": raw,
        "selected_floor_m": rounded,
        "multiplier": float(multiplier),
        "roundup_m": float(roundup_m),
    }


def integration_samples(
    prepared: PreparedCase,
    *,
    max_cells: int = 250_000,
) -> dict[str, Any]:
    """Build deterministic equal-area midpoint quadrature over the wet polygon."""
    domain_xy = Polygon(
        prepared.exterior_xy,
        holes=[values.tolist() for values in prepared.holes_xy],
    )
    if not domain_xy.is_valid or domain_xy.is_empty or domain_xy.area <= 0.0:
        raise ValueError("Prepared projected wet-domain polygon is invalid")
    min_x, min_y, max_x, max_y = domain_xy.bounds
    width = max(float(max_x - min_x), 1.0)
    height = max(float(max_y - min_y), 1.0)
    nx = max(2, int(math.sqrt(max_cells * width / height)))
    ny = max(2, int(max_cells // nx))
    dx = width / nx
    dy = height / ny
    x = min_x + (np.arange(nx, dtype=float) + 0.5) * dx
    y = min_y + (np.arange(ny, dtype=float) + 0.5) * dy
    xx, yy = np.meshgrid(x, y)
    wet = np.asarray(shapely.contains_xy(domain_xy, xx, yy), dtype=bool)
    wet_x = xx[wet]
    wet_y = yy[wet]
    if not len(wet_x):
        raise ValueError("Integration grid contains no wet-domain samples")
    weights = np.full(len(wet_x), dx * dy, dtype=float)
    if prepared.open_boundaries:
        lines = []
        for chain in prepared.open_boundaries:
            for index in chain.exterior_segment_indices:
                lines.append(
                    LineString(
                        [
                            prepared.exterior_xy[index],
                            prepared.exterior_xy[
                                (index + 1) % len(prepared.exterior_xy)
                            ],
                        ]
                    )
                )
        open_geometry = unary_union(lines)
        points = shapely.points(wet_x, wet_y)
        distance = np.asarray(shapely.distance(points, open_geometry), dtype=float)
    else:
        distance = np.full(len(wet_x), np.inf, dtype=float)
    return {
        "x": wet_x,
        "y": wet_y,
        "area_weights_m2": weights,
        "distance_to_obc_m": distance,
        "cell_dx_m": float(dx),
        "cell_dy_m": float(dy),
        "sample_count": int(len(wet_x)),
        "quadrature_area_m2": float(np.sum(weights)),
        "polygon_area_m2": float(domain_xy.area),
        "relative_area_error": float(abs(np.sum(weights) - domain_xy.area) / domain_xy.area),
        "grid_shape": [int(ny), int(nx)],
    }


def bathymetry_coverage_report(
    bathymetry: BathymetryGrid,
    domain_lonlat: Polygon,
) -> dict[str, Any]:
    """Audit coordinate monotonicity, footprint, and finite wet-cell coverage."""
    lon = np.asarray(bathymetry.lon, dtype=float)
    lat = np.asarray(bathymetry.lat, dtype=float)
    depth = np.asarray(bathymetry.depth, dtype=float)
    lon_monotonic = bool(np.all(np.diff(lon) > 0.0) or np.all(np.diff(lon) < 0.0))
    lat_monotonic = bool(np.all(np.diff(lat) > 0.0) or np.all(np.diff(lat) < 0.0))
    lon_step = float(np.quantile(np.abs(np.diff(lon)), 0.95)) if len(lon) > 1 else 0.0
    lat_step = float(np.quantile(np.abs(np.diff(lat)), 0.95)) if len(lat) > 1 else 0.0
    west, south, east, north = domain_lonlat.bounds
    bbox_covers = bool(
        float(np.min(lon)) <= west + lon_step
        and float(np.max(lon)) >= east - lon_step
        and float(np.min(lat)) <= south + lat_step
        and float(np.max(lat)) >= north - lat_step
    )
    stride = max(1, int(math.ceil(math.sqrt(depth.size / 500_000))))
    lon_sample = lon[::stride]
    lat_sample = lat[::stride]
    depth_sample = depth[::stride, ::stride]
    llon, llat = np.meshgrid(lon_sample, lat_sample)
    wet = np.asarray(shapely.contains_xy(domain_lonlat, llon, llat), dtype=bool)
    finite_fraction = (
        float(np.count_nonzero(np.isfinite(depth_sample[wet])) / np.count_nonzero(wet))
        if np.count_nonzero(wet)
        else 0.0
    )
    return {
        "lon_monotonic": lon_monotonic,
        "lat_monotonic": lat_monotonic,
        "raster_bbox_wsen": [
            float(np.min(lon)),
            float(np.min(lat)),
            float(np.max(lon)),
            float(np.max(lat)),
        ],
        "domain_bbox_wsen": [float(west), float(south), float(east), float(north)],
        "bbox_covers_domain_with_one_cell_tolerance": bbox_covers,
        "wet_sample_count": int(np.count_nonzero(wet)),
        "finite_wet_fraction": finite_fraction,
        "passed": bool(
            lon_monotonic
            and lat_monotonic
            and bbox_covers
            and finite_fraction >= 0.95
        ),
    }


def check_case_readiness(
    case_manifest_path: str | Path,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Perform a read-only automatic readiness audit for one regional case."""
    manifest_path = Path(case_manifest_path).resolve()
    workspace = Path(workspace_root).resolve()
    manifest = load_case_manifest(manifest_path)
    blockers: list[str] = []
    warnings: list[str] = []
    geometry_report: dict[str, Any] | None = None
    bathy_report: dict[str, Any] | None = None
    paths: dict[str, Path] = {"case_manifest": manifest_path}
    try:
        _reject_negative_fixture_selection(manifest, workspace)
        if manifest["boundary"]["input_kind"] == "adaptive_v2":
            geometry_values = _load_adaptive_geometry(manifest, workspace)
        else:
            geometry_values = _load_model_loops_geometry(manifest, workspace)
        (
            projection,
            exterior,
            holes,
            kinds,
            hard_anchors,
            open_boundaries,
            domain_lonlat,
            geometry_paths,
            boundary_revalidation,
        ) = geometry_values
        paths.update(geometry_paths)
        domain_xy = Polygon(exterior, holes=[values.tolist() for values in holes])
        geometry_report = {
            "projection_epsg": int(projection.epsg),
            "source_vertex_count": int(
                len(exterior) + sum(len(values) for values in holes)
            ),
            "exterior_segment_count": int(len(exterior)),
            "island_hole_count": int(len(holes)),
            "open_boundary_count": int(len(open_boundaries)),
            "open_boundaries": [asdict(value) for value in open_boundaries],
            "hard_anchor_count": int(len(hard_anchors)),
            "domain_valid": bool(domain_xy.is_valid),
            "domain_area_m2": float(domain_xy.area),
            "open_segment_count": int(sum(value == "open" for value in kinds)),
            "automatic_revalidation": boundary_revalidation,
        }
        if not domain_xy.is_valid or domain_xy.area <= 0.0:
            blockers.append("invalid_or_empty_wet_domain")
        expected_holes = manifest["boundary"].get("expected_island_holes")
        if expected_holes is not None and len(holes) != int(expected_holes):
            blockers.append("island_hole_count_mismatch")
        expected_open = int(manifest["boundary"].get("expected_open_boundary_count", 1))
        if len(open_boundaries) != expected_open:
            blockers.append("open_boundary_count_mismatch")
    except Exception as exc:
        blockers.append(f"boundary_not_ready: {exc}")
        domain_lonlat = None
        projection = None

    bathy_path = resolve_input_path((manifest.get("bathymetry") or {}).get("netcdf"), workspace)
    if bathy_path is None or not bathy_path.exists():
        blockers.append("full_footprint_bathymetry_missing")
    else:
        paths["bathymetry_netcdf"] = bathy_path
        try:
            bathymetry = load_bathymetry(bathy_path)
            if domain_lonlat is None or projection is None:
                bathy_report = {"passed": False, "reason": "boundary_unavailable"}
            else:
                bathy_report = bathymetry_coverage_report(bathymetry, domain_lonlat)
                floor, floor_report = bathymetry_resolution_floor_m(
                    bathymetry,
                    projection,
                )
                bathy_report["resolution_floor"] = floor_report
                bathy_report["bathymetry_floor_m"] = float(floor)
                if not bathy_report["passed"]:
                    blockers.append("bathymetry_coverage_contract_failed")
        except Exception as exc:
            blockers.append(f"bathymetry_unreadable: {exc}")

    input_hashes = {
        key: {"path": str(path), "sha256": file_sha256(path)}
        for key, path in paths.items()
        if path.exists() and path.is_file()
    }
    return {
        "schema_version": "gmsh_case_readiness_v1",
        "case_id": manifest["case_id"],
        "checked_at_utc": utc_now(),
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "warnings": warnings,
        "geometry": geometry_report,
        "bathymetry": bathy_report,
        "input_hashes": input_hashes,
    }


def environment_report() -> dict[str, Any]:
    """Capture the interpreter, platform, Gmsh, and resolved dependencies."""
    from importlib import metadata
    from .gmsh_backend import load_pinned_gmsh

    distributions = {}
    for name in (
        "gmsh",
        "geopandas",
        "matplotlib",
        "netCDF4",
        "numpy",
        "pandas",
        "pydap",
        "pyogrio",
        "pyproj",
        "PyYAML",
        "rasterio",
        "requests",
        "scipy",
        "shapely",
        "triangle",
        "xarray",
    ):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            distributions[name] = None
    gmsh = load_pinned_gmsh()
    return {
        "captured_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "gmsh_version": str(gmsh.__version__),
        "dependencies": distributions,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def _backend_geometry(prepared: PreparedCase):
    from .gmsh_backend import (
        GmshGeometry,
        SourceLoop,
        SourceOpenBoundary as BackendOpenBoundary,
    )

    case_id = str(prepared.manifest["case_id"])
    exterior = SourceLoop(
        loop_id="exterior",
        xy=prepared.exterior_xy,
        segment_kinds=prepared.exterior_segment_kinds,
        source_vertex_ids=tuple(
            f"{case_id}:exterior:{index}"
            for index in range(len(prepared.exterior_xy))
        ),
        role="exterior",
    )
    holes = tuple(
        SourceLoop(
            loop_id=f"island_{index:03d}",
            island_id=f"{index:03d}",
            xy=values,
            segment_kinds=tuple("island" for _ in range(len(values))),
            source_vertex_ids=tuple(
                f"{case_id}:island_{index:03d}:{position}"
                for position in range(len(values))
            ),
            role="island",
        )
        for index, values in enumerate(prepared.holes_xy, start=1)
    )
    open_boundaries = tuple(
        BackendOpenBoundary(
            chain_id=value.chain_id,
            kind=value.kind,
            cyclic=value.cyclic,
            orientation=value.orientation,
            exterior_segment_indices=value.exterior_segment_indices,
        )
        for value in prepared.open_boundaries
    )
    return GmshGeometry(
        exterior=exterior,
        holes=holes,
        open_boundaries=open_boundaries,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _delivered_lineage_manifest(prepared: PreparedCase, result: Any) -> dict[str, Any]:
    source_vertex_expected = (
        len(prepared.exterior_xy)
        + sum(len(values) for values in prepared.holes_xy)
    )
    loops = []
    for delivered in result.delivered_loops:
        loops.append(
            {
                "loop_id": delivered.loop_id,
                "role": delivered.role,
                "island_id": delivered.island_id,
                "orientation": delivered.source_orientation,
                "delivered_node_count": len(delivered.node_ids),
                "nodes": [
                    {
                        "mesh_node_id": item.mesh_node_id,
                        "gmsh_node_tag": item.gmsh_node_tag,
                        "source_segment_id": (
                            f"{item.loop_id}:segment:{item.source_segment_index}"
                        ),
                        "source_segment_index": item.source_segment_index,
                        "interpolation_weight": item.interpolation_weight,
                        "normalized_arclength": item.loop_normalized_arclength,
                        "is_source_vertex": item.is_source_vertex,
                    }
                    for item in delivered.lineage
                ],
            }
        )
    open_boundaries = []
    forcing_compatible = True
    for delivered in result.open_boundaries:
        expected_count = len(delivered.source_segment_indices)
        if not delivered.cyclic:
            expected_count += 1
        inserted = sum(not item.is_source_vertex for item in delivered.lineage)
        unchanged = bool(
            inserted == 0
            and len(delivered.node_ids) == expected_count
        )
        forcing_compatible = forcing_compatible and unchanged
        open_boundaries.append(
            {
                "chain_id": delivered.chain_id,
                "kind": delivered.kind,
                "cyclic": delivered.cyclic,
                "orientation": delivered.orientation,
                "source_segment_indices": list(delivered.source_segment_indices),
                "delivered_node_ids": list(delivered.node_ids),
                "inserted_node_count": int(inserted),
                "source_sequence_unchanged": unchanged,
                "nodes": [
                    {
                        "mesh_node_id": item.mesh_node_id,
                        "source_segment_id": (
                            f"{item.loop_id}:segment:{item.source_segment_index}"
                        ),
                        "source_segment_index": item.source_segment_index,
                        "interpolation_weight": item.interpolation_weight,
                        "normalized_arclength": item.chain_normalized_arclength,
                        "loop_normalized_arclength": item.loop_normalized_arclength,
                        "is_source_vertex": item.is_source_vertex,
                    }
                    for item in delivered.lineage
                ],
            }
        )
    if not result.open_boundaries:
        forcing_compatible = True
    return {
        "schema_version": "gmsh_boundary_remap_v1",
        "source_vertex_count": int(source_vertex_expected),
        "mapped_source_vertex_count": int(len(result.source_vertex_node_ids)),
        "all_source_vertices_retained": bool(
            len(result.source_vertex_node_ids) == source_vertex_expected
        ),
        "hard_anchor_count": int(len(prepared.hard_anchor_vertex_indices)),
        "hard_anchors_retained": bool(
            all(
                f"{prepared.manifest['case_id']}:exterior:{index}"
                in result.source_vertex_node_ids
                for index in prepared.hard_anchor_vertex_indices
            )
        ),
        "forcing_compatible": bool(forcing_compatible),
        "forcing_interpolation_performed": False,
        "loops": loops,
        "open_boundaries": open_boundaries,
    }


def _open_geometry_from_result(result: Any) -> LineString | MultiLineString:
    lines = []
    for boundary in result.open_boundaries:
        node_ids = list(boundary.node_ids)
        coordinates = result.nodes_xy[np.asarray(node_ids, dtype=int) - 1]
        if boundary.cyclic:
            coordinates = np.vstack([coordinates, coordinates[0]])
        lines.append(LineString(coordinates))
    return unary_union(lines) if lines else LineString()


def _target_size_by_triangle(
    result: Any,
    h_uniform_m: float,
    config: BudgetConfig,
) -> np.ndarray:
    triangles_zero = np.asarray(result.triangles_1based, dtype=int) - 1
    centroids = np.mean(result.nodes_xy[triangles_zero], axis=1)
    if result.open_boundaries:
        distance = np.asarray(
            shapely.distance(shapely.points(centroids), _open_geometry_from_result(result)),
            dtype=float,
        )
    else:
        distance = np.full(len(centroids), np.inf, dtype=float)
    return target_size_m(
        distance,
        h_uniform_m,
        near_size_m=config.near_size_m,
        near_distance_m=config.near_distance_m,
        far_distance_m=config.far_distance_m,
        has_open_boundary=bool(result.open_boundaries),
    )


def _roundtrip_report(
    output_path: Path,
    prepared: PreparedCase,
    result: Any,
    depths: np.ndarray,
) -> dict[str, Any]:
    nodes_lonlat = unproject_points(result.nodes_xy, prepared.projection)
    open_chains = [list(boundary.node_ids) for boundary in result.open_boundaries]
    write_2dm(
        output_path,
        nodes_lonlat,
        depths,
        result.triangles_1based,
        np.empty(0, dtype=int),
        mesh_name=f"{prepared.manifest['case_id']}_gmsh_research",
        open_boundary_chains=open_chains,
        open_boundary_ids=range(1, len(open_chains) + 1),
    )
    parsed = read_2dm(output_path)
    parsed_xy = project_points(parsed.nodes_lonlat, prepared.projection)
    shifts = np.linalg.norm(parsed_xy - result.nodes_xy, axis=1)
    chain_equal = bool(
        len(parsed.open_boundary_chains) == len(open_chains)
        and all(
            np.array_equal(actual, np.asarray(expected, dtype=int))
            for actual, expected in zip(parsed.open_boundary_chains, open_chains)
        )
    )
    triangle_equal = bool(
        np.array_equal(parsed.triangles, np.asarray(result.triangles_1based, dtype=int))
    )
    maximum_shift = float(np.max(shifts)) if len(shifts) else 0.0
    return {
        "passed": bool(
            chain_equal
            and triangle_equal
            and len(parsed.nodes_lonlat) == len(result.nodes_xy)
            and maximum_shift < 0.01
        ),
        "open_boundary_chain_count": int(len(open_chains)),
        "nodestring_ids": list(parsed.open_boundary_ids),
        "open_boundary_order_exact": chain_equal,
        "triangle_connectivity_exact": triangle_equal,
        "node_count_exact": bool(len(parsed.nodes_lonlat) == len(result.nodes_xy)),
        "maximum_projected_coordinate_shift_m": maximum_shift,
        "coordinate_shift_threshold_m": 0.01,
    }


def _obc_normality_report(result: Any) -> dict[str, Any]:
    triangles = np.asarray(result.triangles_1based, dtype=int) - 1
    edge_to_opposite: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for a, b, opposite in (
            (triangle[0], triangle[1], triangle[2]),
            (triangle[1], triangle[2], triangle[0]),
            (triangle[2], triangle[0], triangle[1]),
        ):
            edge = tuple(sorted((int(a), int(b))))
            if edge not in edge_to_opposite:
                edge_to_opposite[edge] = int(opposite)
            else:
                edge_to_opposite[edge] = -1
    values: list[float] = []
    chain_reports = []
    for boundary in result.open_boundaries:
        nodes = [int(value) - 1 for value in boundary.node_ids]
        pairs = list(zip(nodes[:-1], nodes[1:]))
        if boundary.cyclic and len(nodes) > 1:
            pairs.append((nodes[-1], nodes[0]))
        chain_values: list[float] = []
        for a, b in pairs:
            opposite = edge_to_opposite.get(tuple(sorted((a, b))), -1)
            if opposite < 0:
                continue
            edge_vector = result.nodes_xy[b] - result.nodes_xy[a]
            midpoint_vector = (
                result.nodes_xy[opposite]
                - 0.5 * (result.nodes_xy[a] + result.nodes_xy[b])
            )
            denom = max(
                float(np.linalg.norm(edge_vector) * np.linalg.norm(midpoint_vector)),
                1.0e-30,
            )
            angle = math.degrees(
                math.acos(
                    float(np.clip(abs(np.dot(edge_vector, midpoint_vector)) / denom, 0.0, 1.0))
                )
            )
            deviation = abs(90.0 - angle)
            chain_values.append(deviation)
            values.append(deviation)
        chain_reports.append(
            {
                "chain_id": boundary.chain_id,
                "edge_count": int(len(chain_values)),
                "median_deviation_deg": (
                    float(np.median(chain_values)) if chain_values else None
                ),
                "maximum_deviation_deg": (
                    float(np.max(chain_values)) if chain_values else None
                ),
            }
        )
    array = np.asarray(values, dtype=float)
    return {
        "definition": (
            "absolute deviation from a 90-degree angle between each OBC edge "
            "and its midpoint-to-adjacent-triangle-apex vector"
        ),
        "advisory_only": True,
        "edge_count": int(len(array)),
        "median_deviation_deg": float(np.median(array)) if len(array) else None,
        "p95_deviation_deg": float(np.quantile(array, 0.95)) if len(array) else None,
        "maximum_deviation_deg": float(np.max(array)) if len(array) else None,
        "count_above_30_deg": int(np.sum(array > 30.0)),
        "fraction_above_30_deg": (
            float(np.mean(array > 30.0)) if len(array) else None
        ),
        "chains": chain_reports,
    }


def _euclidean_leakage_report(
    prepared: PreparedCase,
    quadrature: dict[str, Any],
    *,
    maximum_samples: int = 5_000,
) -> dict[str, Any]:
    if not prepared.open_boundaries:
        return {
            "advisory_only": True,
            "applicable": False,
            "reason": "closed_zero_obc_domain",
        }
    total = int(quadrature["sample_count"])
    indices = np.unique(
        np.linspace(0, total - 1, min(total, maximum_samples)).astype(int)
    )
    points = shapely.points(
        np.asarray(quadrature["x"])[indices],
        np.asarray(quadrature["y"])[indices],
    )
    lines = []
    for chain in prepared.open_boundaries:
        for segment_index in chain.exterior_segment_indices:
            lines.append(
                LineString(
                    [
                        prepared.exterior_xy[segment_index],
                        prepared.exterior_xy[
                            (segment_index + 1) % len(prepared.exterior_xy)
                        ],
                    ]
                )
            )
    open_geometry = unary_union(lines)
    shortest = shapely.shortest_line(points, open_geometry)
    wet_polygon = Polygon(
        prepared.exterior_xy,
        holes=[values.tolist() for values in prepared.holes_xy],
    )
    outside_length = np.asarray(
        shapely.length(shapely.difference(shortest, wet_polygon.buffer(0.05))),
        dtype=float,
    )
    shortcut = outside_length > 1.0
    return {
        "advisory_only": True,
        "applicable": True,
        "definition": (
            "A sampled Euclidean shortest line to an OBC leaves the wet polygon "
            "by more than 1 m, indicating sizing transmitted through land/islands."
        ),
        "sample_count": int(len(indices)),
        "candidate_count": int(np.sum(shortcut)),
        "candidate_fraction": float(np.mean(shortcut)) if len(shortcut) else 0.0,
        "maximum_outside_wet_length_m": (
            float(np.max(outside_length)) if len(outside_length) else 0.0
        ),
        "corrected": False,
    }


def _native_quality_report(result: Any) -> dict[str, Any]:
    return {
        "advisory_only": True,
        "sicn": asdict(result.element_quality.sicn),
        "gamma": asdict(result.element_quality.gamma),
    }


def _write_size_field(
    path: Path,
    prepared: PreparedCase,
    quadrature: dict[str, Any],
    h_uniform_m: float,
    config: BudgetConfig,
) -> Path:
    import xarray as xr

    sizes = target_size_m(
        quadrature["distance_to_obc_m"],
        h_uniform_m,
        near_size_m=config.near_size_m,
        near_distance_m=config.near_distance_m,
        far_distance_m=config.far_distance_m,
        has_open_boundary=bool(prepared.open_boundaries),
    )
    dataset = xr.Dataset(
        {
            "x_m": ("sample", np.asarray(quadrature["x"], dtype=float)),
            "y_m": ("sample", np.asarray(quadrature["y"], dtype=float)),
            "distance_to_obc_m": (
                "sample",
                np.asarray(quadrature["distance_to_obc_m"], dtype=float),
            ),
            "target_size_m": ("sample", sizes),
            "quadrature_area_m2": (
                "sample",
                np.asarray(quadrature["area_weights_m2"], dtype=float),
            ),
        },
        attrs={
            "schema_version": "gmsh_source_aware_size_field_v1",
            "case_id": str(prepared.manifest["case_id"]),
            "projected_crs": prepared.projection.crs.to_string(),
            "uniform_interior_target_m": float(h_uniform_m),
            "near_obc_size_m": float(config.near_size_m),
            "near_obc_distance_m": float(config.near_distance_m),
            "far_obc_distance_m": float(config.far_distance_m),
            "distance_definition": "straight projected Euclidean distance",
        },
    )
    dataset.to_netcdf(path)
    return path


def _write_maps(
    output_dir: Path,
    prepared: PreparedCase,
    result: Any,
    depths: np.ndarray,
    quadrature: dict[str, Any],
    h_uniform_m: float,
    config: BudgetConfig,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from .metrics import triangle_geometry

    outputs: dict[str, str] = {}
    domain_xy = Polygon(
        prepared.exterior_xy,
        holes=[values.tolist() for values in prepared.holes_xy],
    )

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.plot(*domain_xy.exterior.xy, color="#1f2937", linewidth=1.0, label="exterior")
    for ring in domain_xy.interiors:
        ax.plot(*ring.xy, color="#6b7280", linewidth=0.5)
    for boundary in prepared.open_boundaries:
        segments = [
            [
                prepared.exterior_xy[index],
                prepared.exterior_xy[(index + 1) % len(prepared.exterior_xy)],
            ]
            for index in boundary.exterior_segment_indices
        ]
        ax.add_collection(LineCollection(segments, colors="#dc2626", linewidths=2.0))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{prepared.manifest['display_name']} source domain")
    ax.set_xlabel("projected x (m)")
    ax.set_ylabel("projected y (m)")
    path = output_dir / "domain_map.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["domain_map"] = str(path)

    triangles = np.asarray(result.triangles_1based, dtype=int) - 1
    plot_indices = np.unique(
        np.linspace(0, len(triangles) - 1, min(len(triangles), 100_000)).astype(int)
    )
    polygons = result.nodes_xy[triangles[plot_indices]]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    collection = PolyCollection(
        polygons,
        array=np.mean(np.asarray(depths)[triangles[plot_indices]], axis=1),
        cmap="viridis",
        edgecolors=(0.15, 0.18, 0.22, 0.25),
        linewidths=0.08,
    )
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(
        f"{prepared.manifest['display_name']} Gmsh mesh "
        f"({len(result.nodes_xy):,} nodes)"
    )
    fig.colorbar(collection, ax=ax, label="depth positive down (m)")
    path = output_dir / "mesh_map.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["mesh_map"] = str(path)

    quality_values = triangle_geometry(result.nodes_xy, triangles)["quality"]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    collection = PolyCollection(
        polygons,
        array=quality_values[plot_indices],
        cmap="magma_r",
        clim=(0.0, 1.0),
        edgecolors="none",
    )
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{prepared.manifest['display_name']} equilateral quality q")
    fig.colorbar(collection, ax=ax, label="q")
    path = output_dir / "quality_map.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["quality_map"] = str(path)

    sizes = target_size_m(
        quadrature["distance_to_obc_m"],
        h_uniform_m,
        near_size_m=config.near_size_m,
        near_distance_m=config.near_distance_m,
        far_distance_m=config.far_distance_m,
        has_open_boundary=bool(prepared.open_boundaries),
    )
    sample_indices = np.unique(
        np.linspace(
            0,
            len(sizes) - 1,
            min(len(sizes), 100_000),
        ).astype(int)
    )
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    scatter = ax.scatter(
        np.asarray(quadrature["x"])[sample_indices],
        np.asarray(quadrature["y"])[sample_indices],
        c=sizes[sample_indices],
        s=2,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Source-aware target size; h_u={h_uniform_m:.0f} m")
    fig.colorbar(scatter, ax=ax, label="target element size (m)")
    path = output_dir / "size_field_map.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs["size_field_map"] = str(path)
    return outputs


def _nested_number(payload: dict[str, Any], dotted_key: str) -> float | int | None:
    value: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        return value.item() if isinstance(value, np.generic) else value
    return None


def write_contextual_quality_deltas(
    path: str | Path,
    gmsh_quality: dict[str, Any],
    archived_quality_path: str | Path,
) -> dict[str, Any]:
    """Write metric-by-metric contextual deltas with no composite ranking."""
    archived_path = Path(archived_quality_path).resolve()
    archived = json.loads(archived_path.read_text(encoding="utf-8-sig"))
    metrics = (
        "node_count",
        "triangle_count",
        "oceanmesh_quality.q_min",
        "oceanmesh_quality.q_mean",
        "oceanmesh_quality.q_std",
        "oceanmesh_quality.q_l3_sigma",
        "oceanmesh_quality.count_q_below_0_10",
        "min_angle_deg",
        "max_angle_deg",
        "max_bathymetric_slope",
        "max_adjacent_area_change",
        "max_node_valence",
        "topology.connected_component_count",
        "topology.singly_connected_triangle_count",
        "topology.nonmanifold_edge_count",
    )
    comparisons = []
    for metric in metrics:
        gmsh_value = _nested_number(gmsh_quality, metric)
        archived_value = _nested_number(archived, metric)
        comparisons.append(
            {
                "metric": metric,
                "gmsh": gmsh_value,
                "archived_in_house": archived_value,
                "delta_gmsh_minus_archived": (
                    float(gmsh_value) - float(archived_value)
                    if gmsh_value is not None and archived_value is not None
                    else None
                ),
            }
        )
    payload = {
        "schema_version": "gmsh_contextual_quality_deltas_v1",
        "policy": "metric_by_metric_no_composite_winner",
        "archived_mesh_quality": {
            "path": str(archived_path),
            "sha256": file_sha256(archived_path),
            "accepted": archived.get("accepted"),
        },
        "gmsh_accepted": gmsh_quality.get("accepted"),
        "comparisons": comparisons,
        "interpretation": (
            "Signed deltas are descriptive only. Direction of improvement "
            "depends on the metric; no composite score or winner is computed."
        ),
    }
    _write_json(Path(path), payload)
    return payload


def attach_contextual_quality_deltas(
    run_dir: str | Path,
    case_manifest_path: str | Path,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Attach the declared archived-quality comparison to a fresh run."""
    run_path = Path(run_dir).resolve()
    run_manifest_path = run_path / "run_manifest.json"
    quality_path = run_path / "quality.json"
    if not run_manifest_path.exists() or not quality_path.exists():
        raise FileNotFoundError("Run manifest and quality.json are required")
    case = load_case_manifest(case_manifest_path)
    archived_path = resolve_input_path(
        (case.get("context") or {}).get("archived_mesh_quality_json"),
        workspace_root,
    )
    if archived_path is None or not archived_path.exists():
        raise FileNotFoundError("Case has no available archived context quality")
    quality = json.loads(quality_path.read_text(encoding="utf-8-sig"))
    comparison_path = run_path / "contextual_quality_deltas.json"
    comparison = write_contextual_quality_deltas(
        comparison_path,
        quality,
        archived_path,
    )
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8-sig"))
    run_manifest.setdefault("input_hashes", {})["archived_context_quality"] = {
        "path": str(archived_path),
        "sha256": file_sha256(archived_path),
    }
    run_manifest.setdefault("artifacts", {})["contextual_quality_deltas"] = {
        "path": str(comparison_path),
        "sha256": file_sha256(comparison_path),
    }
    run_manifest["contextual_comparison_attached_at_utc"] = utc_now()
    _write_json(run_manifest_path, run_manifest)
    return comparison


def run_gmsh_experiment(
    case_manifest_path: str | Path,
    workspace_root: str | Path,
    output_dir: str | Path,
    *,
    bathymetry_override: str | Path | None = None,
    boundary_loop_override: str | Path | None = None,
    adaptive_resolution_override: str | Path | None = None,
    adaptive_resolution_manifest: str | Path | None = None,
    budget_config: BudgetConfig | None = None,
) -> dict[str, Any]:
    """Execute one fresh, research-only regional Gmsh experiment run."""
    from .gmsh_backend import GmshConfig, measure_boundary_mesh, run_gmsh_attempt

    if adaptive_resolution_manifest is not None:
        if adaptive_resolution_override is not None:
            raise ValueError(
                "adaptive_resolution_manifest and adaptive_resolution_override "
                "are aliases; provide only one"
            )
        adaptive_resolution_override = adaptive_resolution_manifest
    config = budget_config or BudgetConfig()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory must be fresh and empty: {output}")

    readiness = None
    if (
        bathymetry_override is None
        and boundary_loop_override is None
        and adaptive_resolution_override is None
    ):
        readiness = check_case_readiness(case_manifest_path, workspace_root)
        if readiness["status"] != "ready":
            raise ValueError(
                "Automatic case readiness/revalidation failed: "
                + ", ".join(readiness["blockers"])
            )
    prepared = prepare_case(
        case_manifest_path,
        workspace_root,
        bathymetry_override=bathymetry_override,
        boundary_loop_override=boundary_loop_override,
        adaptive_resolution_override=adaptive_resolution_override,
    )
    coverage = bathymetry_coverage_report(
        prepared.bathymetry,
        prepared.source_domain_lonlat,
    )
    if not coverage["passed"]:
        raise ValueError(f"Bathymetry coverage contract failed: {coverage}")
    output.mkdir(parents=True, exist_ok=True)
    if file_sha256(prepared.manifest_path) != prepared.manifest_sha256:
        raise RuntimeError("Case manifest changed while the run was being prepared")
    case_manifest_snapshot = output / "case_manifest.snapshot.json"
    shutil.copy2(prepared.manifest_path, case_manifest_snapshot)
    if file_sha256(case_manifest_snapshot) != prepared.manifest_sha256:
        raise RuntimeError("Case-manifest snapshot hash differs from the loaded input")
    if readiness is None:
        readiness = {
            "schema_version": "gmsh_case_readiness_v1",
            "case_id": prepared.manifest["case_id"],
            "checked_at_utc": utc_now(),
            "status": "ready",
            "blockers": [],
            "warnings": ["explicit_input_override_used"],
            "geometry": {
                "automatic_revalidation": prepared.boundary_revalidation,
            },
            "bathymetry": coverage,
            "input_hashes": {},
        }
    readiness_path = _write_json(output / "case_readiness.json", readiness)
    revalidation_path = _write_json(
        output / "boundary_revalidation.json",
        prepared.boundary_revalidation,
    )

    environment = environment_report()
    geometry = _backend_geometry(prepared)
    floor_m, floor_report = bathymetry_resolution_floor_m(
        prepared.bathymetry,
        prepared.projection,
    )
    field_direction = validate_reversed_threshold_direction(
        floor_m,
        near_size_m=config.near_size_m,
        near_distance_m=config.near_distance_m,
        far_distance_m=config.far_distance_m,
    )
    if not field_direction["passed"]:
        raise RuntimeError("Synthetic Distance/Threshold field-direction test failed")

    floor_gmsh_config = GmshConfig(
        h_uniform_m=floor_m,
        near_size_m=config.near_size_m,
        dist_min_m=config.near_distance_m,
        dist_max_m=config.far_distance_m,
        constant_field=not bool(prepared.open_boundaries),
        model_name=f"{prepared.manifest['case_id']}_boundary_preflight",
    )
    boundary_preflight = measure_boundary_mesh(geometry, floor_gmsh_config)
    quadrature = integration_samples(
        prepared,
        max_cells=config.integration_max_cells,
    )
    h_uniform_m, estimated_nodes = select_uniform_target_m(
        floor_m,
        boundary_preflight.boundary_node_count,
        quadrature["area_weights_m2"],
        quadrature["distance_to_obc_m"],
        has_open_boundary=bool(prepared.open_boundaries),
        config=config,
    )
    selected_direction = validate_reversed_threshold_direction(
        h_uniform_m,
        near_size_m=config.near_size_m,
        near_distance_m=config.near_distance_m,
        far_distance_m=config.far_distance_m,
    )
    if not selected_direction["passed"]:
        raise RuntimeError("Selected Distance/Threshold field-direction test failed")

    preflight_payload = {
        "schema_version": "gmsh_node_budget_preflight_v1",
        "case_id": prepared.manifest["case_id"],
        "budget": asdict(config),
        "bathymetry_resolution_floor": floor_report,
        "boundary_mesh_at_floor": {
            "h_uniform_m": float(floor_m),
            "node_count": int(boundary_preflight.boundary_node_count),
            "loop_node_counts": dict(boundary_preflight.loop_node_counts),
            "open_boundary_node_counts": dict(
                boundary_preflight.open_boundary_node_counts
            ),
        },
        "quadrature": {
            key: value
            for key, value in quadrature.items()
            if key
            not in {
                "x",
                "y",
                "area_weights_m2",
                "distance_to_obc_m",
            }
        },
        "selected_h_uniform_m": float(h_uniform_m),
        "estimated_total_nodes": float(estimated_nodes),
        "preflight_passed": bool(estimated_nodes <= config.preflight_nodes),
        "field_direction_test_at_floor": field_direction,
        "field_direction_test_selected": selected_direction,
    }
    _write_json(output / "node_budget_preflight.json", preflight_payload)
    _write_size_field(
        output / "source_aware_size_field.nc",
        prepared,
        quadrature,
        h_uniform_m,
        config,
    )

    attempts = []
    result = None
    current_h = float(h_uniform_m)
    overflow_after_retry = False
    for attempt_number in (1, 2):
        attempt_config = GmshConfig(
            h_uniform_m=current_h,
            near_size_m=config.near_size_m,
            dist_min_m=config.near_distance_m,
            dist_max_m=config.far_distance_m,
            constant_field=not bool(prepared.open_boundaries),
            model_name=f"{prepared.manifest['case_id']}_attempt_{attempt_number:02d}",
        )
        msh_path = output / f"mesh_attempt_{attempt_number:02d}.msh"
        result = run_gmsh_attempt(geometry, attempt_config, msh_path)
        actual_nodes = int(len(result.nodes_xy))
        attempt_record = {
            "attempt": int(attempt_number),
            "h_uniform_m": float(current_h),
            "node_count": actual_nodes,
            "triangle_count": int(len(result.triangles_1based)),
            "hard_cap": int(config.max_nodes),
            "overflow": bool(actual_nodes > config.max_nodes),
            "msh_path": str(msh_path),
            "msh_sha256": file_sha256(msh_path),
        }
        attempts.append(attempt_record)
        if actual_nodes <= config.max_nodes:
            break
        if attempt_number == 2:
            overflow_after_retry = True
            break
        scale = math.sqrt(actual_nodes / float(config.preflight_nodes))
        current_h = config.step_m * math.ceil(
            (current_h * scale) / config.step_m
        )
    if result is None:
        raise RuntimeError("Gmsh produced no attempt result")

    preflight_payload["attempts"] = attempts
    preflight_payload["final_h_uniform_m"] = float(current_h)
    preflight_payload["overflow_retry_used"] = bool(len(attempts) > 1)
    preflight_payload["overflow_after_retry"] = bool(overflow_after_retry)
    _write_json(output / "node_budget_preflight.json", preflight_payload)
    # The canonical sampled field must describe the delivered attempt.  This
    # intentionally replaces the initial preflight field after a deterministic
    # node-cap retry changes h_u.
    _write_size_field(
        output / "source_aware_size_field.nc",
        prepared,
        quadrature,
        current_h,
        config,
    )

    logger_path = output / "gmsh.log"
    logger_path.write_text("\n".join(result.logger_output) + "\n", encoding="utf-8")
    nodes_lonlat = unproject_points(result.nodes_xy, prepared.projection)
    depths = prepared.bathymetry.sample(
        nodes_lonlat[:, 0],
        nodes_lonlat[:, 1],
    )
    lineage = _delivered_lineage_manifest(prepared, result)
    lineage_path = _write_json(output / "boundary_remap_manifest.json", lineage)
    open_chains = [list(boundary.node_ids) for boundary in result.open_boundaries]
    cyclic_values = [bool(boundary.cyclic) for boundary in result.open_boundaries]
    constraint_chains_zero = [
        [int(value) - 1 for value in delivered.node_ids]
        for delivered in result.delivered_loops
    ]
    target_sizes = _target_size_by_triangle(result, current_h, config)
    constraint_report = {
        "boundary_constraint_recovered": bool(
            lineage["all_source_vertices_retained"]
            and lineage["hard_anchors_retained"]
            and len(result.delivered_loops) == 1 + len(prepared.holes_xy)
        ),
        "all_source_vertices_retained": lineage["all_source_vertices_retained"],
        "hard_anchors_retained": lineage["hard_anchors_retained"],
        "expected_loop_count": int(1 + len(prepared.holes_xy)),
        "delivered_loop_count": int(len(result.delivered_loops)),
    }
    legacy_open = (
        np.asarray(open_chains[0], dtype=int)
        if len(open_chains) == 1
        else np.empty(0, dtype=int)
    )
    quality = evaluate_mesh_quality(
        result.nodes_xy,
        depths,
        result.triangles_1based,
        legacy_open,
        constraint_report,
        constraint_chains=constraint_chains_zero,
        open_boundary_chains=open_chains,
        open_boundary_cyclic=cyclic_values,
        require_open_boundary=bool(prepared.open_boundaries),
        expected_open_boundary_count=int(
            prepared.manifest["boundary"]["expected_open_boundary_count"]
        ),
        enforce_size_error=True,
        enforce_no_unused_nodes=True,
        target_size_by_triangle=target_sizes,
    )
    roundtrip = _roundtrip_report(
        output / f"{prepared.manifest['case_id']}_gmsh.2dm",
        prepared,
        result,
        depths,
    )
    external_failures = []
    if overflow_after_retry:
        external_failures.append("node_cap_overflow_after_deterministic_retry")
    if not roundtrip["passed"]:
        external_failures.append("sms_2dm_roundtrip_failed")
    if not lineage["all_source_vertices_retained"]:
        external_failures.append("source_boundary_vertex_lost")
    if not lineage["hard_anchors_retained"]:
        external_failures.append("protected_anchor_lost")
    if external_failures:
        quality["failure_taxonomy"] = sorted(
            set(quality["failure_taxonomy"] + external_failures)
        )
        quality["accepted"] = False
    quality["sms_2dm_roundtrip"] = roundtrip
    quality["gmsh_native_quality"] = _native_quality_report(result)
    quality["obc_normality"] = _obc_normality_report(result)
    quality["euclidean_through_land_sizing_leakage"] = _euclidean_leakage_report(
        prepared,
        quadrature,
    )
    quality["node_budget_attempts"] = attempts
    quality_path = _write_json(output / "quality.json", quality)
    contextual_path = None
    archived_context_path = resolve_input_path(
        (prepared.manifest.get("context") or {}).get("archived_mesh_quality_json"),
        prepared.workspace_root,
    )
    if archived_context_path is not None and archived_context_path.exists():
        contextual_path = output / "contextual_quality_deltas.json"
        write_contextual_quality_deltas(
            contextual_path,
            quality,
            archived_context_path,
        )
    maps = _write_maps(
        output,
        prepared,
        result,
        depths,
        quadrature,
        current_h,
        config,
    )

    if file_sha256(prepared.manifest_path) != prepared.manifest_sha256:
        raise RuntimeError("Case manifest changed during Gmsh execution")
    input_hashes = {
        "case_manifest": {
            "path": str(case_manifest_snapshot),
            "source_path": str(prepared.manifest_path),
            "sha256": prepared.manifest_sha256,
        },
        "case_manifest_source": {
            "path": str(prepared.manifest_path),
            "sha256": prepared.manifest_sha256,
        },
        **{
            key: {"path": str(path), "sha256": file_sha256(path)}
            for key, path in prepared.input_paths.items()
        },
    }
    if archived_context_path is not None and archived_context_path.exists():
        input_hashes["archived_context_quality"] = {
            "path": str(archived_context_path),
            "sha256": file_sha256(archived_context_path),
        }
    artifact_paths = {
        "msh_4_1": str(result.msh_path),
        "sms_2dm": str(output / f"{prepared.manifest['case_id']}_gmsh.2dm"),
        "node_budget_preflight": str(output / "node_budget_preflight.json"),
        "source_aware_size_field": str(output / "source_aware_size_field.nc"),
        "boundary_remap_manifest": str(lineage_path),
        "quality_json": str(quality_path),
        "gmsh_logger": str(logger_path),
        "case_manifest_snapshot": str(case_manifest_snapshot),
        "case_readiness": str(readiness_path),
        "boundary_revalidation": str(revalidation_path),
        **maps,
    }
    if contextual_path is not None:
        artifact_paths["contextual_quality_deltas"] = str(contextual_path)
    artifacts = {
        key: {
            "path": value,
            "sha256": file_sha256(value),
        }
        for key, value in artifact_paths.items()
    }
    status = "pass" if quality["accepted"] else "needs_review"
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "case_id": prepared.manifest["case_id"],
        "display_name": prepared.manifest["display_name"],
        "research_only": True,
        "production_backend": False,
        "started_from_case_status": prepared.manifest.get("experiment_status"),
        "completed_at_utc": utc_now(),
        "status": status,
        "selected_h_uniform_m": float(current_h),
        "bathymetry_coverage": coverage,
        "boundary_revalidation": prepared.boundary_revalidation,
        "environment": environment,
        "input_hashes": input_hashes,
        "attempts": attempts,
        "forcing_compatible": lineage["forcing_compatible"],
        "quality_accepted": bool(quality["accepted"]),
        "failure_taxonomy": quality["failure_taxonomy"],
        "artifacts": artifacts,
        "license_note": (
            "Gmsh is GPL-licensed and was installed as an isolated dependency; "
            "no Gmsh code or binary is vendored by this source tree."
        ),
    }
    _write_json(output / "run_manifest.json", run_manifest)
    return run_manifest


__all__ = [
    "BudgetConfig",
    "PreparedCase",
    "SCHEMA_VERSION",
    "attach_contextual_quality_deltas",
    "bathymetry_resolution_floor_m",
    "check_case_readiness",
    "environment_report",
    "estimate_node_count",
    "integration_samples",
    "load_case_manifest",
    "prepare_case",
    "run_gmsh_experiment",
    "select_uniform_target_m",
    "target_size_m",
    "validate_reversed_threshold_direction",
    "write_contextual_quality_deltas",
]
