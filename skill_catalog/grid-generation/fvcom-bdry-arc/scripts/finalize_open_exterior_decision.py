#!/usr/bin/env python3
"""Record a hash-bound Codex open-exterior decision and resume boundary QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_bdry_arc.boundary_loops import build_model_boundary_loops  # noqa: E402
from fvcom_bdry_arc.boundary_resolution import (  # noqa: E402
    boundary_resolution_config,
    build_boundary_resolution,
)
from fvcom_bdry_arc.open_exterior import build_open_exterior_contract  # noqa: E402


DECISION_FAILURES = {
    "open_exterior_agent_decision_required",
    "open_exterior_contract_not_downstream_eligible",
    "blocked_by_open_exterior_contract",
    "model_boundary_loop_needs_review",
    "residual_boundary_role_decision_required",
    "residual_boundary_role_pending",
}

RESIDUAL_ROLES = {
    "solid_lagoon_closure",
    "secondary_tidal_obc",
    "invalid_geometry",
}

RESOLUTION_STATUS_FAILURES = {
    "adaptive_boundary_resolution_needs_review",
    "adaptive_boundary_resolution_failed",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def _existing_adaptive_resolution(run_dir: Path, profile: str) -> dict | None:
    path = run_dir / "boundary_resolution" / "boundary_resolution_manifest.json"
    if not path.is_file():
        return None
    value = read_json(path)
    if value.get("profile") != profile:
        return None
    required_outputs = [
        value.get("outputs", {}).get("boundary_resolution_gpkg"),
        value.get("outputs", {}).get("boundary_resolution_nodes_geojson"),
    ]
    if not all(item and Path(item).is_file() for item in required_outputs):
        return None
    return value


def _load_station_screen(path: Path | None, assessed_contract_sha256: str) -> tuple[dict | None, dict]:
    if path is None:
        return None, {"status": "not_run", "path": None, "sha256": None}
    if not path.is_file():
        raise ValueError(f"NOAA CO-OPS station screen does not exist: {path}")
    screen = read_json(path)
    if screen.get("schema_version") != "noaa_coops_tidal_station_screen_v1":
        raise ValueError("unsupported NOAA CO-OPS station-screen schema")
    if screen.get("source_contract_sha256") != assessed_contract_sha256:
        raise ValueError("NOAA CO-OPS station screen is stale for this open-exterior contract")
    return screen, {
        "status": "pass",
        "path": str(path),
        "sha256": sha256_file(path),
        "eligible_station_count": int(screen.get("eligible_station_count", 0)),
    }


def _eligible_station_for_segment(screen: dict | None, segment_id: int) -> dict | None:
    if not screen:
        return None
    component = next(
        (item for item in screen.get("components", []) if int(item.get("segment_id", -1)) == segment_id),
        None,
    )
    if not component:
        return None
    return next(
        (item for item in component.get("candidates", []) if item.get("eligible_for_residual_obc") is True),
        None,
    )


def _apply_residual_roles(
    contract: dict,
    *,
    assessed_contract_sha256: str,
    decision: str,
    requested_roles: dict[int, str],
    station_screen_path: Path | None,
) -> bool:
    screen, screen_binding = _load_station_screen(station_screen_path, assessed_contract_sha256)
    contract["station_screen"] = screen_binding
    components = list(contract.get("residual_components") or [])
    expected_obc = int(contract.get("obc_geometry", {}).get("expected_count", 0) or 0)
    source_delivered_obc = int(contract.get("obc_geometry", {}).get("delivered_count", 0) or 0)
    assigned_solid = 0.0
    assigned_secondary = 0.0
    pending = 0
    invalid = 0
    secondary_count = 0
    for component in components:
        if component.get("classification") == "intentional_open_boundary":
            continue
        segment_id = int(component.get("segment_id", -1))
        role = requested_roles.get(segment_id, "solid_lagoon_closure")
        if role not in RESIDUAL_ROLES:
            raise ValueError(f"Unsupported residual role for segment {segment_id}: {role}")
        geometry = component.get("solid_role_geometry", {})
        if role == "solid_lagoon_closure" and geometry.get("eligible") is not True:
            if decision == "pass":
                raise ValueError(f"Residual segment {segment_id} is not geometrically eligible for a solid closure")
            pending += 1
            continue
        station = None
        if role == "secondary_tidal_obc":
            station = _eligible_station_for_segment(screen, segment_id)
            if station is None and decision == "pass":
                raise ValueError(f"Residual segment {segment_id} has no eligible, hydraulically connected NOAA CO-OPS station")
            if source_delivered_obc + secondary_count >= expected_obc and decision == "pass":
                raise ValueError(
                    "Requested OBC count does not permit another boundary; a nearby gauge is eligibility evidence only"
                )
            secondary_count += 1
            assigned_secondary += float(component.get("length_m", 0.0) or 0.0)
        elif role == "invalid_geometry":
            invalid += 1
        else:
            assigned_solid += float(component.get("length_m", 0.0) or 0.0)
        component["assigned_role"] = role
        component["role_status"] = "accepted" if decision == "pass" and role != "invalid_geometry" else "needs_review"
        component["agent_geometry_confirmation"] = {
            "no_artificial_bar": bool(decision == "pass" and role != "invalid_geometry"),
            "no_protected_feature_conflict": bool(decision == "pass" and role != "invalid_geometry"),
        }
        component["forcing_eligibility"] = (
            {
                "provider": "NOAA CO-OPS",
                "station_id": station.get("station_id"),
                "station_name": station.get("name"),
                "distance_km": station.get("distance_km"),
                "eligibility_only": True,
            }
            if station
            else None
        )
    unassigned_length = float(
        sum(
            float(item.get("length_m", 0.0) or 0.0)
            for item in components
            if item.get("classification") != "intentional_open_boundary"
            and item.get("role_status") not in {"accepted", "needs_review"}
        )
    )
    lengths = contract.get("boundary_lengths", {})
    landward_length = max(float(lengths.get("landward_boundary_length_m", 1.0) or 1.0), 1.0)
    outer_length = max(float(lengths.get("outer_boundary_length_m", 1.0) or 1.0), 1.0)
    tolerance = float(contract.get("hard_metrics", {}).get("absolute_limit_m", 250.0) or 250.0)
    fraction = unassigned_length / landward_length
    coverage = max(0.0, min(1.0, 1.0 - unassigned_length / outer_length))
    hard = contract.setdefault("hard_metrics", {})
    hard.update({
        "absolute_residual_length_m": unassigned_length,
        "absolute_gate_pass": unassigned_length <= tolerance,
        "residual_fraction": fraction,
        "fraction_gate_pass": fraction <= float(hard.get("fraction_limit", 0.001)),
        "coastline_plus_obc_exterior_coverage": coverage,
        "coverage_gate_pass": coverage >= float(hard.get("coverage_minimum", 0.999)),
        "metric_subject": "unassigned_residual",
    })
    hard["all_independent_metric_gates_pass"] = bool(
        hard["absolute_gate_pass"]
        and hard["fraction_gate_pass"]
        and hard["coverage_gate_pass"]
        and hard.get("coastline_source_coverage_gate_pass", True)
    )
    contract["residual_role_summary"] = {
        "pending_count": pending,
        "solid_lagoon_closure_count": int(sum(item.get("assigned_role") == "solid_lagoon_closure" for item in components)),
        "secondary_tidal_obc_count": secondary_count,
        "invalid_geometry_count": invalid,
        "unassigned_residual_length_m": unassigned_length,
        "assigned_solid_length_m": assigned_solid,
        "assigned_secondary_obc_length_m": assigned_secondary,
    }
    contract["obc_geometry"]["source_delivered_count"] = source_delivered_obc
    contract["obc_geometry"]["delivered_count"] = source_delivered_obc + secondary_count
    return bool(
        decision == "pass"
        and pending == 0
        and invalid == 0
        and hard["all_independent_metric_gates_pass"]
        and contract["obc_geometry"]["delivered_count"] == expected_obc
    )


def _parse_roles(values: list[str]) -> dict[int, str]:
    roles: dict[int, str] = {}
    for value in values:
        try:
            segment, role = value.split("=", 1)
            segment_id = int(segment)
        except Exception as exc:
            raise ValueError("--residual-role must use SEGMENT_ID=ROLE") from exc
        if segment_id in roles:
            raise ValueError(f"Duplicate residual role for segment {segment_id}")
        roles[segment_id] = role
    return roles


def _reconcile_resolution_failures(failures: list[str], resolution: dict) -> list[str]:
    """Replace stale derived resolution flags with the latest terminal state."""
    current = [item for item in failures if item not in RESOLUTION_STATUS_FAILURES]
    if resolution.get("final_status") != "pass":
        current.append("adaptive_boundary_resolution_needs_review")
    return list(dict.fromkeys(current))


def _verify_component_maps(contract: dict) -> None:
    maps = contract.get("component_maps", {})
    for component in contract.get("residual_components", []):
        if component.get("classification") == "intentional_open_boundary":
            continue
        segment_id = str(component.get("segment_id"))
        record = maps.get(segment_id) or {}
        path = Path(record.get("path", ""))
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise ValueError(f"Residual component {segment_id} map is missing or stale")


def _clear_loop_role_failures(manifest: dict) -> None:
    loop_path = Path(manifest.get("outputs", {}).get("model_boundary_loop_manifest", ""))
    if not loop_path.is_file():
        return
    loop = read_json(loop_path)
    loop["failure_taxonomy"] = [
        item for item in loop.get("failure_taxonomy", []) if item not in DECISION_FAILURES
    ]
    if not loop["failure_taxonomy"]:
        loop["final_status"] = "pass"
    atomic_json(loop_path, loop)
    if isinstance(manifest.get("model_boundary_loops"), dict):
        manifest["model_boundary_loops"]["final_status"] = loop.get("final_status")
        manifest["model_boundary_loops"]["failure_taxonomy"] = loop.get("failure_taxonomy", [])


def _bind_resolution_contract(resolution: dict, contract_path: Path) -> None:
    if not resolution:
        return
    contract = read_json(contract_path)
    solid_components = [
        {
            "segment_id": int(item.get("segment_id", -1)),
            "role": item.get("assigned_role"),
            "length_m": float(item.get("length_m", 0.0) or 0.0),
            "geometry_lonlat": item.get("geometry_lonlat"),
        }
        for item in contract.get("residual_components", [])
        if item.get("assigned_role") == "solid_lagoon_closure"
    ]
    resolution["open_exterior_contract_binding"] = {
        "schema_version": contract.get("schema_version"),
        "path": str(contract_path),
        "sha256": sha256_file(contract_path),
        "boundary_role_policy": "solid-default",
    }
    resolution["solid_boundary_roles"] = {
        "classification": "fixed_landward_chain",
        "component_count": len(solid_components),
        "total_length_m": float(sum(item["length_m"] for item in solid_components)),
        "components": solid_components,
    }
    resolution.setdefault("qa", {})["solid_residual_boundary_component_count"] = len(solid_components)
    resolution["qa"]["solid_residual_boundary_length_m"] = float(
        sum(item["length_m"] for item in solid_components)
    )
    resolution.setdefault("outputs", {})["open_exterior_contract"] = str(contract_path)
    output = Path(resolution.get("outputs", {}).get("boundary_resolution_manifest", ""))
    if output.name and output.parent.is_dir():
        atomic_json(output, resolution)


def _write_layer(path: Path, layer: str, frame: gpd.GeoDataFrame) -> None:
    if frame.empty:
        return
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    frame.to_file(path, layer=layer, driver="GPKG")


def _append_lines(
    base: gpd.GeoDataFrame,
    additions: list[dict],
) -> gpd.GeoDataFrame:
    if not additions:
        return base
    extra = gpd.GeoDataFrame(additions, geometry="geometry", crs="EPSG:4326")
    if base.empty:
        return extra
    columns = sorted(set(base.columns) | set(extra.columns))
    for column in columns:
        if column not in base.columns:
            base[column] = None
        if column not in extra.columns:
            extra[column] = None
    return gpd.GeoDataFrame(
        pd.concat([base[columns], extra[columns]], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )


def _endpoint_distance_m(first: tuple[float, float], second: tuple[float, float]) -> float:
    mean_lat = math.radians(0.5 * (float(first[1]) + float(second[1])))
    dx = (float(first[0]) - float(second[0])) * 111_320.0 * math.cos(mean_lat)
    dy = (float(first[1]) - float(second[1])) * 110_574.0
    return float(math.hypot(dx, dy))


def _join_open_lines(base: LineString, addition: LineString, tolerance_m: float) -> LineString:
    """Absorb an adjacent intentional-open fragment without creating another OBC."""
    base_coords = list(base.coords)
    addition_coords = list(addition.coords)
    alternatives = [
        (_endpoint_distance_m(base_coords[-1], addition_coords[0]), base_coords, addition_coords),
        (_endpoint_distance_m(base_coords[-1], addition_coords[-1]), base_coords, list(reversed(addition_coords))),
        (_endpoint_distance_m(base_coords[0], addition_coords[-1]), addition_coords, base_coords),
        (_endpoint_distance_m(base_coords[0], addition_coords[0]), list(reversed(addition_coords)), base_coords),
    ]
    gap_m, first, second = min(alternatives, key=lambda item: item[0])
    if gap_m > float(tolerance_m):
        raise ValueError(
            f"intentional open-boundary fragment is {gap_m:.3f} m from the nearest OBC endpoint; "
            f"limit is {float(tolerance_m):.3f} m"
        )
    joined = LineString(first + second)
    if joined.is_empty or not joined.is_simple:
        raise ValueError("intentional open-boundary fragment would create a branched or self-crossing OBC")
    return joined


def _materialize_finalized_package(manifest: dict, contract: dict, run_dir: Path) -> dict:
    """Create a role-resolved package while preserving the candidate GeoPackage."""
    candidate = Path(manifest.get("outputs", {}).get("bdry_arc_package_gpkg", ""))
    if not candidate.is_file():
        raise ValueError("candidate boundary-arc GeoPackage is missing")
    layer_names = list(gpd.list_layers(candidate)["name"])
    layers = {name: gpd.read_file(candidate, layer=name).to_crs("EPSG:4326") for name in layer_names}
    open_layer = layers.get("open_boundary_arc", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")).copy()
    if "obc_id" not in open_layer.columns:
        open_layer["obc_id"] = list(range(len(open_layer)))
    if "is_closed" not in open_layer.columns:
        open_layer["is_closed"] = [bool(getattr(geom, "is_ring", False)) for geom in open_layer.geometry]
    open_layer = open_layer.sort_values("obc_id").reset_index(drop=True)
    open_layer["obc_id"] = list(range(len(open_layer)))
    open_layer["residual_role"] = "primary_delivered_obc"
    open_layer["absorbed_segment_ids"] = ""

    frame_layer = layers.get("frame_clip_boundary_arcs", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")).copy()
    frame_layer = frame_layer.reset_index(drop=True)
    land_layer = layers.get("land_patch_boundary_arcs", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")).copy()
    remove_segments: set[int] = set()
    secondary_rows: list[dict] = []
    solid_rows: list[dict] = []
    next_obc_id = len(open_layer)
    components = sorted(
        contract.get("residual_components", []),
        key=lambda item: int(item.get("segment_id", -1)),
    )
    intentional_absorbed: list[int] = []
    join_tolerance_m = max(
        250.0,
        float(contract.get("hard_metrics", {}).get("absolute_limit_m", 250.0) or 250.0),
    )
    for component in components:
        if component.get("classification") != "intentional_open_boundary":
            continue
        segment_id = int(component.get("segment_id", -1))
        coords = component.get("geometry_lonlat") or []
        if len(coords) < 2:
            raise ValueError(f"intentional open-boundary segment {segment_id} has no materializable geometry")
        fragment = LineString([(float(x), float(y)) for x, y in coords])
        nearest_index = min(
            range(len(open_layer)),
            key=lambda index: min(
                _endpoint_distance_m(tuple(open_layer.iloc[index].geometry.coords[0]), tuple(fragment.coords[0])),
                _endpoint_distance_m(tuple(open_layer.iloc[index].geometry.coords[0]), tuple(fragment.coords[-1])),
                _endpoint_distance_m(tuple(open_layer.iloc[index].geometry.coords[-1]), tuple(fragment.coords[0])),
                _endpoint_distance_m(tuple(open_layer.iloc[index].geometry.coords[-1]), tuple(fragment.coords[-1])),
            ),
        )
        open_layer.at[nearest_index, "geometry"] = _join_open_lines(
            open_layer.iloc[nearest_index].geometry,
            fragment,
            join_tolerance_m,
        )
        prior = str(open_layer.at[nearest_index, "absorbed_segment_ids"] or "")
        open_layer.at[nearest_index, "absorbed_segment_ids"] = ",".join(
            item for item in (prior, str(segment_id)) if item
        )
        remove_segments.add(segment_id)
        intentional_absorbed.append(segment_id)
    for component in components:
        role = component.get("assigned_role")
        if role not in {"secondary_tidal_obc", "solid_lagoon_closure"}:
            continue
        segment_id = int(component.get("segment_id", -1))
        coords = component.get("geometry_lonlat") or []
        if len(coords) < 2:
            raise ValueError(f"accepted residual segment {segment_id} has no materializable geometry")
        geometry = LineString([(float(x), float(y)) for x, y in coords])
        remove_segments.add(segment_id)
        if role == "secondary_tidal_obc":
            secondary_rows.append(
                {
                    "segment_class": "open_boundary",
                    "obc_id": int(next_obc_id),
                    "is_closed": False,
                    "residual_role": role,
                    "source_segment_id": segment_id,
                    "geometry": geometry,
                }
            )
            next_obc_id += 1
        else:
            solid_rows.append(
                {
                    "segment_class": "land_patch_boundary",
                    "residual_role": role,
                    "source_segment_id": segment_id,
                    "geometry": geometry,
                }
            )
    open_layer = _append_lines(open_layer, secondary_rows)
    land_layer = _append_lines(land_layer, solid_rows)
    if remove_segments and not frame_layer.empty:
        segment_column = next(
            (name for name in ("segment_id", "source_segment_id") if name in frame_layer.columns),
            None,
        )
        if segment_column:
            keep = [int(value) not in remove_segments for value in frame_layer[segment_column]]
        else:
            keep = [index not in remove_segments for index in range(len(frame_layer))]
        frame_layer = frame_layer.loc[keep].reset_index(drop=True)
    layers["open_boundary_arc"] = open_layer
    layers["land_patch_boundary_arcs"] = land_layer
    layers["frame_clip_boundary_arcs"] = frame_layer

    final_path = run_dir / "bdry_arc_package_final.gpkg"
    if final_path.exists():
        final_path.unlink()
    for layer_name, frame in layers.items():
        _write_layer(final_path, layer_name, frame)
    chains = [
        {
            "obc_id": int(row.obc_id),
            "is_closed": bool(row.is_closed),
            "length_deg": float(row.geometry.length),
            "residual_role": str(getattr(row, "residual_role", "")),
        }
        for _, row in open_layer.sort_values("obc_id").iterrows()
    ]
    expected = int(contract.get("obc_geometry", {}).get("expected_count", 0))
    if len(chains) != expected:
        raise ValueError(f"finalized OBC count {len(chains)} does not equal requested count {expected}")
    return {
        "candidate_bdry_arc_package_gpkg": str(candidate),
        "bdry_arc_package_final_gpkg": str(final_path),
        "sha256": sha256_file(final_path),
        "delivered_obc_count": int(len(chains)),
        "chains": chains,
        "solid_component_count": int(len(solid_rows)),
        "secondary_obc_count": int(len(secondary_rows)),
        "intentional_open_fragment_count": int(len(intentional_absorbed)),
        "intentional_open_fragment_segment_ids": intentional_absorbed,
    }


def _rebuild_final_loops(manifest: dict, package: dict, manifest_path: Path) -> dict:
    source = dict(manifest)
    source["settings"] = dict(manifest.get("settings", {}))
    source["wet_domain"] = dict(manifest.get("wet_domain", {}))
    source["outputs"] = dict(manifest.get("outputs", {}))
    source["final_status"] = "pass"
    source["failure_taxonomy"] = [
        item for item in source.get("failure_taxonomy", []) if item not in DECISION_FAILURES
    ]
    source.setdefault("outputs", {})["bdry_arc_package_gpkg"] = package["bdry_arc_package_final_gpkg"]
    source["wet_domain"]["candidate_frame_clip_boundary_length_m"] = float(
        source["wet_domain"].get("frame_clip_boundary_length_m", 0.0) or 0.0
    )
    source["wet_domain"]["frame_clip_boundary_length_m"] = 0.0
    source["wet_domain"]["residual_roles_materialized"] = True
    source["wet_domain"]["role_resolved_solid_component_count"] = int(
        package.get("solid_component_count", 0)
    )
    source["wet_domain"]["role_resolved_secondary_obc_count"] = int(
        package.get("secondary_obc_count", 0)
    )
    source_path = manifest_path.parent / "bdry_arc_finalization_source.json"
    atomic_json(source_path, source)
    loop_dir = manifest_path.parent / "model_boundary_loops_final"
    loop = build_model_boundary_loops(
        package["bdry_arc_package_final_gpkg"],
        source_path,
        loop_dir,
        str(manifest.get("name", "fvcom_boundary")),
        target_resolution_m=float(manifest.get("settings", {}).get("target_resolution_m", 250.0)),
        min_island_area_m2=0.0,
        mode=str(manifest.get("settings", {}).get("mode", "test")),
    )
    if loop.get("final_status") != "pass":
        raise ValueError(
            "finalized model-boundary loops did not pass: "
            + ", ".join(loop.get("failure_taxonomy", []))
        )
    return loop


def _rebuild_final_open_exterior(
    manifest: dict,
    package: dict,
    final_loop: dict,
    manifest_path: Path,
) -> dict:
    """Re-audit the role-resolved package before v2 resolution is rebuilt."""
    inputs = manifest.get("inputs", {})
    loop_manifest = final_loop.get("outputs", {}).get("model_boundary_loop_manifest")
    if not loop_manifest:
        loop_manifest = manifest_path.parent / "model_boundary_loops_final" / "model_boundary_loop_manifest.json"
    final_qa = build_open_exterior_contract(
        inputs.get("region_bpoly_json"),
        inputs.get("offshore_artifacts_json"),
        package["bdry_arc_package_final_gpkg"],
        inputs.get("coastline_gpkg"),
        loop_manifest,
        manifest_path.parent / "open_exterior_final",
        manifest,
        frame_clip_policy=str(manifest.get("settings", {}).get("frame_clip_policy", "reject-unintended")),
        residual_boundary_policy="solid-default",
        frame_clip_tolerance_m=manifest.get("settings", {}).get("frame_clip_tolerance_m"),
        adaptive_status="pending",
        expected_obc_count=int(manifest.get("settings", {}).get("expected_obc_count", 1)),
    )
    hard = final_qa.get("hard_metrics", {})
    unexpected = [
        item
        for item in final_qa.get("failure_taxonomy", [])
        if item not in DECISION_FAILURES
    ]
    if unexpected or not hard.get("all_independent_metric_gates_pass"):
        raise ValueError(
            "finalized open-exterior QA did not pass: "
            + ", ".join(unexpected or ["independent_metric_gate_failed"])
        )
    obc = final_qa.get("obc_geometry", {})
    if int(obc.get("delivered_count", -1)) != int(obc.get("expected_count", -2)):
        raise ValueError("finalized open-exterior QA did not preserve the requested OBC count")
    return final_qa


def finalize(
    manifest_path: Path,
    decision: str,
    rationale: str,
    *,
    resume_adaptive: bool,
    residual_roles: dict[int, str] | None = None,
    station_screen_path: Path | None = None,
) -> dict:
    manifest = read_json(manifest_path)
    contract_path = Path(manifest.get("outputs", {}).get("open_exterior_contract", ""))
    map_path = Path(manifest.get("outputs", {}).get("open_exterior_review_map", ""))
    decision_path = Path(manifest.get("outputs", {}).get("open_exterior_agent_decision", ""))
    if not contract_path.is_file() or not map_path.is_file() or not decision_path.parent.is_dir():
        raise ValueError("boundary manifest does not resolve a complete open-exterior evidence package")
    contract = read_json(contract_path)
    schema = contract.get("schema_version")
    if schema not in {"fvcom_open_exterior_contract_v1", "fvcom_open_exterior_contract_v2"}:
        raise ValueError("unsupported open-exterior contract")
    assessed_hash = sha256_file(contract_path)
    role_pass = True
    if schema == "fvcom_open_exterior_contract_v2":
        if decision == "pass":
            _verify_component_maps(contract)
        role_pass = _apply_residual_roles(
            contract,
            assessed_contract_sha256=assessed_hash,
            decision=decision,
            requested_roles=dict(residual_roles or {}),
            station_screen_path=station_screen_path,
        )
    metrics = contract.get("hard_metrics", {})
    source_coverage = contract.get("coastline_source_coverage") or {}
    source_coverage_required = bool(contract.get("coastline_source_coverage_required", False))
    source_coverage_pass = bool(
        not source_coverage_required
        or source_coverage.get("downstream_eligible") is True
    )
    if source_coverage_required:
        for map_record in (source_coverage.get("maps") or {}).values():
            coverage_map = Path(str((map_record or {}).get("path", "")))
            if not coverage_map.is_file() or (map_record or {}).get("sha256") != sha256_file(coverage_map):
                source_coverage_pass = False
    hard_pass = bool(
        metrics.get("absolute_gate_pass")
        and metrics.get("fraction_gate_pass")
        and metrics.get("coverage_gate_pass")
        and metrics.get("all_independent_metric_gates_pass")
        and source_coverage_pass
    )
    if decision == "pass" and (not hard_pass or not role_pass or contract.get("report_only")):
        raise ValueError("Codex cannot pass failed hard metrics or a report-only package")
    if not rationale.strip():
        raise ValueError("a concise visual rationale is required")

    decision_doc = {
        "schema_version": "open_exterior_agent_decision_v2" if schema.endswith("v2") else "open_exterior_agent_decision_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": decision,
        "decision_actor": {"kind": "codex_agent"},
        "assessed_contract_sha256": assessed_hash,
        "inspected_map_sha256": sha256_file(map_path),
        "bound_source_hashes": contract.get("source_hashes", {}),
        "inspected_coastline_coverage_map_sha256": (source_coverage.get("maps", {}).get("whole_domain", {}) or {}).get("sha256"),
        "inspected_coastline_coverage_zoom_sha256": (source_coverage.get("maps", {}).get("source_edge_zoom", {}) or {}).get("sha256"),
        "rationale": rationale.strip(),
        "confirmation": {
            "whole_domain_map_inspected": True,
            "no_artificial_frame_supported_strip": decision == "pass",
        },
        "residual_roles": [
            {
                "segment_id": int(item.get("segment_id", -1)),
                "role": item.get("assigned_role"),
                "component_map_sha256": (contract.get("component_maps", {}).get(str(item.get("segment_id")), {}) or {}).get("sha256"),
                "no_artificial_bar": bool(item.get("agent_geometry_confirmation", {}).get("no_artificial_bar", False)),
                "no_protected_feature_conflict": bool(item.get("agent_geometry_confirmation", {}).get("no_protected_feature_conflict", False)),
            }
            for item in contract.get("residual_components", [])
            if item.get("classification") != "intentional_open_boundary"
        ],
    }
    eligible = bool(
        decision == "pass"
        and hard_pass
        and role_pass
        and source_coverage_pass
        and not contract.get("report_only")
    )
    # Materialize accepted roles and rebuild all downstream geometry before
    # publishing a passing decision.
    resolution = None
    finalized_package = None
    final_loop = None
    if eligible:
        profile = str(
            manifest.get("settings", {}).get("boundary_resolution_profile", "adaptive-coastal-v2")
        )
        if profile != "adaptive-coastal-v2":
            raise ValueError("Open-exterior finalization only supports adaptive-coastal-v2")
        finalized_package = _materialize_finalized_package(manifest, contract, manifest_path.parent)
        final_loop = _rebuild_final_loops(manifest, finalized_package, manifest_path)
        final_open_exterior = _rebuild_final_open_exterior(
            manifest,
            finalized_package,
            final_loop,
            manifest_path,
        )
        loop_outputs = dict(final_loop.get("outputs", {}))
        loop_outputs["model_boundary_loop_manifest"] = str(
            manifest_path.parent / "model_boundary_loops_final" / "model_boundary_loop_manifest.json"
        )
        manifest.setdefault("outputs", {})["candidate_bdry_arc_package_gpkg"] = finalized_package[
            "candidate_bdry_arc_package_gpkg"
        ]
        manifest["outputs"]["bdry_arc_package_final_gpkg"] = finalized_package[
            "bdry_arc_package_final_gpkg"
        ]
        manifest["outputs"]["bdry_arc_package_gpkg"] = finalized_package[
            "bdry_arc_package_final_gpkg"
        ]
        manifest["outputs"].update(loop_outputs)
        manifest["model_boundary_loops"] = {
            "final_status": final_loop.get("final_status"),
            "failure_taxonomy": final_loop.get("failure_taxonomy", []),
            "qa": final_loop.get("qa", {}),
            "outputs": loop_outputs,
        }
        contract["finalized_geometry"] = finalized_package
        contract.setdefault("source_hashes", {})["candidate_bdry_arc_gpkg"] = contract.get(
            "source_hashes", {}
        ).get("bdry_arc_gpkg")
        contract["finalized_source_hashes"] = {
            "bdry_arc_gpkg": finalized_package["sha256"],
            "model_boundary_loops_gpkg": sha256_file(loop_outputs["model_boundary_loops_gpkg"]),
            "model_boundary_loop_manifest": sha256_file(loop_outputs["model_boundary_loop_manifest"]),
            "open_exterior_contract": sha256_file(
                final_open_exterior["outputs"]["open_exterior_contract"]
            ),
        }
        contract["finalized_open_exterior_qa"] = {
            "path": final_open_exterior["outputs"]["open_exterior_contract"],
            "review_map": final_open_exterior["outputs"]["open_exterior_review_map"],
            "schema_version": final_open_exterior.get("schema_version"),
            "hard_metrics": final_open_exterior.get("hard_metrics", {}),
            "obc_geometry": final_open_exterior.get("obc_geometry", {}),
            "failure_taxonomy": final_open_exterior.get("failure_taxonomy", []),
        }
        manifest["outputs"]["finalized_open_exterior_contract"] = final_open_exterior[
            "outputs"
        ]["open_exterior_contract"]
        manifest["outputs"]["finalized_open_exterior_review_map"] = final_open_exterior[
            "outputs"
        ]["open_exterior_review_map"]
        contract.setdefault("obc_geometry", {})["delivered_count"] = finalized_package[
            "delivered_obc_count"
        ]
        contract["obc_geometry"]["chains"] = finalized_package["chains"]
        if resume_adaptive:
            resolution = build_boundary_resolution(
                loop_outputs["model_boundary_loops_gpkg"],
                loop_outputs["model_boundary_loop_manifest"],
                manifest["inputs"]["region_bpoly_json"],
                manifest["inputs"]["coastline_gpkg"],
                manifest_path.parent / "boundary_resolution",
                str(manifest.get("name", "fvcom_boundary")),
                boundary_resolution_config(),
            )

    atomic_json(decision_path, decision_doc)
    decision_hash = sha256_file(decision_path)
    contract["agent_decision"] = {
        "required": True,
        "status": decision,
        "path": str(decision_path),
        "sha256": decision_hash,
        "assessed_contract_sha256": assessed_hash,
        "inspected_map_sha256": decision_doc["inspected_map_sha256"],
        "residual_roles": decision_doc["residual_roles"],
    }
    contract["downstream_eligible"] = eligible
    contract["final_status"] = "pass" if eligible else "needs_review"
    failures = [f for f in contract.get("failure_taxonomy", []) if f not in DECISION_FAILURES]
    if not eligible and "open_exterior_agent_decision_rejected" not in failures:
        failures.append("open_exterior_agent_decision_rejected")
    contract["failure_taxonomy"] = failures
    atomic_json(contract_path, contract)

    if resolution is not None:
        _bind_resolution_contract(resolution, contract_path)

    manifest["open_exterior_contract"] = contract
    manifest["failure_taxonomy"] = [
        f for f in manifest.get("failure_taxonomy", []) if f not in DECISION_FAILURES
    ]
    if resolution is not None:
        manifest["failure_taxonomy"] = _reconcile_resolution_failures(
            manifest["failure_taxonomy"],
            resolution,
        )
    if not eligible:
        if "open_exterior_agent_decision_rejected" not in manifest["failure_taxonomy"]:
            manifest["failure_taxonomy"].append("open_exterior_agent_decision_rejected")
        manifest["final_status"] = "needs_review"
    elif resolution is not None:
        manifest["boundary_resolution"] = resolution
        manifest["outputs"].update(resolution.get("outputs", {}))
    if eligible and not manifest.get("failure_taxonomy"):
        manifest["final_status"] = "pass"
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdry-arc-manifest", required=True, type=Path)
    parser.add_argument("--decision", required=True, choices=("pass", "needs_review"))
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--resume-adaptive", action="store_true")
    parser.add_argument(
        "--residual-role",
        action="append",
        default=[],
        metavar="SEGMENT_ID=ROLE",
        help="Override the solid default for one residual component.",
    )
    parser.add_argument("--station-screen-json", type=Path)
    args = parser.parse_args()
    result = finalize(
        args.bdry_arc_manifest.resolve(),
        args.decision,
        args.rationale,
        resume_adaptive=args.resume_adaptive,
        residual_roles=_parse_roles(args.residual_role),
        station_screen_path=(args.station_screen_json.resolve() if args.station_screen_json else None),
    )
    print(json.dumps({
        "final_status": result.get("final_status"),
        "downstream_eligible": result.get("open_exterior_contract", {}).get("downstream_eligible"),
    }, indent=2))
    return 0 if result.get("open_exterior_contract", {}).get("downstream_eligible") else 2


if __name__ == "__main__":
    raise SystemExit(main())
