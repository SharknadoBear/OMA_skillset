from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import canonical_sha256, file_sha256, read_json, utc_now, write_json


OUTPUT_PACKAGE_SCHEMA = "region_bpoly_output_package_v1"
DELIVERY_MANIFEST_SCHEMA = "region_bpoly_delivery_manifest_v1"


def _freeze_noncoastal_delivery(run_dir: Path, final: dict[str, Any]) -> None:
    """Bind a resolved island/lake delivery without inventing a scientific review."""
    map_path = Path(final.get("final_map_path") or run_dir / "region_bpoly_final_map.png").resolve()
    candidate_path = (run_dir / "region_bpoly_geometry_candidate.json").resolve()
    candidate = {
        "schema_version": "region_bpoly_noncoastal_geometry_candidate_v1",
        "name": final.get("name"),
        "domain_type": final["domain_type"],
        "region_bpoly": final["region_bpoly"],
        "offshore_side_index": (final.get("offshore_boundary_artifacts") or {}).get("selected_side_index"),
        "final_map": str(map_path),
        "final_map_sha256": file_sha256(map_path) if map_path.is_file() else None,
        "map_geography_usable": (final.get("final_map_basemap") or {}).get("geography_usable"),
        "required_feature_coverage": final.get("qa", {}).get("ingredient_coverage"),
        "delivery_warnings": final.get("delivery_warnings", []),
        "scientific_review": None,
        "review_policy": "Resolved geometry provenance only; agent map inspection remains required.",
    }
    # Keep the candidate outside intermediate/: execute-mode cleanup must not
    # erase the evidence referenced by an accepted geometry freeze.
    if candidate_path.exists():
        if read_json(candidate_path) != candidate:
            raise ValueError("Noncoastal geometry evidence is frozen; use a new run directory for a revision")
    else:
        write_json(candidate_path, candidate)
    final["accepted_s1_geometry_freeze"] = {
        "schema_version": "region_bpoly_accepted_geometry_freeze_v1",
        "acceptance_route": "noncoastal_resolved_delivery",
        "region_bpoly_sha256": canonical_sha256(final["region_bpoly"]),
        "candidate_json": str(candidate_path),
        "candidate_json_sha256": file_sha256(candidate_path),
        "scientific_review_json": None,
        "scientific_review_json_sha256": None,
        "review_iteration": None,
        "offshore_side_index": candidate["offshore_side_index"],
        "final_map": candidate["final_map"],
        "final_map_sha256": candidate["final_map_sha256"],
        "downstream_geometry_repair_permitted": False,
    }


def _artifact_record(run_dir: Path, role: str, relative_path: str, required: bool) -> dict[str, Any]:
    path = run_dir / relative_path
    present = path.is_file()
    return {
        "role": role,
        "path": relative_path,
        "required_for_state": bool(required),
        "present": present,
        "bytes": path.stat().st_size if present else None,
        "sha256": file_sha256(path) if present else None,
    }


def write_standard_delivery(run_dir: Path, name: str, final: dict[str, Any], *, write_name_alias: bool = False) -> Path:
    """Write the canonical RegionBPoly package without adding a workflow gate."""
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_path = run_dir / "target_region_features.json"
    final_path = run_dir / "region_bpoly.json"
    manifest_path = run_dir / "region_bpoly_manifest.json"
    alias_name = f"{name}_region_bpoly.json"

    noncoastal_pass = final.get("domain_type") in {"island", "lake"} and final.get("final_status") == "pass"
    if noncoastal_pass:
        _freeze_noncoastal_delivery(run_dir, final)

    write_json(feature_path, final.get("target_region_features", {}))
    final["target_region_features_path"] = str(feature_path)
    final["output_manifest_path"] = str(manifest_path)
    final.setdefault("qa", {}).setdefault("target_region_features", {})["retained_path"] = str(feature_path)

    canonical_files: dict[str, str] = {
        "region_bpoly": "region_bpoly.json",
        "target_region_features": "target_region_features.json",
        "final_map": "region_bpoly_final_map.png",
        "offshore_boundary_artifacts": "offshore_boundary_artifacts.json",
        "manifest": "region_bpoly_manifest.json",
    }
    if noncoastal_pass:
        canonical_files["geometry_candidate"] = "region_bpoly_geometry_candidate.json"
    review = final.get("land_side_visual_review") or {}
    if final.get("domain_type") == "coastal" and final.get("final_status") == "pass":
        canonical_files["land_side_review"] = "region_bpoly_land_side_review.json"
        if review.get("review_map_path"):
            canonical_files["land_side_review_map"] = "region_bpoly_land_side_review.png"
    if final.get("place_discovery") is not None:
        canonical_files["place_discovery"] = "region_place_discovery.json"

    package_state = "accepted_delivery" if final.get("final_status") == "pass" else "internal_review"
    final["output_package"] = {
        "schema_version": OUTPUT_PACKAGE_SCHEMA,
        "canonical_root": ".",
        "package_state": package_state,
        "canonical_files": canonical_files,
        "compatibility_aliases": [alias_name] if write_name_alias else [],
        "package_complete": False,
        "delivery_ready": False,
    }
    write_json(final_path, final)

    required_roles = {
        "region_bpoly",
        "target_region_features",
        "final_map",
        "offshore_boundary_artifacts",
    }
    if noncoastal_pass:
        required_roles.add("geometry_candidate")
    if final.get("domain_type") == "coastal" and final.get("final_status") == "pass":
        required_roles.add("land_side_review")
        if review.get("effective_decision") == "pass" and review.get("review_map_path"):
            required_roles.add("land_side_review_map")
    if final.get("place_discovery") is not None:
        required_roles.add("place_discovery")

    records = [
        _artifact_record(run_dir, role, relative_path, role in required_roles)
        for role, relative_path in canonical_files.items()
        if role != "manifest"
    ]
    package_complete = all(record["present"] for record in records if record["required_for_state"])
    final["output_package"]["package_complete"] = package_complete
    final["output_package"]["delivery_ready"] = bool(final.get("final_status") == "pass" and package_complete)
    write_json(final_path, final)

    if write_name_alias:
        write_json(run_dir / alias_name, final)

    records = [
        _artifact_record(run_dir, role, relative_path, role in required_roles)
        for role, relative_path in canonical_files.items()
        if role != "manifest"
    ]
    if write_name_alias:
        records.append(_artifact_record(run_dir, "compatibility_alias", alias_name, False))

    feature_doc = final.get("target_region_features", {})
    features = feature_doc.get("features", [])
    manifest = {
        "schema_version": DELIVERY_MANIFEST_SCHEMA,
        "object_type": "RegionBPolyDeliveryManifest",
        "name": name,
        "created_at_utc": utc_now(),
        "final_status": final.get("final_status"),
        "package_state": package_state,
        "package_complete": package_complete,
        "delivery_ready": bool(final.get("final_status") == "pass" and package_complete),
        "hash_algorithm": "sha256",
        "manifest_hash_scope": "all listed files; the manifest excludes itself to avoid recursive hashing",
        "feature_plan": {
            "schema_version": feature_doc.get("schema_version"),
            "source_kind": feature_doc.get("source_kind"),
            "source_key": feature_doc.get("source_key"),
            "geometry_status": feature_doc.get("geometry_status"),
            "feature_count": len(features),
            "required_feature_count": sum(bool(feature.get("required", False)) for feature in features),
        },
        "files": records,
    }
    write_json(manifest_path, manifest)
    return final_path
