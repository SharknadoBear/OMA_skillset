from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import file_sha256, utc_now, write_json


OUTPUT_PACKAGE_SCHEMA = "region_bpoly_output_package_v1"
DELIVERY_MANIFEST_SCHEMA = "region_bpoly_delivery_manifest_v1"


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
