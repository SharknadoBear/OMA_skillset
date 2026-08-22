"""Portable standardized FVCOM grid-project management."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .open_exterior import validate_open_exterior_contract


STAGES = {
    "00_request": set(),
    "01_region": {"region_bpoly.json"},
    "02_coastline": {"coastline.gpkg"},
    "03_boundary": {"bdry_arc_manifest.json"},
    "04_bathymetry": {"bathymetry.nc"},
    "05_mesh_intent": {"case_manifest.json", "size_field.nc"},
    "06_raw_mesh": {"raw_mesh.2dm", "raw_mesh_manifest.json"},
    "07_conditioning": {"conditioned_mesh.2dm"},
    "08_audit": {"final_audit.json"},
}
FINAL_COMPANIONS = {
    "mesh_quality": "mesh_quality.json",
    "mesh_conditioning": "mesh_conditioning.json",
    "boundary_nodes": "boundary_nodes.geojson",
    "obc_remap_manifest": "obc_remap_manifest.json",
    "roundtrip_audit": "roundtrip_audit.json",
    "mesh_review_map": "mesh_review_map.png",
}
DEFAULT_MESHER_POLICY = {
    "candidate_id": "gmsh_frontal_delaunay_6",
    "backend": "gmsh",
    "algorithm": 6,
    "algorithm_name": "Frontal-Delaunay",
    "thread_count": 1,
    "random_seed": 1,
    "native_smoothing_steps": 8,
    "fallback": "explicit_only_outside_operational_project",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _root(project: str | Path) -> Path:
    return Path(project).resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _relative(path: Path, root: Path) -> str:
    if not _inside(path, root):
        raise ValueError(f"path escapes project: {path}")
    return path.resolve().relative_to(root.resolve()).as_posix()


def _verified_copy(source: Path, target: Path) -> str:
    if source.is_symlink() or target.is_symlink():
        raise ValueError("symlinks are forbidden by fvcom_grid_project_v1")
    source_hash = sha256_file(source)
    if target.exists():
        if sha256_file(target) != source_hash:
            raise FileExistsError(f"refusing to overwrite a different promoted artifact: {target}")
        return source_hash
    tmp = target.with_name(target.name + ".tmp")
    shutil.copy2(source, tmp)
    if sha256_file(tmp) != source_hash:
        tmp.unlink(missing_ok=True)
        raise IOError("verified copy hash mismatch")
    os.replace(tmp, target)
    return source_hash


def _append_command(root: Path, operation: str, details: dict[str, Any]) -> None:
    with (root / "commands.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"created_utc": utc_now(), "operation": operation, **details}) + "\n")


def init_project(project: str | Path, name: str) -> dict[str, Any]:
    root = _root(project)
    if root.exists() and any(root.iterdir()):
        manifest_path = root / "project_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError("nonempty directory is not an FVCOM grid project")
        manifest = _read(manifest_path)
        if (
            manifest.get("schema_version") != "fvcom_grid_project_v1"
            or manifest.get("name") != name
            or manifest.get("mesher_policy") != DEFAULT_MESHER_POLICY
        ):
            raise ValueError("incompatible project resume")
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        (root / stage / "_work").mkdir(parents=True, exist_ok=True)
    (root / "final").mkdir()
    (root / "logs").mkdir()
    (root / "commands.jsonl").touch()
    manifest = {
        "schema_version": "fvcom_grid_project_v1",
        "name": name,
        "created_utc": utc_now(),
        "layout": {"stages": list(STAGES), "final": "final", "logs": "logs"},
        "mesher_policy": dict(DEFAULT_MESHER_POLICY),
        "selected_artifacts": {},
    }
    _atomic_json(root / "project_manifest.json", manifest)
    _atomic_json(root / "project_status.json", {
        "schema_version": "fvcom_grid_delivery_v1",
        "state": "initialized",
        "fvcom_ready": False,
        "submission_eligible": False,
        "failure_taxonomy": [],
    })
    _append_command(root, "init", {"name": name})
    return manifest


def _resolve_recorded_path(value: str, parent: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = parent / path
    return path.resolve()


def _gmsh6_provenance(
    root: Path,
    source_path: Path,
    candidate_manifest_path: str | Path | None,
) -> dict[str, Any]:
    if candidate_manifest_path is None:
        raise ValueError(
            "raw_mesh.2dm promotion requires --generator-manifest from the "
            "Gmsh Frontal-Delaunay-6 candidate"
        )
    candidate_path = Path(candidate_manifest_path).resolve()
    work_root = (root / "06_raw_mesh" / "_work").resolve()
    if (
        not candidate_path.is_file()
        or candidate_path.is_symlink()
        or not _inside(candidate_path, work_root)
    ):
        raise ValueError("generator manifest must be a regular file under 06_raw_mesh/_work")
    candidate = _read(candidate_path)
    if (
        candidate.get("schema_version") != "fvcom_mesher_candidate_manifest_v1"
        or candidate.get("candidate_id") != DEFAULT_MESHER_POLICY["candidate_id"]
        or candidate.get("raw_stage") is not True
        or candidate.get("common_conditioning_applied") is not False
    ):
        raise ValueError("raw candidate is not the operational Gmsh-6 RAW contract")

    artifacts = candidate.get("artifacts", {})
    mesh_record = artifacts.get("sms_2dm", {})
    recorded_mesh = _resolve_recorded_path(str(mesh_record.get("path", "")), candidate_path.parent)
    source_hash = sha256_file(source_path)
    if (
        recorded_mesh != source_path
        or mesh_record.get("sha256") != source_hash
        or not recorded_mesh.is_file()
    ):
        raise ValueError("candidate mesh path/hash does not match the promoted raw mesh")

    report_record = artifacts.get("generator_report", {})
    report_path = _resolve_recorded_path(str(report_record.get("path", "")), candidate_path.parent)
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or not _inside(report_path, work_root)
        or sha256_file(report_path) != report_record.get("sha256")
    ):
        raise ValueError("Gmsh generator report is missing, unsafe, or stale")
    report = _read(report_path)
    required = {
        "backend": "gmsh",
        "algorithm": 6,
        "algorithm_name": "Frontal-Delaunay",
        "thread_count": 1,
        "random_seed": 1,
        "native_smoothing_steps": 8,
        "algorithm_fallback_enabled": False,
        "raw_stage": True,
        "common_conditioning_applied": False,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("generator report does not satisfy the deterministic Gmsh-6 policy")
    return {
        "schema_version": "fvcom_raw_mesh_provenance_v1",
        "candidate_id": DEFAULT_MESHER_POLICY["candidate_id"],
        "backend": "gmsh",
        "algorithm": 6,
        "algorithm_name": "Frontal-Delaunay",
        "thread_count": 1,
        "random_seed": 1,
        "native_smoothing_steps": 8,
        "algorithm_fallback_enabled": False,
        "raw_stage": True,
        "common_conditioning_applied": False,
        "raw_mesh": {"path": "06_raw_mesh/raw_mesh.2dm", "sha256": source_hash},
        "source_candidate_manifest": {
            "path": _relative(candidate_path, root),
            "sha256": sha256_file(candidate_path),
        },
        "generator_report": {
            "path": _relative(report_path, root),
            "sha256": sha256_file(report_path),
        },
    }


def promote(
    project: str | Path,
    stage: str,
    source: str | Path,
    artifact_name: str,
    *,
    generator_manifest: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(project)
    if stage not in STAGES or artifact_name not in STAGES[stage]:
        raise ValueError(f"{artifact_name} is not canonical for {stage}")
    if (stage, artifact_name) == ("06_raw_mesh", "raw_mesh_manifest.json"):
        raise ValueError(
            "raw_mesh_manifest.json is manager-generated during raw_mesh.2dm promotion"
        )
    source_path = Path(source).resolve()
    work_root = (root / stage / "_work").resolve()
    if not source_path.is_file() or not _inside(source_path, work_root):
        raise ValueError("promotion source must be a regular file under the selected stage _work directory")
    if generator_manifest is not None and (stage, artifact_name) != ("06_raw_mesh", "raw_mesh.2dm"):
        raise ValueError("--generator-manifest is valid only for raw_mesh.2dm promotion")
    raw_provenance = None
    if (stage, artifact_name) == ("06_raw_mesh", "raw_mesh.2dm"):
        raw_provenance = _gmsh6_provenance(root, source_path, generator_manifest)
    target = root / stage / artifact_name
    digest = _verified_copy(source_path, target)
    manifest_path = root / "project_manifest.json"
    manifest = _read(manifest_path)
    record = {"path": _relative(target, root), "sha256": digest, "bytes": target.stat().st_size}
    existing = manifest.setdefault("selected_artifacts", {}).get(artifact_name)
    if existing and existing != record:
        raise ValueError("promoted-artifact lineage is incompatible with resume")
    manifest["selected_artifacts"][artifact_name] = record
    if raw_provenance is not None:
        provenance_path = root / "06_raw_mesh" / "raw_mesh_manifest.json"
        if provenance_path.exists():
            if _read(provenance_path) != raw_provenance:
                raise ValueError("raw-mesh provenance is incompatible with resume")
        else:
            _atomic_json(provenance_path, raw_provenance)
        provenance_record = {
            "path": _relative(provenance_path, root),
            "sha256": sha256_file(provenance_path),
            "bytes": provenance_path.stat().st_size,
        }
        existing_provenance = manifest["selected_artifacts"].get("raw_mesh_manifest.json")
        if existing_provenance and existing_provenance != provenance_record:
            raise ValueError("raw-mesh provenance lineage is incompatible with resume")
        manifest["selected_artifacts"]["raw_mesh_manifest.json"] = provenance_record
    _atomic_json(manifest_path, manifest)
    _append_command(root, "promote", {"stage": stage, "artifact": artifact_name, "sha256": digest})
    return record


def _default_mesher_failures(root: Path, manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if manifest.get("mesher_policy") != DEFAULT_MESHER_POLICY:
        failures.append("default_mesher_policy_missing_or_invalid")
    selected = manifest.get("selected_artifacts", {})
    if "raw_mesh.2dm" not in selected:
        return failures
    record = selected.get("raw_mesh_manifest.json")
    if not isinstance(record, dict):
        failures.append("raw_mesh_provenance_missing")
        return failures
    path = root / str(record.get("path", ""))
    if (
        not _inside(path, root)
        or not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != record.get("sha256")
    ):
        failures.append("raw_mesh_provenance_missing_or_stale")
        return failures
    provenance = _read(path)
    required = {
        "schema_version": "fvcom_raw_mesh_provenance_v1",
        "candidate_id": "gmsh_frontal_delaunay_6",
        "backend": "gmsh",
        "algorithm": 6,
        "algorithm_name": "Frontal-Delaunay",
        "thread_count": 1,
        "random_seed": 1,
        "native_smoothing_steps": 8,
        "algorithm_fallback_enabled": False,
        "raw_stage": True,
        "common_conditioning_applied": False,
    }
    if any(provenance.get(key) != value for key, value in required.items()):
        failures.append("raw_mesh_not_default_gmsh6")
    mesh_record = provenance.get("raw_mesh", {})
    mesh_path = root / str(mesh_record.get("path", ""))
    if (
        not _inside(mesh_path, root)
        or not mesh_path.is_file()
        or sha256_file(mesh_path) != mesh_record.get("sha256")
    ):
        failures.append("raw_mesh_provenance_hash_mismatch")
    return failures


def _selected_mesh(root: Path, explicit: str | Path | None) -> Path | None:
    if explicit:
        mesh = Path(explicit).resolve()
        if not _inside(mesh, root):
            raise ValueError("terminal mesh must be inside the project")
        return mesh
    for relative in ("07_conditioning/conditioned_mesh.2dm", "06_raw_mesh/raw_mesh.2dm"):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def publish(
    project: str | Path,
    *,
    mesh: str | Path | None,
    companions: dict[str, str | Path],
    fvcom_ready: bool,
    submission_eligible: bool,
    obc_status: str,
    forcing_status: str,
    failures: list[str],
    open_exterior_source: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(project)
    manifest = _read(root / "project_manifest.json")
    selected_mesh = _selected_mesh(root, mesh)
    mesher_failures = _default_mesher_failures(root, manifest)
    if (
        selected_mesh is not None
        and "raw_mesh.2dm" not in manifest.get("selected_artifacts", {})
    ):
        mesher_failures.append("raw_mesh_provenance_missing")
    if selected_mesh is not None and mesher_failures:
        raise ValueError(
            "terminal publication requires deterministic Gmsh-6 provenance: "
            + ", ".join(mesher_failures)
        )
    open_audit = None
    if open_exterior_source:
        open_audit = validate_open_exterior_contract(open_exterior_source, required=True)
        if not open_audit["passed"]:
            failures = list(dict.fromkeys(failures + open_audit["failure_taxonomy"]))
            submission_eligible = False
            fvcom_ready = False
        # Delivery manifests remain portable: retain the evidence hash and
        # decision, never an absolute workstation path.
        open_audit = {
            key: value
            for key, value in open_audit.items()
            if key != "contract_path"
        }
    final_dir = root / "final"
    if selected_mesh is not None:
        if selected_mesh.is_symlink() or not selected_mesh.is_file():
            raise ValueError("terminal mesh is not a regular file")
        missing = sorted(set(FINAL_COMPANIONS) - set(companions))
        if missing:
            raise ValueError("terminal mesh publication requires companions: " + ", ".join(missing))
        mesh_hash = _verified_copy(selected_mesh, final_dir / "fvcom_grid.2dm")
        for key, final_name in FINAL_COMPANIONS.items():
            source = Path(companions[key]).resolve()
            if not source.is_file() or not _inside(source, root):
                raise ValueError(f"{key} must be a project-local regular file")
            _verified_copy(source, final_dir / final_name)
    else:
        mesh_hash = None
        submission_eligible = False
        fvcom_ready = False
    if submission_eligible and (not fvcom_ready or selected_mesh is None or failures):
        raise ValueError("submission eligibility requires a ready, failure-free terminal mesh")
    status = {
        "schema_version": "fvcom_grid_delivery_v1",
        "created_utc": utc_now(),
        "state": "mesh_published" if selected_mesh is not None else "failed_pre_mesh",
        "mesh": ({"path": "final/fvcom_grid.2dm", "sha256": mesh_hash} if mesh_hash else None),
        "fvcom_ready": bool(fvcom_ready),
        "submission_eligible": bool(submission_eligible),
        "obc_status": obc_status,
        "forcing_status": forcing_status,
        "selected_stage_hashes": manifest.get("selected_artifacts", {}),
        "open_exterior_audit": open_audit,
        "failure_taxonomy": list(dict.fromkeys(failures)),
    }
    _atomic_json(final_dir / "fvcom_grid_status.json", status)
    _atomic_json(root / "project_status.json", status)
    _append_command(root, "publish", {"mesh_sha256": mesh_hash, "submission_eligible": submission_eligible})
    return status


def validate(project: str | Path, *, require_submission_ready: bool = False) -> dict[str, Any]:
    root = _root(project)
    failures: list[str] = []
    manifest_path = root / "project_manifest.json"
    status_path = root / "project_status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        return {"passed": False, "failure_taxonomy": ["project_contract_files_missing"]}
    manifest = _read(manifest_path)
    status = _read(status_path)
    if manifest.get("schema_version") != "fvcom_grid_project_v1":
        failures.append("project_schema_invalid")
    if status.get("schema_version") != "fvcom_grid_delivery_v1":
        failures.append("delivery_schema_invalid")
    failures.extend(_default_mesher_failures(root, manifest))
    for record in manifest.get("selected_artifacts", {}).values():
        path = root / record["path"]
        if not _inside(path, root) or not path.is_file() or path.is_symlink():
            failures.append("selected_artifact_missing_or_unsafe")
        elif sha256_file(path) != record.get("sha256"):
            failures.append("selected_artifact_hash_stale")
    mesh = status.get("mesh")
    if mesh:
        if "raw_mesh.2dm" not in manifest.get("selected_artifacts", {}):
            failures.append("raw_mesh_provenance_missing")
        path = root / mesh["path"]
        if not path.is_file() or sha256_file(path) != mesh.get("sha256"):
            failures.append("final_mesh_hash_stale")
    if require_submission_ready:
        if status.get("submission_eligible") is not True or status.get("fvcom_ready") is not True:
            failures.append("project_not_submission_ready")
        if not mesh:
            failures.append("submission_mesh_missing")
    return {"passed": not failures, "failure_taxonomy": list(dict.fromkeys(failures)), "status": status}


__all__ = [
    "DEFAULT_MESHER_POLICY",
    "FINAL_COMPANIONS",
    "STAGES",
    "init_project",
    "promote",
    "publish",
    "validate",
]
