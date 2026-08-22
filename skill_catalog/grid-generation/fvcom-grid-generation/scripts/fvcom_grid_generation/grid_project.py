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
    "06_raw_mesh": {"raw_mesh.2dm"},
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
        if manifest.get("schema_version") != "fvcom_grid_project_v1" or manifest.get("name") != name:
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


def promote(project: str | Path, stage: str, source: str | Path, artifact_name: str) -> dict[str, Any]:
    root = _root(project)
    if stage not in STAGES or artifact_name not in STAGES[stage]:
        raise ValueError(f"{artifact_name} is not canonical for {stage}")
    source_path = Path(source).resolve()
    work_root = (root / stage / "_work").resolve()
    if not source_path.is_file() or not _inside(source_path, work_root):
        raise ValueError("promotion source must be a regular file under the selected stage _work directory")
    target = root / stage / artifact_name
    digest = _verified_copy(source_path, target)
    manifest_path = root / "project_manifest.json"
    manifest = _read(manifest_path)
    record = {"path": _relative(target, root), "sha256": digest, "bytes": target.stat().st_size}
    existing = manifest.setdefault("selected_artifacts", {}).get(artifact_name)
    if existing and existing != record:
        raise ValueError("promoted-artifact lineage is incompatible with resume")
    manifest["selected_artifacts"][artifact_name] = record
    _atomic_json(manifest_path, manifest)
    _append_command(root, "promote", {"stage": stage, "artifact": artifact_name, "sha256": digest})
    return record


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
    for record in manifest.get("selected_artifacts", {}).values():
        path = root / record["path"]
        if not _inside(path, root) or not path.is_file() or path.is_symlink():
            failures.append("selected_artifact_missing_or_unsafe")
        elif sha256_file(path) != record.get("sha256"):
            failures.append("selected_artifact_hash_stale")
    mesh = status.get("mesh")
    if mesh:
        path = root / mesh["path"]
        if not path.is_file() or sha256_file(path) != mesh.get("sha256"):
            failures.append("final_mesh_hash_stale")
    if require_submission_ready:
        if status.get("submission_eligible") is not True or status.get("fvcom_ready") is not True:
            failures.append("project_not_submission_ready")
        if not mesh:
            failures.append("submission_mesh_missing")
    return {"passed": not failures, "failure_taxonomy": list(dict.fromkeys(failures)), "status": status}


__all__ = ["FINAL_COMPANIONS", "STAGES", "init_project", "promote", "publish", "validate"]
