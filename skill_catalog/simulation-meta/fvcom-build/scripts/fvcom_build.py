#!/usr/bin/env python3
"""Deterministic FVCOM build-contract planning and evidence validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SCHEMA = "fvcom_build_worker_manifest_v1"
REQUIRED_DEFINES = {
    "DOUBLE_PRECISION",
    "SINGLE_OUTPUT",
    "MULTIPROCESSOR",
    "WET_DRY",
    "LIMITED_NO",
    "GCN",
}
FORBIDDEN_EXACT = {
    "SPHERICAL",
    "TWO_D_MODEL",
    "SEMI_IMPLICIT",
    "ORIG_SED",
    "CSTMS_SED",
    "LAG_PARTICLE",
    "AIR_PRESSURE",
    "WAVE_CURRENT_INTERACTION",
    "WAVE_OFFLINE",
    "ICE",
    "ICING",
    "WATER_QUALITY",
    "DATA_ASSIMILATION",
    "ONLINE_NESTING",
}
FORBIDDEN_PARTS = ("SEDIMENT", "PARTICLE", "RIVER", "BIOGEN", "FABM", "ASSIM")
MODULES = [
    "gcc/12.1.0",
    "intel/2023.2.0",
    "intel-oneapi-mpi/2021.11.0-intel",
    "netcdf-c/4.9.2-cray-mpich-intel",
    "netcdf-fortran/4.6.1-intel",
]
MANDATORY_EVIDENCE = [
    "source_inventory_before.sha256",
    "make.inc.before",
    "make.inc.after",
    "make.inc.diff",
    "make_inc_audit.json",
    "modules.txt",
    "compiler.txt",
    "nf-config.txt",
    "build_commands.txt",
    "build.log",
    "executable.sha256",
    "executable.file.txt",
    "executable.ldd.txt",
    "blank_namelist.nml",
    "smoke.log",
    "freeze.json",
]


class ContractError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def validate_request(request: Dict[str, Any]) -> None:
    if request.get("mode") not in {"barotropic_tide", "benchmark"}:
        raise ContractError("mode must be barotropic_tide or benchmark")
    case_id = request.get("case_id")
    if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id):
        raise ContractError("case_id must be a filesystem-safe non-empty string")
    physics = request.get("physics")
    if not isinstance(physics, dict) or physics.get("formulation") != "three_dimensional_barotropic":
        raise ContractError("physics.formulation must be three_dimensional_barotropic")
    kestrel = request.get("kestrel")
    if not isinstance(kestrel, dict):
        raise ContractError("kestrel object is required")
    source = kestrel.get("source_dir")
    if not isinstance(source, str) or not source.startswith("/home/yhuang168/"):
        raise ContractError("kestrel.source_dir must be an explicit path below /home/yhuang168")


def active_defines(text: str) -> List[str]:
    definitions: List[str] = []
    continuation = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        line = re.split(r"\s+#", line, maxsplit=1)[0]
        continuation += " " + line
        if not line.endswith("\\"):
            definitions.extend(re.findall(r"(?:^|\s)-D\s*([A-Za-z_][A-Za-z0-9_]*)", continuation))
            continuation = ""
    if continuation:
        definitions.extend(re.findall(r"(?:^|\s)-D\s*([A-Za-z_][A-Za-z0-9_]*)", continuation))
    return sorted(set(item.upper() for item in definitions))


def topdir_values(text: str) -> List[str]:
    values: List[str] = []
    for raw in text.splitlines():
        if raw.lstrip().startswith(("#", "!")):
            continue
        match = re.match(r"\s*TOPDIR\s*(?::?=)\s*(.*?)\s*$", raw, flags=re.IGNORECASE)
        if match:
            values.append(match.group(1))
    return values


def audit_make_inc(path: Path, expected_topdir: str) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    defines = active_defines(text)
    missing = sorted(REQUIRED_DEFINES.difference(defines))
    forbidden = sorted(
        name for name in defines
        if name in FORBIDDEN_EXACT or any(part in name for part in FORBIDDEN_PARTS)
    )
    topdirs = topdir_values(text)
    topdir_ok = topdirs == [expected_topdir]
    return {
        "schema": "fvcom_make_inc_audit_v1",
        "path": str(path),
        "sha256": sha256(path),
        "expected_topdir": expected_topdir,
        "active_topdir_values": topdirs,
        "active_defines": defines,
        "required_defines": sorted(REQUIRED_DEFINES),
        "missing_required_defines": missing,
        "forbidden_active_defines": forbidden,
        "topdir_ok": topdir_ok,
        "status": "pass" if topdir_ok and not missing and not forbidden else "fail",
    }


def build_plan(request: Dict[str, Any], local_root: Path, remote_root: str) -> Dict[str, Any]:
    validate_request(request)
    source = request["kestrel"]["source_dir"].rstrip("/")
    case_id = request["case_id"]
    return {
        "schema": "fvcom_build_plan_v1",
        "created_utc": utc_now(),
        "case_id": case_id,
        "source_dir": source,
        "local_artifact_root": str(local_root.resolve()),
        "remote_artifact_root": remote_root.rstrip("/"),
        "fvcom_version_required": "4.3.1",
        "compiler": "mpiifort",
        "compiler_options": ["-O3", "-fpp", "-qopenmp-stubs"],
        "modules": MODULES,
        "required_defines": sorted(REQUIRED_DEFINES),
        "forbidden_defines": sorted(FORBIDDEN_EXACT),
        "authoritative_blank_namelist_command": (
            "srun --account=hindcastra --partition=debug --nodes=1 --ntasks=1 "
            "fvcom_with_init --create_namelist=blank"
        ),
        "phases": [
            "authenticate",
            "inventory_before_mutation",
            "correct_make_inc_with_diff",
            "clean_compile",
            "link_audit",
            "create_blank_namelist",
            "minimal_input_reader_smoke",
            "freeze_executable_hash",
            "download_compact_evidence",
        ],
        "source_edits_after_freeze_allowed": False,
        "production_submission_allowed": False,
    }


def parse_checksum_file(path: Path) -> Tuple[str, str]:
    line = next((item.strip() for item in path.read_text(encoding="utf-8-sig").splitlines() if item.strip()), "")
    match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.*?)\s*$", line)
    if not match:
        raise ContractError(f"Invalid SHA-256 file: {path}")
    return match.group(1).lower(), match.group(2)


def finalize(request: Dict[str, Any], evidence: Path, artifact_root: Path) -> Dict[str, Any]:
    validate_request(request)
    missing = [name for name in MANDATORY_EVIDENCE if not (evidence / name).is_file()]
    blocking: List[str] = []
    warnings: List[str] = []
    hashes: Dict[str, str] = {}
    provenance: Dict[str, Any] = {
        "source_dir": request["kestrel"]["source_dir"],
        "fvcom_version": "4.3.1",
        "modules": MODULES,
        "evidence_dir": str(evidence.resolve()),
        "finalized_utc": utc_now(),
    }
    if missing:
        blocking.append("missing mandatory build evidence: " + ", ".join(missing))
    audit_path = evidence / "make_inc_audit.json"
    if audit_path.is_file():
        audit = load_json(audit_path)
        if audit.get("status") != "pass":
            blocking.append("make.inc audit did not pass")
        hashes["make_inc_sha256"] = str(audit.get("sha256", ""))
    executable_checksum = evidence / "executable.sha256"
    freeze_path = evidence / "freeze.json"
    if executable_checksum.is_file():
        executable_hash, executable_name = parse_checksum_file(executable_checksum)
        hashes["executable_sha256"] = executable_hash
        provenance["executable_name"] = executable_name
    if freeze_path.is_file():
        freeze = load_json(freeze_path)
        frozen_hash = str(freeze.get("executable_sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", frozen_hash):
            blocking.append("freeze.json lacks a valid executable_sha256")
        elif hashes.get("executable_sha256") and frozen_hash != hashes["executable_sha256"]:
            blocking.append("frozen and observed executable hashes differ")
        if freeze.get("source_edits_after_freeze_allowed") is not False:
            blocking.append("freeze record does not forbid post-freeze source edits")
        hashes["freeze_sha256"] = sha256(freeze_path)
    smoke_path = evidence / "smoke.log"
    if smoke_path.is_file():
        smoke = smoke_path.read_text(encoding="utf-8-sig", errors="replace")
        if "FVCOM_INPUT_SMOKE_PASS" not in smoke:
            blocking.append("smoke.log lacks FVCOM_INPUT_SMOKE_PASS")
    file_path = evidence / "executable.file.txt"
    if file_path.is_file() and "executable" not in file_path.read_text(encoding="utf-8-sig", errors="replace").lower():
        warnings.append("file metadata does not contain the word executable")
    ldd_path = evidence / "executable.ldd.txt"
    if ldd_path.is_file() and "not found" in ldd_path.read_text(encoding="utf-8-sig", errors="replace").lower():
        blocking.append("ldd reports an unresolved library")
    blank_path = evidence / "blank_namelist.nml"
    if blank_path.is_file() and "&NML_CASE" not in blank_path.read_text(encoding="utf-8-sig", errors="replace").upper():
        blocking.append("blank_namelist.nml is not an FVCOM namelist")
    artifact_root.mkdir(parents=True, exist_ok=True)
    lineage = {
        "schema": "fvcom_executable_lineage_v1",
        "case_id": request["case_id"],
        "status": "frozen" if not blocking else "incomplete",
        "hashes": hashes,
        "provenance": provenance,
        "blocking_reasons": blocking,
        "warnings": warnings,
    }
    lineage_path = artifact_root / "executable_lineage.json"
    write_json(lineage_path, lineage)
    hashes["lineage_sha256"] = sha256(lineage_path)
    status = "ready" if not blocking else "blocked_compile"
    manifest = {
        "schema": SCHEMA,
        "status": status,
        "artifact_root": str(artifact_root.resolve()),
        "hashes": hashes,
        "provenance": provenance,
        "warnings": warnings,
        "blocking_reasons": blocking,
        "resume_token": f"build:{request['case_id']}:{'frozen' if status == 'ready' else 'evidence'}",
    }
    write_json(artifact_root / "worker_manifest.json", manifest)
    return manifest


def cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-make-inc", help="Audit active TOPDIR and preprocessor definitions")
    audit.add_argument("--make-inc", type=Path, required=True)
    audit.add_argument("--expected-topdir", required=True)
    audit.add_argument("--output", type=Path)
    plan = sub.add_parser("plan", help="Validate a simulation request and emit a build plan")
    plan.add_argument("--request", type=Path, required=True)
    plan.add_argument("--artifact-root", type=Path, required=True)
    plan.add_argument("--remote-artifact-root", required=True)
    plan.add_argument("--output", type=Path, required=True)
    finish = sub.add_parser("finalize", help="Validate downloaded evidence and emit lineage/worker manifests")
    finish.add_argument("--request", type=Path, required=True)
    finish.add_argument("--evidence", type=Path, required=True)
    finish.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "audit-make-inc":
            result = audit_make_inc(args.make_inc, args.expected_topdir)
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "pass" else 2
        if args.command == "plan":
            result = build_plan(load_json(args.request), args.artifact_root, args.remote_artifact_root)
            write_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        result = finalize(load_json(args.request), args.evidence, args.artifact_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "ready" else 3
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
