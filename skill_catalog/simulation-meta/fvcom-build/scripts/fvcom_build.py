#!/usr/bin/env python3
"""Deterministic FVCOM build-contract planning and evidence validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Tuple


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fvcom-run-control" / "scripts"))
from fvcom_profiles import (BAROTROPIC_MODES, COMBINED_MODE, FACTORIAL_PROFILE, PAIR_PROFILE, RIVER_MODE, WIND_MODE,
                            WIND_PROFILE, has_river, is_cold_pair_stage, needs_air_pressure, resolve_request)


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
WIND_INCOMPATIBLE_DEFINES = {"ATMO_TIDE", "MEAN_FLOW"}
FORBIDDEN_PARTS = ("SEDIMENT", "PARTICLE", "PLASTIC", "RIVER", "BIOGEN", "FABM", "ASSIM")
PAPER_TEMPLATE_FIELDS = {
    "NML_PHYSICS": ("FLUME_BOTTOM", "FLUME_DRAG_COEF"),
    "NML_STATION_TIMESERIES": ("READ_GRID_EDGE_FROM_FILE", "GRID_EDGE_FILE_NAME"),
    "NML_ADDITIONAL_MODELS": ("PLASTIC_MODEL",),
}
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
    if request.get("mode") not in BAROTROPIC_MODES | {"benchmark"}:
        raise ContractError("unsupported build mode")
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
    if is_cold_pair_stage(request):
        project = kestrel.get("project_dir", "")
        build_root = kestrel.get("pair_root") if request.get("profile") in {PAIR_PROFILE, WIND_PROFILE, FACTORIAL_PROFILE} else project
        if (not isinstance(build_root, str) or not build_root.startswith(("/scratch/", "/kfs3/scratch/"))
                or not isinstance(project, str) or not isinstance(source, str)
                or any(".." in PurePosixPath(path).parts for path in (build_root, project, source))
                or PurePosixPath(build_root) not in PurePosixPath(source).parents):
            raise ContractError("profile source_dir must be inside the explicit project or pair scratch root")
        if request.get("profile") in {PAIR_PROFILE, WIND_PROFILE, FACTORIAL_PROFILE} and PurePosixPath(project).parent != PurePosixPath(build_root):
            raise ContractError("each paired project_dir must be a child of kestrel.pair_root")
    elif not isinstance(source, str) or not source.startswith("/home/yhuang168/"):
        raise ContractError("kestrel.source_dir must be an explicit path below /home/yhuang168")


def build_environment(request: Dict[str, Any]) -> Dict[str, Any]:
    if not is_cold_pair_stage(request):
        return {"compiler": "mpiifort", "compiler_options": ["-O3", "-fpp", "-qopenmp-stubs"], "modules": MODULES}
    env = request["kestrel"].get("build_environment", {})
    if env.get("os") not in {"rh9", "rh8"} or not env.get("compiler") or not env.get("modules") or not env.get("compiler_options"):
        raise ContractError("profile build requires discovered os, compiler, compiler_options and modules in kestrel.build_environment")
    if env["os"] == "rh8" and not str(env.get("fallback_reason", "")).strip():
        raise ContractError("RH8 fallback requires a documented reason")
    if not isinstance(env["modules"], list) or not all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9._+/-]+", x) for x in env["modules"]):
        raise ContractError("build modules must be explicit safe module names")
    if not isinstance(env["compiler_options"], list) or not all(isinstance(x, str) and x for x in env["compiler_options"]):
        raise ContractError("compiler_options must be a nonempty string list")
    return dict(env)


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


def pressure_build(request: Dict[str, Any]) -> bool:
    return needs_air_pressure(request)


def audit_make_inc(path: Path, expected_topdir: str, air_pressure: bool = False) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    defines = active_defines(text)
    required = REQUIRED_DEFINES | ({"AIR_PRESSURE"} if air_pressure else set())
    forbidden_exact = ((FORBIDDEN_EXACT - {"AIR_PRESSURE"}) | WIND_INCOMPATIBLE_DEFINES
                       if air_pressure else FORBIDDEN_EXACT)
    missing = sorted(required.difference(defines))
    forbidden = sorted(
        name for name in defines
        if name in forbidden_exact or any(part in name for part in FORBIDDEN_PARTS)
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
        "required_defines": sorted(required),
        "missing_required_defines": missing,
        "forbidden_active_defines": forbidden,
        "topdir_ok": topdir_ok,
        "status": "pass" if topdir_ok and not missing and not forbidden else "fail",
    }


def build_plan(request: Dict[str, Any], local_root: Path, remote_root: str) -> Dict[str, Any]:
    request = resolve_request(request)
    validate_request(request)
    environment = build_environment(request)
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
        **({"source_repository": request["kestrel"]["source_repository"],
            "source_commit": request["kestrel"]["source_commit"]}
           if is_cold_pair_stage(request) else {}),
        **environment,
        "required_defines": sorted(REQUIRED_DEFINES | ({"AIR_PRESSURE"} if pressure_build(request) else set())),
        "forbidden_defines": sorted((FORBIDDEN_EXACT - {"AIR_PRESSURE"}) | WIND_INCOMPATIBLE_DEFINES
                                   if pressure_build(request) else FORBIDDEN_EXACT),
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


def finalize(request: Dict[str, Any], evidence: Path, artifact_root: Path, *,
             original_source_dir: str | None = None) -> Dict[str, Any]:
    request = resolve_request(request)
    validate_request(request)
    environment = build_environment(request)
    missing = [name for name in MANDATORY_EVIDENCE if not (evidence / name).is_file()]
    blocking: List[str] = []
    warnings: List[str] = []
    hashes: Dict[str, str] = {}
    provenance: Dict[str, Any] = {
        "source_dir": request["kestrel"]["source_dir"],
        "fvcom_version": "4.3.1",
        **environment,
        "evidence_dir": str(evidence.resolve()),
        "finalized_utc": utc_now(),
    }
    if missing:
        blocking.append("missing mandatory build evidence: " + ", ".join(missing))
    if is_cold_pair_stage(request):
        expected_source = {key: request["kestrel"][key]
                           for key in ("source_repository", "source_commit")}
        provenance.update(expected_source)
        source_path = evidence / "source_revision.json"
        if not source_path.is_file():
            blocking.append("missing source_revision.json: record the observed Git origin and HEAD before build edits")
        else:
            observed_source = load_json(source_path)
            hashes["source_revision_sha256"] = sha256(source_path)
            if any(observed_source.get(key) != value for key, value in expected_source.items()):
                blocking.append("observed source repository/commit does not match the requested source pin")
        for filename in ("os-release.txt", "modules.txt", "build_commands.txt"):
            path = evidence / filename
            if not path.is_file():
                blocking.append(f"missing native build evidence: {filename}")
                continue
            hashes[filename] = sha256(path)
            content = path.read_text(encoding="utf-8-sig", errors="replace")
            if filename == "os-release.txt" and not re.search(r'^VERSION_ID=[\"\']?' + environment["os"][-1] + r'(?:[.\"\']|$)', content, re.M):
                blocking.append("observed OS does not match the selected build environment")
            if filename == "modules.txt" and any(module not in content for module in environment["modules"]):
                blocking.append("observed modules do not match the selected build environment")
            if filename == "build_commands.txt" and environment["compiler"] not in content:
                blocking.append("build transcript does not identify the selected compiler")
    audit_path = evidence / "make_inc_audit.json"
    if audit_path.is_file():
        audit = load_json(audit_path)
        if audit.get("status") != "pass":
            blocking.append("make.inc audit did not pass")
        hashes["make_inc_sha256"] = str(audit.get("sha256", ""))
    if pressure_build(request) and (evidence / "make.inc.after").is_file():
        native = audit_make_inc(evidence / "make.inc.after", original_source_dir or request["kestrel"]["source_dir"], True)
        if native["status"] != "pass" or native["sha256"] != hashes.get("make_inc_sha256"):
            blocking.append("native make.inc does not bind the pressure-enabled feature set")
        log_path = evidence / "build.log"
        if log_path.is_file():
            commands = log_path.read_text(encoding="utf8", errors="replace").splitlines()
            for src in ("mod_obcs.F", "advave_edge_gcn.F", "external_step.F"):
                if not any("cpp " in command and "-DAIR_PRESSURE " in command
                           and src in command.split() for command in commands):
                    blocking.append(f"no observed AIR_PRESSURE preprocessing for {src}")
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
        if is_cold_pair_stage(request) and freeze.get("build_environment") != environment:
            blocking.append("freeze record does not bind the selected native build environment")
        if is_cold_pair_stage(request) and any(
                freeze.get(key) != request["kestrel"][key]
                for key in ("source_repository", "source_commit")):
            blocking.append("freeze record does not bind the requested source repository/commit")
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
    if blank_path.is_file() and is_cold_pair_stage(request):
        blank = blank_path.read_text(encoding="utf-8-sig", errors="replace")
        template_fields = dict(PAPER_TEMPLATE_FIELDS)
        if request.get("profile") in {PAIR_PROFILE, FACTORIAL_PROFILE} or has_river(request):
            template_fields["NML_RIVER_TYPE"] = (
                "RIVER_NUMBER", "RIVER_TS_SETTING", "RIVER_INFLOW_LOCATION", "RIVER_KIND", "RIVER_INFO_FILE",
            )
        if pressure_build(request):
            template_fields["NML_SURFACE_FORCING"] = (
                "WIND_ON", "WIND_TYPE", "WIND_KIND", "WIND_FILE", "AIRPRESSURE_ON",
                "AIRPRESSURE_KIND", "AIRPRESSURE_VALUE", "AIRPRESSURE_FILE",
            )
        for group, keys in template_fields.items():
            body = re.search(r"^\s*&" + group + r"\b(.*?)^\s*/", blank, re.I | re.M | re.S)
            for key in keys:
                if body is None or not re.search(r"\b" + key + r"\s*=", body[1], re.I):
                    blocking.append(f"compiled paper template lacks {group}.{key}")
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


def reuse_build(request: dict, binding_path: Path, evidence: Path, archive: Path,
                binary: Path, template: Path, artifact_root: Path) -> dict:
    """Revalidate original archived evidence and bind the identical binary in a new case."""
    request = resolve_request(request)
    validate_request(request)
    original = load_json(binding_path)
    if original.get("schema") != "fvcom_executable_reuse_binding_v1" or original.get("status") != "frozen":
        raise ContractError("a frozen executable reuse binding is required")
    for key, path in (("executable_sha256", binary), ("blank_template_sha256", template),
                      ("build_evidence_archive_sha256", archive), ("native_freeze_sha256", evidence / "freeze.json")):
        if original.get(key) != sha256(path):
            raise ContractError(f"archived {key} does not match {path.name}")
    with tarfile.open(archive) as tar:
        checked = set()
        for member in tar:
            if member.isdir() and member.name == "evidence":
                continue
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.parent != PurePosixPath("evidence")
                    or name.name in checked or name.name in {".", ".."}):
                raise ContractError("unexpected member in native build evidence archive")
            with tar.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha256(evidence / name.name):
                    raise ContractError(f"extracted evidence drifted for {name.name}")
            checked.add(name.name)
    if not set(MANDATORY_EVIDENCE).issubset(checked):
        raise ContractError("archived native build evidence is incomplete")
    if (original["build_environment"] != build_environment(request)
            or original["source_commit"] != request["kestrel"]["source_commit"]
            or sha256(evidence / "blank_namelist.nml") != sha256(template)
            or parse_checksum_file(evidence / "executable.sha256")[0] != original["executable_sha256"]):
        raise ContractError("archived build, template or executable differs from the request")
    source_bin = PurePosixPath(original["native_executable"])
    manifest = finalize(request, evidence, artifact_root, original_source_dir=str(source_bin.parent))
    if manifest["status"] != "ready":
        raise ContractError("source build evidence failed: " + ", ".join(manifest["blocking_reasons"]))
    audit = load_json(evidence / "make_inc_audit.json")
    if audit["active_defines"] != original["feature_defines"]:
        raise ContractError("archived feature flags differ from the native make audit")
    target = str(PurePosixPath(request["kestrel"]["source_dir"]) / source_bin.name)
    binding = dict(original, native_executable=target, archived_native_executable=str(source_bin),
                   archived_binding_sha256=sha256(binding_path))
    output = artifact_root / "executable_reuse_binding.json"
    if output.exists() and load_json(output) != binding:
        raise ContractError("target already contains a different executable binding")
    write_json(output, binding)
    return binding


def cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-make-inc", help="Audit active TOPDIR and preprocessor definitions")
    audit.add_argument("--make-inc", type=Path, required=True)
    audit.add_argument("--expected-topdir", required=True)
    audit.add_argument("--air-pressure", action="store_true", help="Require the pressure-capable wind-pair build")
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
    reuse = sub.add_parser("reuse", help="Verify a frozen native executable and rebind it to a new project")
    for key in ("request", "binding", "evidence", "archive", "binary", "template", "artifact-root"):
        reuse.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "audit-make-inc":
            result = audit_make_inc(args.make_inc, args.expected_topdir, args.air_pressure)
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "pass" else 2
        if args.command == "plan":
            result = build_plan(load_json(args.request), args.artifact_root, args.remote_artifact_root)
            write_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "reuse":
            result = reuse_build(load_json(args.request), args.binding, args.evidence, args.archive,
                                 args.binary, args.template, args.artifact_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        result = finalize(load_json(args.request), args.evidence, args.artifact_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "ready" else 3
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
