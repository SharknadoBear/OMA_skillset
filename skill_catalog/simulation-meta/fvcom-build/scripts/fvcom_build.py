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
                            WIND_PROFILE, execution_site, has_river, is_cold_pair_stage, needs_air_pressure,
                            resolve_request, site_configuration)


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
    site = execution_site(request)
    config = site_configuration(request)
    if not config:
        raise ContractError(f"{site} object is required")
    source = config.get("source_dir")
    if is_cold_pair_stage(request):
        project = config.get("project_dir", "")
        build_root = config.get("pair_root") if request.get("profile") in {PAIR_PROFILE, WIND_PROFILE, FACTORIAL_PROFILE} else project
        if (not isinstance(build_root, str) or not build_root.startswith(("/pscratch/",) if site == "perlmutter" else ("/scratch/", "/kfs3/scratch/"))
                or not isinstance(project, str) or not isinstance(source, str)
                or any(".." in PurePosixPath(path).parts for path in (build_root, project, source))
                or PurePosixPath(build_root) not in PurePosixPath(source).parents):
            raise ContractError("profile source_dir must be inside the explicit project or pair scratch root")
        if request.get("profile") in {PAIR_PROFILE, WIND_PROFILE, FACTORIAL_PROFILE} and PurePosixPath(project).parent != PurePosixPath(build_root):
            raise ContractError("each paired project_dir must be a child of its site pair_root")
    elif site == "perlmutter":
        if not isinstance(source, str) or not source.startswith("/pscratch/") or ".." in PurePosixPath(source).parts:
            raise ContractError("Perlmutter source_dir must resolve to an explicit PSCRATCH path")
    elif not isinstance(source, str) or not source.startswith("/home/yhuang168/"):
        raise ContractError("kestrel.source_dir must be an explicit path below /home/yhuang168")


def build_environment(request: Dict[str, Any]) -> Dict[str, Any]:
    site = execution_site(request)
    if site == "kestrel" and not is_cold_pair_stage(request):
        return {"compiler": "mpiifort", "compiler_options": ["-O3", "-fpp", "-qopenmp-stubs"], "modules": MODULES}
    env = site_configuration(request).get("build_environment", {})
    if site == "perlmutter":
        if (env.get("site") != site or env.get("os") != "cpe" or env.get("compiler") != "ftn"
                or env.get("programming_environment") not in {"PrgEnv-gnu", "PrgEnv-cray"}):
            raise ContractError("Perlmutter requires a probe-verified CPU GNU/Cray ftn build environment")
    elif env.get("os") not in {"rh9", "rh8"} or not env.get("compiler") or not env.get("modules") or not env.get("compiler_options"):
        raise ContractError("profile build requires discovered os, compiler, compiler_options and modules in kestrel.build_environment")
    if env["os"] == "rh8" and not str(env.get("fallback_reason", "")).strip():
        raise ContractError("RH8 fallback requires a documented reason")
    if not isinstance(env.get("modules"), list) or not env["modules"] or not all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9._+/-]+", x) for x in env["modules"]):
        raise ContractError("build modules must be explicit safe module names")
    if site == "perlmutter":
        names = {item.split("/")[0] for item in env["modules"]}
        supported = ({"cray-netcdf-hdf5parallel", "cray-hdf5-parallel"},
                     {"cray-netcdf", "cray-hdf5"})
        required = {"cpu", env["programming_environment"], "cray-mpich"}
        if (not required <= names or not any(pair <= names for pair in supported)
                or any(name in {"gpu", "cudatoolkit"} or name.startswith("craype-accel-") for name in names)):
            raise ContractError("Perlmutter modules must bind CPU, the selected PrgEnv, MPI and matching HDF5/netCDF")
    if not isinstance(env.get("compiler_options"), list) or not env["compiler_options"] or not all(isinstance(x, str) and x for x in env["compiler_options"]):
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


def audit_make_inc(path: Path, expected_topdir: str, air_pressure: bool = False, spherical: bool = False) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    defines = active_defines(text)
    required = REQUIRED_DEFINES | ({"AIR_PRESSURE"} if air_pressure else set())
    forbidden_exact = ((FORBIDDEN_EXACT - {"AIR_PRESSURE"}) | WIND_INCOMPATIBLE_DEFINES
                       if air_pressure else FORBIDDEN_EXACT)
    if spherical:
        required = required | {"SPHERICAL"}
        forbidden_exact = forbidden_exact - {"SPHERICAL"}
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
    if request.get("profile") == "lake_superior_summer":
        config = site_configuration(request)
        source = PurePosixPath(config["source_dir"])
        project = PurePosixPath(config["project_dir"])
        if source.parent != project or ".." in source.parts:
            raise ContractError("the lake build must use a shared source directory within its PSCRATCH project")
        return {"schema":"fvcom_build_plan_v1","case_id":request["case_id"],"execution_site":"perlmutter",
                "created_utc":utc_now(),"source_commit":config["source_commit"],"source_dir":str(source),
                "local_artifact_root":str(local_root.resolve()),"remote_artifact_root":remote_root.rstrip("/"),
                "required_defines":sorted(REQUIRED_DEFINES | {"AIR_PRESSURE","FRESHWATER_LAKE","SPHERICAL"}),
                "phases":["verify_archived_source","native_cpu_mpi_netcdf","compile","native_generated_blank",
                          "linked_freshwater_eos","positive_and_negative_edge_donor","native_namelist_reader","freeze"],
                "production_submission_allowed":False,"source_edits_after_freeze_allowed":False}
    validate_request(request)
    environment = build_environment(request)
    source = site_configuration(request)["source_dir"].rstrip("/")
    case_id = request["case_id"]
    return {
        "schema": "fvcom_build_plan_v1",
        "created_utc": utc_now(),
        "case_id": case_id,
        "source_dir": source,
        "local_artifact_root": str(local_root.resolve()),
        "remote_artifact_root": remote_root.rstrip("/"),
        "fvcom_version_required": "4.3.1",
        **({"source_repository": site_configuration(request)["source_repository"],
            "source_commit": site_configuration(request)["source_commit"]}
           if is_cold_pair_stage(request) else {}),
        **environment,
        "required_defines": sorted(REQUIRED_DEFINES | ({"AIR_PRESSURE"} if pressure_build(request) else set())),
        "forbidden_defines": sorted((FORBIDDEN_EXACT - {"AIR_PRESSURE"}) | WIND_INCOMPATIBLE_DEFINES
                                   if pressure_build(request) else FORBIDDEN_EXACT),
        "execution_site": execution_site(request),
        "authoritative_blank_namelist_command": (
            "srun --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=cores fvcom_plased --create_namelist=blank"
            if execution_site(request) == "perlmutter" else
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


def check_cpu_reader(text: str, member: str) -> None:
    """Check the two physically distinct configurations sent to the native parser."""
    required = {
        "NML_PHYSICS": {"BAROTROPIC": "t"},
        "NML_SURFACE_FORCING": {"AIRPRESSURE_ON": "t", "HEATING_ON": "f", "PRECIPITATION_ON": "f"},
    }
    surface = required["NML_SURFACE_FORCING"]
    if member == "control":
        surface.update(WIND_ON="f", AIRPRESSURE_KIND="constant", AIRPRESSURE_VALUE=101325.0)
        required["NML_RIVER_TYPE"] = {"RIVER_NUMBER": 0.0}
    elif member == "combined":
        surface.update(WIND_ON="t", WIND_TYPE="stress", WIND_KIND="variable", AIRPRESSURE_KIND="variable")
        required["NML_RIVER_TYPE"] = {"RIVER_KIND": "variable", "RIVER_INFLOW_LOCATION": "node",
                                     "RIVER_TS_SETTING": "calculated", "RIVER_INFO_FILE": "river.nml"}
    else:
        raise ContractError("unknown native parser probe member")
    for group, keys in required.items():
        sections = re.findall(r"^\s*&" + group + r"\b(.*?)^\s*/", text, re.I | re.M | re.S)
        if len(sections) != 1:
            raise ContractError(f"{member} probe needs one {group} namelist")
        values = {key.upper(): value.strip().strip("'\"").lower()
                  for key, value in re.findall(r"^\s*(\w+)\s*=\s*([^,!\r\n]+)", sections[0], re.M)}
        for key, expected in keys.items():
            value = values.get(key)
            if isinstance(expected, float):
                try:
                    value = float((value or "").replace("d", "e"))
                except ValueError:
                    pass
            elif expected in {"t", "f"}:
                value = {".true.": "t", ".false.": "f"}.get(value, value)
            if value != expected:
                raise ContractError(f"{member} probe requires {group}.{key}={expected}")
        if member == "combined" and group == "NML_RIVER_TYPE" and not re.fullmatch(r"[1-9]\d*", values.get("RIVER_NUMBER", "")):
            raise ContractError("combined probe requires a positive river count")
        if member == "combined" and group == "NML_SURFACE_FORCING":
            wind = values.get("WIND_FILE", "")
            if wind in {"", "none"} or wind != values.get("AIRPRESSURE_FILE"):
                raise ContractError("combined probe requires the joint wind and pressure surface input")


def finalize(request: Dict[str, Any], evidence: Path, artifact_root: Path, *,
             original_source_dir: str | None = None) -> Dict[str, Any]:
    request = resolve_request(request)
    validate_request(request)
    environment = build_environment(request)
    site = execution_site(request)
    missing = [name for name in MANDATORY_EVIDENCE if not (evidence / name).is_file()]
    blocking: List[str] = []
    warnings: List[str] = []
    hashes: Dict[str, str] = {}
    provenance: Dict[str, Any] = {
        "source_dir": site_configuration(request)["source_dir"],
        "fvcom_version": "4.3.1",
        **environment,
        "execution_site": site,
        "evidence_dir": str(evidence.resolve()),
        "finalized_utc": utc_now(),
    }
    if missing:
        blocking.append("missing mandatory build evidence: " + ", ".join(missing))
    if is_cold_pair_stage(request):
        expected_source = {key: site_configuration(request)[key]
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
            if filename == "os-release.txt" and site == "kestrel" and not re.search(r'^VERSION_ID=[\"\']?' + environment["os"][-1] + r'(?:[.\"\']|$)', content, re.M):
                blocking.append("observed OS does not match the selected build environment")
            if filename == "modules.txt" and any(module not in content for module in environment["modules"]):
                blocking.append("observed modules do not match the selected build environment")
            if filename == "build_commands.txt" and environment["compiler"] not in content:
                blocking.append("build transcript does not identify the selected compiler")
    if site == "perlmutter":
        for filename in ("nersc_host.txt", "build_job_id.txt", "scheduler_job.txt", "wrapper_link.txt", "python_runtime.txt",
                         "mpi_netcdf_probe.log", "perlmutter_mpi_netcdf.f90", "cpu_probe.nc",
                         "control_reader.log", "combined_reader.log", "probe_control_run.nml", "probe_combined_run.nml"):
            path = evidence / filename
            if not path.is_file() or path.stat().st_size == 0:
                blocking.append(f"missing Perlmutter native evidence: {filename}")
                continue
            hashes[filename] = sha256(path)
            if filename == "nersc_host.txt" and path.read_text().strip() != site:
                blocking.append("native build host is not Perlmutter")
            elif filename == "python_runtime.txt":
                lines = path.read_text().splitlines()
                version = re.match(r"(\d+)\.(\d+)\.", lines[1]) if len(lines) > 1 else None
                if not lines or not lines[0].startswith("/") or not version or tuple(map(int, version.groups())) < (3, 10):
                    blocking.append("native build does not identify a compatible Python interpreter")
            elif filename == "scheduler_job.txt" and not re.search(r"\bFeatures=cpu\b", path.read_text()):
                blocking.append("Slurm accounting does not identify a CPU allocation")
            elif filename == "mpi_netcdf_probe.log" and "FVCOM_MPI_NETCDF_PASS ranks=2 value=3" not in path.read_text().splitlines():
                blocking.append("the two-rank MPI and netCDF readback did not pass")
            elif filename.endswith("_reader.log"):
                content = path.read_text(errors="replace").lower()
                if "subroutine ends: read_name_list;" not in content or "expected_read_end.dat not found" not in content:
                    blocking.append(f"{filename} did not reach the native reader sentinel")
            elif filename.startswith("probe_") and filename.endswith("_run.nml"):
                try:
                    check_cpu_reader(path.read_text(errors="replace"), filename.split("_")[1])
                except ContractError as exc:
                    blocking.append(str(exc))
        modules = evidence / "modules.txt"
        if modules.exists() and re.search(r"^\s*(?:gpu(?:/|$)|cudatoolkit/|craype-accel-)", modules.read_text(), re.M):
            blocking.append("the native CPU environment includes a GPU target")
        compiler = evidence / "compiler.txt"
        compiler_id = "GNU Fortran" if environment["programming_environment"] == "PrgEnv-gnu" else "Cray Fortran"
        if compiler.exists() and compiler_id.lower() not in compiler.read_text(errors="replace").lower():
            blocking.append("native compiler identity differs from the selected CPU programming environment")
        native_freeze = evidence / "freeze.json"
        if native_freeze.is_file():
            record = load_json(native_freeze)
            family = environment["programming_environment"].removeprefix("PrgEnv-")
            job = str(record.get("job_id", ""))
            location = str(PurePosixPath(original_source_dir or site_configuration(request)["source_dir"]) / f"build_{job}_{family}")
            if (not re.fullmatch(r"[0-9]+", job) or record.get("execution_site") != site
                    or record.get("build_source_dir") != location):
                blocking.append("native Perlmutter frozen build is not bound to its source, job and compiler")
            job_id = evidence / "build_job_id.txt"
            scheduler = evidence / "scheduler_job.txt"
            if ((job_id.is_file() and job_id.read_text().strip() != job) or
                    (scheduler.is_file() and f"JobId={job}" not in scheduler.read_text().split())):
                blocking.append("Slurm job and native freeze identify different builds")
            original_source_dir = location
    audit_path = evidence / "make_inc_audit.json"
    if audit_path.is_file():
        audit = load_json(audit_path)
        if audit.get("status") != "pass":
            blocking.append("make.inc audit did not pass")
        hashes["make_inc_sha256"] = str(audit.get("sha256", ""))
    if pressure_build(request) and (evidence / "make.inc.after").is_file():
        native = audit_make_inc(evidence / "make.inc.after", original_source_dir or site_configuration(request)["source_dir"], True)
        if native["status"] != "pass" or native["sha256"] != hashes.get("make_inc_sha256"):
            blocking.append("native make.inc does not bind the pressure-enabled feature set")
        if site == "perlmutter":
            expected = "GFORTRAN" if environment["programming_environment"] == "PrgEnv-gnu" else "CRAY"
            if set(native["active_defines"]).intersection({"GFORTRAN", "CRAY", "IFORT"}) != {expected}:
                blocking.append("native make.inc compiler selector differs from the CPU programming environment")
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
                freeze.get(key) != site_configuration(request)[key]
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
    if original.get("execution_site", "kestrel") != execution_site(request):
        raise ContractError("reusing an executable across execution sites is unsupported")
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
            or original["source_commit"] != site_configuration(request)["source_commit"]
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
    target = str(PurePosixPath(site_configuration(request)["source_dir"]) / source_bin.name)
    binding = dict(original, native_executable=target, archived_native_executable=str(source_bin),
                   archived_binding_sha256=sha256(binding_path))
    output = artifact_root / "executable_reuse_binding.json"
    if output.exists() and load_json(output) != binding:
        raise ContractError("target already contains a different executable binding")
    write_json(output, binding)
    return binding


def cli(argv: Iterable[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["finalize-lake"]:
        from finalize_lake_build import cli as lake_cli
        return lake_cli(argv[1:])
    if argv[:1] == ["salinity"]:
        from salinity_recipe import cli as salinity_cli
        return salinity_cli(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("finalize-lake", help="Freeze a native Perlmutter freshwater baroclinic build")
    sub.add_parser("salinity",help="Prepare or verify a Cartesian, active-salinity budget build")
    audit = sub.add_parser("audit-make-inc", help="Audit active TOPDIR and preprocessor definitions")
    audit.add_argument("--make-inc", type=Path, required=True)
    audit.add_argument("--expected-topdir", required=True)
    audit.add_argument("--air-pressure", action="store_true", help="Require the pressure-capable wind-pair build")
    audit.add_argument("--spherical", action="store_true", help="Require geographic-coordinate model dynamics")
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
            result = audit_make_inc(args.make_inc, args.expected_topdir, args.air_pressure, args.spherical)
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
