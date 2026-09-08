"""Validate an audited regional FVCOM production attempt from frozen artifacts."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
import netCDF4 as nc4

_script_path = Path(__file__).resolve()
for _run_control in (_script_path.parents[2] / "fvcom-run-control" / "scripts",
                     _script_path.parents[3] / "simulation-meta" / "fvcom-run-control" / "scripts"):
    if (_run_control / "fvcom_time_anchor.py").is_file():
        sys.path.insert(0, str(_run_control))
        break
from audit_fvcom_production import parse_namelist, parse_utc, sha256
from fvcom_time_anchor import resolve_anchor


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(value: Any) -> str:
    value = str(value).lower()
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("a required frozen SHA-256 digest is missing or invalid")
    return value


def resolve_project_path(project: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project / path).resolve()


def verify_prerequisites(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project.resolve()
    attempt = resolve_project_path(project, str(args.attempt))
    if not attempt.is_relative_to(project / "run" / args.grid_case):
        raise ValueError("attempt must belong to project/run/<grid-case>")
    request_path = project / "request.json"
    request = read(request_path)
    attempt_manifest_path = attempt / "attempt_manifest.json"
    attempt_manifest = read(attempt_manifest_path)
    if attempt_manifest.get("grid_case") != args.grid_case:
        raise ValueError("attempt grid case differs from requested grid case")
    freeze_path = args.input_freeze.resolve() if args.input_freeze else attempt / "input_freeze.json"
    freeze = read(freeze_path)
    if freeze.get("schema") != "fvcom_input_freeze_v1":
        raise ValueError("input freeze must use fvcom_input_freeze_v1")
    files = freeze.get("files", {})
    if not files or any(Path(name).name != name or "/" in name or "\\" in name for name in files):
        raise ValueError("input freeze must contain flat file basenames")
    computed_bundle = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if computed_bundle != digest(freeze.get("input_bundle_sha256")):
        raise ValueError("input bundle digest is not the canonical frozen file map")
    input_dir = resolve_project_path(project, freeze["input_dir"])
    if not input_dir.is_relative_to(project / "input" / args.grid_case):
        raise ValueError("frozen input directory must belong to this project grid case")
    if resolve_project_path(project, attempt_manifest["input_dir"]) != input_dir:
        raise ValueError("attempt input directory differs from the freeze")
    actual_names = {path.name for path in input_dir.iterdir() if path.is_file()}
    if actual_names != set(files):
        raise ValueError("actual input file inventory differs from the freeze")
    for name, expected in files.items():
        if sha256(input_dir / name) != digest(expected):
            raise ValueError(f"frozen input changed: {name}")
    nml_hash = digest(freeze.get("run_namelist_sha256"))
    nmls = [path for path in attempt.glob("*_run.nml") if sha256(path) == nml_hash]
    if len(nmls) != 1:
        raise ValueError("exactly one attempt run namelist must match the input freeze")
    namelist = nmls[0]
    if attempt_manifest.get("attempt_namelist_sha256", nml_hash) != nml_hash:
        raise ValueError("attempt manifest and frozen run namelist disagree")
    nml = parse_namelist(namelist)
    start = parse_utc(request["period"]["analysis_start"])
    end = parse_utc(request["period"]["analysis_end"])
    if parse_utc(nml["START_DATE"]) != start or parse_utc(nml["END_DATE"]) != end:
        raise ValueError("this runner accepts the requested production interval only")
    if nml.get("STARTUP_TYPE", "").lower() != "hotstart":
        raise ValueError("production validation requires the frozen hotstart attempt")
    binding_path = project / "run/build/executable_reuse_binding.json"
    binding = read(binding_path)
    executable_hash = digest(freeze.get("executable_sha256"))
    if binding.get("status") != "frozen" or digest(binding.get("executable_sha256")) != executable_hash:
        raise ValueError("production executable differs from the verified executable binding")
    mapping_path = input_dir / "station_mapping.json"
    mapping = read(mapping_path)
    station_rows = mapping.get("stations", [])
    keys = [(str(row["station_id"]), str(row["role"])) for row in station_rows]
    if mapping.get("status") != "ready" or not keys or len({x[0] for x in keys}) != len(keys):
        raise ValueError("station mapping must be ready with unique station IDs")
    if any(role not in {"water_level", "current"} for _, role in keys):
        raise ValueError("unsupported station role in frozen mapping")
    if not any(role == "water_level" for _, role in keys):
        raise ValueError("at least one required water-level comparison is mandatory")
    observation_path = args.observation_manifest.resolve() if args.observation_manifest else project / "analysis/observations/observation_manifest.json"
    observation = read(observation_path)
    if not str(observation.get("status", "")).startswith("ready") or observation.get("blocking_reasons"):
        raise ValueError("observation manifest is not ready")
    if observation.get("hashes", {}).get("station_mapping_sha256") != sha256(mapping_path):
        raise ValueError("observation manifest does not bind the actual run station mapping")
    inventory_path = resolve_project_path(project, observation["station_inventory"])
    if observation.get("hashes", {}).get("station_inventory_sha256") != sha256(inventory_path):
        raise ValueError("observation station inventory changed")
    for name, value in observation.items():
        expected = observation.get("hashes", {}).get(name + "_sha256")
        if expected is not None and isinstance(value, str):
            if sha256(resolve_project_path(project, value)) != digest(expected):
                raise ValueError(f"observation evidence changed: {name}")
    inventory = read(inventory_path)
    eligible = {(str(row["id"]), str(row["role"])) for row in inventory.get("stations", []) if row.get("eligible")}
    if eligible != set(keys):
        raise ValueError("frozen mapping and eligible period-screened station inventory disagree")
    for key in ("period_start", "period_end"):
        if key in observation and parse_utc(observation[key]) != (start if key == "period_start" else end):
            raise ValueError("observation period differs from production request")
    forcing_path = args.forcing_manifest.resolve() if args.forcing_manifest else project / "forcing" / args.grid_case / "forcing_manifest.json"
    forcing = read(forcing_path)
    constituents = forcing.get("constituents", [])
    expected_count = request.get("tpxo", {}).get("expected_constituent_count", 22)
    if forcing.get("status") != "ready" or forcing.get("blocking_reasons") or len(constituents) != expected_count or len(set(constituents)) != len(constituents):
        raise ValueError("forcing manifest does not contain the complete ordered constituent set")
    elevation_name = nml.get("OBC_ELEVATION_FILE")
    if elevation_name not in files or forcing.get("hashes", {}).get("forcing_sha256") != files[elevation_name]:
        raise ValueError("forcing manifest does not bind the actual run elevation file")
    provenance = forcing.get("provenance", {})
    rule = "utide_exact_time_nodal_v1"
    flags = [False, False, False, False]
    if provenance.get("reconstruction_rule_version") != rule or provenance.get("utide_ngflags") != flags:
        raise ValueError("forcing predates exact-time nodal reconstruction; preserve it and rebuild dependent stages")
    builder_hash = digest(forcing.get("hashes", {}).get("builder_sha256"))
    with nc4.Dataset(input_dir / elevation_name) as tide:
        if (getattr(tide, "reconstruction_rule_version", None) != rule
                or getattr(tide, "utide_ngflags_json", None) != json.dumps(flags)
                or getattr(tide, "builder_sha256", None) != builder_hash
                or not provenance.get("utide_version")
                or getattr(tide, "utide_version", None) != provenance["utide_version"]):
            raise ValueError("forcing NetCDF and manifest disagree on exact-time nodal reconstruction provenance")
    audit_path = args.production_audit.resolve() if args.production_audit else attempt / "production_audit.json"
    audit = read(audit_path)
    if audit.get("status") != "passed" or audit.get("blocking_reasons") or audit.get("run_namelist_sha256") != nml_hash:
        raise ValueError("production audit is not passing or belongs to another namelist")
    if audit.get("lineage_manifest_sha256") != sha256(freeze_path):
        raise ValueError("production audit must use this exact input freeze as its lineage")
    if audit.get("station_mapping_sha256") != sha256(mapping_path):
        raise ValueError("production audit uses another station mapping")
    if audit.get("audit_rules_version") != "startup_anchor_exact_clock_3d_v1":
        raise ValueError("production audit predates startup-anchored exact-clock and 3D-state checks")
    startup_name = nml.get("STARTUP_FILE", "")
    if startup_name not in files:
        raise ValueError("startup restart is not a frozen flat input file")
    anchor = resolve_anchor(nml, namelist, startup_restart=input_dir / startup_name,
                            expected_sha256=files[startup_name], supplied=audit.get("time_anchor", {}))
    if not audit.get("restart_time", {}).get("exact_clock_verified") or not audit.get("restart_time", {}).get("startup_anchored"):
        raise ValueError("final restart lacks verified anchored clock evidence")
    if parse_utc(audit["restart_time"]["decoded_end_utc"]) != end:
        raise ValueError("final restart does not end at production END_DATE")
    if not audit.get("full_grid_time", {}).get("exact_clock_verified"):
        raise ValueError("full-grid exact clocks were not verified")
    for key, cadence in (("station_time", 360), ("full_grid_time", 10800)):
        info = audit[key]
        if not info.get("startup_anchored") or info.get("iint_at_start") != anchor["iint_at_start"] or info.get("iint_first") != anchor["iint_at_start"]:
            raise ValueError("production output is not anchored to the frozen startup restart")
        count = int(round((end - start).total_seconds() / cadence)) + 1
        if info.get("record_count") != count or info.get("interval_seconds") != cadence:
            raise ValueError(f"production audit lacks the required exact {key} cadence/count")
        if parse_utc(info["decoded_start_utc"]) != start or parse_utc(info["decoded_end_utc"]) != end:
            raise ValueError("production audit coverage differs from request")
    audit_nc = audit["station_netcdf"] + audit["full_grid_netcdf"] + [audit["restart_netcdf"]]
    if not all(row.get("required_variables_finite_and_unmasked") for row in audit_nc):
        raise ValueError("production audit predates mandatory masked-value checks")
    station_paths = list(args.station_netcdf or [])
    if not station_paths:
        for row in audit["station_netcdf"]:
            candidate = Path(row["path"])
            station_paths.append(candidate if candidate.is_file() else attempt / PurePosixPath(row["path"].replace("\\", "/")).name)
    station_paths = [path.resolve() for path in station_paths]
    if len(set(station_paths)) != len(station_paths):
        raise ValueError("station output file list contains duplicates")
    actual_outputs = {sha256(path) for path in station_paths}
    if actual_outputs != {digest(row["sha256"]) for row in audit["station_netcdf"]} or len(station_paths) != len(audit["station_netcdf"]):
        raise ValueError("retrieved station files differ from the audited production outputs")
    return dict(project=project, attempt=attempt, request=request, request_path=request_path,
                namelist=namelist, input_dir=input_dir, freeze=freeze, freeze_path=freeze_path,
                executable_hash=executable_hash, binding_path=binding_path, mapping_path=mapping_path,
                inventory_path=inventory_path, observation_path=observation_path, observation=observation,
                forcing_path=forcing_path, constituents=constituents, audit_path=audit_path,
                station_paths=station_paths, station_keys=keys, start=start, end=end, time_anchor=anchor)


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    context = verify_prerequisites(args)
    scripts = args.skill_dir.resolve() / "scripts"
    sys.path.insert(0, str(scripts))
    condense = importlib.import_module("condense_fvcom_station").condense
    prepare = importlib.import_module("prepare_validation_tables").prepare
    validation_module = importlib.import_module("fvcom_tidal_validation")
    if set(context["constituents"]) - set(validation_module.FREQUENCIES_CPH):
        raise ValueError("validation kernel lacks a forced constituent frequency; do not silently omit it")
    if args.check_only:
        return {"status": "ready_for_validation", "workflow_status": "not_run", "attempt": str(context["attempt"])}
    output = args.output_dir.resolve() if args.output_dir else context["project"] / "analysis" / args.grid_case / f"validation_{context['attempt'].name}"
    if output.exists():
        raise FileExistsError(f"validation output is immutable; choose a new --output-dir: {output}")
    output.mkdir(parents=True)
    stack = [{"path": str(path), "sha256": sha256(path)} for path in context["station_paths"]]
    stack_digest = stack[0]["sha256"] if len(stack) == 1 else hashlib.sha256(json.dumps(stack, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    lineage = {"schema": "fvcom_case_validation_lineage_v1", "case_id": context["request"]["case_id"],
               "grid_case": args.grid_case, "generated_at": datetime.now(timezone.utc).isoformat(),
               "executable_sha256": context["executable_hash"], "executable_binding_sha256": sha256(context["binding_path"]),
               "input_bundle_sha256": context["freeze"]["input_bundle_sha256"],
               "input_freeze_sha256": sha256(context["freeze_path"]), "run_namelist_sha256": sha256(context["namelist"]),
               "model_output_sha256": stack_digest, "model_output_hash_method": "file_sha256" if len(stack) == 1 else "canonical_ordered_path_and_sha256_stack",
               "model_outputs": stack, "observation_manifest_sha256": sha256(context["observation_path"]),
               "station_mapping_sha256": sha256(context["mapping_path"]), "station_inventory_sha256": sha256(context["inventory_path"]),
               "forcing_manifest_sha256": sha256(context["forcing_path"]), "production_audit_sha256": sha256(context["audit_path"]),
               "constituents_forced": context["constituents"],
               "period": [context["start"].isoformat(), context["end"].isoformat()]}
    lineage_path = output / "validation_lineage.json"
    write(lineage_path, lineage)
    condensed_path = output / "condensation_manifest.json"
    condense(context["station_paths"], context["mapping_path"], context["namelist"], output / "condensed", condensed_path,
             time_anchor=context["time_anchor"], startup_restart_path=Path(context["time_anchor"]["startup_restart_path"]))
    tables_path = output / "validation_tables_manifest.json"
    tables = prepare(condensed_path, context["observation_path"].parent, output / "tables", tables_path)
    products = {(row["station_id"], row["role"]): Path(row["path"]) for row in tables["products"]}
    if set(products) != set(context["station_keys"]):
        raise ValueError("prepared tables differ from exact mapped station roles")
    water = [products[key] for key in context["station_keys"] if key[1] == "water_level"]
    currents = [products[key] for key in context["station_keys"] if key[1] == "current"]
    report_path = output / "validation_report.html"
    result = validation_module.validate(water, currents, output, report_path, context["constituents"], lineage_path, context["inventory_path"])
    if result["workflow_status"] != "validation_complete":
        raise ValueError("scientific validation did not complete; partial artifacts are retained")
    availability = "available" if currents else "unavailable_no_eligible_period_screened_profiles"
    result["current_validation_availability"] = availability
    result["threshold_policy"] = "informational_for_initial_regional_accepted_and_fresh_tests; never a stability-tuning gate"
    write(output / "validation_summary.json", result)
    report = report_path.read_text(encoding="utf-8").replace("first two Galveston cases", "initial regional accepted/fresh cases")
    if not currents:
        report = report.replace("</body>", "<p>Current validation is unavailable: no eligible downward-looking profiles overlap this period in the retained wet domain. Inventory and exclusion evidence are preserved above.</p></body>")
    report_path.write_text(report, encoding="utf-8")
    artifacts = {name: {"path": str(output / name), "sha256": sha256(output / name)} for name in
                 ("validation_lineage.json", "condensation_manifest.json", "validation_tables_manifest.json", "validation_summary.json", "validation_report.html")}
    pipeline = {"schema": "fvcom_case_validation_pipeline_v1", "status": "ready", "workflow_status": result["workflow_status"],
                "case_id": context["request"]["case_id"], "grid_case": args.grid_case, "attempt": str(context["attempt"]),
                "scientific_assessment": result["scientific_assessment"], "current_validation_availability": availability,
                "artifacts": artifacts, "blocking_reasons": [], "resume_token": "validation_complete:" + artifacts["validation_summary.json"]["sha256"]}
    write(output / "validation_pipeline_manifest.json", pipeline)
    return pipeline


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--grid-case", required=True)
    p.add_argument("--attempt", type=Path, required=True, help="Absolute or project-relative production attempt")
    p.add_argument("--input-freeze", type=Path)
    p.add_argument("--production-audit", type=Path)
    p.add_argument("--observation-manifest", type=Path)
    p.add_argument("--forcing-manifest", type=Path)
    p.add_argument("--station-netcdf", type=Path, action="append")
    p.add_argument("--skill-dir", type=Path, default=Path.home() / ".codex/skills/fvcom-tidal-validation")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--check-only", action="store_true")
    return p


if __name__ == "__main__":
    print(json.dumps(run_validation(parser().parse_args()), indent=2))
