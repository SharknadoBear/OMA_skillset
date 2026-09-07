from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import itertools
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any


G = 9.81
ALLOWED_STATES = {
    "initialized", "awaiting_authentication", "parallel_preparation",
    "preparation_join", "input_assembly", "stability_smoke",
    "stability_canary", "stability_spinup", "benchmarking", "production",
    "validation", "validation_complete", "waiting_user", "user_stopped",
    "scope_exhausted",
}


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML input requires PyYAML; use JSON or install PyYAML") from exc
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("request must be a mapping")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_request(req: dict[str, Any]) -> None:
    if req.get("schema") != "simulation_request_v1":
        raise ValueError("schema must be simulation_request_v1")
    if req.get("mode") not in {"barotropic_tide", "benchmark"}:
        raise ValueError("mode must be barotropic_tide or benchmark")
    if not str(req.get("case_id", "")).strip():
        raise ValueError("case_id is required")
    if req.get("grid_policy") != "accepted_then_fresh":
        raise ValueError("this workflow requires grid_policy=accepted_then_fresh")
    physics = req.get("physics", {})
    if physics.get("formulation") != "three_dimensional_barotropic":
        raise ValueError("physics.formulation must be three_dimensional_barotropic")
    if req.get("tpxo", {}).get("constituents") != "all_available":
        raise ValueError("tpxo.constituents must be all_available")
    obs = req.get("observations", {})
    if obs.get("provider") != "noaa_coops":
        raise ValueError("observations.provider must be noaa_coops")
    if obs.get("current_policy") != "downward_all_bin_profiles_only":
        raise ValueError("unsupported current policy")


def project_layout(root: Path) -> list[Path]:
    paths = [
        "gridding/accepted_t6v6", "gridding/fresh_reproduction",
        "forcing/shared_source", "forcing/accepted_t6v6", "forcing/fresh_reproduction",
        "run/build", "analysis/observations",
    ]
    for case in ("accepted_t6v6", "fresh_reproduction"):
        paths += [
            f"input/{case}/base", f"input/{case}/attempts", f"input/{case}/final",
            f"run/{case}/attempts", f"run/{case}/benchmark",
            f"output/{case}/station", f"output/{case}/condensed", f"analysis/{case}",
        ]
    return [root / p for p in paths]


def init_project(request_path: Path, project: Path) -> dict[str, Any]:
    req = load_data(request_path)
    validate_request(req)
    project.mkdir(parents=True, exist_ok=True)
    for p in project_layout(project):
        p.mkdir(parents=True, exist_ok=True)
    request_copy = project / "request.json"
    if request_copy.exists() and json.loads(request_copy.read_text(encoding="utf-8")) != req:
        raise FileExistsError("project already contains a different request.json")
    write_json(request_copy, req)
    manifest = {
        "schema": "simulation_project_manifest_v1",
        "case_id": req["case_id"],
        "created_at": utcnow(),
        "request": "request.json",
        "request_sha256": sha256(request_copy),
        "local_root": str(project.resolve()),
        "kestrel_root": f"/scratch/yhuang168/FVCOM_Simulation/{req['case_id']}",
        "grid_order": ["accepted_t6v6", "fresh_reproduction"],
        "immutable_attempts": True,
    }
    if not (project / "project_manifest.json").exists():
        write_json(project / "project_manifest.json", manifest)
    status = {
        "schema": "simulation_project_status_v1",
        "case_id": req["case_id"],
        "state": "initialized",
        "updated_at": utcnow(),
        "active_grid_case": None,
        "active_attempt": None,
        "workflow_status": "initialized",
        "scientific_assessment": None,
        "blocking_reasons": [],
        "history": [],
    }
    if not (project / "project_status.json").exists():
        write_json(project / "project_status.json", status)
    (project / "commands.jsonl").touch(exist_ok=True)
    render_report(project)
    return manifest


def set_state(project: Path, state: str, evidence: str, grid_case: str | None = None,
              attempt: str | None = None, blockers: list[str] | None = None) -> dict[str, Any]:
    if state not in ALLOWED_STATES:
        raise ValueError(f"invalid state {state}")
    path = project / "project_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    changed_at = utcnow()
    status.setdefault("history", []).append({
        "state": status.get("state"), "to_state": state,
        "at": changed_at, "evidence": evidence,
    })
    status.update({
        "state": state, "updated_at": changed_at, "active_grid_case": grid_case,
        "active_attempt": attempt, "blocking_reasons": blockers or [],
        "workflow_status": state,
    })
    write_json(path, status)
    render_report(project)
    return status


def record_command(project: Path, kind: str, command: str, status: str,
                   artifacts: list[str]) -> dict[str, Any]:
    """Append a non-secret local/remote command record to the project ledger."""
    if any(token in command.lower() for token in ("password=", "--password", "otp=", "--otp")):
        raise ValueError("refusing to record a command containing credential-like arguments")
    ledger = project / "commands.jsonl"
    if not ledger.exists():
        raise FileNotFoundError(f"project command ledger does not exist: {ledger}")
    entry = {
        "at": utcnow(),
        "kind": kind,
        "command": command,
        "status": status,
        "artifacts": artifacts,
    }
    with ledger.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def evaluate_preparation_join(manifest_paths: list[Path]) -> dict[str, Any]:
    """Validate the common manifest envelope for the three initial workers."""
    if len(manifest_paths) != 3:
        raise ValueError(f"preparation join requires exactly three worker manifests; found {len(manifest_paths)}")
    required = {
        "status", "artifact_root", "hashes", "provenance", "warnings",
        "blocking_reasons", "resume_token",
    }
    workers = []
    tokens = []
    for path in manifest_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"{path} is missing common worker fields: {missing}")
        if not (value.get("schema") or value.get("schema_version")):
            raise ValueError(f"{path} has no schema identifier")
        token = (
            json.dumps(value["resume_token"], sort_keys=True, separators=(",", ":"))
            if isinstance(value["resume_token"], (dict, list)) else str(value["resume_token"])
        )
        if not token:
            raise ValueError(f"{path} has an empty resume token")
        for logical_name, digest_value in value["hashes"].items():
            digest = str(digest_value).lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{path} hash {logical_name} is not a SHA-256 digest")
        tokens.append(token)
        workers.append({
            "manifest": str(path), "manifest_sha256": sha256(path),
            "schema": value.get("schema") or value.get("schema_version"),
            "worker": value.get("worker") or Path(value["artifact_root"]).name,
            "status": value["status"], "hash_count": len(value["hashes"]),
            "blocking_reasons": value["blocking_reasons"], "resume_token": token,
        })
    if len(set(tokens)) != len(tokens):
        raise ValueError("initial-worker resume tokens must be unique")
    not_ready = [item for item in workers if item["status"] != "ready" or item["hash_count"] == 0]
    blocker_text = json.dumps(not_ready, sort_keys=True).lower()
    user_gate = any(word in blocker_text for word in ("authorization", "authentication", "credential", "locator", "password", "otp"))
    return {
        "schema": "fvcom_preparation_join_v1", "generated_at": utcnow(),
        "status": "ready" if not not_ready else "waiting_user" if user_gate else "blocked",
        "ready": not not_ready, "workers": workers,
        "blocking_reasons": [item for item in not_ready],
    }


def parse_2dm(path: Path) -> tuple[dict[int, tuple[float, float, float]], list[tuple[int, int, int, int]], list[list[int]]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    elems: list[tuple[int, int, int, int]] = []
    chains: list[list[int]] = []
    active_chain: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            parts = raw.split()
            if not parts:
                continue
            if parts[0] == "ND" and len(parts) >= 5:
                nodes[int(parts[1])] = tuple(map(float, parts[2:5]))
            elif parts[0] == "E3T" and len(parts) >= 6:
                elems.append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])))
            elif parts[0] == "NS":
                for token in parts[1:]:
                    value = int(token)
                    active_chain.append(abs(value))
                    if value < 0:
                        chains.append(active_chain)
                        active_chain = []
                        break  # remaining token, when present, is the SMS material id
    if active_chain:
        raise ValueError("unterminated NS nodestring")
    if not nodes or not elems:
        raise ValueError("2DM contains no nodes or E3T elements")
    return nodes, elems, chains


def triangle_altitudes(xy: list[tuple[float, float]]) -> tuple[float, float]:
    lengths = [math.dist(xy[1], xy[2]), math.dist(xy[0], xy[2]), math.dist(xy[0], xy[1])]
    twice_area = abs((xy[1][0] - xy[0][0]) * (xy[2][1] - xy[0][1]) -
                     (xy[2][0] - xy[0][0]) * (xy[1][1] - xy[0][1]))
    if twice_area <= 0 or min(lengths) <= 0:
        raise ValueError("degenerate triangle")
    alts = [twice_area / side for side in lengths]
    return min(alts), twice_area / 2.0


def derive_controls(mesh: Path, source_crs: str | None = None, metric_crs: str | None = None) -> dict[str, Any]:
    nodes, elems, chains = parse_2dm(mesh)
    if source_crs and metric_crs and source_crs != metric_crs:
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise RuntimeError("pyproj is required to derive metric controls from a non-metric mesh") from exc
        transformer = Transformer.from_crs(source_crs, metric_crs, always_xy=True)
        nodes = {nid: (*transformer.transform(p[0], p[1]), p[2]) for nid, p in nodes.items()}
    min_cfl = float("inf")
    min_adv = float("inf")
    limiting_wave: dict[str, Any] = {}
    limiting_adv: dict[str, Any] = {}
    min_alt = float("inf")
    for eid, a, b, c in elems:
        xyz = [nodes[a], nodes[b], nodes[c]]
        altitude, _ = triangle_altitudes([(p[0], p[1]) for p in xyz])
        depth = max(abs(p[2]) for p in xyz)
        if depth <= 0:
            raise ValueError(f"nonpositive depth in element {eid}")
        wave_dt = 0.5 * altitude / math.sqrt(G * depth)
        adv_dt = 0.5 * altitude / 2.0
        if wave_dt < min_cfl:
            min_cfl, limiting_wave = wave_dt, {"element_id": eid, "altitude_m": altitude, "depth_m": depth}
        if adv_dt < min_adv:
            min_adv, limiting_adv = adv_dt, {"element_id": eid, "altitude_m": altitude}
        min_alt = min(min_alt, altitude)
    ext = math.floor(min_cfl * 10.0 + 1e-12) / 10.0
    if ext <= 0:
        raise ValueError("derived EXTSTEP is below 0.1 s")
    internal_limit = min(10.0 * ext, min_adv)
    isplit = max(1, int(math.floor(internal_limit / ext + 1e-12)))
    sponge = []
    for i, chain in enumerate(chains, 1):
        lengths = [math.dist(nodes[a][:2], nodes[b][:2]) for a, b in zip(chain[:-1], chain[1:])]
        if not lengths:
            raise ValueError(f"OBC chain {i} has fewer than two nodes")
        ordered = sorted(lengths)
        n = len(ordered)
        median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
        sponge.append({
            "obc_id": i, "node_count": len(chain), "median_edge_m": median,
            "radius_m": 3.0 * median, "coefficient": 0.0025,
        })
    return {
        "schema": "fvcom_initial_controls_v1", "generated_at": utcnow(),
        "mesh": str(mesh), "mesh_sha256": sha256(mesh),
        "source_crs": source_crs, "metric_crs": metric_crs or source_crs,
        "minimum_element_altitude_m": min_alt,
        "wave_cfl_safety_factor": 0.5, "advective_velocity_m_s": 2.0,
        "extstep_seconds": ext, "isplit": isplit,
        "internal_step_seconds": ext * isplit,
        "limiting_wave_element": limiting_wave, "limiting_advective_element": limiting_adv,
        "sponge": sponge,
    }


def stability_plan(controls: dict[str, Any]) -> dict[str, Any]:
    baseline_isplit = int(controls["isplit"])
    isplits = list(dict.fromkeys([baseline_isplit, max(1, baseline_isplit // 2), 1]))
    ext = [1.0, 0.75, 0.5, 0.25]
    radius = [1.0, 2.0, 4.0]
    coeff = [1.0, 0.5, 2.0]
    baseline = (1.0, baseline_isplit, 1.0, 1.0)
    ordered: list[tuple[float, int, float, float]] = [baseline]
    ordered += [(v, baseline_isplit, 1.0, 1.0) for v in ext[1:]]
    ordered += [(1.0, v, 1.0, 1.0) for v in isplits[1:]]
    ordered += [(1.0, baseline_isplit, v, 1.0) for v in radius[1:]]
    ordered += [(1.0, baseline_isplit, 1.0, v) for v in coeff[1:]]

    def distance(item: tuple[float, int, float, float]) -> tuple[float, float, float, float, float, int]:
        e, i, r, c = item
        components = (
            abs(math.log2(e)), abs(math.log2(i / baseline_isplit)) if i else 99.0,
            abs(math.log2(r)), abs(math.log2(c)),
        )
        return (
            sum(components), *components, ext.index(e),
        )

    rest = sorted(itertools.product(ext, isplits, radius, coeff), key=distance)
    for item in rest:
        if item not in ordered:
            ordered.append(item)
    attempts = []
    for idx, (em, isp, rm, cm) in enumerate(ordered):
        attempts.append({
            "index": idx, "attempt_id": f"attempt_{idx:04d}",
            "extstep_multiplier": em,
            "extstep_seconds": max(0.1, math.floor(float(controls["extstep_seconds"]) * em * 10 + 1e-12) / 10),
            "isplit": isp, "sponge_radius_multiplier": rm,
            "sponge_coefficient_multiplier": cm,
        })
    return {
        "schema": "fvcom_stability_plan_v1", "generated_at": utcnow(),
        "controls_sha256": hashlib.sha256(json.dumps(controls, sort_keys=True).encode()).hexdigest(),
        "attempt_count": len(attempts), "attempts": attempts,
    }


def apply_namelist_controls(text: str, extstep_seconds: float, isplit: int) -> str:
    replacements = {
        "EXTSTEP_SECONDS": f"{extstep_seconds:.1f}",
        "ISPLIT": str(isplit),
    }
    rendered = text
    for key, value in replacements.items():
        pattern = re.compile(rf"(?im)^(\s*{key}\s*=\s*)[^,\r\n]+(,?)")
        rendered, count = pattern.subn(rf"\g<1>{value}\g<2>", rendered)
        if count != 1:
            raise ValueError(f"expected exactly one {key} assignment in base namelist; found {count}")
    return rendered


def apply_namelist_directories(text: str, input_dir: str, output_dir: str = ".") -> str:
    """Bind a cloned attempt namelist to its immutable input package."""
    replacements = {
        "INPUT_DIR": f"'{input_dir}'",
        "OUTPUT_DIR": f"'{output_dir}'",
    }
    rendered = text
    for key, value in replacements.items():
        pattern = re.compile(rf"(?im)^(\s*{key}\s*=\s*)[^,\r\n]+(,?)")
        rendered, count = pattern.subn(rf"\g<1>{value}\g<2>", rendered)
        if count != 1:
            raise ValueError(f"expected exactly one {key} assignment in base namelist; found {count}")
    return rendered


def apply_sponge_multipliers(text: str, radius_multiplier: float,
                             coefficient_multiplier: float) -> str:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("Sponge Node Number"):
        raise ValueError("invalid FVCOM sponge DAT header")
    output = [lines[0]]
    row_count = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"invalid FVCOM sponge DAT row: {line}")
        node, radius, coefficient = fields
        output.append(
            f"{int(node)} {float(radius) * radius_multiplier:.6f} "
            f"{float(coefficient) * coefficient_multiplier:.6f} "
        )
        row_count += 1
    expected = int(lines[0].split("=", 1)[1].strip())
    if row_count != expected:
        raise ValueError(f"sponge row count {row_count} does not match header {expected}")
    return "\n".join(output) + "\n"


def attempt_namelist_sources(args: argparse.Namespace) -> dict[str | None, Path]:
    stage_specs = getattr(args, "stage_namelist", None) or []
    if stage_specs:
        result: dict[str | None, Path] = {}
        for spec in stage_specs:
            if "=" not in spec:
                raise ValueError("--stage-namelist must use STAGE=PATH")
            stage, raw_path = spec.split("=", 1)
            stage = stage.strip().lower()
            if stage not in {"smoke", "canary", "spinup"} or stage in result:
                raise ValueError(f"invalid or duplicate stability stage: {stage}")
            result[stage] = Path(raw_path)
        missing = {"smoke", "canary", "spinup"} - set(result)
        if missing:
            raise ValueError("staged attempts require smoke, canary, and spinup namelists")
        if getattr(args, "base_namelist", None):
            raise ValueError("use either --base-namelist or --stage-namelist, not both")
        return result
    base = getattr(args, "base_namelist", None)
    if not base:
        raise ValueError("one --base-namelist or three --stage-namelist values are required")
    return {None: Path(base)}


def attempt_sbatch(args: argparse.Namespace, aid: str, stage: str | None) -> str:
    suffix = f"_{stage}" if stage else ""
    modules = getattr(args, "module", None) or []
    for name in modules:
        if not re.fullmatch(r"[A-Za-z0-9._+/-]+", name):
            raise ValueError(f"unsafe module name: {name}")
    module_setup = ""
    if modules:
        module_setup = "module purge\n" + "\n".join(f"module load {name}" for name in modules) + "\n"
    return f"""#!/bin/bash
#SBATCH --job-name={args.grid_case}_{aid}{suffix}
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --nodes={int(args.nodes)}
#SBATCH --ntasks={int(args.ranks)}
#SBATCH --time={args.walltime}
#SBATCH --output=stdout.log
#SBATCH --error=stderr.log
set -euo pipefail
{module_setup}module -t list 2>&1
cd \"$SLURM_SUBMIT_DIR\"
srun --ntasks={int(args.ranks)} {args.remote_executable} --casename={args.case_name}
"""


def create_attempt(args: argparse.Namespace) -> dict[str, Any]:
    project = Path(args.project)
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    attempt = plan["attempts"][args.index]
    strategy_attempt_id = attempt["attempt_id"]
    revision = int(getattr(args, "revision", 0))
    if revision < 0:
        raise ValueError("--revision must be nonnegative")
    aid = strategy_attempt_id if revision == 0 else f"{strategy_attempt_id}_r{revision:03d}"
    input_dir = project / "input" / args.grid_case / "attempts" / aid
    run_dir = project / "run" / args.grid_case / "attempts" / aid
    if input_dir.exists() or run_dir.exists():
        raise FileExistsError(f"immutable attempt already exists: {aid}")
    base_input = Path(args.base_input)
    sponge_sources = sorted(base_input.rglob("*_spg.dat"))
    if len(sponge_sources) != 1:
        raise ValueError(f"base input must contain exactly one *_spg.dat; found {len(sponge_sources)}")
    nml_sources = attempt_namelist_sources(args)
    staged = None not in nml_sources
    attempt_input_reference = (
        f"../../../../../input/{args.grid_case}/attempts/{aid}" if staged
        else f"../../../../input/{args.grid_case}/attempts/{aid}"
    )
    rendered_spg = apply_sponge_multipliers(
        sponge_sources[0].read_text(encoding="ascii"),
        attempt["sponge_radius_multiplier"], attempt["sponge_coefficient_multiplier"],
    )
    shutil.copytree(base_input, input_dir)
    run_dir.mkdir(parents=True)
    attempt_spg = input_dir / sponge_sources[0].relative_to(base_input)
    attempt_spg.write_text(rendered_spg, encoding="ascii", newline="\n")
    write_json(run_dir / "control_overrides.json", attempt)
    snippet = (
        "! attempt-specific controls\n"
        f"EXTSTEP_SECONDS = {attempt['extstep_seconds']:.1f}\n"
        f"ISPLIT = {attempt['isplit']}\n"
        f"! SPONGE_RADIUS_MULTIPLIER = {attempt['sponge_radius_multiplier']}\n"
        f"! SPONGE_COEFFICIENT_MULTIPLIER = {attempt['sponge_coefficient_multiplier']}\n"
    )
    (run_dir / "control_overrides.nml.inc").write_text(snippet, encoding="utf-8")
    stage_manifests: dict[str, Any] = {}
    single_nml: Path | None = None
    for stage, nml_src in nml_sources.items():
        rendered_nml = apply_namelist_controls(
            nml_src.read_text(encoding="utf-8"), attempt["extstep_seconds"], attempt["isplit"]
        )
        rendered_nml = apply_namelist_directories(rendered_nml, attempt_input_reference)
        stage_dir = run_dir / stage if stage else run_dir
        stage_dir.mkdir(parents=True, exist_ok=True)
        attempt_nml = stage_dir / f"{args.case_name}_run.nml"
        attempt_nml.write_text(rendered_nml, encoding="utf-8", newline="\n")
        (stage_dir / "job.sbatch").write_text(
            attempt_sbatch(args, aid, stage), encoding="utf-8", newline="\n"
        )
        if stage:
            stage_manifests[stage] = {
                "run_dir": str(stage_dir.relative_to(project)),
                "base_namelist": str(nml_src),
                "base_namelist_sha256": sha256(nml_src),
                "attempt_namelist_sha256": sha256(attempt_nml),
                "status": "prepared",
            }
        else:
            single_nml = attempt_nml
    manifest = {
        "schema": "fvcom_run_attempt_v1", "created_at": utcnow(), "grid_case": args.grid_case,
        "attempt": attempt, "strategy_attempt_id": strategy_attempt_id, "revision": revision,
        "input_dir": str(input_dir.relative_to(project)),
        "run_dir": str(run_dir.relative_to(project)),
        "namelist_input_dir": attempt_input_reference, "namelist_output_dir": ".",
        "modules": list(getattr(args, "module", None) or []),
        "attempt_sponge_sha256": sha256(attempt_spg),
        "plan_sha256": sha256(plan_path), "status": "prepared",
    }
    if staged:
        manifest["stages"] = {stage: stage_manifests[stage] for stage in ("smoke", "canary", "spinup")}
        manifest["stage_order"] = ["smoke", "canary", "spinup"]
    else:
        nml_src = nml_sources[None]
        manifest.update({
            "base_namelist": str(nml_src),
            "base_namelist_sha256": sha256(nml_src),
            "attempt_namelist_sha256": sha256(single_nml),
        })
    write_json(run_dir / "attempt_manifest.json", manifest)
    render_report(project)
    return manifest


BLOWUP_RE = re.compile(
    r"\b(blow\s*up|blowup|floating invalid|segmentation fault|fatal|nan|"
    r"stop running|not good for model)\b",
    re.I,
)
TIME_RE = re.compile(r"(?:time|date)[^\n]{0,60}?(\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)", re.I)


def readable_netcdf(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing_or_empty"
    try:
        from netCDF4 import Dataset  # type: ignore
        with Dataset(path) as ds:
            _ = list(ds.dimensions)
        return True, "netcdf4"
    except ImportError:
        return False, "netcdf4_unavailable"
    except Exception as exc:  # pragma: no cover - library-specific
        return False, f"unreadable:{type(exc).__name__}:{exc}"


def netcdf_contains_final_time(path: Path, expected_final: str) -> tuple[bool, str]:
    """Check common FVCOM time encodings for the requested terminal timestamp."""
    expected = dt.datetime.fromisoformat(expected_final.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    try:
        import netCDF4 as nc4  # type: ignore
        with nc4.Dataset(path) as ds:
            candidates: list[dt.datetime] = []
            if "Times" in ds.variables and ds["Times"].shape[0]:
                encoded = str(nc4.chartostring(ds["Times"][-1]))
                for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                    try:
                        candidates.append(dt.datetime.strptime(encoded.strip(), fmt).replace(tzinfo=dt.timezone.utc))
                        break
                    except ValueError:
                        pass
            if "time" in ds.variables and ds["time"].size and getattr(ds["time"], "units", None):
                value = nc4.num2date(ds["time"][-1], ds["time"].units, getattr(ds["time"], "calendar", "standard"))
                candidates.append(dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=dt.timezone.utc))
            if "Itime" in ds.variables and ds["Itime"].size:
                days = float(ds["Itime"][-1])
                millis = float(ds["Itime2"][-1]) if "Itime2" in ds.variables else 0.0
                candidates.append(dt.datetime(1858, 11, 17, tzinfo=dt.timezone.utc) + dt.timedelta(days=days, milliseconds=millis))
    except Exception as exc:  # pragma: no cover - library-specific
        return False, f"time_check_error:{type(exc).__name__}:{exc}"
    if not candidates:
        return False, "no_supported_time_coordinate"
    latest = max(candidates)
    return abs((latest - expected).total_seconds()) <= 1.0, latest.isoformat().replace("+00:00", "Z")


def audit_run(stdout: Path, stderr: Path | None, exit_code: int, expected_final: str,
              netcdf: list[Path], restart: list[Path]) -> dict[str, Any]:
    out_text = stdout.read_text(encoding="utf-8", errors="replace") if stdout.exists() else ""
    err_text = stderr.read_text(encoding="utf-8", errors="replace") if stderr and stderr.exists() else ""
    combined = out_text + "\n" + err_text
    tada = "TADA!" in combined
    blowup = bool(BLOWUP_RE.search(combined))
    expected = expected_final.replace("Z", "").replace("T", " ")[:16]
    normalized = combined.replace("/", "-").replace("T", " ")
    requested_time_echoed = expected in normalized
    iend_matches = [int(value) for value in re.findall(r"(?im)^\s*!?\s*IEND\s*=\s*(\d+)", combined)]
    # FVCOM variants emit either a legacy D<day>T<time> token or an ISO
    # YYYY-MM-DDT<time> timestamp.  Require an actual progress row reaching
    # IEND, but accept both source-supported renderings.
    progress_matches = [
        int(value) for value in re.findall(
            r"(?im)^\s*!\s*(\d+)\s+(?:D\d+|\d{4}[-/]\d{2}[-/]\d{2})T\d{2}:\d{2}:",
            combined,
        )
    ]
    loop_complete = bool(iend_matches and progress_matches and max(progress_matches) >= max(iend_matches))
    final_present = requested_time_echoed and loop_complete and tada
    products = []
    readable = True
    product_final_checks: list[bool] = []
    for role, paths in (("output", netcdf), ("restart", restart)):
        for p in paths:
            ok, method = readable_netcdf(p)
            readable &= ok
            final_ok, final_check = netcdf_contains_final_time(p, expected_final) if ok else (False, "not_checked")
            product_final_checks.append(final_ok)
            products.append({"role": role, "path": str(p), "readable": ok, "check": method,
                             "final_timestamp_present": final_ok, "final_timestamp_check": final_check,
                             "sha256": sha256(p) if ok and p.exists() else None})
    final_present = final_present or any(product_final_checks)
    reasons = []
    if exit_code != 0:
        reasons.append(f"nonzero_exit:{exit_code}")
    if blowup:
        reasons.append("blowup_or_fatal_text")
    if not tada:
        reasons.append("missing_TADA")
    if not final_present:
        reasons.append("missing_expected_final_time")
    if not readable:
        reasons.append("missing_or_corrupt_required_product")
    stable = not reasons
    return {
        "schema": "fvcom_run_audit_v1", "audited_at": utcnow(), "stable": stable,
        "status": "stable" if stable else "failed", "exit_code": exit_code,
        "tada": tada, "blowup_or_fatal": blowup, "expected_final": expected_final,
        "final_timestamp_present": final_present, "products": products, "failure_reasons": reasons,
        "stdout_sha256": sha256(stdout) if stdout.exists() else None,
        "stderr_sha256": sha256(stderr) if stderr and stderr.exists() else None,
    }


def render_report(project: Path) -> Path:
    manifest = {}
    status = {}
    if (project / "project_manifest.json").exists():
        manifest = json.loads((project / "project_manifest.json").read_text(encoding="utf-8"))
    if (project / "project_status.json").exists():
        status = json.loads((project / "project_status.json").read_text(encoding="utf-8"))
    rows = []
    # Staged stability attempts store status under smoke/canary/spinup, while
    # legacy single-stage attempts store it directly under the attempt root.
    for p in sorted(project.glob("run/*/attempts/attempt_*/**/status.json")):
        try:
            a = json.loads(p.read_text(encoding="utf-8"))
            rows.append((str(p.relative_to(project)), a.get("status"), ", ".join(a.get("failure_reasons", []))))
        except Exception as exc:
            rows.append((str(p.relative_to(project)), "invalid", str(exc)))
    table = "".join(
        f"<tr><td>{html.escape(p)}</td><td>{html.escape(str(s))}</td><td>{html.escape(r)}</td></tr>"
        for p, s, r in rows
    ) or "<tr><td colspan='3'>No completed attempts.</td></tr>"
    benchmark_rows = []
    for p in sorted(project.glob("run/*/benchmark/**/benchmark_submission.json")):
        try:
            submission = json.loads(p.read_text(encoding="utf-8"))
            jobs = submission.get("jobs", [])
            if jobs:
                job_ids = ", ".join(str(job.get("job_id")) for job in jobs if job.get("job_id"))
            else:
                first_job = submission.get("first_job", {})
                job_ids = str(first_job.get("job_id", ""))
            details = {
                "attempt": submission.get("attempt"),
                "execution": submission.get("execution"),
                "requested_walltime": submission.get("requested_walltime"),
                "blocking_reasons": submission.get("blocking_reasons", []),
            }
            benchmark_rows.append((str(p.relative_to(project)), submission.get("status"), job_ids,
                                   json.dumps(details, default=str)))
        except Exception as exc:
            benchmark_rows.append((str(p.relative_to(project)), "invalid", "", str(exc)))
    for p in sorted(project.glob("run/*/benchmark/**/benchmark_record.json")):
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
            details = {
                "slurm_state": record.get("slurm_state"),
                "eligible": record.get("eligible"),
                "tada": record.get("tada"),
                "final_timestamp_present": record.get("final_timestamp_present"),
                "wall_seconds": record.get("wall_seconds"),
            }
            benchmark_rows.append((str(p.relative_to(project)),
                                   "eligible" if record.get("eligible") else "ineligible",
                                   str(record.get("job_id", "")), json.dumps(details, default=str)))
        except Exception as exc:
            benchmark_rows.append((str(p.relative_to(project)), "invalid", "", str(exc)))
    benchmark_table = "".join(
        f"<tr><td>{html.escape(p)}</td><td>{html.escape(str(s))}</td><td>{html.escape(j)}</td><td>{html.escape(d)}</td></tr>"
        for p, s, j, d in benchmark_rows
    ) or "<tr><td colspan='4'>No benchmark attempts.</td></tr>"
    worker_rows = []
    for p in sorted(set(project.glob("gridding/**/worker_manifest.json")) |
                    set(project.glob("forcing/**/worker_manifest.json")) |
                    set(project.glob("run/build/**/worker_manifest.json"))):
        try:
            w = json.loads(p.read_text(encoding="utf-8"))
            reasons = w.get("blocking_reasons", [])
            worker_rows.append((str(p.relative_to(project)), w.get("worker"), w.get("status"), json.dumps(reasons, default=str)))
        except Exception as exc:
            worker_rows.append((str(p.relative_to(project)), "unknown", "invalid", str(exc)))
    worker_table = "".join(
        f"<tr><td>{html.escape(p)}</td><td>{html.escape(str(w))}</td><td>{html.escape(str(s))}</td><td>{html.escape(r)}</td></tr>"
        for p, w, s, r in worker_rows
    ) or "<tr><td colspan='4'>Workers have not returned manifests.</td></tr>"
    observation_note = "Observation preflight not yet available."
    obs_path = project / "analysis" / "observations" / "observation_manifest.json"
    if obs_path.exists():
        try:
            obs = json.loads(obs_path.read_text(encoding="utf-8"))
            observation_note = html.escape(json.dumps({"status": obs.get("status"), "counts": obs.get("counts"), "warnings": obs.get("warnings")}, indent=2))
        except Exception as exc:
            observation_note = html.escape(f"Invalid observation manifest: {exc}")
    blockers = "".join(f"<li>{html.escape(str(x))}</li>" for x in status.get("blocking_reasons", [])) or "<li>None</li>"
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>FVCOM simulation report</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1100px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:.45rem;text-align:left}}code{{background:#eee;padding:.1rem .3rem}}.state{{font-size:1.4rem}}</style></head><body>
<h1>FVCOM Simulation: {html.escape(str(manifest.get('case_id', project.name)))}</h1>
<p class='state'>State: <code>{html.escape(str(status.get('state', 'uninitialized')))}</code></p>
<p>Updated: {html.escape(str(status.get('updated_at', utcnow())))}</p><h2>Blocking reasons</h2><ul>{blockers}</ul>
<h2>Preparation workers</h2><table><thead><tr><th>Manifest</th><th>Worker</th><th>Status</th><th>Blocking evidence</th></tr></thead><tbody>{worker_table}</tbody></table>
<h2>Observation preflight</h2><pre>{observation_note}</pre>
<h2>Stability attempts</h2><table><thead><tr><th>Artifact</th><th>Status</th><th>Reason</th></tr></thead><tbody>{table}</tbody></table>
<h2>Benchmark attempts</h2><table><thead><tr><th>Artifact</th><th>Status</th><th>Slurm jobs</th><th>Evidence</th></tr></thead><tbody>{benchmark_table}</tbody></table>
<h2>Scientific status</h2><p>Workflow: {html.escape(str(status.get('workflow_status')))}; assessment: {html.escape(str(status.get('scientific_assessment')))}</p>
</body></html>"""
    out = project / "report.html"
    out.write_text(body, encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Immutable FVCOM run-control helper")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("init")
    q.add_argument("--request", required=True); q.add_argument("--project", required=True)
    q = sub.add_parser("state")
    q.add_argument("--project", required=True); q.add_argument("--state", required=True, choices=sorted(ALLOWED_STATES))
    q.add_argument("--evidence", required=True); q.add_argument("--grid-case"); q.add_argument("--attempt")
    q.add_argument("--blocker", action="append", default=[])
    q = sub.add_parser("controls")
    q.add_argument("--mesh", required=True); q.add_argument("--output", required=True)
    q.add_argument("--source-crs"); q.add_argument("--metric-crs")
    q = sub.add_parser("plan")
    q.add_argument("--controls", required=True); q.add_argument("--output", required=True)
    q = sub.add_parser("attempt")
    q.add_argument("--project", required=True); q.add_argument("--grid-case", required=True, choices=["accepted_t6v6", "fresh_reproduction"])
    q.add_argument("--plan", required=True); q.add_argument("--index", type=int, required=True)
    q.add_argument("--base-input", required=True); q.add_argument("--base-namelist")
    q.add_argument("--stage-namelist", action="append", default=[])
    q.add_argument("--revision", type=int, default=0)
    q.add_argument("--module", action="append", default=[])
    q.add_argument("--remote-executable", required=True); q.add_argument("--case-name", default="galveston")
    q.add_argument("--account", default="hindcastra"); q.add_argument("--partition", default="standard")
    q.add_argument("--nodes", type=int, default=1); q.add_argument("--ranks", type=int, default=104)
    q.add_argument("--walltime", default="01:00:00")
    q = sub.add_parser("audit")
    q.add_argument("--stdout", required=True); q.add_argument("--stderr")
    q.add_argument("--exit-code", required=True, type=int); q.add_argument("--expected-final", required=True)
    q.add_argument("--netcdf", action="append", default=[]); q.add_argument("--restart", action="append", default=[])
    q.add_argument("--output", required=True)
    q = sub.add_parser("record")
    q.add_argument("--project", required=True); q.add_argument("--kind", required=True)
    q.add_argument("--command", dest="command_text", required=True); q.add_argument("--status", default="success")
    q.add_argument("--artifact", action="append", default=[])
    q = sub.add_parser("join")
    q.add_argument("--manifest", action="append", required=True); q.add_argument("--output", required=True)
    q = sub.add_parser("report"); q.add_argument("--project", required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "init":
        result = init_project(Path(args.request), Path(args.project))
    elif args.command == "state":
        result = set_state(Path(args.project), args.state, args.evidence, args.grid_case, args.attempt, args.blocker)
    elif args.command == "controls":
        if bool(args.source_crs) != bool(args.metric_crs):
            raise ValueError("--source-crs and --metric-crs must be supplied together")
        result = derive_controls(Path(args.mesh), args.source_crs, args.metric_crs); write_json(Path(args.output), result)
    elif args.command == "plan":
        result = stability_plan(load_data(Path(args.controls))); write_json(Path(args.output), result)
    elif args.command == "attempt":
        result = create_attempt(args)
    elif args.command == "audit":
        result = audit_run(Path(args.stdout), Path(args.stderr) if args.stderr else None,
                           args.exit_code, args.expected_final, [Path(x) for x in args.netcdf],
                           [Path(x) for x in args.restart]); write_json(Path(args.output), result)
    elif args.command == "record":
        result = record_command(Path(args.project), args.kind, args.command_text, args.status, args.artifact)
    elif args.command == "join":
        result = evaluate_preparation_join([Path(path) for path in args.manifest]); write_json(Path(args.output), result)
    else:
        result = {"report": str(render_report(Path(args.project)))}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
