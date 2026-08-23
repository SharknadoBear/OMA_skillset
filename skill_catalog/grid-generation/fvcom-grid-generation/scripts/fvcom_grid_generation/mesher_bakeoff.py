"""Generator-neutral, immutable FVCOM mesher bakeoff orchestration.

The module deliberately treats mesh generation and common conditioning as
external adapter commands.  It owns only the experiment contract: immutable
inputs, isolated candidate directories, stage records, and fair comparison.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any


INPUT_BUNDLE_SCHEMA = "fvcom_mesher_input_bundle_v1"
DECLARATION_SCHEMA = "fvcom_mesher_bakeoff_declaration_v1"
RUN_MANIFEST_SCHEMA = "fvcom_mesher_bakeoff_run_v1"
STAGE_RESULT_SCHEMA = "fvcom_mesher_bakeoff_stage_result_v1"
COMPARISON_SCHEMA = "fvcom_mesher_bakeoff_comparison_v1"

RAW = "RAW"
COMMON_CONDITIONED = "COMMON_CONDITIONED"
STAGES = (RAW, COMMON_CONDITIONED)

SUPPORTED_ALGORITHMS: dict[str, dict[str, int | str]] = {
    "clean_room": {"production_reference": "production_reference"},
    "gmsh": {
        "meshadapt": 1,
        "delaunay": 5,
        "frontal_delaunay": 6,
    },
}

FAILURE_CODES = {
    "ARTIFACT_HASH_MISMATCH",
    "COMMAND_FAILED",
    "COMMAND_NOT_FOUND",
    "COMMAND_TIMEOUT",
    "CONTRACT_VIOLATION",
    "EXPECTED_ARTIFACT_MISSING",
    "HARD_GATE_FAILED",
    "INPUT_HASH_MISMATCH",
    "OUTPUT_EXISTS",
    "QA_REPORT_INVALID",
    "UPSTREAM_STAGE_FAILED",
}


class BakeoffContractError(RuntimeError):
    """A deterministic contract failure with a machine-readable category."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return (json.dumps(value, **kwargs) + ("\n" if pretty else "")).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "size_bytes": resolved.stat().st_size,
            "sha256": _file_sha256(resolved),
        }
    if not resolved.is_dir():
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", f"unsupported artifact type: {resolved}"
        )
    entries: list[dict[str, Any]] = []
    total_size = 0
    for item in sorted(
        (candidate for candidate in resolved.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(resolved).as_posix(),
    ):
        size = item.stat().st_size
        total_size += size
        entries.append(
            {
                "relative_path": item.relative_to(resolved).as_posix(),
                "size_bytes": size,
                "sha256": _file_sha256(item),
            }
        )
    return {
        "path": str(resolved),
        "kind": "directory",
        "file_count": len(entries),
        "size_bytes": total_size,
        "sha256": _payload_sha256(entries),
    }


def _artifact_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "kind": record["kind"],
        "size_bytes": int(record["size_bytes"]),
        "sha256": str(record["sha256"]),
    }
    if record["kind"] == "directory":
        identity["file_count"] = int(record["file_count"])
    return identity


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", f"expected a JSON object in {path}"
        )
    return value


def _write_json_fresh(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(_json_bytes(value, pretty=True))
    except FileExistsError as exc:
        raise BakeoffContractError(
            "OUTPUT_EXISTS", f"refusing to overwrite {destination.resolve()}"
        ) from exc


def lock_input_bundle(
    *,
    case_id: str,
    boundary: str | Path,
    bathymetry: str | Path,
    canonical_size_field: str | Path,
    projection: Mapping[str, Any],
    node_budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash the one input bundle that every candidate must consume."""
    if not case_id or not isinstance(case_id, str):
        raise BakeoffContractError("CONTRACT_VIOLATION", "case_id is required")
    if not isinstance(projection, Mapping) or not projection:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "projection must be a non-empty JSON object"
        )
    if not isinstance(node_budget, Mapping):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "node_budget must be a JSON object"
        )
    preflight = node_budget.get(
        "preflight_node_threshold", node_budget.get("preflight_nodes")
    )
    hard_cap = node_budget.get("hard_node_cap", node_budget.get("max_nodes"))
    if (
        isinstance(preflight, bool)
        or isinstance(hard_cap, bool)
        or not isinstance(preflight, int)
        or not isinstance(hard_cap, int)
        or preflight <= 0
        or hard_cap <= 0
        or preflight >= hard_cap
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            "node_budget requires positive preflight and hard caps with "
            "preflight < hard cap",
        )
    artifacts = {
        "bathymetry": _artifact_record(bathymetry),
        "boundary": _artifact_record(boundary),
        "canonical_size_field": _artifact_record(canonical_size_field),
    }
    identity = {
        "schema_version": INPUT_BUNDLE_SCHEMA,
        "case_id": case_id,
        "artifacts": {
            role: _artifact_identity(record)
            for role, record in sorted(artifacts.items())
        },
        "projection": dict(projection),
        "node_budget": dict(node_budget),
    }
    return {
        **identity,
        "artifacts": artifacts,
        "input_bundle_sha256": _payload_sha256(identity),
    }


def write_input_bundle(path: str | Path, bundle: Mapping[str, Any]) -> None:
    validate_input_bundle(bundle)
    _write_json_fresh(path, dict(bundle))


def validate_input_bundle(bundle: Mapping[str, Any]) -> None:
    if bundle.get("schema_version") != INPUT_BUNDLE_SCHEMA:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "unsupported input-bundle schema"
        )
    artifacts = bundle.get("artifacts")
    required = {"boundary", "bathymetry", "canonical_size_field"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            "input bundle must contain boundary, bathymetry, and "
            "canonical_size_field artifacts",
        )
    projection = bundle.get("projection")
    node_budget = bundle.get("node_budget")
    if not isinstance(projection, Mapping) or not projection:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "input bundle projection is invalid"
        )
    if not isinstance(node_budget, Mapping):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "input bundle node budget is invalid"
        )
    preflight = node_budget.get(
        "preflight_node_threshold", node_budget.get("preflight_nodes")
    )
    hard_cap = node_budget.get("hard_node_cap", node_budget.get("max_nodes"))
    if (
        isinstance(preflight, bool)
        or isinstance(hard_cap, bool)
        or not isinstance(preflight, int)
        or not isinstance(hard_cap, int)
        or preflight <= 0
        or hard_cap <= 0
        or preflight >= hard_cap
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "input bundle node budget is invalid"
        )
    identity = {
        "schema_version": INPUT_BUNDLE_SCHEMA,
        "case_id": bundle.get("case_id"),
        "artifacts": {
            role: _artifact_identity(artifacts[role]) for role in sorted(required)
        },
        "projection": bundle.get("projection"),
        "node_budget": bundle.get("node_budget"),
    }
    actual = _payload_sha256(identity)
    if actual != bundle.get("input_bundle_sha256"):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            "input_bundle_sha256 does not match the declared bundle",
        )


def verify_input_bundle_files(bundle: Mapping[str, Any]) -> None:
    """Reject source mutation before any adapter command starts."""
    validate_input_bundle(bundle)
    mismatches: list[dict[str, Any]] = []
    for role, expected in sorted(bundle["artifacts"].items()):
        try:
            current = _artifact_record(expected["path"])
        except FileNotFoundError:
            mismatches.append(
                {
                    "role": role,
                    "path": expected["path"],
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": None,
                }
            )
            continue
        if _artifact_identity(current) != _artifact_identity(expected):
            mismatches.append(
                {
                    "role": role,
                    "path": expected["path"],
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": current["sha256"],
                }
            )
    if mismatches:
        raise BakeoffContractError(
            "INPUT_HASH_MISMATCH",
            json.dumps(mismatches, sort_keys=True, separators=(",", ":")),
        )


def default_candidate_matrix() -> list[dict[str, Any]]:
    """Return the required clean-room/Gmsh generator portfolio."""
    return [
        {
            "candidate_id": "clean_room_reference",
            "mesher_adapter": "clean_room",
            "algorithm": {
                "id": "production_reference",
                "native_code": "production_reference",
            },
        },
        {
            "candidate_id": "gmsh_meshadapt_1",
            "mesher_adapter": "gmsh",
            "algorithm": {"id": "meshadapt", "native_code": 1},
        },
        {
            "candidate_id": "gmsh_delaunay_5",
            "mesher_adapter": "gmsh",
            "algorithm": {"id": "delaunay", "native_code": 5},
        },
        {
            "candidate_id": "gmsh_frontal_delaunay_6",
            "mesher_adapter": "gmsh",
            "algorithm": {"id": "frontal_delaunay", "native_code": 6},
        },
    ]


def _candidate_capability(candidate: Mapping[str, Any]) -> dict[str, Any]:
    capability = candidate.get("capability", {"supported": True})
    if not isinstance(capability, Mapping) or not isinstance(
        capability.get("supported"), bool
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            f"{candidate.get('candidate_id')} capability must declare supported",
        )
    supported = bool(capability["supported"])
    reason = capability.get("reason")
    if not supported and (not isinstance(reason, str) or not reason.strip()):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            f"{candidate.get('candidate_id')} unsupported capability needs a reason",
        )
    if supported and reason is not None and not isinstance(reason, str):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            f"{candidate.get('candidate_id')} capability reason must be text",
        )
    return {"supported": supported, "reason": reason}


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    candidate_id = candidate.get("candidate_id")
    if (
        not isinstance(candidate_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate_id)
        or candidate_id in {".", ".."}
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", f"unsafe candidate_id: {candidate_id!r}"
        )
    adapter = candidate.get("mesher_adapter")
    algorithm = candidate.get("algorithm")
    if adapter not in SUPPORTED_ALGORITHMS or not isinstance(algorithm, Mapping):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            f"unsupported adapter/algorithm for {candidate_id}",
        )
    algorithm_id = algorithm.get("id")
    expected_code = SUPPORTED_ALGORITHMS[adapter].get(str(algorithm_id))
    if expected_code is None or algorithm.get("native_code") != expected_code:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            f"{candidate_id} must declare a supported {adapter} algorithm and "
            "its exact native code",
        )
    _candidate_capability(candidate)


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", f"{label} must be a non-empty relative path"
        )
    path = Path(value)
    if (
        path.is_absolute()
        or path.drive
        or ".." in path.parts
        or path.as_posix() in {".", ""}
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", f"unsafe {label}: {value!r}"
        )
    return path.as_posix()


def _validate_stage_template(stage: Mapping[str, Any], label: str) -> None:
    command = stage.get("command")
    if (
        not isinstance(command, Sequence)
        or isinstance(command, (str, bytes))
        or not command
        or any(not isinstance(token, str) or not token for token in command)
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", f"{label}.command must be a token array"
        )
    artifacts = stage.get("artifacts")
    if (
        not isinstance(artifacts, Sequence)
        or isinstance(artifacts, (str, bytes))
        or not artifacts
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", f"{label}.artifacts must be a non-empty array"
        )
    normalized = [_safe_relative_path(item, f"{label}.artifact") for item in artifacts]
    if len(normalized) != len(set(normalized)):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", f"{label} has duplicate artifacts"
        )
    mesh_artifact = _safe_relative_path(
        stage.get("mesh_artifact"), f"{label}.mesh_artifact"
    )
    qa_report = _safe_relative_path(stage.get("qa_report"), f"{label}.qa_report")
    if mesh_artifact not in normalized or qa_report not in normalized:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            f"{label} must list mesh_artifact and qa_report in artifacts",
        )
    timeout = stage.get("timeout_seconds")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", f"{label}.timeout_seconds must be positive"
        )


def _validate_qa_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schema_version") == "fvcom_grid_quality_policy_v1":
        adapter = policy.get("mesher_bakeoff_adapter")
        if not isinstance(adapter, Mapping):
            raise BakeoffContractError(
                "CONTRACT_VIOLATION",
                "benchmark-first QA policy requires mesher_bakeoff_adapter",
            )
        _validate_qa_policy(
            {
                "policy_id": policy.get("policy_id"),
                "accepted_path": adapter.get("accepted_path"),
                "failure_taxonomy_path": adapter.get("failure_taxonomy_path"),
                "metric_paths": adapter.get("metric_paths"),
                "metric_directions": adapter.get("metric_directions", {}),
                "hard_gates": adapter.get("hard_gate_ids"),
            }
        )
        return
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"]:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "qa_policy.policy_id is required"
        )
    for key in ("accepted_path", "failure_taxonomy_path"):
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise BakeoffContractError(
                "CONTRACT_VIOLATION", f"qa_policy.{key} is required"
            )
    metric_paths = policy.get("metric_paths")
    if not isinstance(metric_paths, Mapping) or not metric_paths:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "qa_policy.metric_paths is required"
        )
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(path, str)
        or not path
        for name, path in metric_paths.items()
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "qa_policy.metric_paths is invalid"
        )
    directions = policy.get("metric_directions", {})
    if not isinstance(directions, Mapping) or any(
        key not in metric_paths
        or value not in {"minimize", "maximize", "report_only"}
        for key, value in directions.items()
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "qa_policy.metric_directions is invalid"
        )
    hard_gates = policy.get("hard_gates")
    if (
        not isinstance(hard_gates, Sequence)
        or isinstance(hard_gates, (str, bytes))
        or not hard_gates
        or any(not isinstance(item, str) or not item for item in hard_gates)
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "qa_policy.hard_gates is required"
        )


def _expand_command(command: Sequence[str], values: Mapping[str, str]) -> list[str]:
    pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise BakeoffContractError(
                "CONTRACT_VIOLATION", f"unknown command placeholder {{{key}}}"
            )
        return values[key]

    return [pattern.sub(replace, token) for token in command]


def _expanded_stage(
    template: Mapping[str, Any],
    *,
    stage_name: str,
    stage_dir: Path,
    values: Mapping[str, str],
) -> dict[str, Any]:
    artifacts = [
        _safe_relative_path(item, f"{stage_name}.artifact")
        for item in template["artifacts"]
    ]
    mesh_artifact = _safe_relative_path(
        template["mesh_artifact"], f"{stage_name}.mesh_artifact"
    )
    stage_values = dict(values)
    stage_values.update(
        {
            "stage": stage_name,
            "stage_dir": str(stage_dir),
            "stage_mesh": str(stage_dir / mesh_artifact),
        }
    )
    return {
        "status": "planned",
        "stage": stage_name,
        "stage_dir": str(stage_dir),
        "command": _expand_command(template["command"], stage_values),
        "artifacts": artifacts,
        "mesh_artifact": mesh_artifact,
        "qa_report": _safe_relative_path(
            template["qa_report"], f"{stage_name}.qa_report"
        ),
        "timeout_seconds": template.get("timeout_seconds"),
    }


def _unsupported_stage(
    *,
    stage_name: str,
    stage_dir: Path,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "stage": stage_name,
        "stage_dir": str(stage_dir),
        "command": [],
        "artifacts": [],
        "mesh_artifact": None,
        "qa_report": None,
        "timeout_seconds": None,
        "capability_reason": reason,
    }


def build_run_manifest(
    declaration: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Build a deterministic, pure plan without touching the output directory."""
    if declaration.get("schema_version") != DECLARATION_SCHEMA:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "unsupported bakeoff declaration schema"
        )
    bundle = declaration.get("input_bundle")
    policy = declaration.get("qa_policy")
    candidates = declaration.get("candidates")
    conditioner = declaration.get("common_conditioner")
    if not isinstance(bundle, Mapping):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "declaration.input_bundle is required"
        )
    validate_input_bundle(bundle)
    if not isinstance(policy, Mapping):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "declaration.qa_policy is required"
        )
    _validate_qa_policy(policy)
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not candidates
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "declaration.candidates is required"
        )
    if not isinstance(conditioner, Mapping):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "declaration.common_conditioner is required"
        )
    _validate_stage_template(conditioner, "common_conditioner")

    normalized_candidates: list[Mapping[str, Any]] = []
    identifiers: set[str] = set()
    identifiers_folded: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise BakeoffContractError(
                "CONTRACT_VIOLATION", "candidate entries must be objects"
            )
        _validate_candidate(candidate)
        capability = _candidate_capability(candidate)
        if capability["supported"]:
            _validate_stage_template(candidate.get("raw", {}), "candidate.raw")
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in identifiers or candidate_id.casefold() in identifiers_folded:
            raise BakeoffContractError(
                "CONTRACT_VIOLATION", f"duplicate candidate_id: {candidate_id}"
            )
        identifiers.add(candidate_id)
        identifiers_folded.add(candidate_id.casefold())
        normalized_candidates.append(candidate)

    run_dir = Path(output_dir).expanduser().resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    policy_value = dict(policy)
    policy_sha = _payload_sha256(policy_value)
    conditioner_sha = _payload_sha256(dict(conditioner))
    artifacts = bundle["artifacts"]
    node_budget = bundle["node_budget"]
    common_values = {
        "run_dir": str(run_dir),
        "run_manifest": str(run_manifest_path),
        "case_id": str(bundle["case_id"]),
        "boundary": str(artifacts["boundary"]["path"]),
        "bathymetry": str(artifacts["bathymetry"]["path"]),
        "size_field": str(artifacts["canonical_size_field"]["path"]),
        "input_bundle_sha256": str(bundle["input_bundle_sha256"]),
        "qa_policy_sha256": policy_sha,
        "preflight_node_threshold": str(
            node_budget.get(
                "preflight_node_threshold", node_budget.get("preflight_nodes")
            )
        ),
        "hard_node_cap": str(
            node_budget.get("hard_node_cap", node_budget.get("max_nodes"))
        ),
    }
    planned_candidates: list[dict[str, Any]] = []
    for candidate in sorted(
        normalized_candidates, key=lambda item: str(item["candidate_id"])
    ):
        candidate_id = str(candidate["candidate_id"])
        candidate_dir = run_dir / "candidates" / candidate_id
        raw_dir = candidate_dir / "raw"
        conditioned_dir = candidate_dir / "common_conditioned"
        algorithm = dict(candidate["algorithm"])
        capability = _candidate_capability(candidate)
        values = {
            **common_values,
            "candidate_id": candidate_id,
            "candidate_dir": str(candidate_dir),
            "mesher_adapter": str(candidate["mesher_adapter"]),
            "algorithm_id": str(algorithm["id"]),
            "algorithm_code": str(algorithm["native_code"]),
        }
        if capability["supported"]:
            raw = _expanded_stage(
                candidate["raw"],
                stage_name=RAW,
                stage_dir=raw_dir,
                values=values,
            )
            conditioned_values = {
                **values,
                "raw_dir": str(raw_dir),
                "raw_mesh": str(raw_dir / raw["mesh_artifact"]),
            }
            common_conditioned = _expanded_stage(
                conditioner,
                stage_name=COMMON_CONDITIONED,
                stage_dir=conditioned_dir,
                values=conditioned_values,
            )
        else:
            reason = str(capability["reason"])
            raw = _unsupported_stage(
                stage_name=RAW, stage_dir=raw_dir, reason=reason
            )
            common_conditioned = _unsupported_stage(
                stage_name=COMMON_CONDITIONED,
                stage_dir=conditioned_dir,
                reason=reason,
            )
        common_conditioned["conditioner_template_sha256"] = conditioner_sha
        planned_candidates.append(
            {
                "candidate_id": candidate_id,
                "mesher_adapter": candidate["mesher_adapter"],
                "algorithm": algorithm,
                "capability": capability,
                "candidate_dir": str(candidate_dir),
                "input_bundle_sha256": bundle["input_bundle_sha256"],
                "qa_policy_sha256": policy_sha,
                "stages": {
                    RAW: raw,
                    COMMON_CONDITIONED: common_conditioned,
                },
            }
        )
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "case_id": bundle["case_id"],
        "run_dir": str(run_dir),
        "input_bundle": dict(bundle),
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "qa_policy": policy_value,
        "qa_policy_sha256": policy_sha,
        "common_conditioner_template_sha256": conditioner_sha,
        "candidate_order": [
            candidate["candidate_id"] for candidate in planned_candidates
        ],
        "candidates": planned_candidates,
        "comparison_policy": {
            "require_identical_input_bundle_sha256": True,
            "require_identical_qa_policy_sha256": True,
            "exclude_hard_gate_failures_from_metric_ordering": True,
            "composite_winner": None,
        },
    }


def plan_bakeoff(
    declaration: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Reserve a fresh run directory and write one immutable run plan."""
    manifest = build_run_manifest(declaration, output_dir)
    run_dir = Path(manifest["run_dir"])
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise BakeoffContractError(
            "OUTPUT_EXISTS", f"refusing to reuse run directory {run_dir}"
        ) from exc
    _write_json_fresh(run_dir / "run_manifest.json", manifest)
    return manifest


def load_declaration(path: str | Path) -> dict[str, Any]:
    """Load a declaration, resolving optional bundle/policy manifests."""
    declaration_path = Path(path).expanduser().resolve()
    declaration = _read_json(declaration_path)
    base = declaration_path.parent
    if "input_bundle_manifest" in declaration:
        reference = Path(declaration.pop("input_bundle_manifest"))
        if not reference.is_absolute():
            reference = base / reference
        declaration["input_bundle"] = _read_json(reference)
    if "qa_policy_manifest" in declaration:
        reference = Path(declaration.pop("qa_policy_manifest"))
        if not reference.is_absolute():
            reference = base / reference
        declaration["qa_policy"] = _read_json(reference)
    return declaration


def _candidate_from_manifest(
    manifest: Mapping[str, Any], candidate_id: str
) -> Mapping[str, Any]:
    candidates = manifest.get("candidates")
    if not isinstance(candidates, Sequence):
        raise BakeoffContractError("CONTRACT_VIOLATION", "invalid run manifest")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", f"unknown candidate_id: {candidate_id}"
        )
    return matches[0]


def _validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "unsupported run-manifest schema"
        )
    bundle = manifest.get("input_bundle")
    policy = manifest.get("qa_policy")
    candidates = manifest.get("candidates")
    if not isinstance(bundle, Mapping):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "run manifest input bundle is invalid"
        )
    validate_input_bundle(bundle)
    if bundle["input_bundle_sha256"] != manifest.get("input_bundle_sha256"):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "run manifest input-bundle hash is detached"
        )
    if not isinstance(policy, Mapping):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "run manifest QA policy is invalid"
        )
    _validate_qa_policy(policy)
    if _payload_sha256(dict(policy)) != manifest.get("qa_policy_sha256"):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "run manifest QA-policy hash is detached"
        )
    if not isinstance(candidates, Sequence) or isinstance(
        candidates, (str, bytes)
    ):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "run manifest candidates are invalid"
        )
    identifiers: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise BakeoffContractError(
                "CONTRACT_VIOLATION", "run manifest candidate is invalid"
            )
        _validate_candidate(candidate)
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in identifiers:
            raise BakeoffContractError(
                "CONTRACT_VIOLATION", f"duplicate candidate: {candidate_id}"
            )
        identifiers.add(candidate_id)
        if (
            candidate.get("input_bundle_sha256")
            != manifest["input_bundle_sha256"]
            or candidate.get("qa_policy_sha256") != manifest["qa_policy_sha256"]
        ):
            raise BakeoffContractError(
                "CONTRACT_VIOLATION",
                f"candidate {candidate_id} is detached from the common contract",
            )
        stages = candidate.get("stages")
        if not isinstance(stages, Mapping) or set(stages) != set(STAGES):
            raise BakeoffContractError(
                "CONTRACT_VIOLATION",
                f"candidate {candidate_id} has invalid stages",
            )
        if (
            stages[COMMON_CONDITIONED].get("conditioner_template_sha256")
            != manifest.get("common_conditioner_template_sha256")
        ):
            raise BakeoffContractError(
                "CONTRACT_VIOLATION",
                f"candidate {candidate_id} has a different conditioner",
            )


def _nested_value(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(path)
        current = current[component]
    return current


def _runtime_qa_policy(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    if policy.get("schema_version") != "fvcom_grid_quality_policy_v1":
        return policy
    adapter = policy.get("mesher_bakeoff_adapter")
    if not isinstance(adapter, Mapping):
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            "benchmark-first QA policy requires mesher_bakeoff_adapter",
        )
    return adapter


def _read_qa_result(
    qa_path: Path, policy: Mapping[str, Any]
) -> tuple[bool, list[str], dict[str, Any]]:
    report = _read_json(qa_path)
    policy = _runtime_qa_policy(policy)
    try:
        accepted = _nested_value(report, str(policy["accepted_path"]))
        failure_taxonomy = _nested_value(
            report, str(policy["failure_taxonomy_path"])
        )
        metrics = {
            name: _nested_value(report, path)
            for name, path in sorted(policy["metric_paths"].items())
        }
    except KeyError as exc:
        raise BakeoffContractError(
            "QA_REPORT_INVALID", f"missing QA path {exc.args[0]!r} in {qa_path}"
        ) from exc
    if not isinstance(accepted, bool):
        raise BakeoffContractError(
            "QA_REPORT_INVALID", "QA accepted value must be boolean"
        )
    if (
        not isinstance(failure_taxonomy, Sequence)
        or isinstance(failure_taxonomy, (str, bytes))
        or any(not isinstance(item, str) for item in failure_taxonomy)
    ):
        raise BakeoffContractError(
            "QA_REPORT_INVALID", "QA failure taxonomy must be a string array"
        )
    _json_bytes(metrics)
    return accepted, list(failure_taxonomy), metrics


def _stage_result_path(stage_spec: Mapping[str, Any]) -> Path:
    return Path(stage_spec["stage_dir"]) / "stage_result.json"


def _verify_result_artifacts(result: Mapping[str, Any]) -> None:
    for artifact in result.get("artifact_hashes", []):
        if not isinstance(artifact, Mapping):
            raise BakeoffContractError(
                "ARTIFACT_HASH_MISMATCH", "malformed artifact record"
            )
        current = _artifact_record(artifact["path"])
        if _artifact_identity(current) != _artifact_identity(artifact):
            raise BakeoffContractError(
                "ARTIFACT_HASH_MISMATCH",
                f"stage artifact changed: {artifact['path']}",
            )


def execute_stage(
    run_manifest_path: str | Path,
    candidate_id: str,
    stage: str,
) -> dict[str, Any]:
    """Execute one isolated candidate stage without mutating the run plan."""
    if stage not in STAGES:
        raise BakeoffContractError("CONTRACT_VIOLATION", f"unknown stage: {stage}")
    manifest_path = Path(run_manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    _validate_run_manifest(manifest)
    candidate = _candidate_from_manifest(manifest, candidate_id)
    stage_spec = candidate["stages"][stage]
    if candidate["input_bundle_sha256"] != manifest["input_bundle_sha256"]:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "candidate input-bundle hash mismatch"
        )
    if candidate["qa_policy_sha256"] != manifest["qa_policy_sha256"]:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "candidate QA-policy hash mismatch"
        )
    if not candidate["capability"]["supported"]:
        stage_dir = Path(stage_spec["stage_dir"])
        try:
            stage_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise BakeoffContractError(
                "OUTPUT_EXISTS", f"refusing to reuse stage directory {stage_dir}"
            ) from exc
        result = {
            "schema_version": STAGE_RESULT_SCHEMA,
            "case_id": manifest["case_id"],
            "candidate_id": candidate_id,
            "mesher_adapter": candidate["mesher_adapter"],
            "algorithm": candidate["algorithm"],
            "stage": stage,
            "status": "unsupported",
            "hard_gate_pass": False,
            "failure_taxonomy": [],
            "capability_supported": False,
            "capability_reason": candidate["capability"]["reason"],
            "metrics": {},
            "command": [],
            "return_code": None,
            "error_message": None,
            "input_bundle_sha256": manifest["input_bundle_sha256"],
            "qa_policy_sha256": manifest["qa_policy_sha256"],
            "run_manifest_sha256": _file_sha256(manifest_path),
            "artifact_hashes": [],
        }
        if stage == COMMON_CONDITIONED:
            result["conditioner_template_sha256"] = manifest[
                "common_conditioner_template_sha256"
            ]
        _write_json_fresh(_stage_result_path(stage_spec), result)
        return result
    verify_input_bundle_files(manifest["input_bundle"])
    if stage == COMMON_CONDITIONED:
        raw_spec = candidate["stages"][RAW]
        raw_result_path = _stage_result_path(raw_spec)
        if not raw_result_path.is_file():
            raise BakeoffContractError(
                "UPSTREAM_STAGE_FAILED", f"missing RAW result for {candidate_id}"
            )
        raw_result = _read_json(raw_result_path)
        if raw_result.get("status") not in {"pass", "needs_review"}:
            raise BakeoffContractError(
                "UPSTREAM_STAGE_FAILED",
                f"RAW stage did not generate a valid mesh for {candidate_id}",
            )
        _verify_result_artifacts(raw_result)

    stage_dir = Path(stage_spec["stage_dir"])
    try:
        stage_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise BakeoffContractError(
            "OUTPUT_EXISTS", f"refusing to reuse stage directory {stage_dir}"
        ) from exc

    command = list(stage_spec["command"])
    stdout_path = stage_dir / "stdout.log"
    stderr_path = stage_dir / "stderr.log"
    taxonomy: list[str] = []
    error_message: str | None = None
    return_code: int | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=stage_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=stage_spec.get("timeout_seconds"),
            check=False,
            shell=False,
        )
        stdout_path.write_bytes(completed.stdout)
        stderr_path.write_bytes(completed.stderr)
        return_code = int(completed.returncode)
        if return_code != 0:
            taxonomy.append("COMMAND_FAILED")
    except FileNotFoundError as exc:
        stdout_path.write_bytes(b"")
        stderr_path.write_text(str(exc), encoding="utf-8")
        taxonomy.append("COMMAND_NOT_FOUND")
        error_message = str(exc)
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_bytes(exc.stdout or b"")
        stderr_path.write_bytes(exc.stderr or b"")
        taxonomy.append("COMMAND_TIMEOUT")
        error_message = str(exc)
    except Exception as exc:  # preserve evidence from an unexpected adapter error
        stdout_path.write_bytes(b"")
        stderr_path.write_text(str(exc), encoding="utf-8")
        taxonomy.append("CONTRACT_VIOLATION")
        error_message = f"{type(exc).__name__}: {exc}"

    expected_paths = [stage_dir / item for item in stage_spec["artifacts"]]
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        taxonomy.append("EXPECTED_ARTIFACT_MISSING")
        error_message = json.dumps(missing, separators=(",", ":"))

    accepted: bool | None = None
    metrics: dict[str, Any] = {}
    if not taxonomy:
        try:
            accepted, qa_failures, metrics = _read_qa_result(
                stage_dir / stage_spec["qa_report"], manifest["qa_policy"]
            )
            taxonomy.extend(qa_failures)
            if not accepted and not taxonomy:
                taxonomy.append("HARD_GATE_FAILED")
        except BakeoffContractError as exc:
            taxonomy.append(exc.code)
            error_message = str(exc)

    artifact_paths = [stdout_path, stderr_path]
    artifact_paths.extend(path for path in expected_paths if path.exists())
    artifact_records = [
        _artifact_record(path)
        for path in sorted(
            {path.resolve() for path in artifact_paths}, key=lambda item: str(item)
        )
    ]
    if accepted is True and not taxonomy:
        status = "pass"
    elif accepted is False and all(
        item not in {
            "COMMAND_FAILED",
            "COMMAND_NOT_FOUND",
            "COMMAND_TIMEOUT",
            "CONTRACT_VIOLATION",
            "EXPECTED_ARTIFACT_MISSING",
            "QA_REPORT_INVALID",
        }
        for item in taxonomy
    ):
        status = "needs_review"
    else:
        status = "failed"
    result = {
        "schema_version": STAGE_RESULT_SCHEMA,
        "case_id": manifest["case_id"],
        "candidate_id": candidate_id,
        "mesher_adapter": candidate["mesher_adapter"],
        "algorithm": candidate["algorithm"],
        "stage": stage,
        "status": status,
        "hard_gate_pass": accepted is True and status == "pass",
        "failure_taxonomy": list(dict.fromkeys(taxonomy)),
        "metrics": metrics,
        "command": command,
        "return_code": return_code,
        "error_message": error_message,
        "input_bundle_sha256": manifest["input_bundle_sha256"],
        "qa_policy_sha256": manifest["qa_policy_sha256"],
        "run_manifest_sha256": _file_sha256(manifest_path),
        "artifact_hashes": artifact_records,
    }
    if stage == COMMON_CONDITIONED:
        result["conditioner_template_sha256"] = manifest[
            "common_conditioner_template_sha256"
        ]
    _write_json_fresh(_stage_result_path(stage_spec), result)
    return result


def _metric_ordering(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    eligible = [row for row in rows if row.get("hard_gate_pass") is True]
    if not eligible:
        return {}
    output: dict[str, list[dict[str, Any]]] = {}
    policy = _runtime_qa_policy(policy)
    directions = policy.get("metric_directions", {})
    for metric in sorted(policy["metric_paths"]):
        direction = directions.get(metric, "report_only")
        if direction == "report_only":
            continue
        values: list[tuple[str, float]] = []
        for row in eligible:
            value = row.get("metrics", {}).get(metric)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                values.append((str(row["candidate_id"]), float(value)))
        values.sort(
            key=lambda item: (
                item[1] if direction == "minimize" else -item[1],
                item[0],
            )
        )
        output[metric] = [
            {"candidate_id": candidate_id, "value": value}
            for candidate_id, value in values
        ]
    return output


def compare_results(
    run_manifest_paths: Iterable[str | Path],
    *,
    stage: str,
) -> dict[str, Any]:
    """Build a per-metric comparison only for identical locked contracts."""
    if stage not in STAGES:
        raise BakeoffContractError("CONTRACT_VIOLATION", f"unknown stage: {stage}")
    paths = sorted(
        {Path(path).expanduser().resolve() for path in run_manifest_paths},
        key=lambda path: str(path),
    )
    if not paths:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION", "at least one run manifest is required"
        )
    manifests = [_read_json(path) for path in paths]
    for manifest in manifests:
        _validate_run_manifest(manifest)
    bundle_hashes = {manifest.get("input_bundle_sha256") for manifest in manifests}
    policy_hashes = {manifest.get("qa_policy_sha256") for manifest in manifests}
    if len(bundle_hashes) != 1:
        raise BakeoffContractError(
            "INPUT_HASH_MISMATCH",
            "comparison candidates do not share input_bundle_sha256",
        )
    if len(policy_hashes) != 1:
        raise BakeoffContractError(
            "CONTRACT_VIOLATION",
            "comparison candidates do not share qa_policy_sha256",
        )
    policy = manifests[0]["qa_policy"]
    rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for manifest_path, manifest in zip(paths, manifests, strict=True):
        plan_sha = _file_sha256(manifest_path)
        for candidate in manifest["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in seen_candidates:
                raise BakeoffContractError(
                    "CONTRACT_VIOLATION",
                    f"duplicate comparison candidate: {candidate_id}",
                )
            seen_candidates.add(candidate_id)
            if candidate["input_bundle_sha256"] not in bundle_hashes:
                raise BakeoffContractError(
                    "INPUT_HASH_MISMATCH",
                    f"candidate {candidate_id} has a different input bundle",
                )
            if candidate["qa_policy_sha256"] not in policy_hashes:
                raise BakeoffContractError(
                    "CONTRACT_VIOLATION",
                    f"candidate {candidate_id} has a different QA policy",
                )
            result_path = _stage_result_path(candidate["stages"][stage])
            if result_path.is_file():
                result = _read_json(result_path)
                if (
                    result.get("schema_version") != STAGE_RESULT_SCHEMA
                    or result.get("candidate_id") != candidate_id
                    or result.get("mesher_adapter")
                    != candidate["mesher_adapter"]
                    or result.get("algorithm") != candidate["algorithm"]
                    or result.get("stage") != stage
                ):
                    raise BakeoffContractError(
                        "CONTRACT_VIOLATION",
                        f"stage result is detached from its candidate: {result_path}",
                    )
                if result.get("run_manifest_sha256") != plan_sha:
                    raise BakeoffContractError(
                        "CONTRACT_VIOLATION",
                        f"stage result is detached from its plan: {result_path}",
                    )
                if result.get("input_bundle_sha256") not in bundle_hashes:
                    raise BakeoffContractError(
                        "INPUT_HASH_MISMATCH",
                        f"stage result has a different input bundle: {result_path}",
                    )
                if result.get("qa_policy_sha256") not in policy_hashes:
                    raise BakeoffContractError(
                        "CONTRACT_VIOLATION",
                        f"stage result has a different QA policy: {result_path}",
                    )
                _verify_result_artifacts(result)
                row = {
                    "candidate_id": candidate_id,
                    "mesher_adapter": candidate["mesher_adapter"],
                    "algorithm": candidate["algorithm"],
                    "status": result["status"],
                    "hard_gate_pass": result["hard_gate_pass"],
                    "failure_taxonomy": result["failure_taxonomy"],
                    "capability_reason": result.get("capability_reason"),
                    "metrics": result["metrics"],
                }
            else:
                unsupported = (
                    candidate["stages"][stage].get("status") == "unsupported"
                )
                row = {
                    "candidate_id": candidate_id,
                    "mesher_adapter": candidate["mesher_adapter"],
                    "algorithm": candidate["algorithm"],
                    "status": "unsupported" if unsupported else "not_run",
                    "hard_gate_pass": False,
                    "failure_taxonomy": [],
                    "capability_reason": (
                        candidate["capability"].get("reason")
                        if unsupported
                        else None
                    ),
                    "metrics": {},
                }
            rows.append(row)
    rows.sort(key=lambda row: row["candidate_id"])
    passers = [
        row["candidate_id"] for row in rows if row["hard_gate_pass"] is True
    ]
    metric_ordering = _metric_ordering(rows, policy)
    return {
        "schema_version": COMPARISON_SCHEMA,
        "case_id": manifests[0]["case_id"],
        "stage": stage,
        "input_bundle_sha256": next(iter(bundle_hashes)),
        "qa_policy_sha256": next(iter(policy_hashes)),
        "run_manifest_sha256": [_file_sha256(path) for path in paths],
        "metric_columns": sorted(policy["metric_paths"]),
        "per_metric_table": rows,
        "hard_gate_passers": passers,
        "per_metric_ordering": metric_ordering,
        "ordering_scope": (
            "hard-gate-passers-only" if passers else "withheld-no-hard-gate-passer"
        ),
        "composite_score": None,
        "composite_winner": None,
    }


def write_comparison(path: str | Path, comparison: Mapping[str, Any]) -> None:
    _write_json_fresh(path, dict(comparison))
