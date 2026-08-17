#!/usr/bin/env python3
"""Run a fresh, sequential FVCOM conditioning campaign from a JSON manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.portfolio_conditioning import (  # noqa: E402
    PortfolioConditioningConfig,
    condition_portfolio_mesh,
)


SCHEMA = "fvcom_conditioning_campaign_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--conditioning-profile",
        choices=(
            "auto",
            "minimal-topology-v1",
            "guarded-v1",
            "aggressive-local-v2",
            "none",
        ),
        default="auto",
    )
    parser.add_argument("--wall-time-s", type=float, default=3_600.0)
    parser.add_argument("--primary-rounds", type=int, default=4)
    parser.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Campaign output must be new: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = _validate_manifest(manifest)
    output.mkdir(parents=True, exist_ok=False)

    source_snapshot = {
        "schema_version": SCHEMA,
        "source_manifest": _artifact(manifest_path),
        "conditioning_profile_requested": str(args.conditioning_profile),
        "wall_time_s_per_case": float(args.wall_time_s),
        "primary_rounds": int(args.primary_rounds),
        "diagnostics_enabled": bool(args.diagnostics),
        "cases": cases,
    }
    _write_json(output / "campaign_input_snapshot.json", source_snapshot)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    status_path = output / "status.json"
    _write_status(status_path, "running", results, len(cases))

    for ordinal, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        case_dir = output / f"{ordinal:02d}_{_slug(case_id)}"
        case_dir.mkdir(parents=False, exist_ok=False)
        case_started = time.perf_counter()
        input_bundle = _input_bundle(case)
        _write_json(case_dir / "input_bundle.json", input_bundle)
        limits = _case_limits(case, args)
        commands = _commands(
            case,
            case_dir,
            str(args.conditioning_profile),
            limits,
        )
        _write_json(case_dir / "commands.json", commands)
        result: dict[str, Any] = {
            "case_id": case_id,
            "display_name": str(case.get("display_name", case_id)),
            "difficulty_rank": int(case["difficulty_rank"]),
            "status": "running",
            "scientific_input_valid": bool(
                case.get("scientific_input_valid", True)
            ),
            "scientific_input_note": case.get("scientific_input_note"),
            "input_bundle_sha256": input_bundle["bundle_sha256"],
            "case_limits": limits,
            "output_directory": str(case_dir),
        }
        _write_json(case_dir / "status.json", result)
        try:
            report = condition_portfolio_mesh(
                case["mesh"],
                case["size_field_nc"],
                case["bathymetry_nc"],
                case_dir / "conditioned",
                name=str(case.get("output_name", case_id)),
                config=PortfolioConditioningConfig(
                    conditioning_profile=str(args.conditioning_profile),
                    primary_rounds=int(limits["primary_rounds"]),
                    max_valence_repairs_per_round=int(
                        limits["max_valence_repairs_per_round"]
                    ),
                    max_valence_flip_batch=int(
                        limits["max_valence_flip_batch"]
                    ),
                    max_valence_cluster_merges_per_round=int(
                        limits["max_valence_cluster_merges_per_round"]
                    ),
                    wall_time_s=float(limits["wall_time_s"]),
                ),
                boundary_contract=_load_optional_json(
                    case.get("boundary_contract_json")
                ),
                source_boundary_metadata=_load_optional_json(
                    case.get("source_boundary_metadata_json")
                ),
                scientific_input_valid=bool(
                    case.get("scientific_input_valid", True)
                ),
                scientific_input_note=case.get("scientific_input_note"),
            )
            diagnostics = (
                _run_diagnostics(case, case_dir)
                if bool(args.diagnostics)
                else {"enabled": False, "commands": []}
            )
            result.update(
                {
                    "status": str(report["status"]),
                    "minimal_local_debt_closed": bool(
                        report["minimal_local_debt_closed"]
                    ),
                    "fvcom_ready": bool(report["fvcom_ready"]),
                    "failure_taxonomy": list(
                        report["fvcom_readiness_failure_taxonomy"]
                    ),
                    "runtime_seconds": float(
                        time.perf_counter() - case_started
                    ),
                    "before": report["raw_global_audit"],
                    "after": report["final_global_audit"],
                    "before_quality": report["raw_quality"],
                    "after_quality": report["final_quality"],
                    "before_edge_size_continuity": report[
                        "raw_edge_size_continuity"
                    ],
                    "after_edge_size_continuity": report[
                        "final_edge_size_continuity"
                    ],
                    "conditioning_report": report["outputs"][
                        "conditioning_report_json"
                    ],
                    "conditioned_2dm": report["outputs"][
                        "conditioned_2dm"
                    ],
                    "mesh_quality_json": report["outputs"][
                        "mesh_quality_json"
                    ],
                    "diagnostics": diagnostics,
                }
            )
        except BaseException as exc:
            failure = {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json(case_dir / "failure.json", failure)
            result.update(
                {
                    "status": "failed",
                    "minimal_local_debt_closed": False,
                    "fvcom_ready": False,
                    "failure_taxonomy": [
                        f"conditioning_exception:{type(exc).__name__}"
                    ],
                    "runtime_seconds": float(
                        time.perf_counter() - case_started
                    ),
                    "failure": failure,
                }
            )
            if case.get("raw_quality_json"):
                raw_quality = _load_optional_json(
                    case.get("raw_quality_json")
                )
                if raw_quality is not None:
                    result["before_quality"] = raw_quality
                    result["before"] = _raw_reference_audit(raw_quality)
        _write_json(case_dir / "status.json", result)
        results.append(result)
        _write_status(status_path, "running", results, len(cases))

    campaign = {
        "schema_version": SCHEMA,
        "status": "complete",
        "runtime_seconds": float(time.perf_counter() - started),
        "case_count": int(len(results)),
        "case_failure_count": int(
            sum(item["status"] == "failed" for item in results)
        ),
        "minimal_local_debt_closed_count": int(
            sum(bool(item.get("minimal_local_debt_closed")) for item in results)
        ),
        "fvcom_ready_count": int(
            sum(bool(item.get("fvcom_ready")) for item in results)
        ),
        "results": results,
    }
    _write_json(output / "campaign.json", campaign)
    _write_csv(output / "campaign.csv", results)
    (output / "REPORT.md").write_text(
        _markdown_report(campaign),
        encoding="utf-8",
    )
    _write_status(status_path, "complete", results, len(cases))
    print(json.dumps(campaign, indent=2, sort_keys=True))
    return 0


def _validate_manifest(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        raise ValueError(f"Campaign manifest must use {SCHEMA}")
    values = payload.get("cases")
    if not isinstance(values, list) or not values:
        raise ValueError("Campaign manifest must contain a nonempty cases list")
    required = {
        "case_id",
        "difficulty_rank",
        "mesh",
        "size_field_nc",
        "bathymetry_nc",
    }
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    output: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict) or not required.issubset(value):
            raise ValueError(
                "Every campaign case requires case_id, difficulty_rank, "
                "mesh, size_field_nc, and bathymetry_nc"
            )
        case_id = str(value["case_id"]).strip()
        rank = int(value["difficulty_rank"])
        if not case_id or case_id in seen_ids or rank in seen_ranks:
            raise ValueError("Case IDs and difficulty ranks must be unique")
        seen_ids.add(case_id)
        seen_ranks.add(rank)
        normalized = dict(value)
        normalized["case_id"] = case_id
        normalized["difficulty_rank"] = rank
        if normalized.get("limits") is not None and not isinstance(
            normalized["limits"],
            dict,
        ):
            raise ValueError("Case limits must be a JSON object")
        if normalized.get("limits") is not None:
            _validate_case_limits_payload(normalized["limits"])
        for key in (
            "mesh",
            "size_field_nc",
            "bathymetry_nc",
            "boundary_contract_json",
            "source_boundary_metadata_json",
            "boundary_nodes_geojson",
            "raw_quality_json",
        ):
            if normalized.get(key) is None:
                continue
            path = Path(str(normalized[key])).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            normalized[key] = str(path)
        output.append(normalized)
    return sorted(output, key=lambda value: int(value["difficulty_rank"]))


def _input_bundle(case: dict[str, Any]) -> dict[str, Any]:
    artifacts = {}
    for key in (
        "mesh",
        "size_field_nc",
        "bathymetry_nc",
        "boundary_contract_json",
        "source_boundary_metadata_json",
        "boundary_nodes_geojson",
        "raw_quality_json",
    ):
        if case.get(key) is not None:
            artifacts[key] = _artifact(Path(str(case[key])))
    contract = {
        "case_id": str(case["case_id"]),
        "difficulty_rank": int(case["difficulty_rank"]),
        "scientific_input_valid": bool(
            case.get("scientific_input_valid", True)
        ),
        "scientific_input_note": case.get("scientific_input_note"),
        "limits": dict(case.get("limits", {})),
        "artifacts": artifacts,
    }
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": "fvcom_conditioning_input_bundle_v1",
        **contract,
        "bundle_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _commands(
    case: dict[str, Any],
    case_dir: Path,
    profile: str,
    limits: dict[str, Any],
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_portfolio_conditioning.py")),
        "--mesh",
        str(case["mesh"]),
        "--size-field-nc",
        str(case["size_field_nc"]),
        "--bathymetry-nc",
        str(case["bathymetry_nc"]),
        "--output-dir",
        str(case_dir / "conditioned"),
        "--conditioning-profile",
        profile,
        "--wall-time-s",
        str(limits["wall_time_s"]),
        "--primary-rounds",
        str(limits["primary_rounds"]),
        "--max-valence-repairs-per-round",
        str(limits["max_valence_repairs_per_round"]),
        "--max-valence-flip-batch",
        str(limits["max_valence_flip_batch"]),
        "--max-valence-cluster-merges-per-round",
        str(limits["max_valence_cluster_merges_per_round"]),
    ]
    for key, flag in (
        ("boundary_contract_json", "--boundary-contract-json"),
        ("source_boundary_metadata_json", "--source-boundary-metadata-json"),
    ):
        if case.get(key):
            command.extend([flag, str(case[key])])
    command.extend(
        [
            "--scientific-input-status",
            (
                "valid"
                if bool(case.get("scientific_input_valid", True))
                else "invalid"
            ),
        ]
    )
    if case.get("scientific_input_note"):
        command.extend(
            ["--scientific-input-note", str(case["scientific_input_note"])]
        )
    return {
        "schema_version": "fvcom_conditioning_commands_v1",
        "equivalent_standalone_command": command,
    }


def _case_limits(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    supplied = dict(case.get("limits", {}))
    limits = {
        "wall_time_s": float(
            supplied.get("wall_time_s", args.wall_time_s)
        ),
        "primary_rounds": int(
            supplied.get("primary_rounds", args.primary_rounds)
        ),
        "max_valence_repairs_per_round": int(
            supplied.get("max_valence_repairs_per_round", 500)
        ),
        "max_valence_flip_batch": int(
            supplied.get("max_valence_flip_batch", 64)
        ),
        "max_valence_cluster_merges_per_round": int(
            supplied.get("max_valence_cluster_merges_per_round", 25)
        ),
    }
    if limits["wall_time_s"] <= 0.0 or limits["primary_rounds"] <= 0:
        raise ValueError("Case wall_time_s and primary_rounds must be positive")
    for name in (
        "max_valence_repairs_per_round",
        "max_valence_flip_batch",
        "max_valence_cluster_merges_per_round",
    ):
        if limits[name] < 0:
            raise ValueError(f"Case {name} must be nonnegative")
    return limits


def _validate_case_limits_payload(values: dict[str, Any]) -> None:
    allowed = {
        "wall_time_s",
        "primary_rounds",
        "max_valence_repairs_per_round",
        "max_valence_flip_batch",
        "max_valence_cluster_merges_per_round",
    }
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            "Unsupported case limit keys: " + ", ".join(unknown)
        )
    if "wall_time_s" in values and float(values["wall_time_s"]) <= 0.0:
        raise ValueError("Case wall_time_s must be positive")
    if "primary_rounds" in values and int(values["primary_rounds"]) <= 0:
        raise ValueError("Case primary_rounds must be positive")
    for name in (
        "max_valence_repairs_per_round",
        "max_valence_flip_batch",
        "max_valence_cluster_merges_per_round",
    ):
        if name in values and int(values[name]) < 0:
            raise ValueError(f"Case {name} must be nonnegative")


def _run_diagnostics(
    case: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    scripts = Path(__file__).resolve().parent
    delivered_boundary = (
        case_dir / "conditioned" / "delivered_boundary_nodes.geojson"
    )
    conditioned_mesh = case_dir / "conditioned" / "conditioned.2dm"
    report = case_dir / "conditioned" / "conditioning_report.json"
    commands: list[dict[str, Any]] = []

    def run(label: str, command: list[str]) -> None:
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(scripts),
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )
        commands.append(
            {
                "label": label,
                "command": command,
                "returncode": int(completed.returncode),
                "runtime_seconds": float(time.perf_counter() - started),
                "stdout": completed.stdout[-8_000:],
                "stderr": completed.stderr[-8_000:],
            }
        )

    before_boundary = case.get("boundary_nodes_geojson")
    high_valence_pairs = [
        ("before", str(case["mesh"]), before_boundary, None),
        (
            "after",
            str(conditioned_mesh),
            str(delivered_boundary),
            str(report),
        ),
    ]
    for stage, mesh, boundary, conditioning_report in high_valence_pairs:
        command = [
            sys.executable,
            str(scripts / "diagnose_high_valence.py"),
            "--mesh",
            mesh,
            "--output-dir",
            str(case_dir / f"diagnostics_{stage}" / "high_valence"),
        ]
        if boundary:
            command.extend(["--boundary-nodes-geojson", str(boundary)])
        if conditioning_report:
            command.extend(
                ["--conditioning-report", str(conditioning_report)]
            )
        run(f"high_valence_{stage}", command)

    if before_boundary:
        for stage, mesh, boundary, conditioning_report in (
            ("before", str(case["mesh"]), before_boundary, None),
            (
                "after",
                str(conditioned_mesh),
                str(delivered_boundary),
                str(report),
            ),
        ):
            command = [
                sys.executable,
                str(scripts / "diagnose_superthin_components.py"),
                "--mesh",
                mesh,
                "--boundary-nodes-geojson",
                str(boundary),
                "--size-field-nc",
                str(case["size_field_nc"]),
                "--output-dir",
                str(case_dir / f"diagnostics_{stage}" / "superthin"),
            ]
            if conditioning_report:
                command.extend(
                    ["--conditioning-report", str(conditioning_report)]
                )
            run(f"superthin_{stage}", command)
    return {
        "enabled": True,
        "all_commands_passed": bool(
            all(item["returncode"] == 0 for item in commands)
        ),
        "commands": commands,
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "difficulty_rank",
        "case_id",
        "status",
        "scientific_input_valid",
        "minimal_local_debt_closed",
        "fvcom_ready",
        "runtime_seconds",
        "nodes_before",
        "nodes_after",
        "triangles_before",
        "triangles_after",
        "superthin_before",
        "superthin_after",
        "valence_above_8_before",
        "valence_above_8_after",
        "q_l3_sigma_before",
        "q_l3_sigma_after",
        "q_min_before",
        "q_min_after",
        "minimum_angle_deg_before",
        "minimum_angle_deg_after",
        "maximum_angle_deg_before",
        "maximum_angle_deg_after",
        "maximum_valence_before",
        "maximum_valence_after",
        "maximum_adjacent_area_change_before",
        "maximum_adjacent_area_change_after",
        "maximum_bathymetric_slope_before",
        "maximum_bathymetric_slope_after",
        "l_over_h_p95_before",
        "l_over_h_p95_after",
        "l_over_h_maximum_before",
        "l_over_h_maximum_after",
        "boundary_first_ring_p95_before",
        "boundary_first_ring_p95_after",
        "boundary_first_ring_maximum_before",
        "boundary_first_ring_maximum_after",
        "components_before",
        "components_after",
        "nonmanifold_edges_before",
        "nonmanifold_edges_after",
        "singly_connected_before",
        "singly_connected_after",
        "failure_taxonomy",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for value in results:
            before = value.get("before", {})
            after = value.get("after", {})
            before_quality = value.get("before_quality", {})
            after_quality = value.get("after_quality", {})
            before_size = before_quality.get("size_error_l_over_h") or {}
            after_size = after_quality.get("size_error_l_over_h") or {}
            before_continuity = value.get(
                "before_edge_size_continuity",
                {},
            )
            after_continuity = value.get(
                "after_edge_size_continuity",
                {},
            )
            before_first_ring = _first_ring_summary(before_continuity)
            after_first_ring = _first_ring_summary(after_continuity)
            writer.writerow(
                {
                    "difficulty_rank": value["difficulty_rank"],
                    "case_id": value["case_id"],
                    "status": value["status"],
                    "scientific_input_valid": value.get(
                        "scientific_input_valid"
                    ),
                    "minimal_local_debt_closed": value.get(
                        "minimal_local_debt_closed"
                    ),
                    "fvcom_ready": value.get("fvcom_ready"),
                    "runtime_seconds": value.get("runtime_seconds"),
                    "nodes_before": before.get(
                        "node_count",
                        before_quality.get("node_count"),
                    ),
                    "nodes_after": after.get(
                        "node_count",
                        after_quality.get("node_count"),
                    ),
                    "triangles_before": before.get(
                        "triangle_count",
                        before_quality.get("triangle_count"),
                    ),
                    "triangles_after": after.get(
                        "triangle_count",
                        after_quality.get("triangle_count"),
                    ),
                    "superthin_before": before.get(
                        "superthin_triangle_count"
                    ),
                    "superthin_after": after.get(
                        "superthin_triangle_count"
                    ),
                    "valence_above_8_before": before.get(
                        "count_valence_above_8"
                    ),
                    "valence_above_8_after": after.get(
                        "count_valence_above_8"
                    ),
                    "q_l3_sigma_before": before.get("q_l3_sigma"),
                    "q_l3_sigma_after": after.get("q_l3_sigma"),
                    "q_min_before": before.get("q_min"),
                    "q_min_after": after.get("q_min"),
                    "minimum_angle_deg_before": before_quality.get(
                        "min_angle_deg"
                    ),
                    "minimum_angle_deg_after": after_quality.get(
                        "min_angle_deg"
                    ),
                    "maximum_angle_deg_before": before_quality.get(
                        "max_angle_deg"
                    ),
                    "maximum_angle_deg_after": after_quality.get(
                        "max_angle_deg"
                    ),
                    "maximum_valence_before": before_quality.get(
                        "max_node_valence"
                    ),
                    "maximum_valence_after": after_quality.get(
                        "max_node_valence"
                    ),
                    "maximum_adjacent_area_change_before": (
                        before_quality.get("max_adjacent_area_change")
                    ),
                    "maximum_adjacent_area_change_after": (
                        after_quality.get("max_adjacent_area_change")
                    ),
                    "maximum_bathymetric_slope_before": before_quality.get(
                        "max_bathymetric_slope"
                    ),
                    "maximum_bathymetric_slope_after": after_quality.get(
                        "max_bathymetric_slope"
                    ),
                    "l_over_h_p95_before": _nested(
                        before_size,
                        "quantiles",
                        "p95",
                    ),
                    "l_over_h_p95_after": _nested(
                        after_size,
                        "quantiles",
                        "p95",
                    ),
                    "l_over_h_maximum_before": before_size.get("maximum"),
                    "l_over_h_maximum_after": after_size.get("maximum"),
                    "boundary_first_ring_p95_before": before_first_ring[0],
                    "boundary_first_ring_p95_after": after_first_ring[0],
                    "boundary_first_ring_maximum_before": before_first_ring[1],
                    "boundary_first_ring_maximum_after": after_first_ring[1],
                    "components_before": before.get(
                        "connected_component_count"
                    ),
                    "components_after": after.get(
                        "connected_component_count"
                    ),
                    "nonmanifold_edges_before": before.get(
                        "nonmanifold_edge_count"
                    ),
                    "nonmanifold_edges_after": after.get(
                        "nonmanifold_edge_count"
                    ),
                    "singly_connected_before": before.get(
                        "singly_connected_triangle_count"
                    ),
                    "singly_connected_after": after.get(
                        "singly_connected_triangle_count"
                    ),
                    "failure_taxonomy": ";".join(
                        map(str, value.get("failure_taxonomy", []))
                    ),
                }
            )


def _markdown_report(campaign: dict[str, Any]) -> str:
    lines = [
        "# Simplified Conditioning Campaign",
        "",
        f"Status: **{campaign['status']}**",
        "",
        (
            f"Cases: {campaign['case_count']}; driver failures: "
            f"{campaign['case_failure_count']}; minimal local closure: "
            f"{campaign['minimal_local_debt_closed_count']}; full FVCOM "
            f"readiness: {campaign['fvcom_ready_count']}."
        ),
        "",
        "| Rank | Case | Status | Minimal | Ready | Runtime s | Nodes pre->post | Triangles pre->post | qL3 pre->post | Superthin pre->post | Valence>8 pre->post | Angle min/max post | Area jump post | L/h p95/max post | Failures |",
        "|---:|---|---|---|---|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for value in campaign["results"]:
        failures = ", ".join(map(str, value.get("failure_taxonomy", [])))
        lines.append(
            "| {rank} | {case} | {status} | {minimal} | {ready} | "
            "{runtime} | {nodes} | {triangles} | {quality} | {thin} | "
            "{valence} | {angles} | {area} | {size} | {failures} |".format(
                rank=value["difficulty_rank"],
                case=value["display_name"],
                status=value["status"],
                minimal=value.get("minimal_local_debt_closed"),
                ready=value.get("fvcom_ready"),
                runtime=_fmt(value.get("runtime_seconds")),
                nodes=_transition(
                    value.get("before_quality", {}).get("node_count"),
                    value.get("after_quality", {}).get("node_count"),
                ),
                triangles=_transition(
                    value.get("before_quality", {}).get("triangle_count"),
                    value.get("after_quality", {}).get("triangle_count"),
                ),
                quality=_transition(
                    value.get("before", {}).get("q_l3_sigma"),
                    value.get("after", {}).get("q_l3_sigma"),
                ),
                thin=_transition(
                    value.get("before", {}).get(
                        "superthin_triangle_count"
                    ),
                    value.get("after", {}).get(
                        "superthin_triangle_count"
                    ),
                ),
                valence=_transition(
                    value.get("before", {}).get("count_valence_above_8"),
                    value.get("after", {}).get("count_valence_above_8"),
                ),
                angles=(
                    f"{_fmt(value.get('after_quality', {}).get('min_angle_deg'))}/"
                    f"{_fmt(value.get('after_quality', {}).get('max_angle_deg'))}"
                ),
                area=_fmt(
                    value.get("after_quality", {}).get(
                        "max_adjacent_area_change"
                    )
                ),
                size=(
                    f"{_fmt(_nested(value.get('after_quality', {}).get('size_error_l_over_h') or {}, 'quantiles', 'p95'))}/"
                    f"{_fmt((value.get('after_quality', {}).get('size_error_l_over_h') or {}).get('maximum'))}"
                ),
                failures=failures.replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "Minimal local closure and full FVCOM readiness are independent. ",
            "No composite cross-region score is computed.",
            "",
        ]
    )
    return "\n".join(lines)


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _raw_reference_audit(quality: dict[str, Any]) -> dict[str, Any]:
    """Extract reportable preconditioning metrics after an early failure."""

    oceanmesh = quality.get("oceanmesh_quality") or {}
    angles = quality.get("angle_statistics") or {}
    valence = quality.get("valence") or {}
    topology = quality.get("topology") or {}
    return {
        "node_count": quality.get("node_count"),
        "triangle_count": quality.get("triangle_count"),
        "q_l3_sigma": oceanmesh.get("q_l3_sigma"),
        "q_min": oceanmesh.get("q_min"),
        "q_p01": _nested(oceanmesh, "q_quantiles", "p01"),
        "superthin_triangle_count": angles.get(
            "count_min_angle_below_5"
        ),
        "count_valence_above_8": valence.get(
            "count_valence_above_8"
        ),
        "maximum_valence": valence.get(
            "max_node_valence",
            quality.get("max_node_valence"),
        ),
        "connected_component_count": topology.get(
            "connected_component_count"
        ),
        "nonmanifold_edge_count": topology.get(
            "nonmanifold_edge_count"
        ),
        "singly_connected_triangle_count": topology.get(
            "singly_connected_triangle_count"
        ),
    }


def _first_ring_summary(value: dict[str, Any]) -> tuple[Any, Any]:
    summary = _nested(
        value,
        "boundary_first_ring_realized_continuity",
        "global",
        "symmetric_ratio",
    )
    if not isinstance(summary, dict):
        return None, None
    return _nested(summary, "quantiles", "p95"), summary.get("maximum")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def _transition(before: Any, after: Any) -> str:
    return f"{_fmt(before)}->{_fmt(after)}"


def _write_status(
    path: Path,
    status: str,
    results: list[dict[str, Any]],
    total: int,
) -> None:
    _write_json(
        path,
        {
            "schema_version": "fvcom_conditioning_campaign_status_v1",
            "status": status,
            "completed_case_count": int(len(results)),
            "total_case_count": int(total),
            "results": results,
        },
    )


def _load_optional_json(path: Any) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "bytes": int(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_"
        for character in value.lower()
    ).strip("_")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
