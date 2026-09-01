#!/usr/bin/env python3
"""Run or resume the autonomous-thin-v1 conditioning workflow.

The command performs the deterministic stages itself.  When residual thin
components require visual classification, it emits hash-bound pending Codex
decision documents and stops with a successful ``agent_decision_required``
status. Supply one completed decision to resume without repeating accepted
earlier stages. GSHHS is the only shoreline source used by this workflow.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.autonomous_thin import sha256_file  # noqa: E402
from fvcom_grid_generation.open_exterior import validate_open_exterior_contract  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--size-field-nc", required=True, type=Path)
    parser.add_argument("--bathymetry-nc", required=True, type=Path)
    parser.add_argument("--boundary-nodes-geojson", required=True, type=Path)
    parser.add_argument("--boundary-contract-json", required=True, type=Path)
    parser.add_argument("--source-boundary-metadata-json", required=True, type=Path)
    parser.add_argument("--region-bpoly-json", required=True, type=Path)
    parser.add_argument("--source-case-manifest", required=True, type=Path)
    parser.add_argument(
        "--source-boundary-resolution-manifest", required=True, type=Path
    )
    parser.add_argument("--gshhs-gpkg", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    open_exterior_audit = validate_open_exterior_contract(
        args.source_boundary_resolution_manifest.resolve(), required=False
    )
    if not open_exterior_audit["passed"]:
        raise ValueError(
            "autonomous-thin-v1 rejected the upstream open-exterior package: "
            + ", ".join(open_exterior_audit["failure_taxonomy"])
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial_dir = output / "01_minimal_conditioning"
    initial_report = initial_dir / "conditioning_report.json"
    history: list[dict[str, Any]] = []
    if not initial_report.is_file():
        if initial_dir.exists() and any(initial_dir.iterdir()):
            raise RuntimeError(
                "incomplete initial-conditioning directory is retained; use a new output directory"
            )
        initial = _run([
            sys.executable,
            str(Path(__file__).with_name("run_portfolio_conditioning.py")),
            "--mesh", str(args.mesh.resolve()),
            "--size-field-nc", str(args.size_field_nc.resolve()),
            "--bathymetry-nc", str(args.bathymetry_nc.resolve()),
            "--boundary-contract-json", str(args.boundary_contract_json.resolve()),
            "--source-boundary-metadata-json",
            str(args.source_boundary_metadata_json.resolve()),
            "--output-dir", str(initial_dir),
            "--conditioning-profile", "minimal-topology-v1",
        ])
        history.append({"stage": "minimal_conditioning", **initial})
        if not initial_report.is_file():
            status = {
                "schema_version": "fvcom_autonomous_thin_workflow_v1",
                "created_at_utc": _utc_now(),
                "status": "minimal_conditioning_failed",
                "history": history,
            }
            _write_json(output / "workflow_status.json", status)
            print(json.dumps(status, indent=2))
            return 2

    conditioned_mesh = initial_dir / "conditioned.2dm"
    if not conditioned_mesh.is_file():
        raise FileNotFoundError(conditioned_mesh)
    gshhs_key = sha256_file(args.gshhs_gpkg)[:12]
    diagnostic_dir = output / f"02_diagnostic_{gshhs_key}"
    diagnostic_path = diagnostic_dir / "thin_v2.json"
    if not diagnostic_path.is_file():
        if diagnostic_dir.exists() and any(diagnostic_dir.iterdir()):
            raise RuntimeError(
                "incomplete diagnostic directory is retained; use a new output directory"
            )
        command = [
            sys.executable,
            str(Path(__file__).with_name("diagnose_autonomous_thin.py")),
            "--mesh", str(conditioned_mesh),
            "--boundary-nodes-geojson", str(args.boundary_nodes_geojson.resolve()),
            "--size-field-nc", str(args.size_field_nc.resolve()),
            "--bathymetry-nc", str(args.bathymetry_nc.resolve()),
            "--conditioning-report", str(initial_report),
            "--gshhs-gpkg", str(args.gshhs_gpkg.resolve()),
            "--region-bpoly-json", str(args.region_bpoly_json.resolve()),
            "--case-manifest-json", str(args.source_case_manifest.resolve()),
            "--boundary-resolution-manifest",
            str(args.source_boundary_resolution_manifest.resolve()),
            "--source-boundary-metadata-json",
            str(args.source_boundary_metadata_json.resolve()),
            "--boundary-contract-json", str(args.boundary_contract_json.resolve()),
            "--output-dir", str(diagnostic_dir),
        ]
        diagnostic = _run(command)
        history.append({"stage": "diagnostic", **diagnostic})
        if not diagnostic_path.is_file():
            raise RuntimeError("autonomous-thin diagnostic was not published")

    diagnostic = _read_json(diagnostic_path)
    if int(diagnostic.get("component_count", -1)) == 0:
        status_name = "pass"
        decision_files: list[str] = []
        closure: dict[str, Any] | None = None
    elif args.decision is None:
        status_name = "agent_decision_required"
        decision_files = [str(value) for value in sorted((diagnostic_dir / "d").glob("*.pending.json"))]
        closure = None
    else:
        closure_dir = output / f"03_closure_{sha256_file(args.decision)[:12]}"
        closure_report = closure_dir / "autonomous_thin_closure.json"
        if not closure_report.is_file():
            if closure_dir.exists() and any(closure_dir.iterdir()):
                raise RuntimeError(
                    "incomplete closure directory is retained; use a new output directory"
                )
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_autonomous_thin_closure.py")),
                "--diagnostic", str(diagnostic_path),
                "--decision", str(args.decision.resolve()),
                "--source-case-manifest", str(args.source_case_manifest.resolve()),
                "--source-boundary-resolution-manifest",
                str(args.source_boundary_resolution_manifest.resolve()),
                "--bathymetry-nc", str(args.bathymetry_nc.resolve()),
                "--workspace-root", str(args.workspace_root.resolve()),
                "--output-dir", str(closure_dir),
            ]
            if args.execute:
                command.append("--execute")
            closure_run = _run(command)
            history.append({"stage": "closure", **closure_run})
            if not closure_report.is_file():
                raise RuntimeError("autonomous-thin closure report was not published")
        closure = _read_json(closure_report)
        status_name = str(closure.get("status", "needs_review"))
        decision_files = [str(args.decision.resolve())]

    status = {
        "schema_version": "fvcom_autonomous_thin_workflow_v1",
        "created_at_utc": _utc_now(),
        "status": status_name,
        "profile": "autonomous-thin-v1",
        "input_mesh_sha256": sha256_file(args.mesh),
        "minimal_conditioning_report": str(initial_report),
        "diagnostic": str(diagnostic_path),
        "diagnostic_sha256": sha256_file(diagnostic_path),
        "pending_or_selected_decisions": decision_files,
        "closure": closure,
        "history": history,
        "routine_human_review_gate": False,
    }
    _write_json(output / "workflow_status.json", status)
    print(json.dumps(status, indent=2))
    return 2 if status_name in {"needs_review", "minimal_conditioning_failed"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
