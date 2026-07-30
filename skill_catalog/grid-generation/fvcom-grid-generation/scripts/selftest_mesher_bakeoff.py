#!/usr/bin/env python3
"""Focused synthetic tests for the generator-neutral bakeoff contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.mesher_bakeoff import (  # noqa: E402
    COMMON_CONDITIONED,
    DECLARATION_SCHEMA,
    RAW,
    BakeoffContractError,
    build_run_manifest,
    compare_results,
    default_candidate_matrix,
    execute_stage,
    lock_input_bundle,
    plan_bakeoff,
)


def _expect_code(code: str, callback) -> None:
    try:
        callback()
    except BakeoffContractError as exc:
        assert exc.code == code, (exc.code, code)
    else:
        raise AssertionError(f"expected {code}")


def _writer_command(*, accepted: bool, score: float) -> list[str]:
    code = (
        "import json,sys;"
        "from pathlib import Path;"
        "Path(sys.argv[1]).write_text('synthetic mesh\\n',encoding='utf-8');"
        "Path(sys.argv[2]).write_text(json.dumps({"
        f"'accepted':{accepted!r},"
        f"'failure_taxonomy':{[] if accepted else ['synthetic_gate']!r},"
        f"'metrics':{{'shape_score':{score!r}}}"
        "},sort_keys=True),encoding='utf-8')"
    )
    return [
        sys.executable,
        "-c",
        code,
        "{stage_mesh}",
        "{stage_dir}/quality.json",
    ]


def main() -> None:
    matrix = default_candidate_matrix()
    assert [
        (item["mesher_adapter"], item["algorithm"]["native_code"])
        for item in matrix
    ] == [
        ("clean_room", "production_reference"),
        ("gmsh", 1),
        ("gmsh", 5),
        ("gmsh", 6),
    ]

    with tempfile.TemporaryDirectory(prefix="fvcom_bakeoff_selftest_") as temporary:
        root = Path(temporary)
        boundary = root / "boundary.json"
        bathymetry = root / "bathymetry.nc"
        size_field = root / "size_field.nc"
        boundary.write_text('{"loop":[[0,0],[1,0],[0,1]]}', encoding="utf-8")
        bathymetry.write_bytes(b"synthetic-bathymetry")
        size_field.write_bytes(b"synthetic-size-field-v4")
        bundle = lock_input_bundle(
            case_id="synthetic_case",
            boundary=boundary,
            bathymetry=bathymetry,
            canonical_size_field=size_field,
            projection={"crs": "EPSG:32618", "units": "m"},
            node_budget={
                "preflight_node_threshold": 135_000,
                "hard_node_cap": 150_000,
            },
        )
        declaration = {
            "schema_version": DECLARATION_SCHEMA,
            "input_bundle": bundle,
            "qa_policy": {
                "policy_id": "synthetic_fvcom_quality_v2",
                "accepted_path": "accepted",
                "failure_taxonomy_path": "failure_taxonomy",
                "hard_gates": ["synthetic_gate"],
                "metric_paths": {"shape_score": "metrics.shape_score"},
                "metric_directions": {"shape_score": "maximize"},
            },
            "common_conditioner": {
                "command": _writer_command(accepted=True, score=0.9),
                "artifacts": ["conditioned.2dm", "quality.json"],
                "mesh_artifact": "conditioned.2dm",
                "qa_report": "quality.json",
            },
            "candidates": [
                {
                    "candidate_id": "clean_room_reference",
                    "mesher_adapter": "clean_room",
                    "algorithm": {
                        "id": "production_reference",
                        "native_code": "production_reference",
                    },
                    "raw": {
                        "command": _writer_command(accepted=False, score=0.4),
                        "artifacts": ["raw.2dm", "quality.json"],
                        "mesh_artifact": "raw.2dm",
                        "qa_report": "quality.json",
                    },
                },
                {
                    "candidate_id": "clean_room_two_obc_unsupported",
                    "mesher_adapter": "clean_room",
                    "algorithm": {
                        "id": "production_reference",
                        "native_code": "production_reference",
                    },
                    "capability": {
                        "supported": False,
                        "reason": "adapter supports at most one noncyclic OBC",
                    },
                }
            ],
        }
        run_dir = root / "fresh_run"

        # The pure planner is byte-for-byte deterministic: no timestamps or UUIDs.
        plan_a = build_run_manifest(declaration, run_dir)
        plan_b = build_run_manifest(declaration, run_dir)
        assert json.dumps(plan_a, sort_keys=True) == json.dumps(plan_b, sort_keys=True)
        unsupported_plan = next(
            candidate
            for candidate in plan_a["candidates"]
            if candidate["candidate_id"] == "clean_room_two_obc_unsupported"
        )
        assert unsupported_plan["stages"][RAW]["status"] == "unsupported"
        assert unsupported_plan["stages"][RAW]["artifacts"] == []

        plan_bakeoff(declaration, run_dir)
        manifest_path = run_dir / "run_manifest.json"
        _expect_code("OUTPUT_EXISTS", lambda: plan_bakeoff(declaration, run_dir))

        # Mutation of any locked source rejects execution before creating a stage.
        bathymetry.write_bytes(b"mutated")
        _expect_code(
            "INPUT_HASH_MISMATCH",
            lambda: execute_stage(manifest_path, "clean_room_reference", RAW),
        )
        bathymetry.write_bytes(b"synthetic-bathymetry")

        raw_result = execute_stage(manifest_path, "clean_room_reference", RAW)
        assert raw_result["stage"] == RAW
        assert raw_result["status"] == "needs_review"
        assert raw_result["failure_taxonomy"] == ["synthetic_gate"]
        _expect_code(
            "OUTPUT_EXISTS",
            lambda: execute_stage(manifest_path, "clean_room_reference", RAW),
        )
        unsupported_raw = execute_stage(
            manifest_path, "clean_room_two_obc_unsupported", RAW
        )
        assert unsupported_raw["status"] == "unsupported"
        assert unsupported_raw["failure_taxonomy"] == []
        assert unsupported_raw["command"] == []
        assert unsupported_raw["artifact_hashes"] == []
        assert not (
            run_dir
            / "candidates"
            / "clean_room_two_obc_unsupported"
            / "raw"
            / "stdout.log"
        ).exists()

        conditioned_result = execute_stage(
            manifest_path, "clean_room_reference", COMMON_CONDITIONED
        )
        assert conditioned_result["stage"] == COMMON_CONDITIONED
        assert conditioned_result["status"] == "pass"
        assert conditioned_result["hard_gate_pass"] is True
        assert conditioned_result["conditioner_template_sha256"]
        unsupported_conditioned = execute_stage(
            manifest_path,
            "clean_room_two_obc_unsupported",
            COMMON_CONDITIONED,
        )
        assert unsupported_conditioned["status"] == "unsupported"
        assert unsupported_conditioned["command"] == []

        raw_comparison = compare_results([manifest_path], stage=RAW)
        assert raw_comparison["hard_gate_passers"] == []
        assert raw_comparison["per_metric_ordering"] == {}
        assert raw_comparison["composite_winner"] is None
        unsupported_row = next(
            row
            for row in raw_comparison["per_metric_table"]
            if row["candidate_id"] == "clean_room_two_obc_unsupported"
        )
        assert unsupported_row["status"] == "unsupported"
        assert unsupported_row["capability_reason"]

        conditioned_comparison = compare_results(
            [manifest_path], stage=COMMON_CONDITIONED
        )
        assert conditioned_comparison["hard_gate_passers"] == [
            "clean_room_reference"
        ]
        assert conditioned_comparison["per_metric_ordering"]["shape_score"][0][
            "candidate_id"
        ] == "clean_room_reference"
        assert conditioned_comparison["composite_score"] is None
        assert conditioned_comparison["composite_winner"] is None

    print("mesher bakeoff selftest passed")


if __name__ == "__main__":
    main()
