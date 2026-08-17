#!/usr/bin/env python3
"""Offline end-to-end test for the sequential conditioning campaign driver."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from selftest_portfolio_conditioning import _write_inputs  # noqa: E402
from run_conditioning_campaign import _raw_reference_audit  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="conditioning_campaign_") as temp:
        root = Path(temp)
        inputs = root / "inputs"
        inputs.mkdir()
        mesh, size_field, bathymetry, _lonlat = _write_inputs(
            inputs,
            open_boundary_chains=[],
            open_boundary_ids=[],
        )
        manifest = root / "campaign.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "fvcom_conditioning_campaign_v1",
                    "cases": [
                        {
                            "case_id": "scientific_failure_second",
                            "display_name": "Scientific Failure Second",
                            "difficulty_rank": 2,
                            "mesh": str(mesh),
                            "size_field_nc": str(size_field),
                            "bathymetry_nc": str(bathymetry),
                            "scientific_input_valid": False,
                            "scientific_input_note": "synthetic rejection",
                            "limits": {
                                "wall_time_s": 30,
                                "primary_rounds": 1,
                                "max_valence_repairs_per_round": 12,
                            },
                        },
                        {
                            "case_id": "valid_first",
                            "display_name": "Valid First",
                            "difficulty_rank": 1,
                            "mesh": str(mesh),
                            "size_field_nc": str(size_field),
                            "bathymetry_nc": str(bathymetry),
                        },
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        output = root / "output"
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("run_conditioning_campaign.py")),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--conditioning-profile",
                "auto",
                "--wall-time-s",
                "60",
                "--primary-rounds",
                "1",
                "--no-diagnostics",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        campaign = json.loads(
            (output / "campaign.json").read_text(encoding="utf-8")
        )
        assert campaign["status"] == "complete"
        assert campaign["case_failure_count"] == 0
        assert campaign["minimal_local_debt_closed_count"] == 2
        assert campaign["fvcom_ready_count"] == 1
        assert [value["case_id"] for value in campaign["results"]] == [
            "valid_first",
            "scientific_failure_second",
        ]
        reference = _raw_reference_audit(
            campaign["results"][0]["before_quality"]
        )
        assert reference["node_count"] > 0
        assert reference["triangle_count"] > 0
        assert reference["q_l3_sigma"] is not None
        assert campaign["results"][0]["fvcom_ready"]
        assert not campaign["results"][1]["fvcom_ready"]
        assert campaign["results"][1]["case_limits"]["wall_time_s"] == 30.0
        assert campaign["results"][1]["case_limits"][
            "max_valence_repairs_per_round"
        ] == 12
        assert "scientific_input_invalid" in campaign["results"][1][
            "failure_taxonomy"
        ]
        assert (output / "campaign.csv").is_file()
        with (output / "campaign.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert all(row["nodes_before"] for row in rows)
        assert all(row["nodes_after"] for row in rows)
        assert all(row["triangles_before"] for row in rows)
        assert all(row["triangles_after"] for row in rows)
        report = (output / "REPORT.md").read_text(encoding="utf-8")
        assert "Minimal local closure and full FVCOM readiness" in report
        assert "No composite cross-region score" in report
        assert "Nodes pre->post" in report
        assert "Triangles pre->post" in report
        assert (output / "01_valid_first" / "conditioned" / "conditioned.2dm").is_file()
        assert (
            output
            / "02_scientific_failure_second"
            / "conditioned"
            / "conditioned.2dm"
        ).is_file()
    print("conditioning campaign self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
