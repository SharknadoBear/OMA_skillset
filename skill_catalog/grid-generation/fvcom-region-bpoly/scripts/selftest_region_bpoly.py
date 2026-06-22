from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from region_bbox.geometry import RegionBPoly

ROOT = Path(__file__).resolve().parent


def run(cmd, expect_ok=True):
    p = subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, text=True, capture_output=True)
    if expect_ok and p.returncode != 0:
        raise AssertionError(f"command failed: {cmd}\nSTDOUT={p.stdout}\nSTDERR={p.stderr}")
    if not expect_ok and p.returncode == 0:
        raise AssertionError(f"command unexpectedly passed: {cmd}")
    return p


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        req = run_dir / "request.json"
        req.write_text(json.dumps({"request": "Puget Sound tidal energy assessment for all tidal channels"}), encoding="utf-8")
        run(["propose_region_bpoly.py", "--request-json", req, "--run-dir", run_dir, "--name", "selftest", "--basemap-provider", "none"])
        cand = run_dir / "selftest_region_bpoly_candidate.json"
        c = json.loads(cand.read_text(encoding="utf-8"))
        assert c["region_bpoly"]["object_type"] == "RegionBPoly"
        assert len(c["region_bpoly"]["polygon_lonlat"]) == 5
        assert c["region_bpoly"]["envelope_bbox"]
        assert c["mission_scope_notes"], "Puget tidal-energy prompt should trigger mission-scope notes"
        assert c["side_focus_mode"] == "fast_open_side"
        assert c["side_focus_count"] == 4
        assert c["ingredient_coverage"]["all_required_inside"], c["ingredient_coverage"]["missing_required_ids"]

        run(["review_region_bpoly.py", "--candidate-json", cand, "--decision", "pass", "--side-review-all-pass"], expect_ok=False)
        run(
            [
                "review_region_bpoly.py",
                "--candidate-json",
                cand,
                "--decision",
                "pass",
                "--map-visibility-status",
                "pass",
                "--side-review-all-pass",
            ],
            expect_ok=False,
        )
        run(
            [
                "classify_region_bpoly_domain.py",
                "--candidate-json",
                cand,
                "--domain-type",
                "coastal",
                "--open-boundary-reference",
                "-125.0",
                "48.3",
            ]
        )
        note = run_dir / "selftest_domain_type_note.json"
        run(
            [
                "review_region_bpoly.py",
                "--candidate-json",
                cand,
                "--decision",
                "pass",
                "--domain-type-note-json",
                note,
                "--map-visibility-status",
                "pass",
                "--mission-scope-status",
                "pass",
                "--side-review-all-pass",
                "--single-open-boundary-status",
                "pass",
            ]
        )
        assert (run_dir / "selftest_region_bpoly.json").exists()
        assert (run_dir / "region_bpoly.json").exists()

        aleut = RegionBPoly([[172.0, 48.9], [-162.0, 49.9], [-161.5, 57.6], [172.0, 56.7]], 172.0)
        assert aleut.crosses_antimeridian()
        assert aleut.contains_lonlat(179.0, 52.0)
        assert aleut.contains_lonlat(-170.0, 53.0)

        exec_dir = run_dir / "execute_case"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Murderkill River DE small estuary salinity intrusion",
                "--run-dir",
                exec_dir,
                "--name",
                "execute_case",
                "--basemap-provider",
                "none",
            ]
        )
        final = json.loads((exec_dir / "region_bpoly.json").read_text(encoding="utf-8"))
        assert final["mode"] == "execute"
        assert final["final_status"] == "pass"
        assert (exec_dir / "region_bpoly_final_map.png").exists()
        assert not (exec_dir / "intermediate").exists()

        test_dir = run_dir / "test_case"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Murderkill River DE small estuary salinity intrusion",
                "--run-dir",
                test_dir,
                "--name",
                "test_case",
                "--mode",
                "test",
                "--basemap-provider",
                "none",
            ]
        )
        test_final = json.loads((test_dir / "region_bpoly.json").read_text(encoding="utf-8"))
        assert test_final["mode"] == "test"
        assert (test_dir / "intermediate").exists()

    print("fvcom-region-bpoly selftest passed")


if __name__ == "__main__":
    main()
