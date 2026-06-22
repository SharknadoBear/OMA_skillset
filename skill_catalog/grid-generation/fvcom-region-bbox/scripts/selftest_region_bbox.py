from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

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
        run(["propose_region_bbox.py", "--request-json", req, "--run-dir", run_dir, "--name", "selftest"])
        cand = run_dir / "selftest_region_box_candidate.json"
        c = json.loads(cand.read_text(encoding="utf-8"))
        assert c["mission_scope_notes"], "Puget tidal-energy prompt should trigger mission-scope notes"
        assert c["side_focus_mode"] == "fast_open_side"
        assert c["side_focus_count"] == 4

        # Missing map visibility blocks pass.
        run(["review_region_bbox.py", "--candidate-json", cand, "--decision", "pass", "--side-review-all-pass"], expect_ok=False)
        # Mission scope blocks pass until explicitly reviewed.
        run(
            [
                "review_region_bbox.py",
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
                "classify_region_domain.py",
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
                "review_region_bbox.py",
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
        final = run_dir / "selftest_region_box.json"
        assert final.exists(), "accepted RegionBox was not written"

        # Full review should generate 12 maps.
        run(["propose_region_bbox.py", "--request-text", "Long Island Sound hypoxia model", "--run-dir", run_dir, "--name", "full", "--full-side-review"])
        full = json.loads((run_dir / "full_region_box_candidate.json").read_text(encoding="utf-8"))
        assert full["side_focus_count"] == 12

    print("fvcom-region-bbox selftest passed")


if __name__ == "__main__":
    main()

