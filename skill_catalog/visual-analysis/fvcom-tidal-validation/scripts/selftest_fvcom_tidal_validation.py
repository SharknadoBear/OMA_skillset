from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fvcom_tidal_validation import FREQUENCIES_CPH, validate


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        t = pd.date_range("2025-04-01", periods=30 * 24 * 10, freq="6min", tz="UTC")
        h = np.arange(len(t)) / 10.0
        eta = 0.8 * np.cos(2 * np.pi * FREQUENCIES_CPH["M2"] * h - np.deg2rad(25))
        water = pd.DataFrame({"time": t, "model": eta + 1.5, "observed": eta + 0.4, "predicted": eta + 0.2})
        water_path = root / "wl001.csv"; water.to_csv(water_path, index=False)
        u = 0.5 * np.cos(2 * np.pi * FREQUENCIES_CPH["M2"] * h)
        v = 0.2 * np.sin(2 * np.pi * FREQUENCIES_CPH["M2"] * h)
        current = pd.DataFrame({"time": t, "model_u": u, "model_v": v, "observed_u": u, "observed_v": v})
        current_path = root / "g06010.csv"; current.to_csv(current_path, index=False)
        lineage_path = root / "lineage.json"
        lineage_path.write_text(json.dumps({
            "executable_sha256": "a" * 64, "input_bundle_sha256": "b" * 64,
            "model_output_sha256": "c" * 64, "observation_manifest_sha256": "d" * 64,
            "station_mapping_sha256": "e" * 64,
        }), encoding="utf-8")
        inventory_path = root / "station_inventory.json"
        inventory_path.write_text(json.dumps({
            "counts": {"water_level_envelope": 2, "water_level_strict_wet": 1,
                       "current_envelope": 2, "current_downward_strict_wet": 1},
            "stations": [
                {"id": "wl001", "role": "water_level", "eligible": True, "exclusion_reason": None},
                {"id": "g06010", "role": "current", "eligible": True, "exclusion_reason": None},
                {"id": "wl002", "role": "water_level", "eligible": False,
                 "exclusion_reason": "outside_actual_wet_mesh"},
                {"id": "gside", "role": "current", "eligible": False,
                 "exclusion_reason": "orientation_side_not_depth_integrated"},
            ],
        }), encoding="utf-8")
        result = validate([water_path], [current_path], root / "analysis", root / "analysis/validation_report.html", list(FREQUENCIES_CPH), lineage_path, inventory_path)
        assert result["workflow_status"] == "validation_complete"
        m = result["water_level"][0]["metrics"]["predicted"]
        assert m["centered_rmse"] < 1e-10 and abs(m["bias"] - 1.3) < 1e-10
        m2 = result["water_level"][0]["harmonics"]["predicted"]["model"]["M2"]
        assert abs(m2["amplitude"] - 0.8) < 1e-10
        assert abs(m2["phase_deg"] - 25.0) < 1e-8
        assert result["current"][0]["metrics"]["vector_rmse"] < 1e-12
        ellipse = result["current"][0]["model_ellipses"]["M2"]
        assert abs(ellipse["major"] - 0.5) < 1e-10
        assert abs(ellipse["minor"] - 0.2) < 1e-10
        unresolved = {(x["constituent"], x["nearest"]) for x in result["water_level"][0]["unresolved"]}
        assert ("K1", "P1") in unresolved or ("P1", "K1") in unresolved
        assert (root / "analysis/validation_report.html").exists()
        assert result["scientific_assessment"] == "diagnostic-pass"
        report_text = (root / "analysis/validation_report.html").read_text(encoding="utf-8")
        assert "src='wl001_water.png'" in report_text and "analysis/analysis" not in report_text
        assert "Resolvable harmonics" in report_text and "Frozen lineage" in report_text
        assert "outside_actual_wet_mesh" in report_text and "orientation_side_not_depth_integrated" in report_text
        incomplete = validate([water_path], [], root / "incomplete", root / "incomplete.html",
                              list(FREQUENCIES_CPH), lineage_path, inventory_path)
        assert incomplete["workflow_status"] == "invalid"
        assert incomplete["station_policy_evidence"]["missing_eligible_stations"] == [
            {"station_id": "g06010", "role": "current"}
        ]
        inventory_path.write_text(json.dumps({'stations': []}), encoding='utf-8')
        empty = validate([], [], root / 'empty', root / 'empty.html',
                         list(FREQUENCIES_CPH), lineage_path, inventory_path)
        assert empty['workflow_status'] == 'invalid', 'No station coverage cannot complete validation'
        inventory_path.write_text(json.dumps({'stations': [
            {'id':'wl001', 'role':'water_level', 'eligible':True},
            {'id':'historic', 'role':'current', 'eligible':False,
             'exclusion_reason':'no_deployment_overlap_requested_period'}
        ]}), encoding='utf-8')
        water_only = validate([water_path], [], root / 'water_only', root / 'water_only.html',
                              list(FREQUENCIES_CPH), lineage_path, inventory_path)
        assert water_only['workflow_status'] == 'validation_complete'
        assert 'Galveston' not in (root / 'water_only.html').read_text()
        print(json.dumps({"status": "pass", "unresolved_count": len(unresolved)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
