"""Exercise project initialization and role handoffs locally without HPC or data access."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fvcom_run_control import evaluate_preparation_join, init_project, render_report, sha256, write_json


class ProjectLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = self.root / "request.json"
        self.project = self.root / "matrix cell" / "case" / "r01"
        self.value = {
            "schema": "simulation_request_v1", "mode": "barotropic_tide",
            "case_id": "layout_test", "project_dir": str(self.project),
            "grid_policy": "accepted_then_fresh",
            "physics": {"formulation": "three_dimensional_barotropic"},
            "tpxo": {"constituents": "all_available"},
            "observations": {"provider": "noaa_coops", "current_policy": "downward_all_bin_profiles_only"},
        }
        write_json(self.request, self.value)

    def test_new_barotropic_handoffs_join_and_appear_in_report(self) -> None:
        manifest = init_project(self.request, self.project)
        self.assertEqual(manifest["local_root"], str(self.project.resolve()))
        self.assertEqual(manifest["agent_layout"]["schema"], "fvcom_barotropic_agent_layout_v1")
        roots = manifest["agent_layout"]["role_roots"]
        self.assertEqual(roots, {
            "lead": "agents/00_lead", "grid": "agents/01_grid",
            "tpxo": "agents/02_tpxo", "configuration_build": "agents/03_configuration_build",
        })
        handoffs = []
        for role, relative_root in roots.items():
            role_root = self.project / relative_root
            self.assertTrue(role_root.is_dir())
            self.assertFalse((role_root / "role_manifest.json").exists())
            if role == "lead":
                continue
            artifact = role_root / "evidence.txt"
            artifact.write_text(f"{role} evidence", encoding="utf-8")
            handoff = role_root / "role_manifest.json"
            write_json(handoff, {
                "schema": "test_worker_v1", "worker": role, "status": "ready",
                "artifact_root": relative_root, "hashes": {"evidence": sha256(artifact)},
                "provenance": {}, "warnings": [], "blocking_reasons": [],
                "resume_token": f"layout_test:{role}:ready",
            })
            handoffs.append(handoff)
        self.assertTrue(evaluate_preparation_join(handoffs)["ready"])
        report = render_report(self.project).read_text(encoding="utf-8")
        for handoff in handoffs:
            self.assertIn(str(handoff.relative_to(self.project)), report)
        for relative_root in ("gridding/fresh_reproduction", "input/accepted_t6v6/base",
                              "forcing/shared_source", "run/build", "output/fresh_reproduction/station"):
            self.assertTrue((self.project / relative_root).is_dir())

    def test_benchmark_does_not_create_barotropic_roles(self) -> None:
        write_json(self.request, dict(self.value, mode="benchmark"))
        manifest = init_project(self.request, self.project)
        self.assertNotIn("agent_layout", manifest)
        self.assertFalse((self.project / "agents").exists())
        self.assertTrue((self.project / "run/accepted_t6v6/benchmark").is_dir())

    def test_reinitialize_preserves_manifest_state_and_role_evidence(self) -> None:
        initial = init_project(self.request, self.project)
        state = self.project / "project_status.json"
        write_json(state, {"state": "parallel_preparation", "history": [{"evidence": "retained"}]})
        evidence = self.project / "agents/01_grid/evidence.txt"
        evidence.write_text("completed grid evidence", encoding="utf-8")
        ledger = self.project / "commands.jsonl"
        ledger.write_text('{"kind":"retained"}\n', encoding="utf-8")
        paths = [self.project / "project_manifest.json", state, evidence, ledger]
        before = {path: path.read_bytes() for path in paths}
        self.assertEqual(init_project(self.request, self.project), initial)
        self.assertEqual({path: path.read_bytes() for path in paths}, before)

    def test_conflicting_request_fails_before_adding_directories(self) -> None:
        self.project.mkdir(parents=True)
        write_json(self.project / "request.json", dict(self.value, case_id="another_trial"))
        before = (self.project / "request.json").read_bytes()
        with self.assertRaises(FileExistsError):
            init_project(self.request, self.project)
        self.assertEqual([path.name for path in self.project.iterdir()], ["request.json"])
        self.assertEqual((self.project / "request.json").read_bytes(), before)

    def test_legacy_handoff_and_manifest_are_preserved(self) -> None:
        legacy = init_project(self.request, self.project)
        del legacy["agent_layout"]
        manifest_path = self.project / "project_manifest.json"
        write_json(manifest_path, legacy)
        handoff = self.project / "gridding/worker_manifest.json"
        write_json(handoff, {"worker": "grid", "status": "ready", "blocking_reasons": []})
        before = {path: path.read_bytes() for path in (manifest_path, handoff)}
        self.assertEqual(init_project(self.request, self.project), legacy)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        report = (self.project / "report.html").read_text(encoding="utf-8")
        self.assertIn(str(handoff.relative_to(self.project)), report)


if __name__ == "__main__":
    unittest.main()
