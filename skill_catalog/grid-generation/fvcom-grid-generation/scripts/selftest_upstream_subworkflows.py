#!/usr/bin/env python3
"""Contract tests for the explicit S1/S2/S3 upstream workflow."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation import workflow  # noqa: E402


class _Progress:
    def update(self, *_args, **_kwargs) -> None:
        return None


def test_request_text_invokes_s1_s2_s3_once_in_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "case"
        calls: list[str] = []
        original_find = workflow._find_skill
        original_run = workflow._run
        original_fetch = workflow._fetch_bathy_sources

        def fake_find(skill_name: str, catalog_relative: tuple[str, str]) -> Path:
            del catalog_relative
            return root / "skills" / skill_name

        def fake_run(cmd: list[str], **_kwargs) -> None:
            script = Path(cmd[1]).name
            run_root = Path(cmd[cmd.index("--run-dir") + 1])
            run_root.mkdir(parents=True, exist_ok=True)
            if script == "run_region_bpoly.py":
                calls.append("fvcom-region-bpoly")
                assert cmd.count("--request-text") == 1
                (run_root / "region_bpoly.json").write_text("{}", encoding="utf-8")
                (run_root / "offshore_boundary_artifacts.json").write_text(
                    "{}", encoding="utf-8"
                )
                return
            if script == "run_bdry_arc.py":
                calls.append("fvcom-bdry-arc")
                bpoly = Path(cmd[cmd.index("--region-bpoly-json") + 1])
                offshore = Path(cmd[cmd.index("--offshore-artifacts-json") + 1])
                assert bpoly == run_dir / "upstream" / "region_bpoly" / "region_bpoly.json"
                assert offshore == run_dir / "upstream" / "region_bpoly" / "offshore_boundary_artifacts.json"
                assert bpoly.exists() and offshore.exists()
                loops = run_root / "model_boundary_loops.gpkg"
                resolution = run_root / "boundary_resolution_manifest.json"
                loops.write_bytes(b"test")
                resolution.write_text(
                    json.dumps({"profile": "adaptive-coastal-v2"}),
                    encoding="utf-8",
                )
                (run_root / "bdry_arc_manifest.json").write_text(
                    json.dumps(
                        {
                            "outputs": {
                                "model_boundary_loops_gpkg": str(loops),
                                "boundary_resolution_manifest": str(resolution),
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return
            raise AssertionError(f"unexpected subprocess: {script}")

        def fake_fetch(
            cudem_skill: Path,
            boundary_loops_gpkg: Path,
            bathy_dir: Path,
            name: str,
            config: workflow.GridConfig,
            progress: _Progress,
        ) -> dict[str, str]:
            del name, config, progress
            calls.append("cudem-bathy")
            assert cudem_skill.name == "cudem-bathy"
            assert boundary_loops_gpkg == run_dir / "upstream" / "bdry_arc" / "model_boundary_loops.gpkg"
            bathy_dir.mkdir(parents=True, exist_ok=True)
            bathy = bathy_dir / "bathy.nc"
            bathy.write_bytes(b"test")
            return {"bathy_nc": str(bathy)}

        workflow._find_skill = fake_find
        workflow._run = fake_run
        workflow._fetch_bathy_sources = fake_fetch
        try:
            result = workflow._resolve_upstream_artifacts(
                run_dir,
                "case",
                "Build an FVCOM tidal model for a scientific region.",
                None,
                None,
                None,
                None,
                None,
                None,
                1000.0,
                workflow.GridConfig(mode="test"),
                _Progress(),
            )
        finally:
            workflow._find_skill = original_find
            workflow._run = original_run
            workflow._fetch_bathy_sources = original_fetch

        assert calls == [
            "fvcom-region-bpoly",
            "fvcom-bdry-arc",
            "cudem-bathy",
        ]
        assert calls.count("fvcom-region-bpoly") == 1
        assert result["source"] == "generated_upstream_chain"
        assert Path(result["region_bpoly_json"]).parent.name == "region_bpoly"
        assert Path(result["bdry_arc_manifest"]).parent.name == "bdry_arc"
        assert Path(result["bathy_nc"]).parent.name == "cudem_bathy"


def test_complete_explicit_artifacts_bypass_subworkflows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        loops = root / "loops.gpkg"
        bathy = root / "bathy.nc"
        resolution = root / "resolution.json"
        for path in (loops, bathy, resolution):
            path.write_bytes(b"test")

        original_find = workflow._find_skill

        def fail_find(*_args, **_kwargs) -> Path:
            raise AssertionError("complete explicit artifacts must not invoke a skill")

        workflow._find_skill = fail_find
        try:
            result = workflow._resolve_upstream_artifacts(
                root / "case",
                "case",
                None,
                None,
                None,
                None,
                loops,
                resolution,
                bathy,
                1000.0,
                workflow.GridConfig(mode="test"),
                _Progress(),
            )
        finally:
            workflow._find_skill = original_find

        assert result["source"] == "supplied_artifacts"
        assert result["boundary_loops_gpkg"] == str(loops)
        assert result["boundary_resolution_manifest"] == str(resolution)
        assert result["bathy_nc"] == str(bathy)


def main() -> int:
    test_request_text_invokes_s1_s2_s3_once_in_order()
    test_complete_explicit_artifacts_bypass_subworkflows()
    print("upstream subworkflow selftests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
