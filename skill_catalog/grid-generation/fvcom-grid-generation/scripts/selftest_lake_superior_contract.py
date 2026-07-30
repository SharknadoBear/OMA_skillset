#!/usr/bin/env python3
"""Focused tests for Lake Superior readiness binding and immutable outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Callable


SCRIPTS = Path(__file__).resolve().parent
RESEARCH_GMSH = SCRIPTS / "research" / "gmsh"
for path in (SCRIPTS, RESEARCH_GMSH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fvcom_grid_generation.gmsh_experiment import (
    assert_readiness_manifest_binding,
    reject_protected_case_input_overrides,
    validate_readiness_artifact,
)
import prepare_lake_superior_bathymetry as bathymetry_preparation
import prepare_lake_superior_boundary as boundary_preparation
import validate_lake_superior_preparation as preparation_validation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expect_raises(
    error_type: type[BaseException],
    action: Callable[[], object],
    message_fragment: str,
) -> None:
    try:
        action()
    except error_type as exc:
        assert message_fragment.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError(f"Expected {error_type.__name__} was not raised")


def _readiness_fixture(root: Path) -> tuple[dict, dict[str, Path], Path]:
    active_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    for name in ("boundary_manifest", "boundary_gpkg", "bathymetry_netcdf"):
        path = root / name
        path.write_bytes(f"immutable-{name}".encode("utf-8"))
        active_paths[name] = path
        artifact_hashes[name] = {"path": str(path), "sha256": _sha256(path)}
    payload = {
        "schema_version": "lake_superior_preparation_readiness_v1",
        "status": "ready",
        "case_id": "lake_superior",
        "checks": {
            "domain_is_valid": True,
            "bathymetry_all_wet_cells_finite": True,
        },
        "failed_checks": [],
        "artifact_hashes": artifact_hashes,
    }
    artifact_path = root / "readiness.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "case_id": "lake_superior",
        "readiness": {
            "validation_artifact": {
                "path": artifact_path.name,
                "sha256": _sha256(artifact_path),
                "schema_version": (
                    "lake_superior_preparation_readiness_v1"
                ),
                "required_status": "ready",
                "required_checks": [
                    "domain_is_valid",
                    "bathymetry_all_wet_cells_finite",
                ],
                "required_input_hashes": list(active_paths),
            }
        },
    }
    return manifest, active_paths, artifact_path


def test_readiness_artifact_binds_report_and_active_inputs() -> None:
    with tempfile.TemporaryDirectory(
        prefix="lake_superior_readiness_contract_"
    ) as temporary:
        root = Path(temporary)
        manifest, active_paths, artifact_path = _readiness_fixture(root)
        report, resolved = validate_readiness_artifact(
            manifest,
            root,
            active_paths,
        )
        assert resolved == artifact_path
        assert report is not None and report["passed"]
        assert report["blockers"] == []

        artifact_path.write_text(
            artifact_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        report, _ = validate_readiness_artifact(manifest, root, active_paths)
        assert report is not None and not report["passed"]
        assert (
            "readiness_validation_artifact_hash_mismatch"
            in report["blockers"]
        )


def test_readiness_artifact_rejects_failed_check_and_stale_input() -> None:
    with tempfile.TemporaryDirectory(
        prefix="lake_superior_stale_contract_"
    ) as temporary:
        root = Path(temporary)
        manifest, active_paths, artifact_path = _readiness_fixture(root)
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload["checks"]["domain_is_valid"] = False
        payload["failed_checks"] = ["domain_is_valid"]
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")
        manifest["readiness"]["validation_artifact"]["sha256"] = _sha256(
            artifact_path
        )
        report, _ = validate_readiness_artifact(manifest, root, active_paths)
        assert report is not None and not report["passed"]
        assert (
            "readiness_validation_required_check_failed:domain_is_valid"
            in report["blockers"]
        )
        assert "readiness_validation_artifact_has_failed_checks" in report[
            "blockers"
        ]

        manifest, active_paths, _ = _readiness_fixture(root)
        active_paths["bathymetry_netcdf"].write_bytes(b"changed-bathymetry")
        report, _ = validate_readiness_artifact(manifest, root, active_paths)
        assert report is not None and not report["passed"]
        assert (
            "readiness_validation_evidence_input_hash_mismatch:"
            "bathymetry_netcdf"
        ) in report["blockers"]


def test_lake_superior_readiness_contract_is_mandatory_and_complete() -> None:
    with tempfile.TemporaryDirectory(
        prefix="lake_superior_required_contract_"
    ) as temporary:
        root = Path(temporary)
        report, resolved = validate_readiness_artifact(
            {"case_id": "lake_superior"},
            root,
            {},
        )
        assert resolved is None
        assert report is not None and not report["passed"]
        assert report["blockers"] == [
            "readiness_validation_artifact_contract_required"
        ]

        manifest, active_paths, _ = _readiness_fixture(root)
        contract = manifest["readiness"]["validation_artifact"]
        contract["schema_version"] = None
        contract["required_status"] = None
        contract["required_checks"] = []
        contract["required_input_hashes"] = []
        report, _ = validate_readiness_artifact(
            manifest,
            root,
            active_paths,
        )
        assert report is not None and not report["passed"]
        for blocker in (
            "readiness_validation_contract_schema_invalid",
            "readiness_validation_contract_status_invalid",
            "readiness_validation_contract_checks_invalid",
            "readiness_validation_contract_input_hashes_invalid",
        ):
            assert blocker in report["blockers"]


def test_lake_superior_rejects_explicit_input_overrides() -> None:
    manifest = {"case_id": "lake_superior"}
    for name in (
        "bathymetry_override",
        "boundary_loop_override",
        "adaptive_resolution_override",
    ):
        values = {name: "replacement-input"}
        _expect_raises(
            ValueError,
            lambda values=values: reject_protected_case_input_overrides(
                manifest,
                **values,
            ),
            "fresh bound case manifest",
        )

    reject_protected_case_input_overrides({"case_id": "delaware_bay"})
    reject_protected_case_input_overrides(
        {"case_id": "delaware_bay"},
        bathymetry_override="research-override.nc",
    )


def test_readiness_hash_binds_the_prepared_manifest() -> None:
    readiness = {
        "input_hashes": {
            "case_manifest": {"sha256": "a" * 64}
        }
    }
    assert_readiness_manifest_binding(readiness, "a" * 64)
    _expect_raises(
        RuntimeError,
        lambda: assert_readiness_manifest_binding(readiness, "b" * 64),
        "changed between readiness",
    )
    _expect_raises(
        RuntimeError,
        lambda: assert_readiness_manifest_binding({}, "a" * 64),
        "does not bind",
    )


def test_lake_superior_output_guards_reject_existing_products() -> None:
    with tempfile.TemporaryDirectory(
        prefix="lake_superior_fresh_guards_"
    ) as temporary:
        root = Path(temporary)
        boundary_output = root / "boundary"
        boundary_output.mkdir()
        _expect_raises(
            FileExistsError,
            lambda: boundary_preparation._require_fresh_output_directory(
                boundary_output
            ),
            "must not already exist",
        )

        source = root / "source.nc"
        domain = root / "domain.gpkg"
        source.write_bytes(b"source")
        domain.write_bytes(b"domain")
        depth_output = root / "depth.nc"
        metadata_output = root / "depth.json"
        depth_output.write_bytes(b"existing")
        _expect_raises(
            FileExistsError,
            lambda: bathymetry_preparation._require_fresh_output_files(
                source=source,
                domain_gpkg=domain,
                output=depth_output,
                metadata_path=metadata_output,
            ),
            "must not already exist",
        )
        _expect_raises(
            ValueError,
            lambda: bathymetry_preparation._require_fresh_output_files(
                source=source,
                domain_gpkg=domain,
                output=root / "fresh.nc",
                metadata_path=source,
            ),
            "immutable input",
        )

        readiness_output = root / "readiness.json"
        readiness_output.write_text("immutable", encoding="utf-8")
        _expect_raises(
            FileExistsError,
            lambda: preparation_validation._require_fresh_output_file(
                readiness_output
            ),
            "must not already exist",
        )


TESTS: tuple[Callable[[], None], ...] = (
    test_readiness_artifact_binds_report_and_active_inputs,
    test_readiness_artifact_rejects_failed_check_and_stale_input,
    test_lake_superior_readiness_contract_is_mandatory_and_complete,
    test_lake_superior_rejects_explicit_input_overrides,
    test_readiness_hash_binds_the_prepared_manifest,
    test_lake_superior_output_guards_reject_existing_products,
)


def main() -> int:
    failures: list[tuple[str, BaseException]] = []
    for test in TESTS:
        try:
            test()
        except BaseException as exc:
            failures.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    if failures:
        print(f"{len(failures)} of {len(TESTS)} Lake Superior tests failed")
        return 1
    print(f"All {len(TESTS)} Lake Superior contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
