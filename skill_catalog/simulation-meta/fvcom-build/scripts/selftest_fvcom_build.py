#!/usr/bin/env python3
"""Focused offline tests for fvcom_build.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import fvcom_build as fb


REQUEST = {
    "mode": "barotropic_tide",
    "case_id": "galveston_tide_2025",
    "physics": {"formulation": "three_dimensional_barotropic"},
    "kestrel": {
        "source_dir": "/home/yhuang168/FVCOM_source_agent_test",
        "account": "hindcastra",
        "partition": "auto",
    },
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fvcom-build-test-") as tmp:
        root = Path(tmp)
        make_inc = root / "make.inc"
        flags = " ".join(f"-D{name}" for name in sorted(fb.REQUIRED_DEFINES))
        write(make_inc, f"TOPDIR = /home/yhuang168/FVCOM_source_agent_test\nDFLAGS = {flags}\n")
        audit = fb.audit_make_inc(make_inc, "/home/yhuang168/FVCOM_source_agent_test")
        assert audit["status"] == "pass", audit
        write(make_inc, make_inc.read_text() + "DFLAGS += -DSPHERICAL -DSEDIMENT -DPLASTIC -DOFFLINE_PLASTIC\n")
        bad = fb.audit_make_inc(make_inc, "/home/yhuang168/FVCOM_source_agent_test")
        assert bad["status"] == "fail"
        assert {"SPHERICAL", "SEDIMENT"}.issubset(set(bad["forbidden_active_defines"]))
        assert {"PLASTIC", "OFFLINE_PLASTIC"}.issubset(set(bad["forbidden_active_defines"]))
        write(make_inc, f"TOPDIR = /scratch/pair/source\nDFLAGS = {flags} -DRIVER_FLOAT\n")
        assert "RIVER_FLOAT" in fb.audit_make_inc(make_inc, "/scratch/pair/source")["forbidden_active_defines"]
        write(make_inc, f"TOPDIR = /scratch/pair/source\nDFLAGS = {flags}\n")
        assert fb.audit_make_inc(make_inc, "/scratch/pair/source", True)["missing_required_defines"] == ["AIR_PRESSURE"]
        write(make_inc, make_inc.read_text() + "DFLAGS += -DAIR_PRESSURE\n")
        assert fb.audit_make_inc(make_inc, "/scratch/pair/source", True)["status"] == "pass"
        assert fb.audit_make_inc(make_inc, "/scratch/pair/source")["status"] == "fail"
        write(make_inc, make_inc.read_text() + "DFLAGS += -DATMO_TIDE -DMEAN_FLOW\n")
        assert set(fb.audit_make_inc(make_inc, "/scratch/pair/source", True)["forbidden_active_defines"]) == fb.WIND_INCOMPATIBLE_DEFINES
        pair = {"case_id": "river", "profile": fb.PAIR_PROFILE, "mode": fb.RIVER_MODE,
                "river": {"provider": "usgs", "station_id": "14246900", "parameter_code": "72137"},
                "kestrel": {"pair_root": "/scratch/pair", "project_dir": "/scratch/pair/river",
                            "source_dir": "/scratch/pair/source",
                            "build_environment": {"os": "rh9", "compiler": "mpiifx",
                               "compiler_options": ["-O3"], "modules": ["oneapi/2025.3.1"]}}}
        pair_plan = fb.build_plan(pair, root / "shared", "/scratch/pair/run/build")
        assert pair_plan["source_commit"] == "f8e4b50d4fd51409febcefc763cbc5dca11870e2"
        for mode in ("barotropic_tide", fb.WIND_MODE):
            wind_pair = {key: value for key, value in pair.items() if key != "river"}
            wind_pair.update(profile=fb.WIND_PROFILE, mode=mode)
            wind_plan = fb.build_plan(wind_pair, root / "shared", "/scratch/pair/run/build")
            assert "AIR_PRESSURE" in wind_plan["required_defines"]
            assert "AIR_PRESSURE" not in wind_plan["forbidden_defines"]
        for mode in (fb.RIVER_MODE, fb.COMBINED_MODE):
            factorial = dict(pair, profile=fb.FACTORIAL_PROFILE, mode=mode)
            plan = fb.build_plan(factorial, root / "shared", "/scratch/pair/run/build")
            assert "AIR_PRESSURE" in plan["required_defines"]
            assert "ATMO_TIDE" in plan["forbidden_defines"]
        for key, invalid in (("project_dir", "/scratch/pair_other/river"),
                             ("source_dir", "/scratch/pair/../other/source"),
                             ("pair_root", "/scratch/other")):
            invalid_request = dict(pair, kestrel=dict(pair["kestrel"], **{key: invalid}))
            try:
                fb.build_plan(invalid_request, root / "shared", "/scratch/pair/run/build")
            except fb.ContractError:
                pass
            else:
                raise AssertionError(f"invalid shared build path accepted: {key}")
        plan = fb.build_plan(REQUEST, root / "out", "/scratch/yhuang168/FVCOM_Simulation/galveston_tide_2025/run/build")
        assert plan["source_edits_after_freeze_allowed"] is False
        evidence = root / "evidence"
        evidence.mkdir()
        for name in fb.MANDATORY_EVIDENCE:
            write(evidence / name, "fixture\n")
        good_hash = "a" * 64
        write(evidence / "make_inc_audit.json", json.dumps({"status": "pass", "sha256": "b" * 64}))
        write(evidence / "executable.sha256", f"{good_hash}  fvcom\n")
        write(evidence / "freeze.json", json.dumps({
            "executable_sha256": good_hash,
            "source_edits_after_freeze_allowed": False,
        }))
        write(evidence / "smoke.log", "FVCOM_INPUT_SMOKE_PASS\n")
        write(evidence / "executable.file.txt", "fvcom: ELF 64-bit executable\n")
        write(evidence / "executable.ldd.txt", "libnetcdff.so => /path/libnetcdff.so\n")
        write(evidence / "blank_namelist.nml", "&NML_CASE\n/\n")
        result = fb.finalize(REQUEST, evidence, root / "final")
        assert result["status"] == "ready", result
        freeze = json.loads((evidence / "freeze.json").read_text())
        freeze["executable_sha256"] = "c" * 64
        write(evidence / "freeze.json", json.dumps(freeze))
        failed = fb.finalize(REQUEST, evidence, root / "bad-final")
        assert failed["status"] == "blocked_compile"
        assert any("hashes differ" in item for item in failed["blocking_reasons"])
    print("fvcom-build selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
