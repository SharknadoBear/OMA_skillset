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
        write(make_inc, make_inc.read_text() + "DFLAGS += -DSPHERICAL -DSEDIMENT\n")
        bad = fb.audit_make_inc(make_inc, "/home/yhuang168/FVCOM_source_agent_test")
        assert bad["status"] == "fail"
        assert {"SPHERICAL", "SEDIMENT"}.issubset(set(bad["forbidden_active_defines"]))
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
