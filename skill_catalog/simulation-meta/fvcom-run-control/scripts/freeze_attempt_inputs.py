"""Create or verify the input freeze consumed by the native and paired FVCOM auditors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from fvcom_profiles import resolve_request
from fvcom_run_control import load_data, sha256, verify_stage_forcing


def freeze_attempt(project: Path, attempt: Path) -> dict:
    project = project.resolve()
    attempt = (project / attempt).resolve()
    manifest = load_data(attempt / "attempt_manifest.json")
    grid = manifest["grid_case"]
    if attempt.parent != project / "run" / grid / "attempts":
        raise ValueError("attempt must belong to this project's grid case")
    inputs = project / "input" / grid / "attempts" / attempt.name
    recorded = (project / manifest["input_dir"].replace("\\", "/")).resolve()
    if inputs != recorded or not inputs.is_dir():
        raise ValueError("input directory differs from the attempt manifest")
    entries = list(inputs.iterdir())
    if any(p.is_symlink() or not p.is_file() for p in entries):
        raise ValueError("the immutable input bundle must consist of flat regular files")
    nml, = attempt.glob("*_run.nml")
    nml_hash = sha256(nml)
    if nml_hash != manifest["attempt_namelist_sha256"]:
        raise ValueError("run namelist changed after attempt creation")
    request = resolve_request(load_data(project / "request.json"))
    verify_stage_forcing(request, nml, inputs)
    binding = load_data(project / "run/build/executable_reuse_binding.json")
    executable = binding.get("executable_sha256", "")
    if binding.get("status") != "frozen" or not re.fullmatch(r"[0-9a-f]{64}", executable):
        raise ValueError("the project needs a frozen executable binding")
    files = {p.name: sha256(p) for p in entries}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    freeze = {
        "schema": "fvcom_input_freeze_v1", "input_dir": inputs.relative_to(project).as_posix(),
        "files": files, "input_bundle_sha256": hashlib.sha256(canonical).hexdigest(),
        "station_mapping_sha256": files["station_mapping.json"], "run_namelist_sha256": nml_hash,
        "executable_sha256": executable,
    }
    if manifest.get("skill_commit"):
        freeze["skill_commit"] = manifest["skill_commit"]
    output = attempt / "input_freeze.json"
    if not output.exists():
        # Publish complete bytes exclusively, including when two callers freeze the same attempt.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=attempt,
                                         delete=False) as temporary:
            json.dump(freeze, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        try:
            try:
                os.link(temporary.name, output)
            except FileExistsError:
                pass
        finally:
            Path(temporary.name).unlink()
    saved = load_data(output)
    if any(saved.get(k) != v for k, v in freeze.items()):
        raise ValueError("the existing freeze differs from the current runtime inputs or binding")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=Path,
                        help="Single-stage run directory, absolute or relative to the project")
    args = parser.parse_args()
    print(freeze_attempt(args.project, args.attempt)["input_bundle_sha256"])
