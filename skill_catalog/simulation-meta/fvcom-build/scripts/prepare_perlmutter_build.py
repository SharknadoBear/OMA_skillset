"""Package verified FVCOM source, native parser inputs and Perlmutter CPU build assets."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

import fvcom_build as build
import fvcom_profiles as profiles


def package(source_package, control_nml, combined_nml, river_nml, output, account, qos="debug"):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]+", account) or qos not in {"debug", "regular"}:
        raise ValueError("Require an explicit Perlmutter CPU account and debug or regular QoS")
    if account.endswith("_g"):
        raise ValueError("Perlmutter CPU builds require a CPU allocation")
    source = build.load_json(source_package / "source_revision.json")
    if build.sha256(source_package / "pinned_source.tgz") != source["source_archive_sha256"]:
        raise ValueError("Source tar differs from the source-revision inventory")
    if not re.fullmatch(r"[0-9a-f]{40}", source["source_commit"]):
        raise ValueError("Require the pinned source commit")
    with tarfile.open(source_package / "pinned_source.tgz") as tar:
        blobs = {}
        for item in tar:
            name = PurePosixPath(item.name)
            if not item.isfile() or name.is_absolute() or ".." in name.parts or item.name in blobs:
                raise ValueError("Expected unique, regular relative source files")
            with tar.extractfile(item) as f:
                blobs[item.name] = hashlib.file_digest(f, "sha256").hexdigest()
    inventory = {name: digest for digest, name in
                 (line.split(None, 1) for line in (source_package / "source_inventory_before.sha256").read_text().splitlines())}
    if blobs != inventory or len(blobs) != source["git_blobs"]:
        raise ValueError("Source tar contents differ from the original Git-blob inventory")
    output.parent.mkdir(exist_ok=True, parents=True)
    if output.exists():
        raise FileExistsError(output)
    assets = Path(__file__).resolve().parents[1] / "assets"
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)
        for name in ("source_revision.json", "source_inventory_before.sha256", "pinned_source.tgz", "no_plastic_guard.patch"):
            shutil.copyfile(source_package / name, stage / name)
        for name in ("perlmutter_build.sh", "perlmutter_make.inc", "perlmutter_mpi_netcdf.f90", "fvcom_mp_sign_kind.patch"):
            (stage / name).write_bytes((assets / name).read_bytes().replace(b"\r\n", b"\n"))
        for module in (profiles, build):
            (stage / Path(module.__file__).name).write_bytes(Path(module.__file__).read_bytes().replace(b"\r\n", b"\n"))
        for label, path in (("control", control_nml), ("combined", combined_nml)):
            text = path.read_text(encoding="utf-8-sig")
            build.check_cpu_reader(text, label)
            for key, value in (("INPUT_DIR","../input"),("OUTPUT_DIR","../output"),("OBC_NODE_LIST_FILE","expected_read_end.dat")):
                text, count = re.subn(r"(?mi)^(\s*" + key + r"\s*=\s*)[^\n]*$", lambda m: m[1]+"'"+value+"',", text)
                if count != 1:
                    raise ValueError(f"{label} native reader probe needs one {key}")
            (stage / f"probe_{label}_run.nml").write_text(text,encoding="utf-8",newline="\n")
        (stage / "river.nml").write_bytes(river_nml.read_bytes())
        job = ["#!/bin/bash -l", "#SBATCH --job-name=fvcom_cpu_build", "#SBATCH --account="+account,
               "#SBATCH --constraint=cpu", "#SBATCH --qos="+qos, "#SBATCH --nodes=1", "#SBATCH --ntasks=2",
               "#SBATCH --cpus-per-task=4", "#SBATCH --time=00:30:00", "#SBATCH --output=build_%j.log",
               "set -euo pipefail", 'exec bash "${1:-$SLURM_SUBMIT_DIR}/perlmutter_build.sh"']
        (stage/"build_cpu.sbatch").write_text("\n".join(job)+"\n",encoding="utf8",newline="\n")
        contents=sorted(stage.iterdir())
        (stage/"payload.sha256").write_text("".join(f"{build.sha256(f)}  {f.name}\n" for f in contents),
                                           encoding="ascii",newline="\n")
        with tarfile.open(output,"x:gz") as tf:
            for file in sorted(stage.iterdir()):
                tf.add(file,arcname=file.name,recursive=False)
    return {"sha256":build.sha256(output),"bytes":output.stat().st_size,"source_commit":source["source_commit"],
            "archive":str(output),"status":"prepared","compute_job_submitted":False}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source-package","control-nml","combined-nml","river-nml","output"):
        p.add_argument("--"+name, type=Path,required=True)
    p.add_argument("--account",required=True)
    p.add_argument("--qos",choices=("debug","regular"),default="debug")
    args=p.parse_args()
    print(json.dumps(package(**vars(args)),indent=2))
