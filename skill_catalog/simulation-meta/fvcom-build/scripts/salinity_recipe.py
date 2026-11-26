"""Package or verify a Cartesian Perlmutter salinity diagnostic against a native parent build."""

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def digest(data):
    return hashlib.sha256(data).hexdigest()


def remote(raw):
    if (
        not re.fullmatch(r"/[A-Za-z0-9._/+-]+", raw)
        or ".." in PurePosixPath(raw).parts
        or raw == "/"
    ):
        raise ValueError("supply an absolute resolved remote path")
    return raw


def prepare(args):
    root = Path(__file__).resolve().parents[1]
    binding = json.loads(args.parent_binding.read_bytes())
    flags = set(
        "AIR_PRESSURE DOUBLE_PRECISION GCN GFORTRAN LIMITED_NO MULTIPROCESSOR SINGLE_OUTPUT WET_DRY".split()
    )
    if (
        binding.get("status") != "frozen"
        or binding.get("execution_site") != "perlmutter"
        or binding.get("source_commit") != "f8e4b50d4fd51409febcefc763cbc5dca11870e2"
        or set(binding["feature_defines"]) != flags
    ):
        raise ValueError(
            "Expected qualified Perlmutter pinned, three-dimensional Cartesian GNU build"
        )
    paths = [
        root / "assets/columbia_salinity_build.sbatch",
        root / "assets/columbia_salinity_ledger.F",
        root / "scripts/columbia_salinity_build.py",
    ]
    files = {p.name: p.read_bytes().replace(b"\r\n", b"\n") for p in paths}
    files.update(
        {
            "parent_binding.json": args.parent_binding.read_bytes(),
            "parent_source.txt": (remote(args.parent_source) + "\n").encode(),
            "parent_evidence.txt": (remote(args.parent_evidence) + "\n").encode(),
        }
    )
    inv = "".join(digest(v) + "  " + k + "\n" for k, v in sorted(files.items()))
    files["payload.sha256"] = inv.encode()
    dest = args.output / ("v_" + digest(inv.encode())[:16])
    dest.mkdir(exist_ok=True, parents=True)
    for name, value in files.items():
        path = dest / name
        if path.exists() and path.read_bytes() != value:
            raise FileExistsError(path)
        path.write_bytes(value)
    return {
        "directory": str(dest),
        "payload_sha256": digest(inv.encode()),
        "submit_script": "columbia_salinity_build.sbatch",
    }


def verify(evidence, accounting):
    get = lambda name: json.loads((evidence / name).read_bytes())
    f = get("freeze.json")
    instrument = get("instrument.json")
    parent = get("parent_binding.json")
    required = {
        "status": "frozen",
        "schema": "fvcom_estuary_salinity_build_v1",
        "execution_site": "perlmutter",
        "source_commit": "f8e4b50d4fd51409febcefc763cbc5dca11870e2",
    }
    if any(f.get(k) != v for k, v in required.items()):
        raise ValueError("Invalid native freeze")
    if not any(
        row.split("|")[:3] == [f["job_id"], "COMPLETED", "0:0"]
        for row in accounting.read_text().splitlines()
    ):
        raise ValueError("Missing completed charged build job")
    for key, name in (
        ("blank_template_sha256", "blank_namelist.nml"),
        ("instrument_sha256", "instrument.json"),
        ("mpi_netcdf_probe_sha256", "mpi_netcdf_probe.log"),
    ):
        if f[key] != digest((evidence / name).read_bytes()):
            raise ValueError("Mismatched native evidence " + name)
    if (
        f["parent_executable_sha256"] != parent["executable_sha256"]
        or f["executable_sha256"]
        != (evidence / "executable.sha256").read_text().split()[0]
        or f["source_commit"] != parent["source_commit"]
        or f["feature_defines"] != parent["feature_defines"]
        or f["build_environment"] != parent["build_environment"]
        or f["ledger_sha256"] != instrument["ledger_sha256"]
        or "instrumented_station_writer_sha256" not in instrument
        or "not found" in (evidence / "ldd.txt").read_text()
        or f["blank_template_sha256"] != parent["blank_template_sha256"]
    ):
        raise ValueError("Salinity instrumentation or native lineage mismatch")
    return f


def cli(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--parent-binding", type=Path, required=True)
    p.add_argument("--parent-source", required=True)
    p.add_argument("--parent-evidence", required=True)
    p.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--evidence", type=Path, required=True)
    v.add_argument("--accounting", type=Path, required=True)
    a = parser.parse_args(argv)
    print(
        json.dumps(
            prepare(a) if a.command == "prepare" else verify(a.evidence, a.accounting),
            indent=2,
        )
    )
    return 0
