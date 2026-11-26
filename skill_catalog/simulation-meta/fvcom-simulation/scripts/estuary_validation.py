"""Validate a returned Columbia salinity pilot against frozen sensor and SSCOFS inputs."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def verified(path, expected):
    value = digest(path)
    if value != expected:
        raise ValueError(f"Changed input: {path}")
    return value


def run(command, *arguments):
    script = Path(__file__).with_name(command + ".py")
    subprocess.run(
        [sys.executable, str(script), *(str(a) for a in arguments)], check=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("attempt", "ofs-manifest", "channels", "sensors", "obc", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--internal-step-seconds", type=float, required=True)
    parser.add_argument("--crs", default="EPSG:32610")
    a = parser.parse_args()
    start = np.datetime64(a.start.rstrip("Z"), "ms")
    end = np.datetime64(a.end.rstrip("Z"), "ms")
    if end - start != np.timedelta64(24, "h"):
        raise ValueError("Expected a 24-hour Columbia smoke test")
    auditpath = a.attempt / "native_audit.json"
    audit = read(auditpath)
    if (
        audit.get("status") != "verified"
        or audit.get("station_records") != 241
        or audit.get("grid_records") != 25
        or audit.get("terminal_seconds") != 86400
    ):
        raise ValueError("Expected successful one-day native execution audit")
    native = audit["closed_output_sha256"]
    for key, value in native.items():
        verified(a.attempt / key, value)
    verified(a.obc, audit["inputs"]["columbia_obc.dat"])
    refs = read(a.ofs_manifest)
    files = []
    times = []
    if not refs.get("complete") or refs.get("failure_count") != 0:
        raise ValueError("Incomplete SSCOFS acquisition")
    wanted = np.arange(
        start + np.timedelta64(1, "h"),
        end + np.timedelta64(1, "h"),
        np.timedelta64(1, "h"),
    )
    for record in refs["records"]:
        clock = np.datetime64(record["valid_time_utc"].rstrip("Z"), "ms")
        if clock not in wanted:
            continue
        if record["guidance"] != "nowcast" or record["product"] != "fields":
            raise ValueError("Expected native SSCOFS nowcast fields")
        path = Path(record["local_path"])
        verified(path, record["sha256"])
        files.append(path)
        times.append(clock)
    if not np.array_equal(sorted(times), wanted):
        raise ValueError("SSCOFS lacks unique post-start hourly records")
    binding = {
        "native_audit_sha256": digest(auditpath),
        "ofs_manifest_sha256": digest(a.ofs_manifest),
        "channels_sha256": digest(a.channels),
        "sensor_spec_sha256": digest(a.sensors),
        "observations": {
            row["id"]: digest(a.sensors.parent / row["csv"]) for row in read(a.sensors)
        },
        "startup_utc": str(start),
        "end_utc": str(end),
        "internal_step_seconds": a.internal_step_seconds,
        "mesh_crs": a.crs,
        "validator_hashes": {
            f.name: digest(f) for f in Path(__file__).parent.glob("estuary_*.py")
        },
    }
    receipt = "validation_receipt.json"
    if a.output.exists():
        prior = read(a.output / receipt)
        if prior["binding"] != binding:
            raise ValueError(
                "Validation already exists with different inputs or code; select a new output directory"
            )
        for k, v in prior["outputs"].items():
            verified(a.output / k, v)
        return
    a.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=a.output.parent) as tmp:
        out = Path(tmp) / "results"
        out.mkdir()
        grid = a.attempt / "columbia_0001.nc"
        run(
            "estuary_sections",
            "--files",
            grid,
            "--channels",
            a.channels,
            "--crs",
            a.crs,
            "--output",
            out / "model_sections",
        )
        run(
            "estuary_sections",
            "--files",
            *files,
            "--channels",
            a.channels,
            "--output",
            out / "ofs_sections",
        )
        run(
            "estuary_comparison",
            "--model",
            out / "model_sections",
            "--reference",
            out / "ofs_sections",
            "--start",
            str(start),
            "--end",
            str(end),
            "--output",
            out,
        )
        run(
            "estuary_gauges",
            "--stations",
            a.attempt / "columbia_station_timeseries.nc",
            "--startup-utc",
            str(start),
            "--internal-step-seconds",
            a.internal_step_seconds,
            "--sensors",
            a.sensors,
            "--output",
            out,
        )
        run(
            "estuary_budget",
            "--grid",
            grid,
            "--obc",
            a.obc,
            "--ledgers",
            a.attempt,
            "--ranks",
            52,
            "--internal-step-seconds",
            a.internal_step_seconds,
            "--output",
            out,
        )
        gauges = read(out / "gauge_metrics.json")
        if any(gauges[row["id"]]["count"] == 0 for row in read(a.sensors)):
            raise ValueError("At least one sensor lacks a valid observation/model pair")
        bounds = np.asarray(
            [row["wet_salinity_range"] for row in audit["history"]], dtype=float
        )
        if bounds.shape != (25, 2) or not np.isfinite(bounds).all():
            raise ValueError("Missing wet-layer salinity extrema in native audit")
        result = {
            "workflow_status": "validation_complete",
            "scientific_assessment": "diagnostic-advisory",
            "assessment_basis": "One model day with a common initial state; gauge errors and the independent physical exchange residual are reported without a calibrated forecast acceptance threshold.",
            "wet_salinity_screen": {
                "accepted_range_psu": [-0.001, 42.0],
                "observed_model_range_psu": [
                    float(bounds[:, 0].min()),
                    float(bounds[:, 1].max()),
                ],
                "failed_clocks": int(
                    np.count_nonzero((bounds[:, 0] < -0.001) | (bounds[:, 1] > 42.0))
                ),
            },
            "job_id": audit["job_id"],
            "binding": binding,
            "outputs": {
                f.relative_to(out).as_posix(): digest(f)
                for f in out.rglob("*")
                if f.is_file()
            },
        }
        (out / receipt).write_text(json.dumps(result, indent=2))
        out.rename(a.output)


if __name__ == "__main__":
    main()
