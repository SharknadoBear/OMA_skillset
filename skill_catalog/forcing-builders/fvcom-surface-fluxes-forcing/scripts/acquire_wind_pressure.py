"""Route complete hourly 10-m wind and MSLP bundles through the installed data connectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ORDER = ("hrrr", "era5", "cfsv2", "cfsr")
HOUR = timedelta(hours=1)


def iso(t):
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def source_file(path, health, expected_sha, base):
    if digest(path) != expected_sha:
        raise ValueError(f"Provider output changed {path.name}")
    report = read(health)
    if report.get("passed", report.get("pass_all")) is not True:
        raise ValueError("Provider health did not pass")
    manifest = health.parent / "run_manifest.json"
    if not manifest.is_file():
        manifest = health.parent / "source_provider_lock.json"
    return dict(
        path=os.path.relpath(path, base),
        sha256=expected_sha,
        health_path=os.path.relpath(health, base),
        health_sha256=digest(health),
        archive_manifest_path=os.path.relpath(manifest, base),
        archive_manifest_sha256=digest(manifest),
    )


def validate_clock(files, folder, start, end):
    import netCDF4 as nc

    found = []
    for row in files:
        with nc.Dataset(folder / row["path"]) as ds:
            var = ds["time"]
            values = nc.num2date(
                var[:],
                var.units,
                calendar=getattr(var, "calendar", "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
            found.extend(v.replace(tzinfo=timezone.utc) for v in values)
    count = int((end - start) / HOUR) + 1
    if sorted(found) != [start + i * HOUR for i in range(count)]:
        raise ValueError("The chosen connector does not supply every requested hour exactly once")


def hrrr_runs(args, start, end, root):
    import hrrr_core as core

    if not core.runtime_preflight()["passed"]:
        raise ValueError("HRRR connector requires its pinned GRIB runtime")
    files = []
    tick = start
    while tick <= end:
        stop = min(end, tick.replace(hour=23))
        directory = (args.hrrr_cache or root) / tick.strftime("%Y%m%d")
        health = directory / "health_check.json"
        request = dict(
            schema_version="hrrr_request_v1",
            domain="conus",
            mode="analysis",
            products=["wind_10m", "mean_sea_level_pressure"],
            start=iso(tick),
            end=iso(stop),
            bbox=args.bbox,
            halo_cells=2,
            missing_policy="error",
            retain_raw_messages=True,
        )
        if not health.is_file():
            plan = core.build_plan(request, directory)
            core.atomic_write_json(directory / "download_plan.json", plan)
            core.execute_plan(plan, directory)
        saved = read(directory / "request.normalized.json")
        norm = core.normalize_request(request)
        for key in ("domain", "mode", "products", "bbox", "halo_cells", "start", "end"):
            if norm[key] != saved[key]:
                raise ValueError(f"HRRR cache request differs at {key}, use a fresh cache")
        if not core.health_run(directory)["passed"]:
            raise ValueError("HRRR cache failed independent health")
        for row in read(directory / "run_manifest.json")["outputs"]:
            files.append(
                source_file(
                    directory / Path(row["path"].replace("\\", "/")).name,
                    health,
                    row["sha256"],
                    args.output,
                )
            )
        tick = stop + HOUR
    return files


def collect(args, provider, start, end, products=None):
    root = args.output / provider
    root.mkdir(exist_ok=True)
    skill_name = {
        "hrrr": "hrrr-fetcher",
        "era5": "era5-fetcher",
        "cfsv2": "cfsv2-fetcher",
        "cfsr": "cfsr-fetcher",
    }[provider]
    scripts = args.skill_root / skill_name / "scripts"
    sys.path.insert(0, str(scripts))
    if provider == "hrrr":
        return hrrr_runs(args, start, end, root)
    if products is None:
        products = ["wind_10m", "mean_sea_level_pressure"]
    if provider == "era5":
        import era5_core as core

        request = dict(
            schema_version="era5_request_v1",
            start=iso(start),
            end=iso(end + HOUR),
            bbox=args.bbox,
            halo_cells=1,
            products=products,
        )
        path = root / "source_request.json"
        write(path, request)
        core.save_plan(path, root)
        core.run(root)
        if not core.health(root)["pass_all"]:
            raise ValueError("ERA5 failed full-period health")
        rows = read(root / "run_manifest.json")["chunks"].values()
        return [
            source_file(
                root / item["output_path"],
                root / "health_check.json",
                item["output_sha256"],
                args.output,
            )
            for item in rows
            if item.get("state") == "complete"
        ]
    import cfs_grib_core as core

    if core.classify_era(start, end) != [(provider, start, end)]:
        raise ValueError("Outside the provider era for the complete requested period")
    if not core.runtime_preflight()["passed"]:
        raise ValueError("CFS connector requires its pinned GRIB runtime")
    west, south, east, north = args.bbox
    # Native Gaussian support must bracket every point. Include two degrees
    # of source halo on these coarser grids and let the sampler check coverage.
    box = [(west - 2) % 360, max(-90, south - 2), (east + 2) % 360, min(90, north + 2)]
    if box[0] > box[2]:
        raise ValueError("CFS connector cannot cover a wrapped request with one grid")
    request = dict(
        schema_version="cfs_atmospheric_request_v2",
        start=iso(start),
        end=iso(end),
        products=products,
        bbox=box,
        output="cfs_fields.nc",
    )
    res = core.execute_request(
        provider, request, root, scripts / (provider + "_fetcher.py"), no_route=True, open_monitor=False
    )
    if res.get("provider") != "ncei":
        raise ValueError("The CFS MSLP bundle requires native NCEI GRIB")
    return [source_file(Path(res["output"]), root / "health_check.json", res["sha256"], args.output)]


def run(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    if (
        any(t.utcoffset() != timedelta(0) or t.minute or t.second or t.microsecond for t in (start, end))
        or start > end
    ):
        raise ValueError("Expected ordered inclusive whole UTC hours")
    if args.worker:
        files = collect(args, args.worker, start, end)
        if not files:
            raise ValueError("Empty provider output")
        validate_clock(files, args.output, start, end)
        write(
            args.output / args.worker / "candidate.json",
            dict(
                schema="fvcom_wind_pressure_source_v1",
                provider=args.worker,
                start_utc=iso(start),
                end_utc=iso(end),
                files=files,
            ),
        )
        return
    attempts = []
    for name in ORDER:
        base = args.output / name
        base.mkdir(exist_ok=True)
        python = args.cfs_python if name in ("cfsv2", "cfsr") else getattr(args, name + "_python")
        command = [
            str(python or sys.executable),
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker",
            name,
        ]
        log = base / "acquire.log"
        with log.open("w", encoding="utf-8") as file:
            try:
                result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, check=False)
                code = result.returncode
            except OSError as exc:
                file.write(str(exc))
                code = 1
        attempts.append({"provider": name, "exit_code": code, "log": str(log.relative_to(args.output))})
        write(args.output / "provider_attempts.json", attempts)
        if code == 0:
            index = read(base / "candidate.json")
            index["attempts"] = attempts
            write(args.output / "source_index.json", index)
            print(name, str(args.output / "source_index.json"))
            return
    raise SystemExit("No complete provider bundle; see provider_attempts.json and per-provider logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--bbox", type=float, nargs=4, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skill-root",
        type=Path,
        default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "skills",
    )
    parser.add_argument("--hrrr-cache", type=Path, help="Existing per-UTC-day HRRR connector runs")
    for name in ("hrrr", "era5", "cfs"):
        parser.add_argument("--" + name + "-python", type=Path)
    parser.add_argument("--worker", choices=ORDER, help=argparse.SUPPRESS)
    run(parser.parse_args())
