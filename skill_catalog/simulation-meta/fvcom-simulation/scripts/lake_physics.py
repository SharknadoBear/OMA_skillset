"""Evaluate same-grid freshwater FVCOM spinup sensitivity and native heat and water budgets."""

import argparse
import csv
import hashlib
import itertools
import json
import re
from functools import lru_cache
from datetime import datetime, timezone, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np

THRESHOLD_KEYS = {
    "volume_weighted_temperature_rms_c": "temperature_3d_rms_C",
    "upper_20m_temperature_rms_c": "upper_20m_rms_C",
    "median_thermocline_abs_difference_m": "median_thermocline_difference_m",
    "lake_mean_elevation_difference_m": "abs_mean_elevation_difference_m",
}
RHO_CP = 1000.0 * 4186.0


@lru_cache(maxsize=256)
def _sha(path, size, mtime_ns):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sha(path):
    stat = path.stat()
    return _sha(path.resolve(), stat.st_size, stat.st_mtime_ns)


def checked(root, obj):
    path = root / obj["path"]
    if sha(path) != obj["sha256"]:
        raise ValueError(f"Expected SHA-256 differs at {path}")
    return path


def utc(value):
    t = datetime.fromisoformat(str(value).replace("/", "-").replace("Z", "+00:00"))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def clocks(ds):
    if "Times" in ds.variables:
        actual = [
            utc(b"".join(row).decode().strip(" \0")) for row in np.ma.asarray(ds["Times"][:]).filled(b" ")
        ]
    else:
        actual = None
    if "Itime" in ds.variables and "Itime2" in ds.variables:
        units = ds["Itime"].units
        if not units.startswith("days since "):
            raise ValueError("Invalid integer origin in native clock")
        origin = utc(units.removeprefix("days since "))
        integer = [
            origin + timedelta(days=int(d), milliseconds=int(ms))
            for d, ms in zip(ds["Itime"][:], ds["Itime2"][:])
        ]
        if actual and actual != integer:
            raise ValueError("Character and integer native UTC clocks disagree")
        return integer
    if actual:
        return actual
    raise ValueError("No precise native UTC clock")


def vals(ds, key, index=slice(None)):
    a = np.ma.asarray(ds[key][index])
    if np.ma.getmaskarray(a).any() or not np.isfinite(a.data).all():
        raise ValueError(f"Missing or nonfinite {key}")
    return np.asarray(a.data)


def nml_value(path, key):
    match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*([^,!\n]+)", path.read_text(), re.M | re.I)
    if not match:
        raise ValueError(f"Missing runtime namelist assignment {key}")
    return match[1].strip().strip("'\x22")


def shared_namelist_sha(path):
    keep = []
    for line in path.read_text().splitlines():
        code = line.split("!", 1)[0].strip()
        if not code:
            continue
        key = code.partition("=")[0].strip().upper()
        if key not in (
            "CASE_TITLE",
            "START_DATE",
            "END_DATE",
            "STARTUP_FILE",
            "RST_FIRST_OUT",
            "NC_FIRST_OUT",
        ):
            keep.append(code)
    return hashlib.sha256("\n".join(keep).encode()).hexdigest()


def setup(root, spec):
    mesh = checked(root, spec["geometry"])
    with nc.Dataset(mesh) as src:
        geom = {k: vals(src, k) for k in ("nv", "h", "lon", "lat", "siglev", "art1")}
    geom["fraction"] = -np.diff(geom["siglev"].astype("f8"), axis=0)
    if (
        min(geom["art1"]) <= 0
        or np.any(geom["fraction"] <= 0)
        or not np.allclose(geom["fraction"].sum(axis=0), 1)
    ):
        raise ValueError("Invalid spherical ART1 or layer thickness")
    return geom


def member(root, record, geom, start, end):
    nml = checked(root, record["namelist"])
    assert_job = record["job"]
    acct = checked(root, assert_job["accounting"]).read_text()
    pattern = (
        r"(?m)^\s*"
        + re.escape(str(assert_job["job_id"]))
        + r"(?:\||\s)+[^\n]*\bCOMPLETED\b[^\n]*(?:\||\s)0:0\b"
    )
    if re.search(pattern, acct, re.I) is None:
        raise ValueError("No successful exact parent Slurm job row with exit 0:0")
    accounting = {}
    rows = csv.DictReader(acct.splitlines(), delimiter="|" if "|" in acct.splitlines()[0] else "\t")
    for row in rows:
        if row.get("JobID") == str(assert_job["job_id"]) and row.get("NNodes") and row.get("ElapsedRaw"):
            wall = int(row["ElapsedRaw"])
            nodes = int(row["NNodes"])
            accounting = dict(
                elapsed_seconds=wall, allocated_nodes=nodes, allocated_node_hours=wall * nodes / 3600.0
            )
    if nml_value(nml, "GRID_FILE_UNITS").lower() != "degrees" or float(nml_value(nml, "IRAMP")) != 0:
        raise ValueError("Budget policy requires native spherical grid and zero forcing ramp")
    run_start = utc(nml_value(nml, "START_DATE"))
    if run_start > start or utc(nml_value(nml, "END_DATE")) < end:
        raise ValueError("Run namelist does not span comparison")
    freeze = json.loads(checked(root, record["freeze"]).read_text())
    binary = freeze.get("executable_sha256") or freeze.get("binary_sha256")
    if not binary:
        raise ValueError("Missing frozen executable identity")
    attempt = json.loads(checked(root, record["attempt"]).read_text())
    submission = json.loads(checked(root, record["job"]["submission"]).read_text())
    if (
        not attempt.get("site")
        or not attempt.get("account")
        or submission.get("account") != attempt["account"]
        or attempt["executable_sha256"] != binary
        or attempt["run_namelist_sha256"] != record["namelist"]["sha256"]
        or str(submission["job_id"]) != str(assert_job["job_id"])
        or attempt["job_sha256"] != submission["script_sha256"]
    ):
        raise ValueError("Site, account, executable, namelist or submitted job script binding differs")
    native = json.loads(checked(root, record["native_audit"]).read_text())
    if (
        native.get("schema") != "fvcom_lake_native_audit_v1"
        or native.get("status") != "verified"
        or native["attempt_sha256"] != record["attempt"]["sha256"]
        or native["executable_sha256"] != binary
        or str(native["job_id"]) != str(assert_job["job_id"])
        or native["startup_sha256"] != attempt["files"][nml_value(nml, "STARTUP_FILE")]
        or utc(native["first_utc"]) != run_start
        or utc(native["last_utc"]) != utc(nml_value(nml, "END_DATE"))
    ):
        raise ValueError("Native run eligibility audit is missing, failed or bound to a different run")
    fields = {}
    for obj in record["outputs"]:
        path = checked(root, obj)
        if native["output_files"].get(path.name, {}).get("sha256") != obj["sha256"]:
            raise ValueError("History stack bytes differ from the native producing job audit")
        with nc.Dataset(path) as ds:
            times = clocks(ds)
            if not any(start <= t <= end for t in times):
                continue
            for k in ("nv", "h", "lon", "lat", "siglev"):
                new, old = vals(ds, k), geom[k]
                if k == "lon":
                    match = np.max(abs((new.astype(float) - old + 180) % 360 - 180)) < 3.2e-5
                elif k == "lat":
                    match = np.max(abs(new - old)) < 4.5e-6
                elif k == "siglev":
                    match = np.allclose(new, old, rtol=0, atol=1e-6)
                else:
                    match = np.array_equal(new, old)
                if not match:
                    raise ValueError(f"Output mesh differs at {k}")
            for i, t in enumerate(times):
                if start <= t <= end:
                    if t in fields:
                        raise ValueError(f"Duplicate model UTC {t}")
                    fields[t] = (path, i)
    expected = [start + timedelta(hours=6 * i) for i in range(int((end - start).total_seconds() / 21600) + 1)]
    if sorted(fields) != expected:
        raise ValueError("Missing or extra native six-hourly clock")
    bindings = {}
    for key, obj in record["forcing"].items():
        path = checked(root, obj)
        if attempt["files"].get(path.name) != obj["sha256"]:
            raise ValueError("Staged forcing differs from the submitted attempt")
        bindings[key] = obj["sha256"]
    if not {"atmosphere", "river", "pe"} <= set(bindings):
        raise ValueError("Missing common forcing identity")
    common = {name: h for name, h in attempt["files"].items() if name != nml_value(nml, "STARTUP_FILE")}
    return dict(
        fields=fields,
        run_start=run_start,
        nml=nml,
        native=native,
        identity=dict(
            job_id=assert_job["job_id"],
            executable_sha256=binary,
            mpi_ranks=attempt["mpi_ranks"],
            common_namelist_sha256=shared_namelist_sha(nml),
            accounting=accounting,
            common_forcing=bindings,
            common_inputs=common,
            native_audit_sha256=record["native_audit"]["sha256"],
            namelist_sha256=record["namelist"]["sha256"],
            output_sha256=[r["sha256"] for r in record["outputs"]],
        ),
    )


def read_record(member, t, geom, extra=()):
    path, i = member["fields"][t]
    with nc.Dataset(path) as ds:
        a = {
            k: vals(ds, k, i)
            for k in ("temp", "salinity", "zeta", "wet_nodes", "wet_cells", "u", "v", *extra)
        }
        a["area"] = vals(ds, "art1").astype("f8")
        if a["area"].shape != geom["art1"].shape or not np.all(a["area"] > 0):
            raise ValueError("Malformed model ART1")
        if any(not np.isin(a[k], [0, 1]).all() for k in ("wet_nodes", "wet_cells")):
            raise ValueError("Invalid wet flags")
        a["D"] = geom["h"].astype("f8") + a["zeta"]
        wet = a["wet_nodes"] == 1
        if not wet.any() or np.min(a["D"][wet]) <= 0 or abs(a["salinity"][:, wet]).max() > 1e-7:
            raise ValueError("Failed freshwater or positive wet thickness")
        if np.min(a["temp"][:, wet]) < -2 or np.max(a["temp"][:, wet]) > 40:
            raise ValueError("Wet temperatures outside liquid freshwater bounds")
        if max(np.max(abs(a["u"])), np.max(abs(a["v"]))) >= 10:
            raise ValueError("Extreme model currents")
    return a


def spherical_cell_area(geom):
    lo, la = np.deg2rad(geom["lon"].astype(float)), np.deg2rad(geom["lat"].astype(float))
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)
    a, b, c = xyz[geom["nv"] - 1]
    det = abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    dot = lambda m, n: np.einsum("ij,ij->i", m, n)
    return 2 * 6371000.0**2 * np.arctan2(det, 1 + dot(a, b) + dot(b, c) + dot(c, a))


def thermocline(t, f, d):
    edges = f * d
    mid = np.cumsum(edges, axis=0) - edges / 2
    differences = (t[:-1] - t[1:]) / np.diff(mid, axis=0)
    layer = np.argmax(differences, axis=0)
    node = np.arange(t.shape[1])
    # Summer thermocline requires warmer surface, positively stratified fresh water
    # and a resolved vertical gradient. Ties choose the shallower interface.
    strat = (d >= 4) & (t[0] >= 4) & (t[0] - t[-1] >= 1) & (differences[layer, node] >= 0.1)
    return (mid[layer, node] + mid[layer + 1, node]) / 2, strat


def weighted_median(x, w):
    order = np.argsort(x)
    center = np.searchsorted(np.cumsum(w[order]), w.sum() / 2)
    return float(x[order[center]])


def compare_pair(a, b, geom, start, end, limits):
    result = []
    face_area = spherical_cell_area(geom)
    nv = geom["nv"] - 1
    for day in range((end - start).days):
        numer = np.zeros(4)
        denom = np.zeros(4)
        elevation = 0.0
        area_time = 0.0
        depths = []
        weights = []
        for step, tw in zip(range(5), (0.5, 1, 1, 1, 0.5)):
            t = start + timedelta(days=day, hours=6 * step)
            x, y = (read_record(m, t, geom) for m in (a, b))
            if not np.array_equal(x["area"], y["area"]):
                raise ValueError("Different native areas at common nodes")
            wet = (x["wet_nodes"] * y["wet_nodes"]).astype(bool)
            d = np.minimum(x["D"], y["D"])[wet]
            ar = x["area"][wet]
            f = geom["fraction"][:, wet]
            h = f * d
            shallow = np.minimum(np.cumsum(h, axis=0), 20.0)
            upper = np.diff(shallow, axis=0, prepend=np.zeros((1, len(d))))
            delta = x["temp"][:, wet].astype(float) - y["temp"][:, wet]
            for i, hk in enumerate((h, upper)):
                vol = ar * hk
                numer[i] += tw * np.sum(vol * delta**2)
                denom[i] += tw * vol.sum()
            z1, s1 = thermocline(x["temp"][:, wet], f, x["D"][wet])
            z2, s2 = thermocline(y["temp"][:, wet], f, y["D"][wet])
            strat = s1 & s2
            if strat.any():
                depths.append(abs(z1[strat] - z2[strat]))
                weights.append(tw * ar[strat])
            elevation += tw * np.dot(ar, x["zeta"][wet].astype(float) - y["zeta"][wet])
            area_time += tw * ar.sum()
            cw = (x["wet_cells"] * y["wet_cells"]).astype(bool)
            nodes = nv[:, cw]
            fdepth = np.minimum(x["D"], y["D"])
            thickness = sum(geom["fraction"][:, v] * fdepth[v] for v in nodes) / 3
            vol = face_area[cw] * thickness
            for j, name in enumerate(("u", "v"), 2):
                numer[j] += tw * np.sum(vol * (x[name][:, cw].astype(float) - y[name][:, cw]) ** 2)
                denom[j] += tw * vol.sum()
        if np.any(denom <= 0) or area_time <= 0:
            raise ValueError("No positive volume on common wet support")
        rms = np.sqrt(numer / denom)
        thermo = weighted_median(np.concatenate(depths), np.concatenate(weights)) if depths else None
        metrics = dict(
            temperature_3d_rms_C=rms[0],
            upper_20m_rms_C=rms[1],
            eastward_rms_m_s=rms[2],
            northward_rms_m_s=rms[3],
            median_thermocline_difference_m=thermo,
            abs_mean_elevation_difference_m=abs(elevation / area_time),
        )
        tests = {k: metrics[k] is not None and bool(metrics[k] <= limit) for k, limit in limits.items()}
        result.append(
            dict(
                start_utc=(start + timedelta(days=day)).isoformat(),
                metrics=metrics,
                jointly_stratified_area_fraction=float(sum(w.sum() for w in weights) / area_time),
                tests=tests,
                passed=all(tests.values()),
            )
        )
    return result


def policy(root, spec):
    plan = json.loads(checked(root, spec["stage_plan"]).read_text())
    start, end = utc(plan["comparison_start_utc"]), utc(plan["comparison_end_utc"])
    if (
        end <= start
        or (end - start).total_seconds() % 86400
        or start.hour
        or start.minute
        or start.second
        or start.microsecond
    ):
        raise ValueError("Spinup comparisons require complete UTC days")
    lengths = []
    for stage in plan["stages"]:
        if stage["name"].startswith("spinup_"):
            delta = start - utc(stage["start_utc"])
            if delta.days <= 0 or delta.seconds or delta.microseconds or utc(stage["end_utc"]) < end:
                raise ValueError(
                    "Spinup trial must start whole days before the common comparison and cover it"
                )
            lengths.append(delta.days)
    if len(lengths) < 2 or len(lengths) != len(set(lengths)):
        raise ValueError("Require at least two distinct preregistered spinup lengths")
    limits = {out: float(plan["convergence"][key]) for key, out in THRESHOLD_KEYS.items()}
    if any(not np.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError("Invalid preregistered convergence threshold")
    return start, end, sorted(lengths), limits


def spinup(root, spec, geom):
    start, end, lengths, limits = policy(root, spec)
    available = {}
    rejected = {}
    for day in lengths:
        try:
            m = member(root, spec["members"][str(day)], geom, start, end)
            if m["run_start"] != start - timedelta(days=day):
                raise ValueError("Wrong spinup initialization date")
            native = m["native"]
            matches = [r for r in native.get("restart_receipts", {}).values() if utc(r["utc"]) == start]
            if len(matches) != 1:
                raise ValueError("Expected one registered restart at the comparison start")
            restart = matches[0]
            basename = Path(restart["native_path"]).name
            evidence = native["output_files"].get(basename, {})
            if (
                restart.get("schema") != "fvcom_lake_restart_v1"
                or restart.get("status") != "verified"
                or str(restart["job_id"]) != str(native["job_id"])
                or restart.get("attempt") != native["attempt"]
                or restart["sha256"] != evidence.get("sha256")
                or evidence.get("count") != 1
                or utc(evidence["first"]) != start
                or utc(evidence["last"]) != start
            ):
                raise ValueError("Comparison restart does not match the producing job and native output")
            checked(root, restart)
            m["comparison_restart"] = restart
            for t in m["fields"]:
                read_record(m, t, geom)
            available[day] = m
        except (KeyError, ValueError, OSError) as err:
            rejected[day] = spec.get("excluded", {}).get(str(day), str(err))
    results = {}
    for left, right in itertools.combinations(sorted(available), 2):
        a, b = available[left], available[right]
        if any(
            a["identity"][k] != b["identity"][k]
            for k in ("executable_sha256", "common_inputs", "common_namelist_sha256", "mpi_ranks")
        ):
            results[f"{left}-{right}"] = dict(
                passed=False, reason="Native binary, common settings, rank count or inputs differ"
            )
            continue
        days = compare_pair(a, b, geom, start, end, limits)
        results[f"{left}-{right}"] = dict(passed=all(v["passed"] for v in days), utc_days=days)
    selected = next(
        (
            day
            for day in lengths[:-1]
            if day in available
            and all(
                results.get(f"{day}-{longer}", {}).get("passed", False) for longer in lengths if longer > day
            )
        ),
        None,
    )
    status = "converged"
    if selected is None:
        selected = lengths[-1] if lengths[-1] in available else None
        status = "advisory_unresolved" if selected else "blocked"
    return dict(
        schema="fvcom_lake_spinup_sensitivity_v1",
        status=status,
        selection_days=selected,
        selected_restart=available[selected]["comparison_restart"] if selected is not None else None,
        thresholds=limits,
        stage_plan_sha256=spec["stage_plan"]["sha256"],
        comparison_start_utc=start.isoformat(),
        comparison_end_utc=end.isoformat(),
        method="24h intervals, trapezoidal 6h time weights; common sigma layers, lesser physical wet thickness, native ART1; thermocline median weighted by jointly stratified area and time",
        thermocline="strongest decrease in temperature per metre between adjacent layer centers, depth midpoint, shallower tie; >=4m water, surface >=4C, surface-bottom >=1C and peak >=0.1C/m in both",
        accepted={day: m["identity"] for day, m in available.items()},
        rejected=rejected,
        pairs=results,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("spinup", "budget"))
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    root = args.spec.parent
    geom = setup(root, spec)
    if args.action == "spinup":
        result = spinup(root, spec, geom)
    else:
        from lake_budget import budget

        result = budget(root, spec, geom)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
