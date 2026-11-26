"""Verify and integrate hourly, rank-distributed FVCOM freshwater COARE26Z ledgers."""
import argparse
from datetime import timedelta
import json
from pathlib import Path

import numpy as np

from lake_physics import nml_value, sha, utc


HEAT = ("area_s", "sensible_J", "latent_J", "lw_J", "sw_J", "net_J", "guarded_net_J",
        "bed_J", "rain_m3", "vapor_m3", "precip_m3", "evap_m3", "latent_from_mass_J",
        "condensation_m3", "pe_heat_J")
EXTERNAL = ("area_s", "rain_m3", "vapor_m3", "condensation_m3", "evap_m3")


def steps_per_hour(dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time step")
    ticks = round(3600 / dt)
    if ticks < 1 or abs(ticks * dt - 3600) > 1.e-7:
        raise ValueError("Time step must divide an hour")
    return ticks


def reduce_ranks(root, tag, columns, ranks, hours, hourly_ticks, absolute_iint, hashes):
    files = sorted(root.glob(f"coare_{tag}_rank_*.dat"))
    rank_ids = [int(p.stem.rsplit("_", 1)[1]) for p in files]
    if rank_ids not in (list(range(ranks)), list(range(1, ranks + 1))):
        raise ValueError(f"Missing or duplicate {tag} rank")
    cumulative = np.zeros((hours, len(columns)))
    ticks = np.arange(1, hours + 1) * hourly_ticks
    for file in files:
        if sha(file) != hashes[file.name]:
            raise ValueError(f"Native ledger digest differs for {file.name}")
        rows = np.loadtxt(file, ndmin=2)
        if (rows.shape != (hours, len(columns) + 2) or not np.isfinite(rows).all()
                or not np.array_equal(rows[:, 0], ticks)
                or not np.array_equal(rows[:, 1], absolute_iint)):
            raise ValueError(f"Invalid hourly ledger or model clock in {file.name}")
        cumulative += rows[:, 2:]
    increments = np.diff(cumulative, axis=0, prepend=np.zeros((1, len(columns))))
    return [dict(zip(columns, row.tolist())) for row in increments]


def validate_hour(h, e):
    if min(h["area_s"], e["area_s"]) <= 0:
        raise ValueError("Nonpositive wet-area exposure")
    for error in (h["net_J"] - sum(h[k] for k in ("sensible_J", "latent_J", "lw_J", "sw_J")),
                  h["latent_J"] - h["latent_from_mass_J"]):
        if abs(error) / h["area_s"] >= .001:
            raise ValueError("Native heat components or latent energy-to-mass conversion disagree")
    for row in (h, e):
        if min(row["rain_m3"], row["condensation_m3"]) < -.001 or row["evap_m3"] > .001:
            raise ValueError("Invalid raw rain, condensation or evaporation sign")
        if abs(row["vapor_m3"] - row["condensation_m3"] - row["evap_m3"]) > max(
                .05, 2.e-6 * (row["condensation_m3"] - row["evap_m3"])):
            raise ValueError("Signed vapor does not sum to condensation and evaporation")
    if abs(h["precip_m3"] - h["rain_m3"] - h["condensation_m3"]) > max(.05, 2.e-6 * h["precip_m3"]):
        raise ValueError("Tracer rain and condensation disagree")


def integrate(root, audit_path):
    audit = json.loads(audit_path.read_bytes())
    attempt_path = root / "attempt_manifest.json"
    attempt = json.loads(attempt_path.read_bytes())
    nml = root / "superior_run.nml"
    if (audit["schema"] != "fvcom_coare_native_audit_v1" or audit["status"] != "verified"
            or sha(attempt_path) != audit["attempt_sha256"] or sha(nml) != attempt["run_namelist_sha256"]
            or audit["executable_sha256"] != attempt["executable_sha256"]):
        raise ValueError("Native audit, executable or namelist differs from the registered attempt")
    start, end = utc(audit["first_utc"]), utc(audit["last_utc"])
    duration = (end - start).total_seconds()
    if duration <= 0 or duration % 3600 or start != utc(nml_value(nml, "START_DATE")) or end != utc(nml_value(nml, "END_DATE")):
        raise ValueError("Ledger interval must match the model and contain whole hours")
    hours = int(duration / 3600)
    ranks = int(attempt["mpi_ranks"])
    if ranks < 1 or len(audit["ledger_sha256"]) != 2 * ranks:
        raise ValueError("Native audit does not cover every heat and volume rank")
    dte = float(nml_value(nml, "EXTSTEP_SECONDS"))
    dti = dte * int(nml_value(nml, "ISPLIT"))
    absolute = audit["initial_iint"] + np.arange(1, hours + 1) * steps_per_hour(dti)
    heat = reduce_ranks(root, "heat", HEAT, ranks, hours, steps_per_hour(dti), absolute, audit["ledger_sha256"])
    ext = reduce_ranks(root, "ext", EXTERNAL, ranks, hours, steps_per_hour(dte), absolute, audit["ledger_sha256"])
    rows = []
    for i, (h, e) in enumerate(zip(heat, ext)):
        validate_hour(h, e)
        rows.append(dict(end_utc=(start + timedelta(hours=i + 1)).isoformat(), absolute_iint=int(absolute[i]),
                         internal=h, external=e))
    return dict(schema="fvcom_lake_coare_hourly_ledger_v1", job_id=audit["job_id"],
                executable_sha256=audit["executable_sha256"], native_audit_sha256=sha(audit_path),
                internal_steps=hours * steps_per_hour(dti), external_steps=hours * steps_per_hour(dte),
                hourly=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--native-audit", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = json.dumps(integrate(args.run, args.native_audit), indent=2, allow_nan=False) + "\n"
    if args.output.exists() and args.output.read_text() != result:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
