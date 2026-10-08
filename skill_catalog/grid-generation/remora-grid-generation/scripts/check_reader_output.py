#!/usr/bin/env python3
"""Verify REMORA initialization output and bind real executable evidence to a grid."""

import argparse
import re
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from remora_grid import sha, read_json, write_json, file_record


def check(grid_path, returned, job_id, output):
    grid_path = Path(grid_path).resolve()
    root = Path(returned).resolve()
    output = Path(output).resolve()
    history = root / "case/reader_initialized_his_d01.nc"
    log = root / f"reader.{job_id}.log"
    accounting = root / "reader_accounting.txt"
    inputs = root / "case/inputs.reader"
    fixture = read_json(root / "case/fixture.json")
    if fixture["grid_sha256"] != sha(grid_path):
        raise ValueError("Reader fixture used a different grid")
    if fixture["inputs_sha256"] != sha(inputs):
        raise ValueError("Reader configuration changed")
    if not re.search(r"^remora.max_step\s*=\s*0\s*$", inputs.read_text(), re.M):
        raise ValueError("Not an initialization-only configuration")
    staged = (root / "case/staged.sha256").read_text()
    if not re.search(
        r"^" + re.escape(sha(grid_path)) + r"\s+remora_grid.nc\s*$", staged, re.M
    ):
        raise ValueError("Staged grid identity not confirmed")
    records = [
        line.split("|") for line in accounting.read_text().splitlines() if line.strip()
    ]
    for wanted in [str(job_id), str(job_id) + ".0"]:
        row = next((r for r in records if r[0] == wanted), None)
        if row is None or row[2:4] != ["COMPLETED", "0:0"]:
            raise ValueError(
                "Missing successful scheduler and executable exit evidence"
            )
    text = log.read_text()
    required = [
        "Loading initial bathymetry from NetCDF file remora_grid.nc",
        "Loading grid variables from NetCDF file remora_grid.nc",
        "Loading masks from NetCDF file remora_grid.nc",
        "Loading initial coriolis from NetCDF file remora_grid.nc",
        "AMReX (",
    ]
    if not all(x in text for x in required) or "finalized" not in text:
        raise ValueError("Grid ingestion or finalization missing from log")
    expected = (root / "initialized.sha256").read_text().split()[0]
    if sha(history) != expected:
        raise ValueError("Returned model output hash mismatch")
    fields = {}
    with Dataset(grid_path) as src, Dataset(history) as dst:
        if dst["ocean_time"].shape != (1,) or float(dst["ocean_time"][0]) != 0:
            raise ValueError("Unexpected model output times")
        for name in [
            "h",
            "pm",
            "pn",
            "f",
            "mask_rho",
            "mask_u",
            "mask_v",
            "x_rho",
            "y_rho",
            "x_u",
            "y_u",
            "x_v",
            "y_v",
            "x_psi",
            "y_psi",
            "s_rho",
            "s_w",
            "Cs_r",
            "Cs_w",
            "theta_s",
            "theta_b",
            "hc",
        ]:
            x = np.asarray(src[name][:])
            v = dst[name][:]
            if np.ma.getmaskarray(v).any():
                raise ValueError("Missing initialized field " + name)
            y = np.asarray(v)
            y = y[0] if y.ndim == x.ndim + 1 else y
            if x.shape != y.shape or not np.isfinite(y).all():
                raise ValueError("Invalid initialized shape/data " + name)
            full_error = float(np.max(np.abs(x - y)))
            # Model boundary fill extrapolates f in the one-cell outer halo.
            if name == "f":
                x = x[1:-1, 1:-1]
                y = y[1:-1, 1:-1]
            np.testing.assert_allclose(
                y, x, rtol=1e-12, atol=1e-12, err_msg="Reader changed " + name
            )
            fields[name] = {
                "max_abs_error_compared": float(np.max(np.abs(x - y))),
                "max_abs_error_full_array": full_error,
                "comparison": (
                    "physical interior; extrapolated halo reported separately"
                    if name == "f"
                    else "full field"
                ),
            }
    evidence = [
        history,
        log,
        accounting,
        inputs,
        root / "case/fixture.json",
        root / "case/staged.sha256",
        root / "source_revision.txt",
        root / "submodules.txt",
        root / "modules.txt",
        root / "executable.sha256",
        root / "initialized.sha256",
    ]
    receipt = {
        "schema": "remora_reader_check_v1",
        "status": "pass",
        "grid_sha256": sha(grid_path),
        "job_id": str(job_id),
        "source_revision": (root / "source_revision.txt").read_text().strip(),
        "executable_sha256": (root / "executable.sha256").read_text().split()[0],
        "scope": "One MPI rank, one grid level, initialization only, zero time steps; synthetic constant IC and wall boundaries",
        "comparisons": fields,
        "evidence": [file_record(p, output.parent) for p in evidence],
        "limitations": [
            "No time-integration or nesting validation",
            "Layer depths were checked locally; executable comparison includes its stretching curves and parameters",
            "Coriolis halo is extrapolated by model boundary fill; physical interior agrees exactly",
        ],
    }
    write_json(output, receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", required=True)
    p.add_argument("--returned-dir", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    r = check(a.grid, a.returned_dir, a.job_id, a.output)
    print(
        "Reader verification: "
        + r["status"]
        + "; "
        + str(len(r["comparisons"]))
        + " fields compared"
    )
