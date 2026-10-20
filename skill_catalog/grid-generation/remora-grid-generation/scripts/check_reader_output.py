#!/usr/bin/env python3
"""Verify initialization output with explicit, version-bound halo and GPU checks."""

import argparse
import re
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from remora_grid import sha, read_json, write_json, file_record

PINNED_SOURCE = "8f1f606ece6d6d9fb3fe0954cc500e4c63fb79cc"
INIT_SOURCE_SHA256 = "b1b2f8ab8c01bb93752ddd35eb1ff5b8fcc69cd71f85d57f7308909aee9aa062"
FIELDS = [
    "h", "pm", "pn", "f", "mask_rho", "mask_u", "mask_v",
    "x_rho", "y_rho", "x_u", "y_u", "x_v", "y_v", "x_psi", "y_psi",
    "s_rho", "s_w", "Cs_r", "Cs_w", "theta_s", "theta_b", "hc",
]


def metric_with_outer_ghosts(field, extra=0):
    """Pinned metric extrapolation, including corners.

    REMORA_init_from_netcdf.cpp, extrapolate_metric_to_physical_boundaries,
    starts at dom_lo-2 and dom_hi+2 (lines 1240-1266). Thus the input rho
    halo at dom_lo-1/dom_hi+1 survives unchanged. Only deeper, unexported
    AMReX ghosts repeat that first halo. Never replace the first halo with
    the physical interior, even if uniform-grid metrics nearly coincide.
    """
    field = np.asarray(field)
    if field.ndim != 2 or min(field.shape) < 3 or extra < 0:
        raise ValueError("Invalid metric field or ghost count")
    return np.pad(field, extra, mode="edge") if extra else field.copy()


def compare_field(name, x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid initialized shape/data " + name)
    expected = x
    result = {"max_abs_error_full_array": float(np.max(np.abs(x - y)))}
    if name in ("pm", "pn", "f"):
        if x.ndim != 2 or min(x.shape) < 3:
            raise ValueError("Invalid rho shape for " + name)
        np.testing.assert_array_equal(y[1:-1, 1:-1], x[1:-1, 1:-1],
                                      err_msg="Reader changed physical interior " + name)
        halo = np.ones(x.shape, bool)
        halo[1:-1, 1:-1] = False
        if name in ("pm", "pn"):
            expected = metric_with_outer_ghosts(x)
            policy = "one-cell input halo retained exactly; deeper ghosts clamp to first input halo"
        else:
            # Initialized f receives first-order physical-domain extrapolation,
            # unlike pm/pn. Also verified in retained zero-step Delaware output.
            expected = np.pad(x[1:-1, 1:-1], 1, mode="edge")
            policy = "one-cell halo clamps to nearest physical interior, including corners"
        np.testing.assert_array_equal(y[halo], expected[halo],
                                      err_msg="Reader halo disagrees with explicit policy: " + name)
        result.update(physical_interior_exact=True, halo_exact=True, halo_policy=policy,
                      halo_max_abs_error_vs_expected=float(np.max(np.abs(y[halo] - expected[halo]))))
    else:
        np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12,
                                   err_msg="Reader changed " + name)
    result["max_abs_error_compared"] = float(np.max(np.abs(y - expected)))
    result["comparison"] = "full field against explicit initialized expectation"
    return result


def sha_manifest(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+[* ]?(.+)", line.strip())
        if not match:
            raise ValueError("Malformed checksum manifest: " + str(path))
        name = match[2].removeprefix("./")
        if name in result:
            raise ValueError("Duplicate checksum entry " + name)
        result[name] = match[1].lower()
    return result


def _input_values(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = [p.strip() for p in line.split("=", 1)]
        if key in values:
            raise ValueError("Duplicate input key " + key)
        values[key] = value
    for key in ("remora.max_step", "remora.stop_time"):
        if key not in values or float(values[key]) != 0:
            raise ValueError("Not an initialization-only configuration: " + key)
    for key, value in {
        "remora.ic_type": "netcdf", "remora.mask_type": "netcdf",
        "remora.nc_grid_file_0": "remora_grid.nc",
        "remora.nc_init_file_0": "reader_initial.nc",
        "remora.use_curvilinear_grid": "true", "amr.max_level": "0",
    }.items():
        if values.get(key) != value:
            raise ValueError("Unexpected reader input " + key)
    return values


def check(grid_path, returned, job_id, output, *, expected_source_revision=None,
          expected_executable_sha256=None, expected_amrex_revision=None,
          require_gpu=False, execution_receipt=None):
    grid_path, root, output = Path(grid_path).resolve(), Path(returned).resolve(), Path(output).resolve()
    if not re.fullmatch(r"\d+", str(job_id)):
        raise ValueError("Invalid Slurm job ID")
    history = root / "case/reader_initialized_his_d01.nc"
    log = root / f"reader.{job_id}.log"
    accounting = root / "reader_accounting.txt"
    inputs = root / "case/inputs.reader"
    fixture = read_json(root / "case/fixture.json")
    grid_hash = sha(grid_path)
    if fixture["grid_sha256"] != grid_hash or fixture["inputs_sha256"] != sha(inputs):
        raise ValueError("Reader grid/configuration identity mismatch")
    values = _input_values(inputs)
    staged = sha_manifest(root / "case/staged.sha256")
    if staged.get("remora_grid.nc") != grid_hash:
        raise ValueError("Staged grid identity not confirmed")
    for name, expected in (("reader_initial.nc", fixture["initial_sha256"]),
                           ("inputs.reader", fixture["inputs_sha256"])):
        if staged.get(name) != expected:
            raise ValueError("Staged fixture identity not confirmed: " + name)
    for name in ("remora_grid.nc", "reader_initial.nc"):
        path = root / "case" / name
        if path.exists() and sha(path) != staged[name]:
            raise ValueError("Returned fixture changed: " + name)

    source = (root / "source_revision.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Invalid source revision")
    if source != PINNED_SOURCE:
        raise ValueError("Halo policy has not been established for source " + source)
    executable_records = sha_manifest(root / "executable.sha256")
    if len(executable_records) != 1:
        raise ValueError("Expected exactly one executable checksum")
    executable = next(iter(executable_records.values()))
    if expected_source_revision and source != expected_source_revision:
        raise ValueError("Unexpected REMORA source revision")
    if expected_executable_sha256 and executable != expected_executable_sha256.lower():
        raise ValueError("Unexpected executable SHA-256")
    submodules = (root / "submodules.txt").read_text()
    if expected_amrex_revision and not re.search(
            r"^\s*" + re.escape(expected_amrex_revision) + r"\s+Submodules/AMReX(?:\s|$)",
            submodules, re.M):
        raise ValueError("Unexpected AMReX revision")
    records = [line.split("|") for line in accounting.read_text().splitlines() if line.strip()]
    rows = {}
    for wanted in (str(job_id), str(job_id) + ".0"):
        matches = [r for r in records if r[0] == wanted]
        if len(matches) != 1 or matches[0][2:4] != ["COMPLETED", "0:0"]:
            raise ValueError("Missing successful scheduler and executable exit evidence")
        rows[wanted] = matches[0]
    text = log.read_text(errors="replace")
    required = [
        "Loading initial bathymetry from NetCDF file remora_grid.nc",
        "Loading grid variables from NetCDF file remora_grid.nc",
        "Loading masks from NetCDF file remora_grid.nc",
        "Loading initial coriolis from NetCDF file remora_grid.nc", "AMReX (",
    ]
    if not all(x in text for x in required) or "finalized" not in text:
        raise ValueError("Grid ingestion or finalization missing from log")
    if re.search(r"STEP\s*[=:]?\s*[1-9]\d*", text, re.I):
        raise ValueError("Model log records a positive time step")
    output_manifest = sha_manifest(root / "initialized.sha256")
    if len(output_manifest) != 1 or sha(history) != next(iter(output_manifest.values())):
        raise ValueError("Returned model output hash mismatch")

    gpu_verified = False
    extra_evidence = []
    if require_gpu:
        if not expected_source_revision or not expected_executable_sha256 or not expected_amrex_revision:
            raise ValueError("GPU verification requires expected source, AMReX and executable identities")
        for name in ("remora_grid.nc", "reader_initial.nc"):
            if not (root / "case" / name).is_file():
                raise ValueError("GPU receipt requires returned exact fixture file " + name)
        if "Initializing CUDA..." not in text or "CUDA initialized with 1 device." not in text:
            raise ValueError("No positive one-device CUDA runtime evidence")
        job_row = rows[str(job_id)]
        if len(job_row) < 8 or not re.search(r"(?:^|,)gres/gpu(?:[^=,]*)?=1(?:,|$)", job_row[5]):
            raise ValueError("Accounting does not establish a one-GPU allocation")
        devices = root / "gpu_devices.csv"
        if not devices.exists() or not re.search(r"GPU-[0-9a-fA-F-]+", devices.read_text()):
            raise ValueError("Missing GPU device inventory")
        extra_evidence.append(devices)
        gpu_verified = True
    if execution_receipt:
        receipt_path = Path(execution_receipt)
        execution = read_json(receipt_path)
        for key, expected in {"job_id": str(job_id), "grid_sha256": grid_hash,
                              "source_revision": source, "executable_sha256": executable,
                              "history_sha256": sha(history), "max_step": 0,
                              "model_exit_status": 0}.items():
            if execution.get(key) != expected:
                raise ValueError("Execution receipt mismatch: " + key)
        extra_evidence.append(receipt_path)

    fields = {}
    with Dataset(grid_path) as src, Dataset(history) as dst:
        n_cell = [len(src.dimensions["xi_rho"])-2, len(src.dimensions["eta_rho"])-2,
                  len(src.dimensions["s_rho"])]
        if fixture["n_cell"] != n_cell or list(map(int, values["remora.n_cell"].split())) != n_cell:
            raise ValueError("Reader changed exact grid dimensions")
        if dst["ocean_time"].shape != (1,) or float(dst["ocean_time"][0]) != 0:
            raise ValueError("Unexpected model output times")
        for name in FIELDS:
            x, y = src[name][:], dst[name][:]
            if np.ma.getmaskarray(x).any() or np.ma.getmaskarray(y).any():
                raise ValueError("Missing initialized field " + name)
            y = y[0] if y.ndim == x.ndim + 1 else y
            fields[name] = compare_field(name, x, y)
        if "logical_extent_m" in fixture:
            lengths = [float(src[name][...]) for name in ("xl", "el")]
            if fixture["logical_extent_m"] != lengths:
                raise ValueError("Logical extent receipt disagrees with grid")
            np.testing.assert_array_equal(list(map(float, values["remora.prob_hi"].split()))[:2], lengths)
    evidence = [
        history, log, accounting, inputs, root / "case/fixture.json",
        root / "case/staged.sha256", root / "source_revision.txt",
        root / "submodules.txt", root / "modules.txt", root / "executable.sha256",
        root / "initialized.sha256",
    ] + extra_evidence
    receipt = {
        "schema": "remora_reader_check_v1", "verification_version": 2, "status": "pass",
        "grid_sha256": grid_hash, "job_id": str(job_id), "source_revision": source,
        "executable_sha256": executable, "gpu_runtime_verified": gpu_verified,
        "expected_identity_checks": {
            "source_revision": expected_source_revision, "executable_sha256": expected_executable_sha256,
            "amrex_revision": expected_amrex_revision},
        "halo_source": {"revision": PINNED_SOURCE, "file": "Source/Initialization/REMORA_init_from_netcdf.cpp",
                        "sha256": INIT_SOURCE_SHA256, "function": "extrapolate_metric_to_physical_boundaries"},
        "scope": "One MPI rank, one grid level, initialization only, zero time steps; exact grid; synthetic constant IC and wall boundaries",
        "comparisons": fields, "evidence": [file_record(p, output.parent) for p in evidence],
        "limitations": [
            "No time-integration or nesting validation",
            "Layer depths are checked locally; executable comparisons include stretching curves and parameters",
            "pm/pn first input halo is retained exactly; deeper extrapolated AMReX ghosts are not exported",
            "Coriolis first halo is explicitly checked against nearest-interior extrapolation",
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
    p.add_argument("--expected-source-revision")
    p.add_argument("--expected-executable-sha256")
    p.add_argument("--expected-amrex-revision")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--execution-receipt")
    a = p.parse_args()
    result = check(a.grid, a.returned_dir, a.job_id, a.output,
                   expected_source_revision=a.expected_source_revision,
                   expected_executable_sha256=a.expected_executable_sha256,
                   expected_amrex_revision=a.expected_amrex_revision,
                   require_gpu=a.require_gpu, execution_receipt=a.execution_receipt)
    print("Reader verification: " + result["status"] + "; " + str(len(result["comparisons"])) + " fields compared")
