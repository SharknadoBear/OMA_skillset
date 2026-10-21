#!/usr/bin/env python3
"""Make explicitly synthetic initialization-only input for a real grid reader check."""

import argparse
from pathlib import Path
import shutil
import json
import numpy as np
from netCDF4 import Dataset
from remora_grid import sha, readback


def create(grid_path, output):
    grid_path = Path(grid_path).resolve()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("remora_grid.nc", "reader_initial.nc", "inputs.reader", "fixture.json"):
        if (out / name).exists():
            raise ValueError("Use a new fixture directory; refusing to replace " + name)
    result = readback(grid_path)
    nx, ny, N = result["n_cell"]
    with Dataset(grid_path) as grid:
        # xl/el are logical extents; nominal spacing is not a graded-grid extent.
        extents = [float(np.asarray(grid[name][...]).item()) for name in ("xl", "el")]
        if not all(np.isfinite(extents)) or min(extents) <= 0:
            raise ValueError("Grid xl and el must be finite positive logical extents")
    original_sha = sha(grid_path)
    shutil.copy2(grid_path, out / "remora_grid.nc")
    if sha(out / "remora_grid.nc") != original_sha:
        raise ValueError("Exact grid copy failed")
    with Dataset(grid_path) as grid, Dataset(
        out / "reader_initial.nc", "w", format="NETCDF3_64BIT_DATA"
    ) as ds:
        for key, dim in grid.dimensions.items():
            ds.createDimension(key, len(dim))
        ds.createDimension("ocean_time", 1)
        t = ds.createVariable("ocean_time", "f8", ("ocean_time",))
        t[:] = 0
        t.units = "seconds since 2000-01-01 00:00:00"
        for name, tag, value, three in [
            ("temp", "rho", 10.0, True),
            ("salt", "rho", 32.0, True),
            ("u", "u", 0.0, True),
            ("v", "v", 0.0, True),
            ("ubar", "u", 0.0, False),
            ("vbar", "v", 0.0, False),
            ("zeta", "rho", 0.0, False),
        ]:
            dims = (
                ("ocean_time",)
                + (("s_rho",) if three else ())
                + ("eta_" + tag, "xi_" + tag)
            )
            ds.createVariable(name, "f8", dims)[:] = value
        ds.purpose = (
            "SYNTHETIC reader-only constant state; not scientific initial conditions"
        )
        hmax = float(grid["h"][:].max())
        theta_s = float(grid.theta_s)
        theta_b = float(grid.theta_b)
        hc = float(grid.hc_m)
    text = f"""# SYNTHETIC INITIALIZATION-ONLY READER CHECK; not a simulation setup
remora.prob_name = BlankProblem
remora.max_step = 0
remora.stop_time = 0
remora.n_cell = {nx} {ny} {N}
remora.prob_lo = 0 0 {-hmax:.12g}
remora.prob_hi = {extents[0]:.17g} {extents[1]:.17g} 0
remora.is_periodic = 0 0 0
remora.bc.xlo.type = slipwall
remora.bc.xhi.type = slipwall
remora.bc.ylo.type = slipwall
remora.bc.yhi.type = slipwall
remora.ic_type = netcdf
remora.nc_init_file_0 = reader_initial.nc
remora.nc_grid_file_0 = remora_grid.nc
remora.mask_type = netcdf
remora.check_mask_consistency = true
remora.mask_consistency = abort
remora.use_curvilinear_grid = true
remora.use_coriolis = true
remora.coriolis_type = netcdf
remora.theta_s = {theta_s}
remora.theta_b = {theta_b}
remora.tcline = {hc}
remora.fixed_dt = 1
remora.ndtfast = 1
remora.v = 1
remora.check_int = -1
remora.plot_int = 1
remora.plot_file = reader_initialized
remora.plotfile_type = netcdf
remora.plot_vars_3d = temp salt z_cc
remora.plot_vars_2d = zeta mask_rho mask_u mask_v
amr.max_level = 0
amr.blocking_factor = 1
amr.max_grid_size = 512
"""
    # These bytes are consumed by POSIX checksum and exact-line guards on HPC.
    # Explicit LF also makes fixture hashes independent of the local OS.
    (out / "inputs.reader").write_text(text, encoding="utf-8", newline="\n")
    receipt = {
        "purpose": "initialization_only_no_time_steps",
        "schema": "remora_reader_fixture_v2",
        "grid_sha256": original_sha,
        "exact_grid_copy": True,
        "n_cell": [nx, ny, N],
        "logical_extent_m": extents,
        "logical_extent_source": "NetCDF scalar variables xl and el",
        "max_step": 0,
        "stop_time": 0,
        "initial_sha256": sha(out / "reader_initial.nc"),
        "inputs_sha256": sha(out / "inputs.reader"),
    }
    (out / "fixture.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    print(json.dumps(create(a.grid, a.output_dir), indent=2))
