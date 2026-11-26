"""Audit native FVCOM salinity storage and estimate physical exchange on metric open edges."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from estuary_sections import native_times, numeric
from netCDF4 import Dataset


def open_edges(xy, triangles, obc):
    pairs = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    cells = np.tile(np.arange(len(triangles)), 3)
    keys = np.sort(pairs, axis=1)
    _, first, count = np.unique(keys, axis=0, return_index=True, return_counts=True)
    keep = first[(count == 1) & np.isin(keys[first], obc).all(axis=1)]
    ends, owners = keys[keep], cells[keep]
    if not len(ends) or not np.array_equal(np.unique(ends), np.unique(obc)):
        raise ValueError("Open nodes do not all belong to resolved exterior edges")
    a, b = xy[ends].transpose(1, 0, 2)
    tangent = b - a
    # The vector already includes edge length.
    normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    inward = xy[triangles[owners]].mean(axis=1) - (a + b) * 0.5
    normals[(normals * inward).sum(axis=1) > 0] *= -1
    return ends, owners, normals


def read_ledgers(directory, kind):
    files = sorted(directory.glob("salt_" + kind + "_rank_*.dat"))
    if not files:
        raise ValueError("Missing pinned per-rank " + kind + " ledgers")
    values = [np.loadtxt(f, ndmin=2) for f in files]
    if any(not np.array_equal(v[:, :2], values[0][:, :2]) for v in values):
        raise ValueError("Contradictory per-rank clocks")
    return (
        [f.name.rsplit("_", 1)[-1] for f in files],
        values[0][:, 0],
        np.stack([v[-1, 2:] for v in values]),
    )


def estimate(path, obc_file):
    obc = np.loadtxt(obc_file, skiprows=1, ndmin=2, dtype=int)[:, 1] - 1
    with Dataset(path) as d:
        xy = np.column_stack([numeric(d["x"]), numeric(d["y"])])
        nv = np.asarray(d["nv"][:]).T - 1
        ends, owners, normals = open_edges(xy, nv, obc)
        nodes, back = np.unique(ends, return_inverse=True)
        back = back.reshape(ends.shape)
        elems, eback = np.unique(owners, return_inverse=True)
        h = numeric(d["h"], nodes)
        sig = numeric(d["siglev"])
        if sig.ndim == 1:
            sig = np.broadcast_to(sig[:, None], (len(sig), len(xy)))
        dsigma = np.abs(np.diff(sig[:, nodes], axis=0))
        if not np.allclose(dsigma.sum(axis=0), 1.0):
            raise ValueError("Incomplete water columns")
        ts = native_times(d)
        result = []
        for i, t in enumerate(ts):
            eta = numeric(d["zeta"], (i, nodes))
            salt = numeric(d["salinity"], (i, slice(None), nodes))
            wet_node = np.asarray(d["wet_nodes"][i, nodes]) == 1
            wet_cell = np.asarray(d["wet_cells"][i, elems])[eback] == 1
            dz = dsigma * (h + eta)[None, :]
            if np.any(dz[:, wet_node] < 0):
                raise ValueError("Negative thickness at a wet OBC node")
            dz[:, ~wet_node] = 0.0
            u, v = (
                numeric(d[k], (i, slice(None), elems))[:, eback] for k in ("u", "v")
            )
            vel = u * normals[:, 0] + v * normals[:, 1]
            q = vel * dz[:, back].mean(axis=2)
            qs = vel * (dz * salt)[:, back].mean(axis=2)
            q[:, ~wet_cell] = 0
            qs[:, ~wet_cell] = 0
            if not np.isfinite(qs).all():
                raise ValueError("Non-finite edge flux")
            result.append(
                [q[q > 0].sum(), -q[q < 0].sum(), qs[q > 0].sum(), -qs[q < 0].sum()]
            )
    return ts, np.array(result), len(ends)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["grid", "obc", "ledgers", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--ranks", type=int, required=True)
    p.add_argument("--internal-step-seconds", type=float, required=True)
    a = p.parse_args()
    ts, flux, edges = estimate(a.grid, a.obc)
    ranks, time, stages = read_ledgers(a.ledgers, "stage")
    others, btime, boundary = read_ledgers(a.ledgers, "boundary")
    seconds = np.asarray((ts - ts[0]) / np.timedelta64(1, "s"))
    if (
        ranks != others
        or len(ranks) != a.ranks
        or not np.array_equal(time, btime)
        or len(ts) < 2
        or np.any(np.diff(seconds) <= 0)
        or not np.isclose(seconds[-1], time[-1] * a.internal_step_seconds)
    ):
        raise ValueError("Incomplete rank or time coverage")
    integrated = np.sum(
        0.5 * (flux[1:] + flux[:-1]) * np.diff(seconds)[:, None], axis=0
    )
    stage = stages.sum(axis=0)
    bc = boundary.sum(axis=0)
    salt_change = stage[3] - stage[2]
    volume_change = stage[1] - stage[0]
    river_vol, river_salt = bc[9], bc[10]
    stage_residual = salt_change - stage[4:].sum()

    def balance(vout, vin, sout, sin):
        return {
            "ocean_volume_out_m3": vout,
            "ocean_volume_in_m3": vin,
            "ocean_salt_out_psu_m3": sout,
            "ocean_salt_in_psu_m3": sin,
            "water_residual_m3": volume_change - (vin - vout + river_vol),
            "salt_residual_psu_m3": salt_change - (sin - sout + river_salt),
            "salt_residual_percent_gross_exchange": 100
            * (salt_change - (sin - sout + river_salt))
            / (sin + sout)
            if sin + sout
            else None,
        }

    result = {
        "records": len(ts),
        "open_edges": edges,
        "ranks": len(ranks),
        "river_volume_m3": river_vol,
        "river_salt_psu_m3": river_salt,
        "initial_native_volume_m3": stage[0],
        "final_native_volume_m3": stage[1],
        "volume_change_m3": volume_change,
        "initial_native_salt_psu_m3": stage[2],
        "final_native_salt_psu_m3": stage[3],
        "salt_change_psu_m3": salt_change,
        "hourly_layer_face_exchange": balance(*integrated),
        "native_depth_mean_exchange": balance(*bc[:4]),
        "native_ocean_column_replacement_out_psu_m3": bc[4],
        "native_ocean_column_replacement_in_psu_m3": bc[5],
        "native_stage_changes_psu_m3": dict(
            zip(
                [
                    "between_steps",
                    "advection",
                    "before_positivity",
                    "positivity",
                    "vertical_diffusion_and_bed",
                    "before_ocean_boundary",
                    "ocean_boundary",
                    "assigned_state",
                ],
                stage[4:].tolist(),
                strict=True,
            )
        ),
        "native_stage_telescoping_residual_psu_m3": stage_residual,
        "interpretation": "The 3-D edge estimate uses interior element velocities, half-edge nodal thickness and salinity, and trapezoidal integration on saved clocks. The native step estimate uses depth-mean transport and a layer-weighted boundary concentration. Neither is an exact numerical face flux. Stage telescoping tests bookkeeping; ocean replacements are diagnostics of a numerical operation and are not added twice.",
    }
    a.output.mkdir(exist_ok=True, parents=True)
    (a.output / "salt_budget.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    with (a.output / "ocean_fluxes.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(
            ["time", "out_m3_s", "in_m3_s", "salt_out_psu_m3_s", "salt_in_psu_m3_s"]
        )
        w.writerows((str(t), *v) for t, v in zip(ts, flux))


if __name__ == "__main__":
    main()
