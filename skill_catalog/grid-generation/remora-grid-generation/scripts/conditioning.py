"""Wet-neighbor log conditioning and explicitly evidenced source-cell repair.

The smoother uses REMORA's Vtransform=2 interfaces at zero free surface.  It
does not modify a mask, fill missing coverage, or rescale water volume.
"""

from __future__ import annotations

import numpy as np

from grid_math import stretching, vertical_diagnostics


def _integer(value, name, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _binary_mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must be a matching binary mask")
    return value.astype(bool, copy=False)


def _pair_haney(a, b, sw, cw, hc):
    """Actual interface equation, using two bounded (N+1, pair_count) arrays."""
    za = a[None, :] * (hc * sw[:, None] + cw[:, None] * a[None, :]) / (hc + a)[None, :]
    zb = b[None, :] * (hc * sw[:, None] + cw[:, None] * b[None, :]) / (hc + b)[None, :]
    da, db = np.diff(za, axis=0), np.diff(zb, axis=0)
    if (
        not np.isfinite(za).all()
        or not np.isfinite(zb).all()
        or np.any(da <= 0)
        or np.any(db <= 0)
    ):
        raise ValueError("Invalid vertical interfaces during depth conditioning")
    return np.max(
        np.abs(za[1:] - zb[1:] + za[:-1] - zb[:-1]) / (da + db),
        axis=0,
    )


def _pair_rx0(a, b):
    # Scaling avoids overflow for two individually finite positive depths.
    scale = np.maximum(a, b)
    aa, bb = a / scale, b / scale
    return np.abs(aa - bb) / (aa + bb)


def _pair_sets(shape, mask):
    """Fixed axis/parity order; each set consists of disjoint wet-wet pairs."""
    indices = np.arange(np.prod(shape)).reshape(shape)
    for axis in (0, 1):
        for parity in (0, 1):
            left = [slice(None), slice(None)]
            right = left.copy()
            left[axis] = slice(parity, shape[axis] - 1, 2)
            right[axis] = slice(parity + 1, shape[axis], 2)
            left, right = tuple(left), tuple(right)
            wet = mask[left] & mask[right]
            yield indices[left][wet], indices[right][wet]


def smooth_depth_vertical(
    h,
    mask,
    area,
    rx0_max,
    rx1_max,
    N,
    theta_s,
    theta_b,
    hc,
    hmin=2.0,
    max_iter=4000,
    *,
    pair_block_size=4096,
):
    """Return ``(conditioned_depth, report)`` or fail on a bounded budget.

    Inputs must already respect hmin. Dry-cell values, including NaNs, remain
    untouched. For each violating wet pair, the difference of natural logs is
    reduced toward zero while its area-weighted log mean is fixed. The rx0
    constraint supplies an analytic bracket; bisection retains a feasible
    endpoint for the actual interface Haney constraint. Corrections use a
    small internal margin, while final acceptance requires rx0 <= rx0_max
    and rx1 < rx1_max, with no acceptance tolerance.

    Disjoint red/black pairs have a deterministic order. Bounded pair blocks
    limit vertical working memory without changing pair order or arithmetic.
    No water-volume conservation adjustment is applied.
    """
    h = np.array(h, dtype=float, copy=True)
    if h.ndim != 2 or not all(h.shape):
        raise ValueError("Expected a nonempty two-dimensional depth field")
    mask = _binary_mask(mask, h.shape, "mask")
    area = np.asarray(area, dtype=float)
    if area.shape != h.shape or not mask.any():
        raise ValueError("Expected matching areas and at least one wet cell")
    if not np.isfinite(hmin) or hmin <= 0:
        raise ValueError("hmin must be finite and positive")
    if not np.isfinite(h[mask]).all() or np.any(h[mask] < hmin):
        raise ValueError("Wet depths must be finite and already at or above hmin")
    if not np.isfinite(area[mask]).all() or np.any(area[mask] <= 0):
        raise ValueError("Wet areas must be finite and positive")
    if not np.isfinite(rx0_max) or not 0 < rx0_max < 1:
        raise ValueError("rx0_max must lie strictly between zero and one")
    if not np.isfinite(rx1_max) or rx1_max <= 0:
        raise ValueError("rx1_max must be finite and positive")
    if not np.isfinite(hc) or hc < 0:
        raise ValueError("hc must be finite and nonnegative")
    N = _integer(N, "N", 1)
    max_iter = _integer(max_iter, "max_iter", 0)
    pair_block_size = _integer(pair_block_size, "pair_block_size", 1)
    _, sw, _, cw = stretching(N, theta_s, theta_b)
    initial = h.copy()
    flat = h.ravel()
    weights = area.ravel()
    u = np.zeros(h.size, dtype=float)
    u[mask.ravel()] = np.log(flat[mask.ravel()])
    # A convex combination cannot go below the original wet minimum. This
    # floor guards exp(log(minimum)) roundoff only; it is never a rescaling.
    floor = float(h[mask].min())
    pairs = list(_pair_sets(h.shape, mask))
    touched = np.zeros(h.size, bool)
    start_volume = float(np.sum(h[mask] * area[mask]))
    if not np.isfinite(start_volume) or start_volume <= 0:
        raise ValueError("Wet volume must be finite and positive")
    initial_log_sum = float(np.sum(area[mask] * np.log(h[mask])))
    # Explicit strictness margin is far larger than interface roundoff but
    # small compared with a scientific rx1 target such as 7 or 9.
    rx0_inner = rx0_max * (1.0 - 1.0e-10)
    rx1_inner = rx1_max * (1.0 - 1.0e-10)
    max_jump = np.log1p(rx0_inner) - np.log1p(-rx0_inner)
    corrections = 0
    bisection_steps = 0

    def limits():
        rx0, rx1 = 0.0, 0.0
        for left, right in pairs:
            for start in range(0, len(left), pair_block_size):
                aa = flat[left[start : start + pair_block_size]]
                bb = flat[right[start : start + pair_block_size]]
                rx0 = max(rx0, float(_pair_rx0(aa, bb).max()))
                rx1 = max(rx1, float(_pair_haney(aa, bb, sw, cw, hc).max()))
        return rx0, rx1

    initial_rx0, initial_rx1 = limits()
    rx0, rx1 = initial_rx0, initial_rx1
    for iteration in range(max_iter + 1):
        if rx0 <= rx0_max and rx1 < rx1_max:
            # Independently use the production row-block diagnostic for final
            # acceptance; dry depths are replaced only in this temporary view.
            diagnostic_h = np.where(mask, h, hmin)
            diagnostics = vertical_diagnostics(
                diagnostic_h, N, theta_s, theta_b, hc, mask=mask
            )
            actual_rx1 = diagnostics["haney_rx1_diagnostic"]
            if not actual_rx1 < rx1_max:
                raise ValueError(
                    "Independent interface check failed strict rx1 acceptance: "
                    f"{actual_rx1} >= {rx1_max}"
                )
            final_volume = float(np.sum(h[mask] * area[mask]))
            delta = h[mask] - initial[mask]
            return h, {
                "method": "area_weighted_log_rx0_rx1_pair_projection_v2",
                "space": "natural_log_depth",
                "vertical_transform": "REMORA_Vtransform_2_zeta_0",
                "volume_policy": "report_only_no_rescaling",
                "converged": True,
                "iterations": iteration,
                "max_iterations": max_iter,
                "pair_corrections": corrections,
                "bisection_steps": bisection_steps,
                "modified_wet_cells": int(touched.sum()),
                "mask_unchanged": True,
                "hmin_m": float(hmin),
                "minimum_wet_depth_m": float(h[mask].min()),
                "rx0_max": float(rx0_max),
                "rx1_max": float(rx1_max),
                "internal_rx0_target": float(rx0_inner),
                "internal_rx1_target": float(rx1_inner),
                "initial_rx0": initial_rx0,
                "initial_haney_rx1": initial_rx1,
                "rx0": rx0,
                "haney_rx1_diagnostic": actual_rx1,
                "rx1": actual_rx1,
                "strict_rx1_pass": True,
                "initial_volume_m3": start_volume,
                "final_volume_m3": final_volume,
                "volume_relative_change": (final_volume - start_volume) / start_volume,
                "area_weighted_log_sum_initial": initial_log_sum,
                "area_weighted_log_sum_final": float(
                    np.sum(area[mask] * np.log(h[mask]))
                ),
                "maximum_absolute_depth_change_m": float(np.abs(delta).max()),
                "rms_depth_change_m": float(np.sqrt(np.mean(delta * delta))),
            }
        if iteration == max_iter:
            break
        for left, right in pairs:
            for start in range(0, len(left), pair_block_size):
                ia = left[start : start + pair_block_size]
                ib = right[start : start + pair_block_size]
                bad = (_pair_rx0(flat[ia], flat[ib]) > rx0_inner) | (
                    _pair_haney(flat[ia], flat[ib], sw, cw, hc) >= rx1_inner
                )
                if not bad.any():
                    continue
                ia, ib = ia[bad], ib[bad]
                aa, bb = u[ia], u[ib]
                # Normalize areas first to avoid overflow in A+B.
                scale = np.maximum(weights[ia], weights[ib])
                wa, wb = weights[ia] / scale, weights[ib] / scale
                wa, wb = wa / (wa + wb), wb / (wa + wb)
                mean = wa * aa + wb * bb
                jump = aa - bb
                upper = np.minimum(1.0, max_jump / np.abs(jump))
                lower = np.zeros_like(upper)

                def at(fraction):
                    separation = jump * fraction
                    return (
                        np.maximum(np.exp(mean + wb * separation), floor),
                        np.maximum(np.exp(mean - wa * separation), floor),
                    )

                ca, cb = at(upper)
                feasible = (_pair_rx0(ca, cb) <= rx0_inner) & (
                    _pair_haney(ca, cb, sw, cw, hc) <= rx1_inner
                )
                lower[feasible] = upper[feasible]
                active = ~feasible
                # Each retained lower endpoint is feasible. 52 iterations
                # resolve the fractional bracket to double precision.
                for _ in range(52):
                    if not active.any():
                        break
                    middle = (lower + upper) / 2.0
                    ca, cb = at(middle)
                    good = (_pair_rx0(ca, cb) <= rx0_inner) & (
                        _pair_haney(ca, cb, sw, cw, hc) <= rx1_inner
                    )
                    lower[active & good] = middle[active & good]
                    upper[active & ~good] = middle[active & ~good]
                    bisection_steps += 1
                ca, cb = at(lower)
                flat[ia], flat[ib] = ca, cb
                # Keep the log representation synchronized with rounded
                # positive output depths before the next pair is processed.
                u[ia], u[ib] = np.log(ca), np.log(cb)
                touched[ia], touched[ib] = True, True
                corrections += len(ia)
        rx0, rx1 = limits()
    raise ValueError(
        "Vertical depth conditioning did not converge within "
        f"{max_iter} iterations: rx0={rx0:.17g} (<= {rx0_max}), "
        f"interface rx1={rx1:.17g} (< {rx1_max})"
    )


def repair_source_artifacts(
    elevation,
    mapped_land,
    suspect_cells,
    *,
    supporting_source,
    supporting_elevation,
    radius_cells=2,
    min_neighbors=3,
):
    """Repair explicitly selected finite source artifacts before resampling.

    Elevations are positive-up metres on a regular source grid. Each selected
    cell must be outside mapped land and independently supported as wet by a
    co-registered elevation source. Missing primary cells are never repaired.
    Original finite negative primary cells supply inverse-index-distance
    squared interpolation; selected cells never become donors. Local
    four-neighbor reachability prevents interpolation through mapped land.
    This is an evidence-bound interpolation utility, not an artifact detector.
    """
    elevation = np.array(elevation, dtype=float, copy=True)
    if elevation.ndim != 2 or not all(elevation.shape):
        raise ValueError("Source elevation must be a nonempty 2D array")
    land = _binary_mask(mapped_land, elevation.shape, "mapped_land")
    selected = _binary_mask(suspect_cells, elevation.shape, "suspect_cells")
    support = np.asarray(supporting_elevation, dtype=float)
    if support.shape != elevation.shape:
        raise ValueError("Supporting source must have the same geometry")
    if not isinstance(supporting_source, dict) or not all(
        isinstance(supporting_source.get(key), str) and supporting_source[key].strip()
        for key in ("reference", "rationale")
    ):
        raise ValueError("Declare supporting_source reference and rationale")
    radius_cells = _integer(radius_cells, "radius_cells", 1)
    min_neighbors = _integer(min_neighbors, "min_neighbors", 3)
    if not selected.any():
        raise ValueError("Explicitly select at least one suspect source cell")
    if np.any(land & selected):
        raise ValueError("Selected source artifacts must not be mapped real land")
    if not np.isfinite(elevation[selected]).all():
        raise ValueError("Artifact repair cannot fill missing primary coverage")
    if not np.isfinite(support[selected]).all() or np.any(support[selected] >= 0):
        raise ValueError("Every selected cell needs finite independent wet evidence")
    donors = np.isfinite(elevation) & (elevation < 0) & ~land & ~selected
    traversable = donors | selected
    output = elevation.copy()
    records = []
    ny, nx = elevation.shape
    for row, col in np.argwhere(selected):
        row, col = int(row), int(col)
        r0, r1 = max(0, row - radius_cells), min(ny, row + radius_cells + 1)
        c0, c1 = max(0, col - radius_cells), min(nx, col + radius_cells + 1)
        todo, visited = [(row, col)], {(row, col)}
        neighbors = []
        while todo:
            j, i = todo.pop()
            if donors[j, i]:
                neighbors.append((j, i))
            for jj, ii in ((j - 1, i), (j, i - 1), (j, i + 1), (j + 1, i)):
                if (
                    r0 <= jj < r1
                    and c0 <= ii < c1
                    and (jj, ii) not in visited
                    and traversable[jj, ii]
                ):
                    visited.add((jj, ii))
                    todo.append((jj, ii))
        neighbors.sort()
        if len(neighbors) < min_neighbors:
            raise ValueError(
                f"Selected cell {(row, col)} has only {len(neighbors)} "
                "reachable local wet source donors"
            )
        jj, ii = np.asarray(neighbors).T
        weight = 1.0 / ((jj - row) ** 2 + (ii - col) ** 2)
        # Center the weighted average to preserve a constant donor field
        # exactly and reduce cancellation when elevations have a large offset.
        baseline = float(elevation[jj[0], ii[0]])
        repaired = baseline + float(
            np.sum(weight * (elevation[jj, ii] - baseline)) / weight.sum()
        )
        output[row, col] = repaired
        records.append(
            {
                "row": row,
                "column": col,
                "original_elevation_m": float(elevation[row, col]),
                "repaired_elevation_m": repaired,
                "supporting_elevation_m": float(support[row, col]),
                "donor_count": len(neighbors),
                "donor_indices": [[int(j), int(i)] for j, i in neighbors],
            }
        )
    return output, {
        "method": "explicit_source_artifact_local_wet_idw_v2",
        "stage": "source_grid_before_horizontal_interpolation",
        "supporting_source": dict(supporting_source),
        "radius_cells": radius_cells,
        "min_neighbors": min_neighbors,
        "distance_metric": "source_grid_index_distance_squared",
        "selected_cells": int(selected.sum()),
        "changed_cells": int(np.count_nonzero(output[selected] != elevation[selected])),
        "mapped_land_unchanged": True,
        "missing_coverage_filled": 0,
        "cells": records,
    }


def _source_artifact_repair_cli(args):
    """Write a new source NetCDF, complete repair receipt, and review map."""
    import hashlib
    import json
    import shutil
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from netCDF4 import Dataset

    def sha(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def read_source(path, variable, positive):
        with Dataset(path) as source:
            z = source[variable]
            if getattr(z, "units", "").lower() not in (
                "m",
                "meter",
                "meters",
                "metre",
                "metres",
            ):
                raise ValueError("Source elevation units must explicitly be metres")
            if getattr(z, "positive", positive).lower() != positive:
                raise ValueError("Declared elevation sign conflicts with metadata")
            lon, lat = np.asarray(source["lon"][:]), np.asarray(source["lat"][:])
            data = np.asarray(np.ma.filled(z[:], np.nan), dtype=float)
            if (
                lon.ndim != 1
                or lat.ndim != 1
                or data.shape != (len(lat), len(lon))
                or not np.isfinite(lon).all()
                or not np.isfinite(lat).all()
                or not (np.all(np.diff(lon) > 0) or np.all(np.diff(lon) < 0))
                or not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0))
            ):
                raise ValueError(
                    "Expected rectilinear monotone lon/lat source geometry"
                )
        return lon, lat, data if positive == "up" else -data

    primary, support_path = Path(args.primary).resolve(), Path(args.support).resolve()
    if primary == support_path or sha(primary) == sha(support_path):
        raise ValueError("Supporting evidence must be a distinct independent source")
    selection = Path(args.selection_npz).resolve()
    out = Path(args.output_dir).resolve()
    output_nc, output_json, output_map = (
        out / "repaired_source.nc",
        out / "source_artifact_repair.json",
        out / "source_artifact_repair.png",
    )
    if any(path.exists() for path in (output_nc, output_json, output_map)):
        raise ValueError(
            "Source repair outputs already exist; use a new output directory"
        )
    # Never permit case artifacts inside any installed skill, even if a user
    # accidentally points this standalone utility at the script directory.
    installed_root = Path(__file__).resolve().parents[2]
    if out == installed_root or installed_root in out.parents:
        raise ValueError("Write source repair artifacts outside installed skills")
    lon, lat, original = read_source(primary, args.variable, args.primary_positive)
    support_lon, support_lat, evidence = read_source(
        support_path, args.support_variable, args.support_positive
    )
    if not np.array_equal(lon, support_lon) or not np.array_equal(lat, support_lat):
        raise ValueError(
            "Primary and evidence grids must have identical lon/lat geometry"
        )
    with np.load(selection, allow_pickle=False) as arrays:
        suspect, land = arrays["suspect_cells"], arrays["mapped_land"]
    repaired, report = repair_source_artifacts(
        original,
        land,
        suspect,
        supporting_source={
            "reference": args.support_reference,
            "rationale": args.rationale,
            "path": str(support_path),
            "sha256": sha(support_path),
        },
        supporting_elevation=evidence,
        radius_cells=args.radius_cells,
        min_neighbors=args.min_neighbors,
    )
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(primary, output_nc)
    with Dataset(output_nc, "r+") as target:
        encoded = repaired if args.primary_positive == "up" else -repaired
        # Assign only selected finite cells; preserve original fill values,
        # encodings, other variables, and nonselected samples exactly.
        variable = target[args.variable]
        for j, i in np.argwhere(suspect):
            variable[int(j), int(i)] = encoded[int(j), int(i)]
        target.source_artifact_repair_method = report["method"]
        target.source_artifact_repair_support_sha256 = sha(support_path)
    _, _, reopened = read_source(output_nc, args.variable, args.primary_positive)
    np.testing.assert_allclose(reopened, repaired, rtol=1e-6, atol=1e-6, equal_nan=True)
    # Show the selected region plus context; retain every modified cell.
    selected_rows, selected_cols = np.nonzero(suspect)
    r0, r1 = max(0, int(selected_rows.min()) - 4), min(
        len(lat), int(selected_rows.max()) + 5
    )
    c0, c1 = max(0, int(selected_cols.min()) - 4), min(
        len(lon), int(selected_cols.max()) + 5
    )
    window = (slice(r0, r1), slice(c0, c1))
    # Artifact spikes must not compress all scientifically relevant depths
    # into one color. Use the same support/repaired range for every panel and
    # explicitly mark saturation of outlying original values.
    reference_values = np.stack((repaired[window], evidence[window]))
    vmin = float(np.nanmin(reference_values))
    vmax = float(np.nanmax(reference_values))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, data, title in zip(
        axes,
        (original, repaired, evidence),
        ("Original source", "Repaired source", "Independent supporting source"),
    ):
        image = axis.pcolormesh(
            lon[c0:c1],
            lat[r0:r1],
            data[window],
            shading="auto",
            cmap="terrain",
            vmin=vmin,
            vmax=vmax,
        )
        axis.scatter(
            lon[selected_cols],
            lat[selected_rows],
            s=30,
            facecolors="none",
            edgecolors="magenta",
            label="Explicit suspect cells",
        )
        axis.set(title=title, xlabel="Longitude", ylabel="Latitude")
        axis.legend(fontsize=8)
    fig.suptitle(
        "Shared scale from repaired/supporting source; original outliers saturate",
        fontsize=11,
    )
    fig.colorbar(image, ax=axes, label="Elevation (m; positive up)", extend="both")
    fig.savefig(output_map, dpi=160)
    plt.close(fig)
    report.update(
        schema="remora_source_artifact_repair_v2",
        primary_source={
            "path": str(primary),
            "sha256": sha(primary),
            "variable": args.variable,
        },
        selection={"path": str(selection), "sha256": sha(selection)},
        output_source={"path": str(output_nc), "sha256": sha(output_nc)},
        review_map={"path": str(output_map), "sha256": sha(output_map)},
        readback_passed=True,
        visual_review_status="pending",
    )
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source": str(output_nc),
                "report": str(output_json),
                "map": str(output_map),
            }
        )
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    repair = commands.add_parser(
        "source-artifact-repair", help=repair_source_artifacts.__doc__
    )
    repair.add_argument("--primary", required=True)
    repair.add_argument("--variable", required=True)
    repair.add_argument("--primary-positive", choices=("up", "down"), default="up")
    repair.add_argument("--support", required=True)
    repair.add_argument("--support-variable", required=True)
    repair.add_argument("--support-positive", choices=("up", "down"), default="up")
    repair.add_argument(
        "--selection-npz",
        required=True,
        help="NPZ containing matching suspect_cells and mapped_land binary masks",
    )
    repair.add_argument("--support-reference", required=True)
    repair.add_argument("--rationale", required=True)
    repair.add_argument("--radius-cells", type=int, default=2)
    repair.add_argument("--min-neighbors", type=int, default=3)
    repair.add_argument("--output-dir", required=True)
    repair.set_defaults(run=_source_artifact_repair_cli)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
