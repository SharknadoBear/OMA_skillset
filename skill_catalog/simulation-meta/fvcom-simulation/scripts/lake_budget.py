"""Reconstruct signed heat and water budgets using native state and source intervals."""

import csv
import json
from datetime import timedelta

import netCDF4 as nc
import numpy as np

from lake_physics import RHO_CP, checked, clocks, member, nml_value, read_record, sha, utc, vals


def integral_product(a, b, c, d, seconds):
    """Exact integral of two linear functions with endpoint values a,b and c,d."""
    return seconds * (a * c + ((b - a) * c + a * (d - c)) / 2 + (b - a) * (d - c) / 3)


def interval_indices(ts, t0, t1):
    ix = [i for i, t in enumerate(ts) if t0 <= t <= t1]
    if len(ix) < 2 or ts[ix[0]] != t0 or ts[ix[-1]] != t1:
        raise ValueError("Forcing misses a required interval endpoint")
    return ix


def snapshot(s, f):
    volume = s["area"] * s["D"] * s["wet_nodes"]
    return float(volume.sum()), float(RHO_CP * np.sum(s["temp"] * f * volume))


def budget(root, spec, g):
    start, end = utc(spec["start_utc"]), utc(spec["end_utc"])
    if end <= start or (end - start).total_seconds() % 21600:
        raise ValueError("Use six-hour boundaries")
    rec = spec["member"]
    run = member(root, rec, g, start, end)
    river_manifest = json.loads(checked(root, spec["river_manifest"]).read_text())
    surface_manifest = json.loads(checked(root, spec["surface_manifest"]).read_text())
    rpath = checked(root, rec["forcing"]["river"])
    ppath = checked(root, rec["forcing"]["pe"])
    apath = checked(root, rec["forcing"]["atmosphere"])
    csvpath = checked(root, spec["surface_budget"])
    for path, item in (
        (rpath, river_manifest),
        (apath, surface_manifest),
        (ppath, surface_manifest),
        (csvpath, surface_manifest),
    ):
        if item["files"].get(path.name) != sha(path):
            raise ValueError("Forcing or interval inventory disagrees with source manifest")
    entries = river_manifest["entries"]
    if not entries or [e["index"] for e in entries] != list(range(1, len(entries) + 1)):
        raise ValueError("River source ordering differs")
    nodes = np.asarray([item["edge_nodes"] for item in entries]).T - 1
    faces = np.asarray([item["cell_id"] for item in entries]) - 1
    nv = g["nv"] - 1
    if any(not set(ab).issubset(nv[:, face]) for ab, face in zip(nodes.T, faces)):
        raise ValueError("River source nodes differ from receiving face")
    f = g["fraction"]
    layer_fraction = f[:, nv[:, faces]].mean(axis=1)
    if not np.allclose(layer_fraction.sum(axis=0), 1):
        raise ValueError("Invalid native river layer weights")
    frac = float(nml_value(run["nml"], "HEATING_LONGWAVE_PERCTAGE"))
    l1 = float(nml_value(run["nml"], "HEATING_LONGWAVE_LENGTHSCALE"))
    l2 = float(nml_value(run["nml"], "HEATING_SHORTWAVE_LENGTHSCALE"))
    if not 0 <= frac <= 1 or min(l1, l2) <= 0:
        raise ValueError("Invalid runtime optical attenuation")
    with csvpath.open(newline="") as stream:
        rows = {utc(row["start_utc"]): row for row in csv.DictReader(stream)}
    result = []
    native_flux_checks = []
    with nc.Dataset(rpath) as rf, nc.Dataset(apath) as af, nc.Dataset(ppath) as pf:
        rt, at, pt = map(clocks, (rf, af, pf))
        if len(rt) != river_manifest["count"] or len(rf.dimensions["rivers"]) != len(entries):
            raise ValueError("River clock or source count differs from the producing manifest")
        names = [
            b"".join(row).decode().strip(" \0") for row in np.ma.asarray(rf["river_names"][:]).filled(b" ")
        ]
        if names != [e["river_name"] for e in entries]:
            raise ValueError("Native river names/order differ")
        if any(ts != sorted(set(ts)) for ts in (rt, at, pt)):
            raise ValueError("Duplicate or nonmonotone source time")
        t0 = start
        prev = read_record(run, start, g, ("net_heat_flux", "short_wave"))
        while t0 < end:
            t1 = t0 + timedelta(hours=6)
            cur = read_record(run, t1, g, ("net_heat_flux", "short_wave"))
            area = prev["area"]
            if not np.array_equal(area, cur["area"]):
                raise ValueError("Native ART1 changes with time")
            source_area = g["art1"].astype("f8")
            ef = area[nodes] / area[nodes].sum(axis=0)

            def donor(state):
                return (np.sum(state["temp"][:, nodes] * ef[np.newaxis, :, :], axis=1) * layer_fraction).sum(
                    axis=0
                )

            d0, d1 = donor(prev), donor(cur)
            v0, e0 = snapshot(prev, f)
            v1, e1 = snapshot(cur, f)
            initial = lambda t: (t - t0).total_seconds() / 21600.0
            mix = lambda x, y, w: (1 - w) * x + w * y
            ai = interval_indices(at, t0, t1)
            pi = interval_indices(pt, t0, t1)
            ri = interval_indices(rt, t0, t1)
            heat_surface = sw_bottom = pe_heat = pe_vol = net_river = incoming_heat = outgoing_heat = 0.0
            reference_heat = reference_water = reference_rain = reference_evap = reference_cond = 0.0
            for i, j in zip(ai, ai[1:]):
                a, b = at[i], at[j]
                secs = (b - a).total_seconds()
                row = rows[a]
                if utc(row["end_utc"]) != b or secs != 3600:
                    raise ValueError("Hourly surface forcing & reference intervals disagree")
                reference_heat += float(row["net_heat_j"])
                reference_rain += float(row["rain_m3"])
                reference_evap += float(row["evaporated_m3"])
                reference_cond += float(row["condensed_m3"])
                q0, q1 = vals(af, "net_heat_flux", i), vals(af, "net_heat_flux", j)
                sw0, sw1 = vals(af, "short_wave", i), vals(af, "short_wave", j)
                w0, w1 = initial(a), initial(b)
                wet0 = mix(prev["wet_nodes"], cur["wet_nodes"], w0)
                wet1 = mix(prev["wet_nodes"], cur["wet_nodes"], w1)
                heat_surface += np.dot(area, integral_product(q0, q1, wet0, wet1, secs))
                # The native radiation profile leaves a depth-dependent shortwave flux at the bed.
                dep0 = mix(prev["D"], cur["D"], w0)
                dep1 = mix(prev["D"], cur["D"], w1)
                att = lambda dep: frac * np.exp(-np.maximum(dep, 0) / l1) + (1 - frac) * np.exp(
                    -np.maximum(dep, 0) / l2
                )
                sw_bottom += secs / 2 * np.dot(area, sw0 * wet0 * att(dep0) + sw1 * wet1 * att(dep1))
            for i, j in zip(pi, pi[1:]):
                a, b = pt[i], pt[j]
                secs = (b - a).total_seconds()
                q0 = vals(pf, "precip", i) + vals(pf, "evap", i)
                q1 = vals(pf, "precip", j) + vals(pf, "evap", j)
                w0, w1 = initial(a), initial(b)
                wt0 = mix(prev["wet_nodes"], cur["wet_nodes"], w0)
                wt1 = mix(prev["wet_nodes"], cur["wet_nodes"], w1)
                temp0 = mix(prev["temp"][0], cur["temp"][0], w0)
                temp1 = mix(prev["temp"][0], cur["temp"][0], w1)
                pe_vol += np.dot(area, integral_product(q0, q1, wt0, wt1, secs))
                pe_heat += RHO_CP * np.dot(area, integral_product(q0 * wt0, q1 * wt1, temp0, temp1, secs))
            for i, j in zip(ri, ri[1:]):
                a, b = rt[i], rt[j]
                secs = (b - a).total_seconds()
                q0, q1 = vals(rf, "river_flux", i).astype(float), vals(rf, "river_flux", j).astype(float)
                if np.any(q0 * q1 < 0):
                    raise ValueError(
                        "Intra-interval sign switching requires donor-specific split integration"
                    )
                neg = (q0 + q1) < 0
                if not prev["wet_nodes"][nodes[:, neg]].all() or not cur["wet_nodes"][nodes[:, neg]].all():
                    raise ValueError("Dry withdrawal endpoint")
                net_river += secs / 2 * np.sum(q0 + q1)
                temperatures0 = np.where(neg, mix(d0, d1, initial(a)), vals(rf, "river_temp", i))
                temperatures1 = np.where(neg, mix(d0, d1, initial(b)), vals(rf, "river_temp", j))
                advective = RHO_CP * integral_product(q0, q1, temperatures0, temperatures1, secs)
                incoming_heat += advective[~neg].sum()
                outgoing_heat += advective[neg].sum()
            for t, state in [(t0, prev), (t1, cur)] if t0 == start else [(t1, cur)]:
                q = vals(af, "net_heat_flux", at.index(t))
                wet = state["wet_nodes"].astype(bool)
                err = state["net_heat_flux"][wet] - q[wet]
                placeholder = (
                    t == run["run_start"]
                    and np.all(state["net_heat_flux"] == 0)
                    and np.all(state["short_wave"] == 0)
                    and np.any(q[wet] != 0)
                )
                native_flux_checks.append(
                    dict(
                        utc=t.isoformat(),
                        role="initial_zero_output_before_first_forcing_update"
                        if placeholder
                        else "model_readback",
                        flux_rmse_w_m2=float(np.sqrt(np.average(err**2, weights=area[wet]))),
                        prescribed_mean_w_m2=float(np.average(q[wet], weights=area[wet])),
                        model_mean_w_m2=float(np.average(state["net_heat_flux"][wet], weights=area[wet])),
                    )
                )
            reference_water = reference_rain - reference_evap + reference_cond
            dv = v1 - v0
            de = e1 - e0
            residual_v = dv - net_river - pe_vol
            residual_e = de - (heat_surface - sw_bottom + incoming_heat + outgoing_heat + pe_heat)
            result.append(
                dict(
                    start_utc=t0.isoformat(),
                    end_utc=t1.isoformat(),
                    storage_change_m3=dv,
                    storage_change_j=de,
                    river_net_m3=net_river,
                    precip_minus_evap_native_wet_m3=pe_vol,
                    source_precip_minus_evap_m3=reference_water,
                    source_precip_m3=reference_rain,
                    source_evaporated_m3=reference_evap,
                    source_condensed_m3=reference_cond,
                    source_surface_heat_j=reference_heat,
                    prescribed_heat_on_native_wet_area_j=heat_surface,
                    transmitted_shortwave_at_bed_j=sw_bottom,
                    river_in_heat_j=incoming_heat,
                    river_donor_out_heat_j=outgoing_heat,
                    precip_evap_advective_heat_j=pe_heat,
                    water_residual_m3=residual_v,
                    heat_residual_j=residual_e,
                    wet_changes=int(np.count_nonzero(prev["wet_nodes"] != cur["wet_nodes"])),
                )
            )
            prev = cur
            t0 = t1
    keys = [k for k in result[0] if k not in ("start_utc", "end_utc")]
    totals = {k: float(sum(row[k] for row in result)) for k in keys}
    seconds = (end - start).total_seconds()
    totals["water_residual_equivalent_depth_m"] = totals["water_residual_m3"] / area.sum()
    totals["heat_residual_equivalent_w_m2"] = totals["heat_residual_j"] / (area.sum() * seconds)
    return dict(
        schema="fvcom_lake_native_budget_v1",
        status="diagnostic",
        start_utc=start.isoformat(),
        end_utc=end.isoformat(),
        member=run["identity"],
        rho_kg_m3=1000.0,
        cp_j_kg_k=4186.0,
        native_area_m2=float(area.sum()),
        source_area_m2=float(source_area.sum()),
        scalar_positivity_control=nml_value(run["nml"], "SCALAR_POSITIVITY_CONTROL").strip(".").upper()
        in ("T", "TRUE"),
        noflux_bottom_condition=nml_value(run["nml"], "NOFLUX_BOT_CONDITION").strip(".").upper()
        in ("T", "TRUE"),
        outlet_heat="local temperature with native ART1 endpoint fractions and uniform DZ1 layer fractions; donor temperatures linearly interpolated from six-hourly model output",
        notes="Heat relative to 0 C. Signed P/E transports local surface temperature. Atmospheric and river clocks retain native sampling. Wet exposure and model donor fields vary linearly between output records. Bed shortwave is reconstructed from namelist attenuation. The residual includes the native scalar limiter and terrain-following bottom correction if enabled. Attribution to these operations requires within-step native diagnostics.",
        optical_longwave_fraction=frac,
        attenuation_m=[l1, l2],
        native_output_heat_checks=native_flux_checks,
        intervals=result,
        totals=totals,
    )
