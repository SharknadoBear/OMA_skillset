"""Build and recheck a type-1 FVCOM offshore boundary for the Cartesian wind experiment."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4 as nc
import numpy as np

from glorys_obc import DATASET, PRODUCT, TIDES_AND_IB, daily_midpoints, monthly_days, prepare
from hycom_obc import corrected, prepare_hycom, validate_policy
from waterlevel_core import (DAY_MS, Boundary, decode_netcdf_times, iso_utc,
                             sha256_file, temporal_interpolate, write_fvcom_forcing, write_json_atomic)

MANIFEST = "wind_boundary_manifest.json"
SCHEMA = "fvcom_wind_boundary_v1"
POLICY = dict(native_obc="type_1_timeseries_direct_clamp", offshore_tides="absent_after_source_processing",
              offshore_inverse_barometer="absent_after_source_processing", boundary_inverse_barometer_terms=1,
              interior_pressure_gradient=True, reference_pa=101325.0, density_kg_m3=1025.0,
              gravity_m_s2=9.81, native_ib_evidence="Cartesian FVCOM mod_setup.F, mod_main.F, mod_obcs.F, advave_edge_gcn.F")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(var, selection=slice(None)):
    value = np.ma.asarray(var[selection], dtype="f8").filled(np.nan)
    if not np.isfinite(value).all():
        raise ValueError(f"Missing or nonfinite {var.name}")
    return value


def read_tide(path):
    with nc.Dataset(path) as ds:
        clock, _, _ = decode_netcdf_times(ds, "Itime")
        ids = np.asarray(ds["obc_nodes"][:], dtype=int)
        if (ds["elevation"].units not in ("m", "meters") or ds["elevation"].dimensions != ("time", "nobc")
                or np.any(np.diff(clock) <= 0) or len(ids) != len(set(ids)) or np.any(ids < 1)):
            raise ValueError("Invalid TPXO elevation geometry, clock, or units")
        return clock, ids, finite(ds["elevation"]), ds.obc_points_sha256


def read_dat_ids(path):
    lines = Path(path).read_text().splitlines()
    rows = np.loadtxt(lines[1:], dtype=int, ndmin=2)
    if (rows.shape != (int(lines[0].split("=")[1]), 3) or not np.all(rows[:, 2] == 1)
            or not np.array_equal(rows[:, 0], np.arange(1, len(rows)+1))):
        raise ValueError("Expected an ordered type-1 prescribed sea-level OBC DAT")
    return rows[:, 1]


def pressure_at_tide_clock(path, ids, clock, coordinates):
    with nc.Dataset(path) as ds:
        field = ds["air_pressure"]
        if field.units != "Pa" or field.dimensions != ("time", "node"):
            raise ValueError("Require absolute nodal air_pressure(time,node) in Pa")
        node_ids = np.asarray(ds["node_id"][:], dtype=int)
        if not np.array_equal(node_ids, np.arange(1, len(node_ids)+1)) or np.max(ids) > len(node_ids):
            raise ValueError("Surface pressure does not index the frozen one-based FVCOM nodes")
        pressure_coords = np.column_stack((finite(ds["lon"], ids-1), finite(ds["lat"], ids-1)))
        if not np.allclose(pressure_coords, coordinates, rtol=0, atol=1e-5):
            raise ValueError("Pressure nodes differ geographically from ordered offshore boundary points")
        native_t, _, _ = decode_netcdf_times(ds, "Itime")
        if len(native_t) < 2 or np.any(np.diff(native_t) != DAY_MS // 24):
            raise ValueError("Require gap-free hourly atmospheric pressure")
        native_p = finite(field, (slice(None), ids-1))
        if np.any(native_p < 80000) or np.any(native_p > 110000):
            raise ValueError("Expected absolute sea-level pressure in Pa")
    return temporal_interpolate(native_t, native_p, clock)[0]


def calculate(clock, pressure, daily_t, daily_z, monthly_t, monthly_z, source):
    if not np.isfinite(daily_z).all() or not np.isfinite(monthly_z).all():
        raise ValueError("Unresolved offshore SSH")
    baseline = np.average(monthly_z, axis=0, weights=monthly_days(monthly_t))
    native_clock = np.asarray(daily_t,dtype="i8")
    if source["provider"] == "GLORYS":
        sample_clock = daily_midpoints(native_clock)
    else:
        sample_clock = native_clock
        if len(native_clock)<2 or np.any(np.diff(native_clock) != 1000*source["run"]["cadence_seconds"]):
            raise ValueError("HYCOM source clock differs from documented cadence")
    run = temporal_interpolate(sample_clock, daily_z, clock)[0]
    ib = -(pressure - POLICY["reference_pa"]) / (POLICY["density_kg_m3"] * POLICY["gravity_m_s2"])
    return baseline, run - baseline[None, :], ib


def file_ref(path):
    return dict(path=path.name, sha256=sha256_file(path))


def verify_pressure_owner(paths, manifest):
    receipt = paths["pressure"].with_name("atmosphere_manifest.json")
    if sha256_file(receipt) != manifest["atmosphere_manifest_sha256"]:
        raise ValueError("Surface producer receipt has changed")
    owner = load_json(receipt)
    if (owner["schema"] != "fvcom_atmosphere_manifest_v1" or owner["status"] != "ready"
            or owner["pressure_reference"] != "absolute_mean_sea_level"
            or owner["files"]["surface"] != manifest["files"]["pressure"]
            or owner["mesh_sha256"] != manifest["files"]["mesh"]["sha256"]
            or owner["obc_sha256"] != manifest["files"]["obc"]["sha256"]
            or owner["provider"] != manifest["atmosphere_provider"]):
        raise ValueError("Pressure field and frozen mesh/OBC identities differ from surface producer receipt")


def record(varset, name, values, axes, units):
    arr = np.asarray(values)
    v = varset.createVariable(name, arr.dtype, axes, zlib=True, complevel=2, shuffle=True)
    v.units = units
    v[:] = arr


def write_components(path, clock, ids, coords, daily_t, dz, monthly_t, mz, baseline, anomaly, ib, correction_fields):
    with nc.Dataset(path, "w") as ds:
        ds.title = "Ordered water-level components for a barotropic wind and pressure experiment"
        for name, length in (("time", len(clock)), ("nobc", len(ids)), ("daily", len(daily_t)), ("monthly", len(monthly_t))):
            ds.createDimension(name, length)
        for name, values, axes, units in (
                ("obc_nodes", ids, ("nobc",), "1"), ("longitude", coords[:,0], ("nobc",), "degrees_east"),
                ("latitude", coords[:,1], ("nobc",), "degrees_north"),
                ("time", clock, ("time",), "milliseconds since 1970-01-01 00:00:00"),
                ("daily_native_time", daily_t, ("daily",), "milliseconds since 1970-01-01 00:00:00"),
                ("monthly_native_time", monthly_t, ("monthly",), "milliseconds since 1970-01-01 00:00:00"),
                ("daily_zos", dz, ("daily","nobc"), "m"), ("monthly_zos", mz, ("monthly","nobc"), "m"),
                ("reference_zos", baseline, ("nobc",), "m"), ("offshore_anomaly", anomaly, ("time","nobc"), "m"),
                ("inverse_barometer", ib, ("time","nobc"), "m")):
            record(ds, name, values, axes, units)
        for period, fields in correction_fields.items():
            for term, arr in fields.items():
                record(ds, f"hycom_{period}_{term}", arr, ("daily" if period == "run" else "monthly","nobc"), "m")


def namelist_bindings(values):
    def scalar(group, key):
        return str(values[group][key]).strip().strip("'\u0022")
    obc = "NML_OPEN_BOUNDARY_CONTROL"
    surface = "NML_SURFACE_FORCING"
    for group, key in ((obc,"OBC_ELEVATION_FORCING_ON"), (surface,"AIRPRESSURE_ON"), (surface,"WIND_ON")):
        if scalar(group, key).strip(".").lower() not in ("true", "t"):
            raise ValueError(f"{key} must be enabled for the wind boundary")
    return {"mesh":scalar("NML_GRID_COORDINATES","GRID_FILE"), "obc":scalar(obc,"OBC_NODE_LIST_FILE"),
            "combined":scalar(obc,"OBC_ELEVATION_FILE"), "pressure":scalar(surface,"AIRPRESSURE_FILE")}


def audit(paths, manifest):
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "ready" or manifest["physics"] != POLICY:
        raise ValueError("Invalid wind boundary certificate or inverse-barometer policy")
    source = manifest["offshore"]
    if source.get("provider") == "HYCOM":
        validate_policy(source)
    elif (source.get("provider") != "GLORYS" or source["product"] != PRODUCT or source["tide_and_inverse_barometer"] != "both_absent"
            or source["tide_and_inverse_barometer_evidence"] != TIDES_AND_IB
            or source["daily"]["dataset_id"] != DATASET+"P1D-m" or source["monthly"]["dataset_id"] != DATASET+"P1M-m"):
        raise ValueError("Offshore source physics does not support one explicit boundary inverse barometer")
    for role, path in paths.items():
        if sha256_file(path) != manifest["files"][role]["sha256"]:
            raise ValueError(f"Changed wind boundary dependency: {role}")
    verify_pressure_owner(paths, manifest)
    clock, ids, tide, points_hash = read_tide(paths["pure_tpxo"])
    if points_hash != manifest["points_sha256"] or not np.array_equal(ids, read_dat_ids(paths["obc"])):
        raise ValueError("Terminal OBC geometry or points do not match the certified tide")
    if len(clock) < 2 or np.any(np.diff(clock) != 360000):
        raise ValueError("Wind experiment requires six-minute boundary forcing")
    with nc.Dataset(paths["components"]) as ds, nc.Dataset(paths["combined"]) as combined:
        coords = np.column_stack((finite(ds["longitude"]),finite(ds["latitude"])))
        pressure = pressure_at_tide_clock(paths["pressure"], ids, clock, coords)
        if (not np.array_equal(ds["obc_nodes"][:], ids) or not np.array_equal(ds["time"][:], clock)
                or not np.array_equal(combined["obc_nodes"][:], ids)
                or not np.array_equal(decode_netcdf_times(combined, "Itime")[0], clock)
                or combined.data_model != "NETCDF3_CLASSIC"):
            raise ValueError("Staged treatment geometry, UTC clock or encoding changed")
        run_ssh, ref_ssh = finite(ds["daily_zos"]), finite(ds["monthly_zos"])
        if source["provider"] == "HYCOM":
            for period, expected_ssh in (("run",run_ssh), ("reference",ref_ssh)):
                prefix = "hycom_"+period+"_"
                fields = {name[len(prefix):]:finite(v) for name,v in ds.variables.items() if name.startswith(prefix)}
                if not np.array_equal(corrected(fields, source[period]),expected_ssh):
                    raise ValueError(f"Incorrect {period} HYCOM tide or pressure correction")
        baseline, anomaly, ib = calculate(clock, pressure, ds["daily_native_time"][:], run_ssh,
                                          ds["monthly_native_time"][:], ref_ssh, source)
        for name, expected in (("reference_zos",baseline), ("offshore_anomaly",anomaly), ("inverse_barometer",ib)):
            if not np.array_equal(finite(ds[name]), expected):
                raise ValueError(f"Incorrect staged boundary component: {name}")
        expected = (tide + anomaly + ib).astype("f4")
        if not np.array_equal(finite(combined["elevation"]), expected):
            raise ValueError("Combined boundary differs from TPXO + offshore anomaly + one pressure head")
    return paths["pure_tpxo"]


def verify_wind_boundary(input_root: Path, namelist_values: dict) -> Path:
    root = Path(input_root)
    manifest = load_json(root / MANIFEST)
    if set(manifest["files"]) != {"mesh", "obc", "pressure", "combined", "pure_tpxo", "components"}:
        raise ValueError("Wind boundary stage requires six bound input files")
    bindings = namelist_bindings(namelist_values)
    paths = {}
    for role, ref in manifest["files"].items():
        name = ref["path"]
        if Path(name).name != name or not name or role in bindings and name != bindings[role]:
            raise ValueError(f"Unbound wind boundary file: {role}")
        paths[role] = root / name
    if len(set(paths.values())) != len(paths):
        raise ValueError("Wind boundary artifact roles must be distinct")
    return audit(paths, manifest)


def build(args):
    target = args.output_dir
    target.mkdir(parents=True, exist_ok=True)
    tide_manifest = load_json(args.tide_manifest)
    if (tide_manifest["status"] != "ready" or tide_manifest["health"]["status"] != "pass"
            or tide_manifest["hashes"]["forcing_sha256"] != sha256_file(args.tide)):
        raise ValueError("Require the original verified TPXO forcing and its owner manifest")
    clock, ids, tide, points_hash = read_tide(args.tide)
    if sha256_file(args.points) != points_hash or not np.array_equal(read_dat_ids(args.obc), ids):
        raise ValueError("Ordered terminal points/OBC do not match the original TPXO forcing")
    with args.points.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not np.array_equal([int(r["node_id"]) for r in rows], ids):
        raise ValueError("TPXO coordinate rows must follow exact terminal OBC order")
    coords = np.array([[float(r["longitude"]), float(r["latitude"])] for r in rows])
    hycom = getattr(args,"hycom_package",None)
    if bool(hycom) == bool(args.daily or args.monthly) or not hycom and not (args.daily and args.monthly):
        raise ValueError("Choose both GLORYS --daily and --monthly or an evidenced --hycom-package")
    correction_fields = {}
    if hycom:
        daily_t, dz, monthly_t, mz, source, correction_fields = prepare_hycom(hycom,coords,args.nearest_limit_km)
    else:
        daily_t, dz, monthly_t, mz, source = prepare(args.daily,args.monthly,coords,args.nearest_limit_km)
    pressure = pressure_at_tide_clock(args.pressure, ids, clock, coords)
    baseline, anomaly, ib = calculate(clock, pressure, daily_t, dz, monthly_t, mz, source)
    with TemporaryDirectory(dir=target) as temp:
        stage = Path(temp)
        paths = dict(pure_tpxo=stage/"pure_tpxo.nc", components=stage/"wind_boundary_components.nc",
                     combined=stage/args.tide.name, pressure=args.pressure, mesh=args.mesh, obc=args.obc)
        shutil.copyfile(args.tide, paths["pure_tpxo"])
        write_components(paths["components"], clock, ids, coords, daily_t, dz, monthly_t, mz, baseline, anomaly, ib, correction_fields)
        boundary = Boundary(ids, coords[:,0], coords[:,1], (np.arange(len(ids)),), str(args.points))
        write_fvcom_forcing(paths["combined"], boundary, clock, tide+anomaly+ib, case_name="Combined prescribed sea level",
                            vertical_datum="TPXO zero plus offshore anomaly from local 1993-2012 mean and pressure head",
                            source_name=paths["components"], source_sha256=sha256_file(paths["components"]),
                            spatial_method="ordered common wet offshore stencil and exact FVCOM nodal pressure")
        def limits(v):
            return [float(np.min(v)), float(np.max(v))]
        manifest = dict(schema=SCHEMA, status="ready", physics=POLICY, offshore=source, points_sha256=points_hash,
                        files={key:file_ref(path) for key,path in paths.items()}, tide_manifest_sha256=sha256_file(args.tide_manifest),
                        start_utc=iso_utc(clock[:1])[0], end_utc=iso_utc(clock[-1:])[0], time_count=len(clock),
                        node_count=len(ids), reference_days=int(monthly_days(monthly_t).sum()),
                        ranges_m=dict(tide=limits(tide), offshore_anomaly=limits(anomaly), inverse_barometer=limits(ib),
                                      combined=limits(tide+anomaly+ib), reference_zos=limits(baseline)))
        surface_receipt = args.pressure.with_name("atmosphere_manifest.json")
        manifest.update(atmosphere_manifest_sha256=sha256_file(surface_receipt),atmosphere_provider=load_json(surface_receipt)["provider"])
        audit(paths, manifest)
        for key in ("pure_tpxo","components","combined"):
            os.replace(paths[key], target / paths[key].name)
        write_json_atomic(target / MANIFEST, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tide", "tide-manifest", "points", "pressure", "mesh", "obc", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("daily", "monthly", "hycom-package"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--nearest-limit-km", type=float, required=True, help="Maximum fallback distance established for the OBC")
    result = build(parser.parse_args())
    print(json.dumps({key:result[key] for key in ("status","node_count","time_count","ranges_m")}, indent=2))


if __name__ == "__main__":
    main()
