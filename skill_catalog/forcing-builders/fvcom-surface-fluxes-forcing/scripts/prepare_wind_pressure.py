"""Prepare model-frame PSIMPLE_DRAG and absolute MSLP on a projected FVCOM mesh."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
from pyproj import CRS, Geod, Proj, Transformer
from scipy.interpolate import RegularGridInterpolator

from surface_fluxes_core import (
    MeshGeometry,
    PreparedSurfaceData,
    _time_variables,
    sha256_file,
    write_prepared_bundle,
)
from surface_fluxes_validation import _decode_exact_time, validate_forcing_files

PROVIDERS = ("hrrr", "era5", "cfsv2", "cfsr")
NAMES = {
    "hrrr": ("eastward_wind_10m", "northward_wind_10m", "air_pressure_at_mean_sea_level"),
    "era5": ("u10", "v10", "msl"),
    "cfsv2": ("eastward_wind", "northward_wind", "air_pressure_at_mean_sea_level"),
    "cfsr": ("eastward_wind", "northward_wind", "air_pressure_at_mean_sea_level"),
}
FIELDS = ("eastward_stress", "northward_stress", "absolute_air_pressure")


def require(predicate, message):
    if not predicate:
        raise ValueError(message)


def utc(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    require(stamp.utcoffset() is not None, "Explicit UTC offset required")
    return stamp.astimezone(timezone.utc)


def hour_axis(start, end):
    a, b = utc(start), utc(end)
    require(
        a <= b and not any((a.minute, a.second, a.microsecond, b.minute, b.second, b.microsecond)),
        "Request endpoints must be ordered whole UTC hours",
    )
    return np.arange(int(a.timestamp()), int(b.timestamp()) + 1, 3600, dtype="i8") * 1000


def stamp(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def record(path):
    return {"path": path.name, "sha256": sha256_file(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metric_mesh(path, model_crs):
    crs = CRS.from_user_input(model_crs)
    require(
        crs.is_projected and all(axis.unit_conversion_factor == 1 for axis in crs.axis_info[:2]),
        "Model Cartesian coordinates must use a projected CRS in meters",
    )
    with Path(path).open(encoding="ascii") as stream:
        nn = int(stream.readline().split("=")[1])
        ne = int(stream.readline().split("=")[1])
        elements = np.loadtxt(stream, max_rows=ne, dtype=int, ndmin=2)
        nodes = np.loadtxt(stream, ndmin=2)
    require(
        nodes.shape == (nn, 4)
        and elements.shape == (ne, 4)
        and np.array_equal(nodes[:, 0], np.arange(1, nn + 1))
        and np.array_equal(elements[:, 0], np.arange(1, ne + 1)),
        "Runtime mesh index/shape mismatch",
    )
    tri = elements[:, 1:] - 1
    require(tri.min() >= 0 and tri.max() < nn and np.isfinite(nodes).all(), "Invalid runtime mesh")
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    xy = nodes[:, 1:3]
    lon, lat = inverse.transform(*xy.T, errcheck=True)
    lonc, latc = inverse.transform(*xy[tri].mean(axis=1).T, errcheck=True)
    mesh = MeshGeometry(nodes[:, 0].astype(int), elements[:, 0], lon, lat, lonc, latc, tri, str(path))
    return mesh, crs


def obc_ids(path, count):
    with Path(path).open(encoding="ascii") as stream:
        number = int(stream.readline().split("=")[1])
        rows = np.loadtxt(stream, dtype=int, ndmin=2)
    require(
        rows.shape == (number, 3) and np.array_equal(rows[:, 0], np.arange(1, number + 1)),
        "Unexpected runtime ocean OBC rows",
    )
    ids = rows[:, 1]
    require(
        len(np.unique(ids)) == len(ids) and min(ids) > 0 and max(ids) <= count, "Invalid ocean OBC IDs"
    )
    return ids


def model_basis(crs, lon, lat):
    factors = Proj(crs).get_factors(lon, lat, errcheck=True)
    angle = np.asarray(factors.meridian_convergence) * np.pi / 180
    require(
        np.max(np.abs(factors.angular_distortion)) < 1e-4,
        "Wind rotation requires a locally conformal model CRS",
    )
    c, s = np.cos(angle), np.sin(angle)
    project = Transformer.from_crs(4326, crs, always_xy=True)
    x1, y1 = project.transform(lon, lat, errcheck=True)
    az = np.broadcast_to(90.0, np.shape(lon))
    e_lon, e_lat, _ = Geod(ellps="WGS84").fwd(lon, lat, az, np.full_like(az, 10.0))
    x2, y2 = project.transform(e_lon, e_lat, errcheck=True)
    length = np.hypot(x2 - x1, y2 - y1)
    error = np.max(np.hypot((x2 - x1) / length - c, (y2 - y1) / length - s))
    require(error < 1e-5, "Projected basis disagrees with an independent true-east geodesic")
    return c, s


def psimple_drag(u, v):
    speed = np.hypot(u, v)
    cd = np.where(speed >= 11.0, (0.49 + 0.065 * np.minimum(speed, 25.0)) * 1e-3, 1.2e-3)
    factor = 1.2 * cd * speed
    return factor * u, factor * v


def source_law(path):
    text = Path(path).read_text(encoding="ascii", errors="replace").lower()
    body = text.split("subroutine psimple_drag(", 1)[-1].split("end subroutine", 1)[0]
    lines = [re.sub(r"\s+", "", line.partition("!")[0]) for line in body.splitlines()]
    for line in (
        "tx=spdx(i)",
        "ty=spdy(i)",
        "wds=sqrt(tx*tx+ty*ty)",
        "cd=1.2e-3",
        "if(wds>=11.0_sp)cd=(0.49_sp+0.065_sp*wds)*1.e-3_sp",
        "if(wds>=25.0_sp)cd=(0.49_sp+0.065_sp*25.0_sp)*1.e-3_sp",
        "strx(i)=1.2_sp*cd*tx*wds",
        "stry(i)=1.2_sp*cd*ty*wds",
    ):
        require(line in lines, f"Cannot verify PSIMPLE_DRAG statement {line}")
    return record(Path(path))


def array(variable):
    values = np.ma.asarray(variable[:], dtype=float).filled(np.nan)
    require(np.isfinite(values).all(), f"Missing or nonfinite {variable.name}")
    return values


def source_time(ds):
    t = ds["time"]
    times = nc.num2date(
        array(t),
        t.units,
        calendar=getattr(t, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return np.array(
        [round(x.replace(tzinfo=timezone.utc).timestamp() * 1000) for x in times], dtype="i8"
    )


def geographic_points(lon, lat, longitude_axis):
    centre = longitude_axis.mean()
    return np.column_stack((lat, (lon - centre + 180.0) % 360.0 - 180.0 + centre))


def coordinates(ds, provider, mesh):
    if provider == "hrrr":
        require(
            getattr(ds, "domain", None) == "conus" and getattr(ds, "time_mode", None) == "analysis",
            "Expected a native HRRR CONUS analysis grid",
        )
        x, y = array(ds["x"]), array(ds["y"])
        project = Transformer.from_crs(4326, ds["crs"].crs_wkt, always_xy=True)

        def points(lon, lat):
            xq, yq = project.transform(lon, lat, errcheck=True)
            return np.column_stack((yq, xq))

    else:
        x, y = array(ds["longitude"]), array(ds["latitude"])

        def points(lon, lat):
            return geographic_points(lon, lat, x)

    require(len(x) > 1 and len(y) > 1, "Insufficient native interpolation support")
    pressure_axes = (y, x)
    if provider in ("cfsv2", "cfsr") and ds[NAMES[provider][2]].dimensions[1:] == (
        "mslp_latitude",
        "mslp_longitude",
    ):
        pressure_axes = tuple(array(ds[dim]) for dim in ds[NAMES[provider][2]].dimensions[1:])
        require(all(len(axis) > 1 for axis in pressure_axes), "Insufficient native pressure support")
        nodes = geographic_points(mesh.lon, mesh.lat, pressure_axes[1])
    else:
        nodes = points(mesh.lon, mesh.lat)
    return (y, x), points(mesh.lonc, mesh.latc), pressure_axes, nodes


def checked_fields(ds, provider, times):
    u, v, p = (ds[n] for n in NAMES[provider])
    spatial = ("y", "x") if provider == "hrrr" else ("latitude", "longitude")
    pressure_dimensions = {("time", *spatial)}
    if provider in ("cfsv2", "cfsr"):
        pressure_dimensions.add(("time", "mslp_latitude", "mslp_longitude"))
    require(
        u.dimensions == v.dimensions == ("time", *spatial) and p.dimensions in pressure_dimensions,
        "Unexpected provider field dimensions",
    )
    require(
        u.units.replace(" ", "") in ("ms-1", "ms**-1", "m/s") and v.units == u.units and p.units == "Pa",
        "Require matched 10-m wind in m/s and mean sea-level pressure in Pa",
    )
    if provider == "hrrr":
        for var, name, level in zip(
            (u, v, p),
            ("UGRD", "VGRD", "MSLMA"),
            ("10 m above ground", "10 m above ground", "mean sea level"),
        ):
            rows = json.loads(var.source_grib_metadata)
            require(len(rows) == len(times), "Missing HRRR per-hour source metadata")
            for row, ms in zip(rows, times):
                require(
                    row["idx_short_name"] == name
                    and row["source_idx_level"] == level
                    and row["forecast_time"] == 0
                    and row["step_type"] == "instant"
                    and utc(row["forecast_reference_time"]) == utc(stamp(ms)) == utc(row["valid_time"]),
                    f"Invalid f00 time, parameter or level for {name}",
                )
                if name != "MSLMA":
                    expected = (
                        "grid_to_earth_tangent_basis"
                        if row["uv_relative_to_grid"]
                        else "source_earth_relative"
                    )
                    require(
                        row["wind_rotation"] == expected,
                        "Expected paired winds converted to true east/north exactly once",
                    )
    elif provider == "era5":
        require(
            getattr(ds, "schema_version", "") == "era5_fields_v1"
            and (u.standard_name, v.standard_name) == ("eastward_wind", "northward_wind"),
            "Expected ERA5 connector output with geographical 10-m wind",
        )
    else:
        require(
            getattr(ds, "model", "") == provider and getattr(ds, "source_provider", "") == "ncei",
            "Expected the indicated NCEI CFS era and a whole-provider product",
        )
        pressure = json.loads(p.source_grib_metadata)
        require(
            pressure["short_name"].lower() in ("prmsl", "msl")
            and pressure["type_of_level"].lower() == "meansea",
            "CFS pressure is not native PRMSL",
        )
        for var in (u, v):
            wind = json.loads(var.source_grib_metadata)
            require(
                wind["type_of_level"].lower() == "heightaboveground" and wind["level"] == 10,
                "CFS wind is not at 10 m",
            )
            require(
                wind["uv_relative_to_grid"] == 0 or wind["grid_type"] in ("regular_ll", "regular_gg"),
                "CFS grid-relative wind is not on an unrotated geographic grid",
            )
    return array(u), array(v), array(p)


def write_pressure_slice(path, clock, ids, pressure):
    with nc.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("time", None)
        ds.createDimension("DateStrLen", 26)
        ds.createDimension("nobc", len(ids))
        _time_variables(ds, clock)
        ds.createVariable("obc_nodes", "i4", ("nobc",))[:] = ids
        var = ds.createVariable("air_pressure", "f4", ("time", "nobc"))
        var.units = "Pa"
        var.standard_name = "air_pressure_at_mean_sea_level"
        var[:] = pressure[:, ids - 1]


def verify_bundle(manifest_path, grid_path, obc_path, *, start=None, end=None):
    manifest_path = Path(manifest_path)
    m = read(manifest_path)
    require(
        m["schema"] == "fvcom_atmosphere_manifest_v1"
        and m["status"] == "ready"
        and m["provider"] in PROVIDERS,
        "Invalid atmosphere manifest",
    )
    require(
        m["mesh_sha256"] == sha256_file(grid_path) and m["obc_sha256"] == sha256_file(obc_path),
        "Atmosphere belongs to another runtime mesh or OBC",
    )
    require(
        m["wind_vector_frame"] == "model_xy" and m["pressure_reference"] == "absolute_mean_sea_level",
        "Unexpected stress frame or pressure reference",
    )
    paths = {}
    for key in ("surface", "obc_pressure"):
        row = m["files"][key]
        require(Path(row["path"]).name == row["path"], "Expected flat runtime product name")
        paths[key] = manifest_path.parent / row["path"]
        require(sha256_file(paths[key]) == row["sha256"], f"Changed {key} product")
    clock = hour_axis(m["start_utc"], m["end_utc"])
    require(m["records"] == len(clock) and m["cadence_seconds"] == 3600, "Incomplete declared coverage")
    if start is not None or end is not None:
        require(
            np.array_equal(hour_axis(start or m["start_utc"], end or m["end_utc"]), clock),
            "Forcing UTC endpoints differ from the specified interval",
        )
    with nc.Dataset(paths["surface"]) as ds, nc.Dataset(paths["obc_pressure"]) as cut:
        require(
            ds.data_model == "NETCDF3_CLASSIC"
            and ds.source == "FVCOM grid (unstructured) surface forcing"
            and ds.wind_vector_frame == "model_xy"
            and ds.pressure_reference == "absolute_mean_sea_level"
            and ds.source_provider == m["provider"]
            and ds.source_sha256 == m["source_index_sha256"],
            "Native file metadata disagrees with provider or frame",
        )
        require(
            np.array_equal(_decode_exact_time(ds), clock)
            and np.array_equal(_decode_exact_time(cut), clock),
            "Native clocks differ",
        )
        with Path(grid_path).open(encoding="ascii") as file:
            nn, ne = (int(file.readline().split("=")[1]) for _ in range(2))
        require(
            np.array_equal(ds["node_id"][:], np.arange(1, nn + 1))
            and np.array_equal(ds["element_id"][:], np.arange(1, ne + 1)),
            "Native ID order differs",
        )
        ids = obc_ids(obc_path, nn)
        require(np.array_equal(cut["obc_nodes"][:], ids), "OBC pressure order differs")
        for key, dim in (("uwind_stress", "nele"), ("vwind_stress", "nele"), ("air_pressure", "node")):
            var = ds[key]
            require(var.units == "Pa" and var.dimensions == ("time", dim), f"Invalid native field {key}")
            for i in range(0, len(clock), 24):
                values = np.ma.asarray(var[i : i + 24, :], dtype=float).filled(np.nan)
                require(np.all(np.isfinite(values)), f"Nonfinite {key}")
                if key == "air_pressure":
                    require(values.min() >= 50000 and values.max() <= 120000, "Implausible MSLP in Pa")
                    require(
                        np.array_equal(values[:, ids - 1], cut["air_pressure"][i : i + 24, :]),
                        "OBC slice differs from nodal pressure",
                    )
    return m


def prepare(source_index, grd, obc, model_crs, mod_bulk, start, end, case_name, output):
    index = read(source_index)
    provider = index["provider"]
    require(
        index["schema"] == "fvcom_wind_pressure_source_v1" and provider in PROVIDERS and index["files"],
        "Require a verified whole-provider source index",
    )
    clock = hour_axis(start, end)
    law = source_law(mod_bulk)
    mesh, crs = metric_mesh(grd, model_crs)
    ids = obc_ids(obc, len(mesh.node_ids))
    c, s = model_basis(crs, mesh.lonc, mesh.latc)
    source_root = Path(source_index).resolve().parent
    output = Path(output).resolve()
    require(not output.exists(), "Choose a fresh bundle directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".atmosphere-", dir=output.parent
    ) as temp, ExitStack() as mappings:
        work = Path(temp)
        fields = {}
        for field in FIELDS:
            width = len(mesh.node_ids) if field == FIELDS[2] else len(mesh.element_ids)
            fields[field] = np.memmap(
                work / (field + ".bin"), mode="w+", dtype="f4", shape=(len(clock), width)
            )
            mappings.callback(fields[field]._mmap.close)
        seen = np.zeros(len(clock), dtype=bool)
        wind_min, wind_max = np.inf, -np.inf
        health_keys = set()
        for row in index["files"]:
            data_path = source_root / row["path"]
            health_path = source_root / row["health_path"]
            require(sha256_file(data_path) == row["sha256"], f"Changed provider data {data_path.name}")
            if row["health_path"] not in health_keys:
                require(sha256_file(health_path) == row["health_sha256"], "Changed connector health")
                require(
                    sha256_file(source_root / row["archive_manifest_path"])
                    == row["archive_manifest_sha256"],
                    "Changed per-provider archive provenance",
                )
                health = read(health_path)
                require(health.get("passed", health.get("pass_all")) is True, "Provider health failed")
                health_keys.add(row["health_path"])
            with nc.Dataset(data_path) as ds:
                t = source_time(ds)
                pos = np.searchsorted(clock, t)
                require(
                    len(t)
                    and pos.max() < len(clock)
                    and np.array_equal(clock[pos], t)
                    and len(np.unique(pos)) == len(pos)
                    and not seen[pos].any(),
                    "Gap, duplicate or out-of-window source hours",
                )
                axes, elements, pressure_axes, nodes = coordinates(ds, provider, mesh)
                u, v, p = checked_fields(ds, provider, t)
                for k, i in enumerate(pos):
                    ue = RegularGridInterpolator(axes, u[k], bounds_error=True)(elements)
                    vn = RegularGridInterpolator(axes, v[k], bounds_error=True)(elements)
                    wind_min = min(wind_min, float(np.hypot(ue, vn).min()))
                    wind_max = max(wind_max, float(np.hypot(ue, vn).max()))
                    tx, ty = psimple_drag(ue * c - vn * s, ue * s + vn * c)
                    fields[FIELDS[0]][i] = tx
                    fields[FIELDS[1]][i] = ty
                    fields[FIELDS[2]][i] = RegularGridInterpolator(
                        pressure_axes, p[k], bounds_error=True
                    )(nodes)
                seen[pos] = True
        require(seen.all(), "Selected provider does not cover every requested hour")
        package = work / "bundle"
        package.mkdir()
        obc_file = package / "ordered_obc_pressure.nc"
        write_pressure_slice(obc_file, clock, ids, fields[FIELDS[2]])
        data = PreparedSurfaceData(
            "fvcom",
            clock,
            fields,
            {k: "Pa" for k in fields},
            mesh.lat,
            mesh.lon,
            mesh,
            str(source_index),
            sha256_file(source_index),
            (
                "Bilinear sampling on the provider grid without extrapolation",
                "True east/north to projected model x/y at element centroids",
                "Source-matched PSIMPLE_DRAG with positive wind-on-ocean stress",
            ),
        )
        bundle = write_prepared_bundle(
            data, package, case_name=case_name, packages=("wind", "pressure"), wind_mode="stress"
        )
        path = bundle.package_files["wind"]
        del data
        mappings.close()
        fields.clear()
        with nc.Dataset(path, "r+") as ds:
            ds.wind_vector_frame = "model_xy"
            ds.model_horizontal_crs = crs.to_string()
            ds.pressure_reference = "absolute_mean_sea_level"
            ds.source_provider = provider
            ds["air_pressure"].standard_name = "air_pressure_at_mean_sea_level"
            for name, component in (("uwind_stress", "model_x"), ("vwind_stress", "model_y")):
                ds[name].source_role = component + "_surface_stress"
        check = validate_forcing_files(
            [path], package / "qa", model_start_ms=clock[0], model_end_ms=clock[-1], mesh=mesh
        )
        for row in check["files"]:
            row["path"] = Path(row["path"]).relative_to(package).as_posix()
        check["qa_artifacts"] = {
            key: Path(value).relative_to(package).as_posix()
            for key, value in check["qa_artifacts"].items()
        }
        write(package / "surface_validation.json", check)
        (package / "surface.nml.inc").write_text(bundle.namelist, encoding="utf-8")
        manifest = dict(
            schema="fvcom_atmosphere_manifest_v1",
            status="ready",
            provider=provider,
            mesh_sha256=sha256_file(grd),
            obc_sha256=sha256_file(obc),
            source_index_sha256=sha256_file(source_index),
            start_utc=stamp(clock[0]),
            end_utc=stamp(clock[-1]),
            cadence_seconds=3600,
            records=len(clock),
            wind_vector_frame="model_xy",
            pressure_reference="absolute_mean_sea_level",
            model_horizontal_crs=crs.to_string(),
            drag_source=law,
            earth_wind_speed_range_m_s=[wind_min, wind_max],
            projected_grid_convergence_deg=[
                float(np.rad2deg(np.arctan2(s, c)).min()),
                float(np.rad2deg(np.arctan2(s, c)).max()),
            ],
            files={
                "surface": record(path),
                "obc_pressure": record(obc_file),
                "validation": record(package / "surface_validation.json"),
            },
        )
        write(package / "atmosphere_manifest.json", manifest)
        verify_bundle(package / "atmosphere_manifest.json", grd, obc, start=start, end=end)
        package.rename(output)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-index", "grd", "obc", "mod-bulk", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("model-crs", "start", "end", "case-name"):
        p.add_argument("--" + key, required=True)
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
