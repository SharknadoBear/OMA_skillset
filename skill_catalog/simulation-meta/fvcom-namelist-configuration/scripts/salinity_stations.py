"""Append SATURN03 to the 53 retained geometric Columbia station definitions."""

import argparse, json
from pathlib import Path


def prepare(base, startup, out):
    import netCDF4 as nc
    import numpy as np
    from pyproj import Transformer

    mapping = json.loads((base / "station_mapping.json").read_bytes())
    dat = (base / "columbia_station.dat").read_text().strip().splitlines()
    if (
        mapping["station_count"] != 53
        or len(mapping["stations"]) != 53
        or len(dat) != 54
        or [r.split()[-1] for r in dat[1:]]
        != [str(r["station_id"]) for r in mapping["stations"]]
    ):
        raise ValueError("Unexpected retained station inventory")
    with nc.Dataset(startup) as f:
        nv = np.asarray(f["nv"][:], dtype=int).T - 1
        xy = np.column_stack((f["x"][:], f["y"][:])).astype(float)
        point = np.asarray(
            Transformer.from_crs(4326, 32610, always_xy=True).transform(
                -123.94001, 46.19973
            )
        )
        u = xy[nv[:, 1]] - xy[nv[:, 0]]
        v = xy[nv[:, 2]] - xy[nv[:, 0]]
        z = point - xy[nv[:, 0]]

        def cross(a, b):
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

        det = cross(u, v)
        if np.any(det == 0):
            raise ValueError("Degenerate native face")
        a = cross(z, v) / det
        b = cross(u, z) / det
        matches = np.flatnonzero((a >= 0) & (b >= 0) & (a + b <= 1)) + 1
        if matches.tolist() != [19424]:
            raise ValueError("SATURN03 does not occupy exactly native cell 19424")
        nodes = nv[19423]
        depth = float(np.mean(f["h"][nodes]))
    mapping["stations"].append(
        dict(
            name="SATURN03",
            station_id="SATURN03",
            role="salinity_observation",
            validation_eligible=True,
            cell_id=19424,
            longitude=-123.94001,
            latitude=46.19973,
            depth_m=depth,
            node_ids=(nodes + 1).tolist(),
            location_policy="contains instrument position; native station salinity averages three cell nodes",
        )
    )
    mapping["station_count"] += 1
    dat.append(f"54 -123.940010 46.199730 19424 {depth:.6f} SATURN03")
    out.mkdir(exist_ok=True, parents=True)
    files = {
        "columbia_station.dat": "\n".join(dat) + "\n",
        "station_mapping.json": json.dumps(
            mapping, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
    }
    for name, value in files.items():
        path = out / name
        if path.exists() and path.read_bytes() != value.encode():
            raise FileExistsError(path)
        path.write_text(value, newline="\n")
    return mapping["stations"][-1]


def cli(argv):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "startup", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args(argv)
    print(json.dumps(prepare(a.base, a.startup, a.output), indent=2))
    return 0
