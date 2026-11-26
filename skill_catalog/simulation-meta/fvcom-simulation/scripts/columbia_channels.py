"""Recreate the retained Columbia North and South validation transects from bathymetry alone."""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from netCDF4 import Dataset
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        required=True,
        type=Path,
        help="Retained Columbia FVCOM grid NetCDF, EPSG:32610",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory receiving channels.geojson and fixed_channel_transects.png",
    )
    args = parser.parse_args()
    root = args.output
    source = args.grid
    root.mkdir(parents=True, exist_ok=True)
    project = Transformer.from_crs(4326, 32610, always_xy=True)
    world = Transformer.from_crs(32610, 4326, always_xy=True)
    shared = [(-124.076, 46.254), (-124.038, 46.258), (-124.007, 46.25)]
    south = [
        (-123.965, 46.221),
        (-123.927, 46.199),
        (-123.901, 46.190),
        (-123.851, 46.196),
        (-123.817, 46.200),
        (-123.772, 46.214),
        (-123.724, 46.237),
        (-123.669, 46.255),
    ]
    north = [
        (-123.953, 46.247),
        (-123.908, 46.231),
        (-123.881, 46.234),
        (-123.846, 46.259),
        (-123.807, 46.269),
        (-123.764, 46.272),
        (-123.726, 46.269),
        (-123.702, 46.263),
        (-123.669, 46.255),
    ]
    tail = [
        (-123.618, 46.257),
        (-123.566, 46.257),
        (-123.516, 46.268),
        (-123.485, 46.272),
        (-123.456, 46.260),
    ]
    with Dataset(source) as d:
        xy = np.column_stack([d["x"][:], d["y"][:]]).astype(float)
        nv = np.array(d["nv"][:], dtype=int).T - 1
        h = np.asarray(d["h"][:], dtype=float)
    if xy.shape != (24911, 2) or nv.shape != (44958, 3):
        raise ValueError("Expected the retained Columbia pilot mesh")
    centers = xy[nv].mean(axis=1)
    depth = h[nv].mean(axis=1)
    tri = Triangulation(*xy.T, triangles=nv)
    neighbors = tri.neighbors
    i, j = np.where(neighbors >= 0)
    k = neighbors[i, j]
    distance = np.linalg.norm(centers[i] - centers[k], axis=1)
    weight = distance * (1 + (12 / np.maximum(1, np.minimum(depth[i], depth[k]))) ** 2)
    graph = coo_matrix((weight, (i, k)), shape=(len(nv), len(nv))).tocsr()
    tree = cKDTree(centers)
    result = []
    fig, ax = plt.subplots(figsize=(11, 4))
    lon, lat = world.transform(*xy.T)
    ax.tripcolor(
        Triangulation(lon, lat, triangles=nv),
        h,
        cmap="Blues",
        vmin=0,
        vmax=20,
        shading="gouraud",
    )
    for name, branch, color in [
        ("South", south, "darkorange"),
        ("North", north, "crimson"),
    ]:
        pos = shared + branch + tail
        separation, anchors = tree.query(
            np.column_stack(project.transform(*np.array(pos).T))
        )
        if np.any(separation > 1500):
            raise ValueError(
                "Incorrect projected grid or channel waypoint outside the mesh"
            )
        path = [anchors[0]]
        for start, end in itertools.pairwise(anchors):
            _, pred = shortest_path(
                graph, directed=False, indices=start, return_predecessors=True
            )
            seg = [end]
            while seg[-1] != start:
                node = pred[seg[-1]]
                if node < 0:
                    raise ValueError("Disconnected anchors")
                seg.append(node)
            path.extend(reversed(seg[:-1]))
        pts = centers[path]
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        km = np.r_[np.arange(0, d[-1], 200.0), d[-1]]
        line = np.column_stack([np.interp(km, d, z) for z in pts.T])
        lon1, lat1 = world.transform(*line.T)
        ax.plot(lon1, lat1, color=color, linewidth=1.1, label=name)
        result.append(
            {
                "type": "Feature",
                "properties": {
                    "channel": name,
                    "distance_km": (km / 1000).tolist(),
                    "node_spacing_m": 200,
                    "waypoints": pos,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": np.column_stack([lon1, lat1]).tolist(),
                },
            }
        )
    ax.set(
        xlim=(-124.11, -123.44),
        ylim=(46.16, 46.299),
        xlabel="Longitude",
        ylabel="Latitude",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "fixed_channel_transects.png", dpi=160)
    (root / "channels.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": result}, indent=2)
    )


if __name__ == "__main__":
    main()
