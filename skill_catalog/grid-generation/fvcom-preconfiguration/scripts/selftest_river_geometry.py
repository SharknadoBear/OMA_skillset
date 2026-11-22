"""Exercise the intake geometry, ocean separation and oriented control-volume boundary."""
import tempfile
from pathlib import Path

import numpy as np
from pyproj import Transformer

from prepare_river_geometry import prepare


def run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        grid, obc = root / "grid.dat", root / "obc.dat"
        nx, ny = 6, 4
        xy = [(500000 + 100 * i, 5100000 + 100 * j) for j in range(ny) for i in range(nx)]
        elements = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i + 1
                elements.extend(((a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)))
        rows = [f"Node Number = {len(xy)}", f"Cell Number = {len(elements)}"]
        rows.extend(f"{i} {a} {b} {c}" for i, (a, b, c) in enumerate(elements, 1))
        rows.extend(f"{i} {x} {y} 10" for i, (x, y) in enumerate(xy, 1))
        grid.write_text("\n".join(rows) + "\n", encoding="ascii")
        ocean = [1 + j * nx for j in range(ny)]
        obc.write_text("OBC Node Number = 4\n" + "".join(f"{i} {n} 1\n" for i, n in enumerate(ocean, 1)), encoding="ascii")
        ll = Transformer.from_crs(32610, 4326, always_xy=True).transform
        request = dict(horizontal_crs="EPSG:32610", cutoff_description="eastern test face", gauge_lonlat=ll(500600, 5100150),
                       cutoff_line_lonlat=[ll(500500, 5100000), ll(500500, 5100300)], cutoff_tolerance_m=1,
                       section_line_lonlat=[ll(500245, 5099900), ll(500245, 5100400)], wet_depth_floor_m=2)
        a, b = prepare(grid, obc, request)
        assert np.isclose(a["source_area_m2"], 3000.0)
        assert np.allclose([n["weight"] for n in a["nodes"]], [1 / 6, 1 / 3, 1 / 3, 1 / 6])
        assert set(ocean).isdisjoint(n["node_id"] for n in a["nodes"])
        # A 1 m/s westward flow through the stair-stepped cut transports 3000 m3/s,
        # regardless of which native edges form it; normals cancel lateral steps.
        transport = sum(-e["outward_normal_east"] * e["length_m"] * 10 for e in b["section"])
        assert np.isclose(transport, 3000) and 0 < len(b["upstream_element_ids"]) < len(elements)
        for override in ({"wet_depth_floor_m": 11}, {"section_line_lonlat": [ll(500245, 5100150), ll(500245, 5100180)]},
                         {"cutoff_line_lonlat": [ll(500000, 5100000), ll(500000, 5100300)]}, {"horizontal_crs": "EPSG:4326"}):
            try:
                prepare(grid, obc, {**request, **override})
            except ValueError:
                pass
            else:
                raise AssertionError(f"Expected rejection for {override}")
    print("PASS river intake, oriented section, wetness and OBC guards")


if __name__ == "__main__":
    run()
