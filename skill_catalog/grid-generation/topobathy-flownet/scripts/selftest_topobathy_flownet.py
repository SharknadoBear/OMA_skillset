"""Self-tests for topology, physical threshold, UTM, and schema behavior."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from topobathy_flownet_core import (
    SCHEMA_VERSION,
    affine_cell_area_m2,
    automatic_utm_epsg,
    build_arc_records,
    network_dat_rows,
    source_area_to_cells,
)


def lookup(values):
    def sample(points):
        return [values[(float(point[0]), float(point[1]))] for point in points]

    return sample


def test_physical_threshold() -> None:
    cell_area = affine_cell_area_m2(20.0, 2.0, 1.0, -10.0)
    assert math.isclose(cell_area, 202.0)
    assert source_area_to_cells(0.001, cell_area) == 5
    assert source_area_to_cells(1.0, 100.0) == 10_000


def test_automatic_utm() -> None:
    assert automatic_utm_epsg(-122.4, 37.8) == 32610
    assert automatic_utm_epsg(151.2, -33.9) == 32756


def test_topology_and_longest_path_order() -> None:
    # Two headwaters join, then a longer branch joins one step downstream.
    lines = [
        [(0, 2), (1, 1)],
        [(0, 0), (1, 1)],
        [(1, 1), (2, 1)],
        [(2, 2), (2, 1)],
        [(2, 1), (3, 0)],
    ]
    elevations = {
        (0.0, 2.0): 30.0,
        (0.0, 0.0): 28.0,
        (1.0, 1.0): 20.0,
        (2.0, 2.0): 25.0,
        (2.0, 1.0): 10.0,
        (3.0, 0.0): 0.0,
    }
    accumulations = {
        (0.0, 2.0): 1.0,
        (0.0, 0.0): 1.0,
        (1.0, 1.0): 2.0,
        (2.0, 2.0): 1.0,
        (2.0, 1.0): 3.0,
        (3.0, 0.0): 4.0,
    }
    records, qa = build_arc_records(
        lines,
        lookup(elevations),
        lookup(accumulations),
        cell_area_m2=100.0,
        node_tolerance_m=0.01,
    )
    assert len(records) == 5
    assert qa["headwater_segments"] == 3
    assert qa["terminal_segments"] == 1
    assert not qa["has_cycle_or_unresolved_topology"]
    assert not qa["segorder_errors"]
    outlet = next(record for record in records if record["downarc"] == -1)
    assert outlet["segorder"] == 3
    assert outlet["drainage_area_m2"] == 400.0
    assert all(record["SELEV"] >= record["EELEV"] for record in records)
    assert len(network_dat_rows(records)) == len(records)


def test_downhill_reorientation_and_deterministic_ids() -> None:
    lines_a = [[(1, 0), (0, 0)], [(2, 0), (1, 0)]]
    lines_b = list(reversed(lines_a))
    elevations = {(0.0, 0.0): 20.0, (1.0, 0.0): 10.0, (2.0, 0.0): 0.0}
    accumulations = {(0.0, 0.0): 1.0, (1.0, 0.0): 2.0, (2.0, 0.0): 3.0}
    first, qa_first = build_arc_records(
        lines_a,
        lookup(elevations),
        lookup(accumulations),
        cell_area_m2=25.0,
        node_tolerance_m=0.01,
    )
    second, qa_second = build_arc_records(
        lines_b,
        lookup(elevations),
        lookup(accumulations),
        cell_area_m2=25.0,
        node_tolerance_m=0.01,
    )
    compact = lambda records: [
        (record["arcid"], record["from_node"], record["to_node"], record["downarc"], record["segorder"])
        for record in records
    ]
    assert compact(first) == compact(second)
    assert not qa_first["segorder_errors"]
    assert not qa_second["segorder_errors"]


def test_manifest_schema_literal() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "schema.json"
        path.write_text(json.dumps({"schema": SCHEMA_VERSION}), encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8"))["schema"] == "topobathy_flownet_v1"


def main() -> None:
    tests = [
        test_physical_threshold,
        test_automatic_utm,
        test_topology_and_longest_path_order,
        test_downhill_reorientation_and_deterministic_ids,
        test_manifest_schema_literal,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} topobathy-flownet self-tests")


if __name__ == "__main__":
    main()
