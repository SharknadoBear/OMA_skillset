from __future__ import annotations

from pathlib import Path
import sys

from shapely.geometry import Polygon, box


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fvcom_bdry_arc.boundary_resolution import (  # noqa: E402
    _build_resolved_domain_from_serialized_chains,
)


def _ring(minx: float, miny: float, maxx: float, maxy: float):
    return list(box(minx, miny, maxx, maxy).exterior.coords)[:-1]


def test_exterior_conflict_is_classified_before_interior_union() -> None:
    outer = _ring(0.0, 0.0, 100.0, 100.0)
    ordinary = [
        _ring(30.0 + 8.0 * i, 10.0 + 15.0 * j, 34.0 + 8.0 * i, 14.0 + 15.0 * j)
        for j in range(5)
        for i in range(8)
    ]
    valid_near_coast = _ring(10.0, 1.0, 20.0, 5.0)
    exterior_crossing = _ring(15.0, -2.0, 25.0, 3.0)
    chains = ordinary + [valid_near_coast, exterior_crossing]

    # This reproduces the historical failure mechanism: blanket repair of the
    # invalid 42-chain aggregate dissolves the crossing and valid coastal
    # features together, publishing only 40 reference holes.
    legacy = Polygon(outer, holes=chains).buffer(0)
    assert isinstance(legacy, Polygon)
    assert len(legacy.interiors) == 40

    resolved, report = _build_resolved_domain_from_serialized_chains(outer, chains)
    assert resolved.is_valid
    assert list(resolved.exterior.coords) == list(Polygon(outer).exterior.coords)
    assert len(resolved.interiors) == 41
    assert report["source_island_chain_count"] == 42
    assert report["strictly_contained_chain_count"] == 41
    assert report["exterior_conflict_chain_count"] == 1
    assert report["reference_hole_count"] == 41
    assert report["interior_union_merge_delta"] == 0
    assert report["exterior_conflicts"] == [
        {
            "chain_index_one_based": 42,
            "relation": "touches_serialized_exterior",
            "area_m2": 50.0,
        }
    ]


def test_overlapping_strictly_interior_chains_remain_a_merge_reference() -> None:
    outer = _ring(0.0, 0.0, 100.0, 100.0)
    chains = [_ring(10.0, 10.0, 20.0, 20.0), _ring(19.0, 12.0, 25.0, 18.0)]
    resolved, report = _build_resolved_domain_from_serialized_chains(outer, chains)
    assert resolved.is_valid
    assert len(resolved.interiors) == 1
    assert report["strictly_contained_chain_count"] == 2
    assert report["interior_union_merge_delta"] == 1
    assert report["exterior_conflicts"] == []


def main() -> int:
    test_exterior_conflict_is_classified_before_interior_union()
    test_overlapping_strictly_interior_chains_remain_a_merge_reference()
    print("resolved-domain serialization selftests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
