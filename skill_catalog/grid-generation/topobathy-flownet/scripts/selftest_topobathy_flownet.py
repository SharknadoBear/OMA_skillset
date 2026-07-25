"""Self-tests for topology, physical threshold, UTM, and schema behavior."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import run_topobathy_flownet as runner

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


def test_regular_lonlat_netcdf_ascending_lat_and_sign() -> None:
    import geopandas as gpd
    import numpy as np
    import rasterio
    import xarray as xr
    from pyproj import CRS
    from shapely.geometry import box

    longitudes = np.array([-123.2, -123.1, -123.0, -122.9], dtype="float64")
    latitudes = np.array([37.0, 37.1, 37.2, 37.3], dtype="float64")
    positive_down_depth = np.arange(1, 17, dtype="float32").reshape(4, 4)
    positive_down_depth[1, 2] = np.nan
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        netcdf_path = root / "ascending_lat.nc"
        xr.Dataset(
            {
                "depth": (
                    ("lat", "lon"),
                    positive_down_depth,
                    {"units": "m", "positive": "down"},
                )
            },
            coords={"lat": latitudes, "lon": longitudes},
        ).to_netcdf(netcdf_path)

        values, transform, crs, adapter = runner.load_regular_lonlat_netcdf(netcdf_path, "depth")
        assert values.shape == positive_down_depth.shape
        assert math.isclose(values[0, 0], positive_down_depth[-1, 0])
        assert np.isnan(values[-2, 2])
        center_x, center_y = transform * (0.5, 0.5)
        assert math.isclose(center_x, longitudes[0])
        assert math.isclose(center_y, latitudes[-1])
        assert crs.to_epsg() == 4326
        assert adapter["axis_normalization"]["latitude_reversed"]
        assert not adapter["axis_normalization"]["longitude_reversed"]
        assert adapter["nonfinite_nodata_count"] == 1
        assert adapter["source_fill_value"] is None

        mask_path = root / "mask.gpkg"
        mask = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(-123.24, 36.96, -122.86, 37.34)],
            crs="EPSG:4326",
        )
        mask.to_file(mask_path, layer="mask", driver="GPKG")
        projected_surface = root / "projected.tif"
        projected_mask = root / "projected_mask.gpkg"
        prep = runner.prepare_projected_inputs(
            surface_path=netcdf_path,
            surface_variable="depth",
            surface_positive="down",
            mask=runner.load_mask(mask_path, "mask"),
            target_crs=CRS.from_epsg(32610),
            target_resolution_m=2_000.0,
            projected_surface=projected_surface,
            projected_mask=projected_mask,
        )
        assert prep["input_adapter"]["name"] == "xarray_regular_lonlat_netcdf_v1"
        assert prep["input_adapter"]["selected_variable"] == "depth"
        with rasterio.open(projected_surface) as dataset:
            projected_values = dataset.read(1, masked=True).compressed()
        assert projected_values.size > 0
        assert float(projected_values.max()) < 0.0


def test_windows_grass_argv_preserves_spaced_paths() -> None:
    surface = r"C:\Users\Bear\OneDrive - PNNL\A&B (test)\projected surface.tif"
    location = r"C:\Users\Bear\OneDrive - PNNL\run folder\grassdata\topobathy_flownet"
    logical = ["grass85", "-c", surface, location, "-e"]
    process_argv = runner.build_grass_direct_argv(
        logical,
        grass_python=r"C:\OSGeo4W\bin\python3.exe",
        grass_script=r"C:\OSGeo4W\apps\grass\grass85\etc\grass85.py",
    )
    assert process_argv == [
        r"C:\OSGeo4W\bin\python3.exe",
        r"C:\OSGeo4W\apps\grass\grass85\etc\grass85.py",
        "-c",
        surface,
        location,
        "-e",
    ]
    assert process_argv[3] == surface
    assert process_argv[4] == location


def main() -> None:
    tests = [
        test_physical_threshold,
        test_automatic_utm,
        test_topology_and_longest_path_order,
        test_downhill_reorientation_and_deterministic_ids,
        test_manifest_schema_literal,
        test_regular_lonlat_netcdf_ascending_lat_and_sign,
        test_windows_grass_argv_preserves_spaced_paths,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} topobathy-flownet self-tests")


if __name__ == "__main__":
    main()
