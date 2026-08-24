#!/usr/bin/env python3
"""Selftests for the GSHHS coastline connector."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gshhs_coastline.fetch import fetch_gshhs_bbox  # noqa: E402
from gshhs_coastline.quality import summarize_product  # noqa: E402
from gshhs_coastline.sources import (  # noqa: E402
    expand_centered_topology_bbox,
    find_gshhs_cache,
    split_bbox_antimeridian,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def _require_cache():
    cache = find_gshhs_cache()
    assert cache is not None, "Expected local GSHHS cache for selftests."
    assert "h" in cache.available_resolutions, f"Expected high-resolution cache, found {cache.available_resolutions}"
    assert "f" in cache.available_resolutions, f"Expected full-resolution cache, found {cache.available_resolutions}"
    return cache


def test_cache_discovery() -> None:
    cache = _require_cache()
    assert (cache.gshhs_dir / "h" / "GSHHS_h_L1.shp").exists()
    assert (cache.gshhs_dir / "f" / "GSHHS_f_L1.shp").exists()


def test_bbox_clip_and_coastline_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = fetch_gshhs_bbox(
            (-75.35, 38.75, -74.95, 39.10),
            run_dir=tmp,
            name="delaware_micro",
            resolution="h",
            levels="1",
            formats="gpkg,geojson",
            make_plot=False,
            quiet=True,
        )
        assert not result.land_gdf.empty
        assert not result.coastline_gdf.empty
        assert "Polygon" in set(result.land_gdf.geom_type)
        assert Path(result.manifest["outputs"]["gpkg"]).exists()
        assert Path(result.manifest["outputs"]["land_geojson"]).exists()
        assert Path(result.manifest["outputs"]["coastline_geojson"]).exists()
        summary = summarize_product(result.manifest["outputs"]["gpkg"])
        assert summary["status"] == "pass", summary


def test_full_resolution_clip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = fetch_gshhs_bbox(
            (-75.35, 38.75, -74.95, 39.10),
            run_dir=tmp,
            name="delaware_full",
            resolution="f",
            levels="1",
            formats="gpkg",
            make_plot=False,
            quiet=True,
        )
        assert result.manifest["source"]["selected_resolution"] == "f"
        assert int(result.manifest["quality"]["land_polygons"]["feature_count"]) > 0


def test_empty_bbox_behavior() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = fetch_gshhs_bbox(
            (-140.0, 0.0, -139.8, 0.2),
            run_dir=tmp,
            name="open_ocean",
            resolution="h",
            levels="1",
            formats="gpkg",
            make_plot=False,
            quiet=True,
        )
        assert result.land_gdf.empty
        assert "GeoDataFrame is empty." in result.manifest["quality"]["land_polygons"]["warnings"]


def test_antimeridian_split() -> None:
    parts, meta = split_bbox_antimeridian((170.0, -10.0, -170.0, 10.0))
    assert len(parts) == 2
    assert meta["antimeridian_split"] is True


def test_centered_topology_coverage_contract() -> None:
    source, coverage = expand_centered_topology_bbox(
        (-75.8, 38.1, -74.7, 40.4), coverage_factor=3.0, lookahead_km=0.0
    )
    assert coverage["coverage_factor_lon"] == 3.0
    assert coverage["coverage_factor_lat"] == 3.0
    assert coverage["model_bbox_centrally_contained"] is True
    assert coverage["margins_degrees"]["west"] == coverage["margins_degrees"]["east"]
    assert coverage["margins_degrees"]["south"] == coverage["margins_degrees"]["north"]
    assert abs((source[0] + source[2]) / 2.0 - (-75.25)) < 1.0e-12
    try:
        expand_centered_topology_bbox((-75.8, 38.1, -74.7, 40.4), coverage_factor=1.99)
    except ValueError:
        pass
    else:
        raise AssertionError("coverage factor below two must fail")

    dateline_source, dateline = expand_centered_topology_bbox(
        (179.2, -1.0, -179.2, 1.0), coverage_factor=3.0, lookahead_km=100.0
    )
    dateline_parts, dateline_split = split_bbox_antimeridian(dateline_source)
    assert dateline["center_lonlat_unwrapped"][0] == 180.0
    assert dateline_split["antimeridian_split"] is True
    assert len(dateline_parts) == 2

    _high_lat_source, high_lat = expand_centered_topology_bbox(
        (-151.0, 59.0, -149.0, 60.0), coverage_factor=3.0, lookahead_km=100.0
    )
    assert high_lat["downstream_topology_eligible"] is True
    assert high_lat["coverage_factor_lon"] >= 3.0
    assert high_lat["coverage_factor_lat"] >= 3.0


def test_topology_product_layers_and_physical_coastline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = fetch_gshhs_bbox(
            model_bbox=(-75.25, 38.82, -75.05, 39.02),
            coverage_factor=3.0,
            lookahead_km=0.0,
            run_dir=tmp,
            name="delaware_topology",
            resolution="h",
            levels="1",
            formats="gpkg",
            make_plot=False,
            quiet=True,
        )
        assert result.manifest["schema_version"] == "gshhs_coastline_fetch_v2"
        coverage = result.manifest["topology_coverage"]
        assert coverage["downstream_topology_eligible"] is True, coverage
        assert coverage["physical_coastline_source_frame_overlap_m"] <= 1.0
        assert coverage["source_bbox_handling"]["parts"]
        assert coverage["source_component_sha256"]
        gpkg = Path(result.manifest["outputs"]["gpkg"])
        layers = set(gpd.list_layers(gpkg)["name"].tolist())
        assert {"land_polygons", "coastline_lines", "request_bbox", "source_footprint", "source_frame", "model_bbox"}.issubset(layers)
        summary = summarize_product(gpkg, result.manifest)
        assert summary["status"] == "pass", summary


def test_health_check_script() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = fetch_gshhs_bbox(
            (-75.35, 38.75, -74.95, 39.10),
            run_dir=tmp,
            name="health_case",
            resolution="h",
            levels="1",
            formats="gpkg",
            make_plot=False,
            quiet=True,
        )
        manifest = Path(result.manifest["outputs"]["manifest_json"])
        out = Path(tmp) / "health_check.json"
        plots = Path(tmp) / "health_plots"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "check_download_health.py"),
                "--request",
                str(manifest),
                "--run-dir",
                tmp,
                "--output",
                str(out),
                "--plots-dir",
                str(plots),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["status"] == "pass", data
        assert data["plots"]


def main() -> None:
    test_cache_discovery()
    test_bbox_clip_and_coastline_outputs()
    test_full_resolution_clip()
    test_empty_bbox_behavior()
    test_antimeridian_split()
    test_centered_topology_coverage_contract()
    test_topology_product_layers_and_physical_coastline()
    test_health_check_script()
    print("gshhs-coastline selftests passed")


if __name__ == "__main__":
    main()
