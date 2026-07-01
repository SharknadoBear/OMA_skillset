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
from gshhs_coastline.sources import find_gshhs_cache, split_bbox_antimeridian  # noqa: E402


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
    test_health_check_script()
    print("gshhs-coastline selftests passed")


if __name__ == "__main__":
    main()
