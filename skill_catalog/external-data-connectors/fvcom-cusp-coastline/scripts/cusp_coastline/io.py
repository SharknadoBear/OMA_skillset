"""I/O helpers for clipped CUSP products."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import geopandas as gpd


def write_gpkg_layer(gdf: gpd.GeoDataFrame, path: str | Path, *, layer: str = "coastline") -> Path:
    """Write one GeoPackage layer, replacing an existing file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")
    return path


def write_geo_outputs(
    gdf: gpd.GeoDataFrame,
    run_dir: str | Path,
    name: str,
    *,
    formats: tuple[str, ...] = ("shapefile", "gpkg", "geojson"),
) -> dict[str, str]:
    """Write requested geospatial outputs and return paths by format."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    stem = f"{name}_cusp_coastline"

    if "gpkg" in formats:
        gpkg = run_dir / f"{stem}.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        gdf.to_file(gpkg, layer="coastline", driver="GPKG", engine="pyogrio")
        outputs["gpkg"] = str(gpkg)

    if "geojson" in formats:
        geojson = run_dir / f"{stem}.geojson"
        if geojson.exists():
            geojson.unlink()
        gdf.to_file(geojson, driver="GeoJSON", engine="pyogrio")
        outputs["geojson"] = str(geojson)

    if "shapefile" in formats:
        zip_path = run_dir / f"{stem}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with tempfile.TemporaryDirectory(prefix=f"{stem}_") as tmp:
            tmp_dir = Path(tmp)
            shp = tmp_dir / f"{stem}.shp"
            gdf.to_file(shp, driver="ESRI Shapefile", engine="pyogrio")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for path in sorted(tmp_dir.glob(f"{stem}.*")):
                    zf.write(path, arcname=path.name)
        outputs["shapefile"] = str(zip_path)

    return outputs


def write_metadata(metadata: dict[str, object], path: str | Path) -> Path:
    """Write metadata JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return path


def copy_tree_clean(src: str | Path, dst: str | Path) -> None:
    """Replace a directory tree with another copy."""

    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
