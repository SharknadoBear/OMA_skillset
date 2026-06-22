"""Fetch, cache, clip, and export NOAA CUSP coastline vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
import zipfile

import geopandas as gpd
from shapely.geometry import box

from .io import write_geo_outputs, write_gpkg_layer, write_metadata
from .merge import merge_with_fallback
from .osm import fetch_osm_coastline
from .plot import plot_coastline_satellite, plot_merged_coastline_satellite
from .progress import ProgressReporter, normalize_timeout
from .quality import summarize_quality
from .sources import build_region_index, load_region_index, save_region_index, select_region, validate_bbox
from .visual_qa import write_visual_review_files


@dataclass
class CuspFetchResult:
    """Result paths and metadata for one CUSP bbox fetch."""

    metadata: dict[str, object]
    outputs: dict[str, str]


def ensure_region_zip(
    region: dict[str, object],
    cache_dir: str | Path,
    *,
    client_timeout_s: float | int | None = 0,
    reporter: ProgressReporter | None = None,
) -> Path:
    """Download a regional ZIP into cache if needed."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_name = str(region.get("zip_name") or Path(str(region["url"])).name)
    zip_path = cache_dir / zip_name
    expected = region.get("http", {}).get("content_length") if isinstance(region.get("http"), dict) else None
    if zip_path.exists() and (not expected or zip_path.stat().st_size == expected):
        if reporter:
            reporter.event(
                "cusp-download",
                "using cached CUSP region ZIP",
                path=str(zip_path),
                size_mb=round(zip_path.stat().st_size / 1048576.0, 3),
            )
        return zip_path

    tmp_path = zip_path.with_suffix(zip_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    timeout = normalize_timeout(client_timeout_s)
    kwargs = {"timeout": timeout} if timeout is not None else {}
    if reporter:
        reporter.event(
            "cusp-download",
            "downloading CUSP region ZIP",
            url=region["url"],
            target=str(zip_path),
            expected_mb=round(float(expected) / 1048576.0, 3) if expected else None,
            client_timeout_s=timeout,
        )
    heartbeat = reporter.background_heartbeat("cusp-download", "waiting for CUSP ZIP response", url=region["url"]) if reporter else None
    if heartbeat:
        heartbeat.__enter__()
    try:
        response_ctx = urlopen(str(region["url"]), **kwargs)
        downloaded = 0
        with response_ctx as response, tmp_path.open("wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if reporter:
                    reporter.heartbeat(
                        "cusp-download",
                        "downloaded CUSP region ZIP",
                        downloaded_mb=round(downloaded / 1048576.0, 3),
                        expected_mb=round(float(expected) / 1048576.0, 3) if expected else None,
                    )
    finally:
        if heartbeat:
            heartbeat.__exit__(None, None, None)
    tmp_path.replace(zip_path)
    if reporter:
        reporter.event("cusp-download", "cached CUSP region ZIP", path=str(zip_path), size_mb=round(zip_path.stat().st_size / 1048576.0, 3))
    return zip_path


def extract_region_zip(
    zip_path: str | Path,
    extract_root: str | Path,
    region_key: str,
    *,
    reporter: ProgressReporter | None = None,
) -> Path:
    """Extract a regional ZIP and return the shapefile path."""

    zip_path = Path(zip_path)
    extract_dir = Path(extract_root) / region_key
    shp_name = zip_path.with_suffix(".shp").name
    shp_path = extract_dir / shp_name
    if shp_path.exists():
        if reporter:
            reporter.event("cusp-extract", "using extracted CUSP shapefile", path=str(shp_path))
        return shp_path
    extract_dir.mkdir(parents=True, exist_ok=True)
    if reporter:
        reporter.event("cusp-extract", "extracting CUSP region ZIP", zip_path=str(zip_path), extract_dir=str(extract_dir))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    candidates = sorted(extract_dir.glob("*.shp"))
    if not candidates:
        raise FileNotFoundError(f"no shapefile found after extracting {zip_path}")
    if reporter:
        reporter.event("cusp-extract", "found extracted CUSP shapefile", path=str(candidates[0]))
    return candidates[0]


def read_and_clip(
    shp_path: str | Path,
    bbox: tuple[float, float, float, float],
    *,
    reporter: ProgressReporter | None = None,
) -> gpd.GeoDataFrame:
    """Read CUSP features intersecting bbox and clip geometries to the bbox."""

    shp_path = Path(shp_path)
    if reporter:
        reporter.event("cusp-read", "reading CUSP features intersecting bbox", path=str(shp_path), bbox=bbox)
    gdf = gpd.read_file(shp_path, bbox=bbox, engine="pyogrio")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4269")
    gdf = gdf.to_crs("EPSG:4326")
    if gdf.empty:
        return gdf

    bbox_poly = box(*bbox)
    clipped = gdf.copy()
    clipped["geometry"] = clipped.geometry.intersection(bbox_poly)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
    clipped = clipped[clipped.geometry.geom_type.isin(["LineString", "MultiLineString", "GeometryCollection"])].copy()
    clipped = clipped.explode(index_parts=False, ignore_index=True)
    clipped = clipped[clipped.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    clipped = clipped.reset_index(drop=True)
    if reporter:
        reporter.event("cusp-read", "clipped CUSP features", raw_features=len(gdf), clipped_features=len(clipped))
    return clipped


def _parse_formats(formats: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(formats, str):
        values = tuple(x.strip().lower() for x in formats.split(",") if x.strip())
    else:
        values = tuple(str(x).strip().lower() for x in formats if str(x).strip())
    valid = {"shapefile", "gpkg", "geojson"}
    invalid = sorted(set(values) - valid)
    if invalid:
        raise ValueError(f"unsupported formats: {invalid}")
    return values or ("shapefile", "gpkg", "geojson")


def fetch_cusp_bbox(
    index_path: str | Path,
    bbox: tuple[float, float, float, float],
    *,
    run_dir: str | Path,
    name: str,
    region: str = "auto",
    formats: str | tuple[str, ...] | list[str] = ("shapefile", "gpkg", "geojson"),
    basemap_provider: str = "Esri.WorldImagery",
    allow_no_basemap: bool = False,
    make_plot: bool = True,
    fallback_policy: str = "none",
    merge_tolerance_m: float = 75.0,
    snap_tolerance_m: float = 100.0,
    min_fallback_fragment_m: float = 100.0,
    refresh_osm: bool = False,
    heartbeat_seconds: float = 30.0,
    client_timeout_s: float | int | None = 0,
    overpass_timeout_s: float | int | None = 0,
    progress_jsonl: str | Path | None = None,
    quiet: bool = False,
) -> CuspFetchResult:
    """Fetch and clip CUSP coastline for a bbox."""

    bbox = validate_bbox(bbox)
    fallback_policy = fallback_policy.lower().strip()
    valid_policies = {"none", "osm-overpass", "auto"}
    if fallback_policy not in valid_policies:
        raise ValueError(f"fallback_policy must be one of {sorted(valid_policies)}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(progress_jsonl) if progress_jsonl else run_dir / f"{name}_progress.jsonl"
    reporter = ProgressReporter(progress_path, heartbeat_seconds=heartbeat_seconds, quiet=quiet)
    index_path = Path(index_path)
    if not index_path.exists():
        with reporter.stage("cusp-index", "build missing CUSP region index", output=str(index_path)):
            save_region_index(build_region_index(client_timeout_s=client_timeout_s, reporter=reporter), index_path)
    with reporter.stage("cusp-index", "load/select CUSP region", index=str(index_path), region=region):
        index = load_region_index(index_path)
        selected = select_region(index, bbox, requested=region)
        reporter.event("cusp-index", "selected CUSP region", region=selected.get("key"), url=selected.get("url"))

    cache_root = index_path.parent
    with reporter.stage("cusp-download", "ensure CUSP region ZIP", region=selected.get("key")):
        zip_path = ensure_region_zip(selected, cache_root / "region_zips", client_timeout_s=client_timeout_s, reporter=reporter)
    with reporter.stage("cusp-extract", "ensure CUSP shapefile extraction", region=selected.get("key")):
        shp_path = extract_region_zip(zip_path, cache_root / "extracted", str(selected["key"]), reporter=reporter)
    with reporter.stage("cusp-read", "read and clip CUSP shoreline", bbox=bbox):
        clipped = read_and_clip(shp_path, bbox, reporter=reporter)

    with reporter.stage("write-primary", "write CUSP vector outputs"):
        outputs = write_geo_outputs(clipped, run_dir, name, formats=_parse_formats(formats))
    with reporter.stage("quality", "summarize CUSP geometry quality"):
        quality = summarize_quality(clipped, bbox)
    warnings = list(quality.get("warnings", []))

    if make_plot:
        png = run_dir / f"{name}_cusp_satellite.png"
        with reporter.stage("plot", "render CUSP satellite diagnostic", output=str(png)):
            _, plot_warnings = plot_coastline_satellite(
                clipped,
                bbox,
                png,
                title=f"CUSP shoreline: {name}",
                basemap_provider=basemap_provider,
                allow_no_basemap=allow_no_basemap,
            )
        outputs["satellite_png"] = str(png)
        warnings.extend(plot_warnings)

    fallback_metadata: dict[str, object] = {"policy": fallback_policy, "used": False}
    merge_report: dict[str, object] | None = None
    if fallback_policy in {"osm-overpass", "auto"}:
        primary_path = run_dir / f"{name}_cusp_primary.gpkg"
        with reporter.stage("write-primary", "write CUSP primary fallback layer", output=str(primary_path)):
            write_gpkg_layer(clipped, primary_path, layer="coastline")
        outputs["cusp_primary_gpkg"] = str(primary_path)

        with reporter.stage("osm", "fetch/convert OSM Overpass coastline candidates", bbox=bbox):
            fallback_candidates, osm_meta = fetch_osm_coastline(
                bbox,
                cache_root / "osm_overpass",
                name=name,
                refresh=refresh_osm,
                client_timeout_s=client_timeout_s,
                overpass_timeout_s=overpass_timeout_s,
                reporter=reporter,
            )
        fallback_path = run_dir / f"{name}_fallback_candidates.gpkg"
        with reporter.stage("write-fallback", "write fallback candidate layer", output=str(fallback_path), features=len(fallback_candidates)):
            write_gpkg_layer(fallback_candidates, fallback_path, layer="coastline")
        outputs["fallback_candidates_gpkg"] = str(fallback_path)

        with reporter.stage("merge", "merge CUSP primary and OSM fallback", primary_features=len(clipped), fallback_features=len(fallback_candidates)):
            merged, retained_fallback, merge_report = merge_with_fallback(
                clipped,
                fallback_candidates,
                bbox,
                merge_tolerance_m=merge_tolerance_m,
                snap_tolerance_m=snap_tolerance_m,
                min_fallback_fragment_m=min_fallback_fragment_m,
                reporter=reporter,
            )
        merged_path = run_dir / f"{name}_merged_coastline.gpkg"
        with reporter.stage("write-merged", "write merged coastline layer", output=str(merged_path), features=len(merged)):
            write_gpkg_layer(merged, merged_path, layer="coastline")
        outputs["merged_coastline_gpkg"] = str(merged_path)

        merge_report_path = run_dir / f"{name}_merge_report.json"
        write_metadata(merge_report, merge_report_path)
        outputs["merge_report_json"] = str(merge_report_path)
        warnings.extend(merge_report.get("warnings", []))

        if make_plot:
            merged_png = run_dir / f"{name}_merged_satellite.png"
            with reporter.stage("plot", "render merged satellite diagnostic", output=str(merged_png)):
                _, merged_plot_warnings = plot_merged_coastline_satellite(
                    clipped,
                    retained_fallback,
                    bbox,
                    merged_png,
                    title=f"Merged shoreline: {name}",
                    basemap_provider=basemap_provider,
                    allow_no_basemap=allow_no_basemap,
                )
            outputs["merged_satellite_png"] = str(merged_png)
            warnings.extend(merged_plot_warnings)

        fallback_metadata = {
            "policy": fallback_policy,
            "used": bool(merge_report["fallback_retained_count"] > 0),
            "osm": osm_meta,
            "merged_quality": summarize_quality(merged, bbox),
            "retained_fallback_feature_count": int(len(retained_fallback)),
        }

    if make_plot and "satellite_png" in outputs:
        image_paths = {"cusp_satellite_png": outputs["satellite_png"]}
        if "merged_satellite_png" in outputs:
            image_paths["merged_satellite_png"] = outputs["merged_satellite_png"]
        vector_keys = (
            "gpkg",
            "geojson",
            "shapefile",
            "cusp_primary_gpkg",
            "fallback_candidates_gpkg",
            "merged_coastline_gpkg",
        )
        vector_paths = {key: outputs[key] for key in vector_keys if key in outputs}
        with reporter.stage("visual-review", "write visual review manifest"):
            outputs.update(
                write_visual_review_files(
                    run_dir,
                    name,
                    bbox=bbox,
                    image_paths=image_paths,
                    vector_paths=vector_paths,
                    context={
                        "selected_region_key": selected.get("key"),
                        "fallback_policy": fallback_policy,
                        "quality_feature_count": quality.get("feature_count"),
                        "quality_total_length_m_web_mercator": quality.get("total_length_m_web_mercator"),
                        "agent_gate": "Numeric checks are not sufficient; inspect satellite overlay before production acceptance.",
                    },
                )
            )

    source_http = selected.get("http") if isinstance(selected.get("http"), dict) else {}
    outputs["progress_jsonl"] = str(progress_path)
    metadata = {
        "name": name,
        "bbox_wsen": bbox,
        "selected_region": {
            "key": selected.get("key"),
            "label": selected.get("label"),
            "url": selected.get("url"),
            "zip_name": selected.get("zip_name"),
            "last_modified": source_http.get("last_modified"),
            "content_length": source_http.get("content_length"),
            "size_mb": source_http.get("size_mb"),
        },
        "source": {
            "product": "NOAA NGS Continually Updated Shoreline Product (CUSP)",
            "source_page": index.get("source_page"),
            "metadata_url": index.get("metadata_url"),
            "mode": "official_nsde_region_zip",
            "cached_zip": str(zip_path),
            "extracted_shapefile": str(shp_path),
        },
        "crs": {
            "source_metadata": "NAD83 / EPSG:4269",
            "output": "EPSG:4326",
        },
        "quality": quality,
        "fallback": fallback_metadata,
        "merge_report": merge_report,
        "progress": {
            "progress_jsonl": str(progress_path),
            "heartbeat_seconds": heartbeat_seconds,
            "client_timeout_s": normalize_timeout(client_timeout_s),
            "overpass_timeout_s": normalize_timeout(overpass_timeout_s),
            "stage_elapsed_seconds": reporter.stage_elapsed,
        },
        "outputs": outputs,
        "warnings": warnings,
    }
    metadata_path = run_dir / f"{name}_metadata.json"
    write_metadata(metadata, metadata_path)
    outputs["metadata_json"] = str(metadata_path)
    metadata["outputs"] = outputs
    write_metadata(metadata, metadata_path)
    return CuspFetchResult(metadata=metadata, outputs=outputs)
