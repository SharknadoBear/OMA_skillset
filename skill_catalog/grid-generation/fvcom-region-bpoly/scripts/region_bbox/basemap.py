from __future__ import annotations

import math
from pathlib import Path

_COASTLINE_CACHE = {}
_ONLINE_PROVIDER_FAILURES: set[str] = set()


def _display_longitude_segments(bbox) -> list[dict]:
    """Split a continuous display bbox into native [-180, 180] requests."""
    west, south, east, north = map(float, bbox)
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("basemap bbox must contain finite coordinates")
    if east <= west or north <= south:
        raise ValueError("basemap bbox must have increasing longitude and latitude")
    if east - west > 360.0 + 1.0e-9:
        raise ValueError("basemap display longitude span must not exceed 360 degrees")

    cuts = [west]
    first_k = math.floor((west + 180.0) / 360.0) + 1
    last_k = math.ceil((east + 180.0) / 360.0)
    for k in range(first_k, last_k):
        boundary = -180.0 + 360.0 * k
        if west < boundary < east:
            cuts.append(boundary)
    cuts.append(east)

    segments: list[dict] = []
    for display_west, display_east in zip(cuts[:-1], cuts[1:]):
        midpoint = 0.5 * (display_west + display_east)
        display_shift = 360.0 * math.floor((midpoint + 180.0) / 360.0)
        native_west = max(-180.0, min(180.0, display_west - display_shift))
        native_east = max(-180.0, min(180.0, display_east - display_shift))
        segments.append(
            {
                "native_bbox": [native_west, south, native_east, north],
                "display_bbox": [display_west, south, display_east, north],
                "display_shift_deg": display_shift,
            }
        )
    return segments


def _draw_base_fill(ax) -> None:
    ax.set_facecolor("#d9edf7")


def _source_bounds_cover_bbox(source_gdf, bbox, tolerance_deg: float = 1.0e-6) -> bool:
    """Return true only when the loaded source footprint covers the map bbox."""
    if source_gdf.empty:
        return False
    try:
        minx, miny, maxx, maxy = map(float, source_gdf.total_bounds)
        west, south, east, north = map(float, bbox)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (minx, miny, maxx, maxy, west, south, east, north)):
        return False
    if west <= east:
        lon_covered = minx <= west + tolerance_deg and maxx >= east - tolerance_deg
    else:
        # A wrapped review window needs coverage on both sides of the dateline.
        lon_covered = minx <= -180.0 + tolerance_deg and maxx >= 180.0 - tolerance_deg
    lat_covered = miny <= south + tolerance_deg and maxy >= north - tolerance_deg
    return bool(lon_covered and lat_covered)


def _workspace_roots(extra_roots: list[Path] | None = None) -> list[Path]:
    roots: list[Path] = []
    starts = [*(extra_roots or []), Path.cwd(), Path(__file__).resolve()]
    for start in starts:
        start = Path(start)
        if start.suffix and not start.is_dir():
            start = start.parent
        for path in [start, *start.parents]:
            if path not in roots:
                roots.append(path)
    return roots


def _coastline_candidates(extra_roots: list[Path] | None = None) -> list[Path]:
    rels = [
        Path("Workspace/Preprocessing/fvcom-cusp-coastline/cache/gshhg/GSHHS_shp/f/GSHHS_f_L1.shp"),
        Path("Workspace/Preprocessing/fvcom-cusp-coastline/cache/gshhg/GSHHS_shp/h/GSHHS_h_L1.shp"),
        Path("Workspace/Preprocessing/fvcom-cusp-coastline/cache/ne_10m_coastline.zip"),
    ]
    out: list[Path] = []
    for root in _workspace_roots(extra_roots):
        for rel in rels:
            path = root / rel
            if path.exists() and path not in out:
                out.append(path)
    return out


def _draw_offline_coastline(ax, bbox, search_roots: list[Path] | None = None) -> dict | None:
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"status": f"geopandas unavailable: {exc}"}

    segments = _display_longitude_segments(bbox)
    for source in _coastline_candidates(search_roots):
        try:
            key = str(source)
            if key not in _COASTLINE_CACHE:
                _COASTLINE_CACHE[key] = gpd.read_file(source)
            source_gdf = _COASTLINE_CACHE[key]
            if source_gdf.crs is not None and not source_gdf.crs.is_geographic:
                source_gdf = source_gdf.to_crs("EPSG:4326")

            clipped_segments = []
            source_covers_all = True
            for segment in segments:
                native_bbox = segment["native_bbox"]
                if not _source_bounds_cover_bbox(source_gdf, native_bbox):
                    source_covers_all = False
                    break
                west, south, east, north = native_bbox
                subset = source_gdf.cx[west:east, south:north].copy()
                if subset.empty:
                    continue
                from shapely.affinity import translate
                from shapely.geometry import box

                clip_box = box(west, south, east, north)
                subset.geometry = subset.geometry.intersection(clip_box)
                subset = subset[~subset.geometry.is_empty].copy()
                shift = float(segment["display_shift_deg"])
                if shift:
                    subset.geometry = subset.geometry.apply(lambda geom: translate(geom, xoff=shift))
                if not subset.empty:
                    clipped_segments.append(subset)

            if not source_covers_all:
                continue
            if not clipped_segments:
                # Empty geometry is positive water-only evidence only when the
                # loaded source footprint independently covers every native
                # longitude segment of the review bbox.
                return {
                    "status": "ok",
                    "source": str(source),
                    "feature_count": 0,
                    "coverage_kind": "water_only",
                    "antimeridian_composited": len(segments) > 1 or any(segment["display_shift_deg"] for segment in segments),
                    "display_coverage_fraction": 1.0,
                    "longitude_segments": segments,
                }
            feature_count = 0
            for gdf in clipped_segments:
                feature_count += int(len(gdf))
                geom_types = set(gdf.geom_type.dropna())
                if any("Polygon" in geom_type for geom_type in geom_types):
                    gdf.plot(ax=ax, facecolor="#eef2e6", edgecolor="#495057", linewidth=0.45, alpha=0.9, zorder=0.5)
                else:
                    gdf.plot(ax=ax, color="#495057", linewidth=0.45, alpha=0.9, zorder=0.5)
            return {
                "status": "ok",
                "source": str(source),
                "feature_count": feature_count,
                "coverage_kind": "coastline_or_land",
                "antimeridian_composited": len(segments) > 1 or any(segment["display_shift_deg"] for segment in segments),
                "display_coverage_fraction": 1.0,
                "longitude_segments": segments,
            }
        except Exception:
            continue
    return None


def _draw_minimal_background(ax) -> None:
    ax.grid(True, color="white", lw=0.8, alpha=0.9)
    ax.grid(True, color="0.72", lw=0.35, alpha=0.6)


def _provider_chain(provider_norm: str):
    import xyzservices.providers as xyz

    if provider_norm.startswith("sat"):
        return [(xyz.Esri.WorldImagery, "Esri World Imagery")]
    if provider_norm in {"topo", "topographic", "regional_context", "regional-context"}:
        return [
            (xyz.Esri.WorldTopoMap, "Esri World Topographic Map"),
            (xyz.OpenTopoMap, "OpenTopoMap"),
            (xyz.CartoDB.Voyager, "CartoDB Voyager"),
            (xyz.OpenStreetMap.Mapnik, "OpenStreetMap Standard"),
        ]
    if provider_norm in {"open_topo", "open-topo", "opentopomap"}:
        return [(xyz.OpenTopoMap, "OpenTopoMap")]
    if provider_norm in {"road", "street", "road_detail"}:
        return [
            (xyz.Esri.WorldStreetMap, "Esri World Street Map"),
            (xyz.CartoDB.Voyager, "CartoDB Voyager"),
            (xyz.OpenStreetMap.Mapnik, "OpenStreetMap Standard"),
        ]
    if provider_norm in {"esri_street", "esri-street", "worldstreetmap", "world_street_map"}:
        return [(xyz.Esri.WorldStreetMap, "Esri World Street Map")]
    if provider_norm in {"voyager", "cartodb_voyager", "cartodb-voyager"}:
        return [(xyz.CartoDB.Voyager, "CartoDB Voyager")]
    if provider_norm in {"osm", "openstreetmap"}:
        return [(xyz.OpenStreetMap.Mapnik, "OpenStreetMap Standard")]
    if provider_norm in {"light", "positron", "carto"}:
        return [(xyz.CartoDB.Positron, "CartoDB Positron")]
    return [(xyz.Esri.WorldTopoMap, "Esri World Topographic Map")]


def _estimate_tile_count(west: float, south: float, east: float, north: float, zoom: int) -> int:
    try:
        import mercantile

        return sum(
            len(list(mercantile.tiles(*segment["native_bbox"], [zoom])))
            for segment in _display_longitude_segments([west, south, east, north])
        )
    except Exception:
        return 0


def _automatic_zoom(west: float, south: float, east: float, north: float) -> int:
    lon_length = max(abs(float(east) - float(west)), 1.0e-9)
    lat_length = max(abs(float(north) - float(south)), 1.0e-9)
    zoom_lon = math.ceil(math.log2(720.0 / lon_length))
    zoom_lat = math.ceil(math.log2(720.0 / lat_length))
    return int(min(zoom_lon, zoom_lat))


def _bounded_zoom(west: float, south: float, east: float, north: float, zoom: int | None) -> tuple[int | str, dict]:
    if zoom is None:
        effective = _automatic_zoom(west, south, east, north)
        requested_zoom = None
    else:
        effective = int(zoom)
        requested_zoom = int(zoom)
    max_tiles = 64
    tile_count = _estimate_tile_count(west, south, east, north, effective)
    while tile_count > max_tiles and effective > 10:
        effective -= 1
        tile_count = _estimate_tile_count(west, south, east, north, effective)
    return effective, {
        "requested_zoom": requested_zoom,
        "effective_zoom": effective,
        "tile_count": tile_count,
        "max_tile_count": max_tiles,
        "adjusted": requested_zoom is not None and effective != requested_zoom,
    }


def _warp_tiles_edge_safe(image, extent, target_crs="EPSG:4326"):
    """Warp tile pixels without extending half a pixel across the world edge."""
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT

    height, width, band_count = image.shape
    min_x, max_x, min_y, max_y = map(float, extent)
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=band_count,
            dtype=str(image.dtype.name),
            crs="EPSG:3857",
            transform=transform,
        ) as raster:
            raster.write(image.transpose(2, 0, 1))
        with memfile.open() as raster:
            with WarpedVRT(raster, crs=target_crs, resampling=Resampling.bilinear) as warped:
                warped_image = warped.read().transpose(1, 2, 0)
                bounds = warped.bounds
    return warped_image, (bounds.left, bounds.right, bounds.bottom, bounds.top)


def _add_online_tiles(ax, source, zoom: int | None) -> dict:
    from contextily.plotting import add_attribution
    import contextily.tile as tile
    from contextily.tile import bounds2img

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    effective_zoom, zoom_meta = _bounded_zoom(xmin, ymin, xmax, ymax, zoom)
    segments = _display_longitude_segments([xmin, ymin, xmax, ymax])
    original_get = tile.requests.get

    def get_with_timeout(url, **kwargs):
        kwargs.setdefault("timeout", 3.0)
        return original_get(url, **kwargs)

    tile.requests.get = get_with_timeout
    try:
        rendered_segments = []
        rendered_images = []
        for segment in segments:
            image, extent = bounds2img(
                *segment["native_bbox"],
                zoom=effective_zoom,
                source=source,
                ll=True,
                wait=0,
                max_retries=0,
                n_connections=2,
                use_cache=False,
            )
            image, extent = _warp_tiles_edge_safe(image, extent, target_crs="EPSG:4326")
            if image.shape[2] == 1:
                image = image[:, :, 0]
            shift = float(segment["display_shift_deg"])
            displayed_extent = (extent[0] + shift, extent[1] + shift, extent[2], extent[3])
            display_west, display_south, display_east, display_north = segment["display_bbox"]
            tolerance = max(1.0e-5, (display_east - display_west) * 1.0e-6)
            covered = bool(
                displayed_extent[0] <= display_west + tolerance
                and displayed_extent[1] >= display_east - tolerance
                and displayed_extent[2] <= display_south + tolerance
                and displayed_extent[3] >= display_north - tolerance
            )
            if not covered:
                raise RuntimeError(
                    f"tile segment does not cover requested display bbox: requested={segment['display_bbox']} rendered={displayed_extent}"
                )
            rendered_images.append((image, displayed_extent))
            rendered_segments.append(
                {
                    **segment,
                    "warped_native_extent": list(map(float, extent)),
                    "displayed_extent": list(map(float, displayed_extent)),
                    "pixel_shape": list(image.shape),
                    "coverage_pass": True,
                }
            )
    finally:
        tile.requests.get = original_get
    for image, displayed_extent in rendered_images:
        ax.imshow(image, extent=displayed_extent, interpolation="bilinear", aspect=ax.get_aspect())
    attribution = source.get("attribution") if hasattr(source, "get") else None
    if attribution:
        add_attribution(ax, attribution, font_size=6)
    zoom_meta.update(
        {
            "antimeridian_composited": len(segments) > 1 or any(segment["display_shift_deg"] for segment in segments),
            "display_coverage_fraction": 1.0,
            "longitude_segment_count": len(rendered_segments),
            "longitude_segments": rendered_segments,
        }
    )
    return zoom_meta


def add_basemap(
    ax,
    bbox,
    provider: str = "topo",
    zoom: int | None = None,
    search_roots: list[Path] | None = None,
) -> dict:
    """Add required map context. Never leaves a blank background."""
    requested_provider = provider
    provider_norm = (provider or "topo").lower()
    if provider_norm == "auto":
        provider_norm = "topo"
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    _draw_base_fill(ax)

    if provider_norm not in {"none", "off", "false", "", "fallback", "offline", "built-in", "builtin"}:
        try:
            failures: list[dict] = []
            chain = _provider_chain(provider_norm)
            for source, label in chain:
                if label in _ONLINE_PROVIDER_FAILURES:
                    failures.append({"provider": label, "error": "skipped_after_prior_failure"})
                    continue
                try:
                    zoom_meta = _add_online_tiles(ax, source, zoom)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    return {
                        "enabled": True,
                        "required": True,
                        "provider": requested_provider,
                        "resolved_provider": provider_norm,
                        "provider_chain": [item[1] for item in chain],
                        "selected_provider": label,
                        "source": label,
                        "status": "ok",
                        "geography_usable": True,
                        "zoom": zoom_meta.get("effective_zoom"),
                        "requested_zoom": zoom_meta.get("requested_zoom"),
                        "zoom_adjusted": zoom_meta.get("adjusted"),
                        "tile_count": zoom_meta.get("tile_count"),
                        "max_tile_count": zoom_meta.get("max_tile_count"),
                        "antimeridian_composited": bool(zoom_meta.get("antimeridian_composited", False)),
                        "display_coverage_fraction": zoom_meta.get("display_coverage_fraction", 1.0),
                        "longitude_segment_count": zoom_meta.get("longitude_segment_count", 1),
                        "longitude_segments": zoom_meta.get("longitude_segments", []),
                        "request_timeout_seconds": 3.0,
                        "provider_failures": failures,
                    }
                except Exception as exc:  # pragma: no cover - depends on optional online tiles
                    _ONLINE_PROVIDER_FAILURES.add(label)
                    failures.append({"provider": label, "error": str(exc)})
            online_status = "online tiles unavailable from provider chain"
        except Exception as exc:  # pragma: no cover - depends on optional online tiles
            online_status = f"online tiles unavailable: {exc}"
            failures = [{"provider": requested_provider, "error": str(exc)}]
    else:
        online_status = "online tiles skipped; using required offline fallback"
        failures = []

    coast = _draw_offline_coastline(ax, bbox, search_roots=search_roots)
    if coast and coast.get("status") == "ok":
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return {
            "enabled": True,
            "required": True,
            "provider": requested_provider,
            "resolved_provider": provider_norm,
            "source": "offline coastline background",
            "status": coast["status"],
            "geography_usable": True,
            "coastline_source": coast.get("source"),
            "coastline_feature_count": coast.get("feature_count"),
            "coastline_coverage_kind": coast.get("coverage_kind"),
            "antimeridian_composited": bool(coast.get("antimeridian_composited", False)),
            "display_coverage_fraction": coast.get("display_coverage_fraction", 1.0),
            "longitude_segments": coast.get("longitude_segments", []),
            "tile_status": online_status,
            "zoom": zoom,
            "provider_failures": failures,
        }

    _draw_minimal_background(ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    status = coast.get("status") if coast else "no offline coastline cache found"
    return {
        "enabled": True,
        "required": True,
        "provider": requested_provider,
        "resolved_provider": provider_norm,
        "source": "minimal geographic background",
        "status": "fallback_minimal",
        "geography_usable": False,
        "coastline_status": status,
        "tile_status": online_status,
        "zoom": zoom,
        "provider_failures": failures,
    }
