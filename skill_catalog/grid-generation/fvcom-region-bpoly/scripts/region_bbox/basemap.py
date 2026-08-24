from __future__ import annotations

import math
from pathlib import Path

_COASTLINE_CACHE = {}
_ONLINE_PROVIDER_FAILURES: set[str] = set()


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

    west, south, east, north = map(float, bbox)
    for source in _coastline_candidates(search_roots):
        try:
            key = str(source)
            if key not in _COASTLINE_CACHE:
                _COASTLINE_CACHE[key] = gpd.read_file(source)
            source_gdf = _COASTLINE_CACHE[key]
            gdf = source_gdf.cx[west:east, south:north]
            if gdf.empty:
                # Empty geometry is positive water-only evidence only when the
                # loaded source footprint independently covers the review bbox.
                # A clipped regional subset outside this window is no evidence.
                if not _source_bounds_cover_bbox(source_gdf, bbox):
                    continue
                return {
                    "status": "ok",
                    "source": str(source),
                    "feature_count": 0,
                    "coverage_kind": "water_only",
                }
            geom_types = set(gdf.geom_type.dropna())
            if any("Polygon" in geom_type for geom_type in geom_types):
                gdf.plot(ax=ax, facecolor="#eef2e6", edgecolor="#495057", linewidth=0.45, alpha=0.9, zorder=0.5)
            else:
                gdf.plot(ax=ax, color="#495057", linewidth=0.45, alpha=0.9, zorder=0.5)
            return {
                "status": "ok",
                "source": str(source),
                "feature_count": int(len(gdf)),
                "coverage_kind": "coastline_or_land",
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

        return len(list(mercantile.tiles(west, south, east, north, [zoom])))
    except Exception:
        return 0


def _bounded_zoom(west: float, south: float, east: float, north: float, zoom: int | None) -> tuple[int | str, dict]:
    if zoom is None:
        return "auto", {"requested_zoom": None, "effective_zoom": "auto", "tile_count": None, "max_tile_count": 64, "adjusted": False}
    effective = int(zoom)
    max_tiles = 64
    tile_count = _estimate_tile_count(west, south, east, north, effective)
    while tile_count > max_tiles and effective > 10:
        effective -= 1
        tile_count = _estimate_tile_count(west, south, east, north, effective)
    return effective, {
        "requested_zoom": int(zoom),
        "effective_zoom": effective,
        "tile_count": tile_count,
        "max_tile_count": max_tiles,
        "adjusted": effective != int(zoom),
    }


def _add_online_tiles(ax, source, zoom: int | None) -> dict:
    from contextily.plotting import add_attribution, warp_tiles
    import contextily.tile as tile
    from contextily.tile import bounds2img

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    effective_zoom, zoom_meta = _bounded_zoom(xmin, ymin, xmax, ymax, zoom)
    original_get = tile.requests.get

    def get_with_timeout(url, **kwargs):
        kwargs.setdefault("timeout", 3.0)
        return original_get(url, **kwargs)

    tile.requests.get = get_with_timeout
    try:
        image, extent = bounds2img(
            xmin,
            ymin,
            xmax,
            ymax,
            zoom=effective_zoom,
            source=source,
            ll=True,
            wait=0,
            max_retries=0,
            n_connections=2,
            use_cache=False,
        )
    finally:
        tile.requests.get = original_get
    image, extent = warp_tiles(image, extent, t_crs="EPSG:4326")
    if image.shape[2] == 1:
        image = image[:, :, 0]
    ax.imshow(image, extent=extent, interpolation="bilinear", aspect=ax.get_aspect())
    attribution = source.get("attribution") if hasattr(source, "get") else None
    if attribution:
        add_attribution(ax, attribution, font_size=6)
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
