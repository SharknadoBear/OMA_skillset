from __future__ import annotations


def add_basemap(ax, bbox, provider: str = "street", zoom: int | None = None) -> dict:
    """Best-effort street/satellite basemap. Never fails the workflow."""
    if provider in {"none", "off", "false", ""}:
        return {"enabled": False, "provider": provider, "status": "disabled"}
    try:
        import contextily as cx
        import xyzservices.providers as xyz

        source = xyz.OpenStreetMap.Mapnik
        label = "OpenStreetMap Standard"
        if provider.lower().startswith("sat"):
            source = xyz.Esri.WorldImagery
            label = "Esri World Imagery"
        cx.add_basemap(ax, crs="EPSG:4326", source=source, zoom=zoom, attribution_size=6)
        return {"enabled": True, "provider": provider, "source": label, "status": "ok", "zoom": zoom}
    except Exception as exc:  # pragma: no cover - depends on optional online tiles
        return {"enabled": False, "provider": provider, "status": f"unavailable: {exc}"}

