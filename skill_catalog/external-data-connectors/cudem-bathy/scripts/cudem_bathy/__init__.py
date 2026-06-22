"""NOAA CUDEM bathymetry fetch, mosaic, and FVCOM interpolation helpers."""

from .catalog import build_tile_index, load_tile_index, save_tile_index
from .bathy_fetch import BathyFetchResult, fetch_bathy_bbox
from .fetch import FetchResult, fetch_cudem_bbox
from .interp import interpolate_to_points
from .sources import (
    BathySourceRecord,
    build_bathy_source_index,
    load_bathy_source_index,
    save_bathy_source_index,
)
from .tiles import (
    COLLECTION_ORDER,
    NoCoverageError,
    TileRecord,
    parse_cudem_tile_name,
    select_tiles,
)

__all__ = [
    "COLLECTION_ORDER",
    "BathyFetchResult",
    "BathySourceRecord",
    "FetchResult",
    "NoCoverageError",
    "TileRecord",
    "build_bathy_source_index",
    "build_tile_index",
    "fetch_bathy_bbox",
    "fetch_cudem_bbox",
    "interpolate_to_points",
    "load_bathy_source_index",
    "load_tile_index",
    "parse_cudem_tile_name",
    "save_bathy_source_index",
    "save_tile_index",
    "select_tiles",
]
