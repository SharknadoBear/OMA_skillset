"""NOAA CUDEM bathymetry fetch, mosaic, and FVCOM interpolation helpers."""

from .catalog import build_tile_index, load_tile_index, save_tile_index
from .fetch import FetchResult, fetch_cudem_bbox
from .interp import interpolate_to_points
from .tiles import (
    COLLECTION_ORDER,
    NoCoverageError,
    TileRecord,
    parse_cudem_tile_name,
    select_tiles,
)

__all__ = [
    "COLLECTION_ORDER",
    "FetchResult",
    "NoCoverageError",
    "TileRecord",
    "build_tile_index",
    "fetch_cudem_bbox",
    "interpolate_to_points",
    "load_tile_index",
    "parse_cudem_tile_name",
    "save_tile_index",
    "select_tiles",
]
