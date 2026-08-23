"""CFSR fetcher package."""

from .cfs_grib_core import (
    PRODUCTS,
    build_plan as build_atmospheric_plan,
    execute_request as execute_atmospheric_request,
    fetch_plan as fetch_atmospheric_plan,
    health as health_atmospheric,
    normalize_request as normalize_atmospheric_request,
    runtime_preflight,
)

__all__ = [
    "PRODUCTS",
    "build_atmospheric_plan",
    "execute_atmospheric_request",
    "fetch_atmospheric_plan",
    "health_atmospheric",
    "normalize_atmospheric_request",
    "runtime_preflight",
]
