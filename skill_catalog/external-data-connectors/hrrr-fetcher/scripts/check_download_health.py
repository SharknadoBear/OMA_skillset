#!/usr/bin/env python3
"""Compatibility entry point for HRRR run health checks."""

from __future__ import annotations

import sys

try:
    from .hrrr_fetcher import main
except ImportError:
    from hrrr_fetcher import main


if __name__ == "__main__":
    raise SystemExit(main(["health", *sys.argv[1:]]))
