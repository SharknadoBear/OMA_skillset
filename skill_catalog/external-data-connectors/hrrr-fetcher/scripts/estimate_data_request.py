#!/usr/bin/env python3
"""Compatibility entry point for HRRR estimate-first planning."""

from __future__ import annotations

import sys

try:
    from .hrrr_fetcher import main
except ImportError:
    from hrrr_fetcher import main


if __name__ == "__main__":
    raise SystemExit(main(["estimate", *sys.argv[1:]]))
