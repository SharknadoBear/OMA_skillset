from __future__ import annotations

from pathlib import Path


def read_2dm_bbox(path: str | Path) -> list[float]:
    lons, lats = [], []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "ND":
                lons.append(float(parts[2]))
                lats.append(float(parts[3]))
    if not lons:
        raise ValueError(f"No ND nodes found in {path}")
    return [min(lons), min(lats), max(lons), max(lats)]

