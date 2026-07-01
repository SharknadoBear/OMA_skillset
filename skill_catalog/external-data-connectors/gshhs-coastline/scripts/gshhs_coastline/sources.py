from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GSHHG_VERSION = "2.3.7"
GSHHG_ZIP_NAME = f"gshhg-shp-{GSHHG_VERSION}.zip"
GSHHG_ZIP_URL = f"https://ftp.soest.hawaii.edu/gshhg/{GSHHG_ZIP_NAME}"
GSHHG_ZIP_ESTIMATED_BYTES = 142 * 1024 * 1024
RESOLUTION_ORDER = ("c", "l", "i", "h", "f")


@dataclass(frozen=True)
class SourceLocation:
    root: Path
    gshhs_dir: Path
    source_kind: str
    available_resolutions: tuple[str, ...]


def workspace_roots() -> list[Path]:
    roots: list[Path] = []
    starts = [Path.cwd(), Path(__file__).resolve()]
    for start in starts:
        for path in [start, *start.parents]:
            if path not in roots:
                roots.append(path)
    return roots


def default_cache_dir() -> Path:
    return Path("Workspace/Preprocessing/fvcom-gshhs-coastline/cache/gshhg")


def _candidate_dirs(cache_dir: str | Path | None = None) -> list[tuple[Path, str]]:
    rels = []
    if cache_dir:
        requested = Path(cache_dir)
        rels.extend([(requested / "GSHHS_shp", "requested_cache"), (requested, "requested_cache")])
    rels.extend(
        [
            (default_cache_dir() / "GSHHS_shp", "default_cache"),
            (default_cache_dir(), "default_cache"),
            (
                Path("Workspace/Preprocessing/fvcom-cusp-coastline/cache/gshhg/GSHHS_shp"),
                "legacy_cusp_cache",
            ),
            (Path("Workspace/Preprocessing/fvcom-cusp-coastline/cache/gshhg"), "legacy_cusp_cache"),
        ]
    )

    out: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for root in workspace_roots():
        for rel, kind in rels:
            path = rel if rel.is_absolute() else root / rel
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved not in seen:
                seen.add(resolved)
                out.append((path, kind))
    return out


def available_resolutions(gshhs_dir: Path) -> tuple[str, ...]:
    found: list[str] = []
    for resolution in RESOLUTION_ORDER:
        path = shapefile_path(gshhs_dir, resolution, 1)
        if path.exists():
            found.append(resolution)
    return tuple(found)


def shapefile_path(gshhs_dir: Path, resolution: str, level: int) -> Path:
    return gshhs_dir / resolution / f"GSHHS_{resolution}_L{int(level)}.shp"


def find_gshhs_cache(cache_dir: str | Path | None = None) -> SourceLocation | None:
    for candidate, kind in _candidate_dirs(cache_dir):
        gshhs_dir = candidate / "GSHHS_shp" if (candidate / "GSHHS_shp").exists() else candidate
        if not gshhs_dir.exists():
            continue
        resolutions = available_resolutions(gshhs_dir)
        if resolutions:
            return SourceLocation(
                root=candidate,
                gshhs_dir=gshhs_dir,
                source_kind=kind,
                available_resolutions=resolutions,
            )
    return None


def ensure_gshhs_cache(
    cache_dir: str | Path | None = None,
    *,
    force_download: bool = False,
    quiet: bool = False,
) -> tuple[SourceLocation, dict[str, object]]:
    existing = None if force_download else find_gshhs_cache(cache_dir)
    if existing is not None:
        return existing, {
            "cache_status": "found",
            "download_performed": False,
            "source_url": GSHHG_ZIP_URL,
            "cache_root": str(existing.root),
            "gshhs_dir": str(existing.gshhs_dir),
            "available_resolutions": list(existing.available_resolutions),
        }

    target_root = Path(cache_dir) if cache_dir else default_cache_dir()
    target_root.mkdir(parents=True, exist_ok=True)
    zip_path = target_root / GSHHG_ZIP_NAME
    if not zip_path.exists() or force_download:
        if not quiet:
            print(f"Downloading {GSHHG_ZIP_URL} -> {zip_path}")
        urllib.request.urlretrieve(GSHHG_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [name for name in zf.namelist() if name.startswith("GSHHS_shp/")]
        zf.extractall(target_root, members)

    location = find_gshhs_cache(target_root)
    if location is None:
        raise RuntimeError(f"Downloaded archive did not create a usable GSHHS_shp cache under {target_root}")
    return location, {
        "cache_status": "downloaded",
        "download_performed": True,
        "source_url": GSHHG_ZIP_URL,
        "zip_path": str(zip_path),
        "cache_root": str(location.root),
        "gshhs_dir": str(location.gshhs_dir),
        "available_resolutions": list(location.available_resolutions),
    }


def choose_resolution(
    requested: str,
    bbox: tuple[float, float, float, float],
    available: Iterable[str],
) -> tuple[str, list[str]]:
    requested = requested.lower().strip()
    available_set = set(available)
    warnings: list[str] = []
    if requested != "auto":
        if requested in available_set:
            return requested, warnings
        for fallback in ("f", "h", "i", "l", "c"):
            if fallback in available_set:
                warnings.append(f"Requested resolution {requested!r} was unavailable; using {fallback!r}.")
                return fallback, warnings
        raise FileNotFoundError(f"No GSHHS resolutions available for requested {requested!r}.")

    west, south, east, north = bbox
    spans = [abs(east - west), abs(north - south)]
    target = "f" if max(spans) <= 1.0 else "h"
    if target in available_set:
        return target, warnings
    for fallback in ("f", "h", "i", "l", "c"):
        if fallback in available_set:
            warnings.append(f"Auto target {target!r} was unavailable; using {fallback!r}.")
            return fallback, warnings
    raise FileNotFoundError("No GSHHS resolutions available.")


def parse_levels(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        out = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        out = [int(part) for part in value]
    return sorted(set(out or [1]))


def split_bbox_antimeridian(bbox: tuple[float, float, float, float]) -> tuple[list[tuple[float, float, float, float]], dict[str, object]]:
    west, south, east, north = map(float, bbox)
    if south > north:
        raise ValueError(f"Invalid bbox latitude order: {bbox}")
    metadata = {"input_bbox_wsen": [west, south, east, north], "antimeridian_split": False, "parts": []}
    if west <= east and west >= -180.0 and east <= 180.0:
        part = (west, south, east, north)
        metadata["parts"] = [list(part)]
        return [part], metadata

    def norm(lon: float) -> float:
        while lon < -180.0:
            lon += 360.0
        while lon > 180.0:
            lon -= 360.0
        return lon

    west_n = norm(west)
    east_n = norm(east)
    if west_n <= east_n and abs(east - west) < 360.0:
        parts = [(west_n, south, east_n, north)]
    else:
        parts = [(west_n, south, 180.0, north), (-180.0, south, east_n, north)]
    metadata["antimeridian_split"] = len(parts) > 1
    metadata["parts"] = [list(part) for part in parts]
    return parts, metadata


def write_json(path: str | Path, data: dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def local_free_bytes(path: str | Path) -> int:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(target).free)
