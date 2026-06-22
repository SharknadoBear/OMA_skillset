"""
cbofs_fetcher.py
================
Fetch CBOFS (Chesapeake Bay OFS) salinity and temperature for the
Chesapeake Canal open boundary condition (F05).

Forcing component: F05 -- T/S at Chesapeake and Delaware Canal boundary.

Design
------
*Streaming download-extract-delete* pattern: for each time snapshot the
caller requests, one CBOFS netCDF file is downloaded to a temporary path,
the requested fields are extracted at the precomputed canal grid node, and
the file is deleted immediately (``finally`` block).  No permanent copies of
raw CBOFS files are kept on disk.

Variable selector
-----------------
Pass ``variables`` to control which fields are extracted per file.  Each
variable maps to a specific ROMS C-grid type:

    CBOFS_VARIABLE_GRIDS = {
        'salt': 'rho',   # 3-D  (ocean_time, s_rho, eta_rho, xi_rho)
        'temp': 'rho',   # 3-D  (ocean_time, s_rho, eta_rho, xi_rho)
        'zeta': 'rho',   # 2-D  (ocean_time, eta_rho, xi_rho)
        'u'   : 'u',     # 3-D  (ocean_time, s_rho, eta_u, xi_u) -- staggered
        'v'   : 'v',     # 3-D  (ocean_time, s_rho, eta_v, xi_v) -- staggered
    }

If only one variable is needed (e.g. just 'zeta'), the function downloads
the file once and returns only that variable.  For a combined extraction
(e.g. ['salt', 'temp', 'zeta']), the file is still downloaded only once
per time snapshot, keeping total bandwidth to a minimum.

Probe results (nos.cbofs.fields.f000.20180101.t00z.nc)
-------------------------------------------------------
  xi_rho=332, eta_rho=291, N=20 (s_rho=20)
  theta_s=4.5, theta_b=0.95, hc=2.0
  Canal rho-node: i_rho=228, j_rho=283 (lon=-75.8108, lat=39.5301)
  ocean_time: seconds since epoch → 2018-01-01T00:00:00

NCEI archive URL
----------------
Base:  https://www.ncei.noaa.gov/oa/prod-model/
       operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/
       access/chesapeake-bay-operational-forecast-system-cbofs/
File:  YYYY/MM/nos.cbofs.fields.f{FHH}.YYYYMMDD.t{CC}z.nc
       e.g. 2018/01/nos.cbofs.fields.f000.20180101.t00z.nc

Old MATLAB reference
--------------------
  data_source/cbofs_results_2019.mat, cbofs_results_2020.mat  -- pre-extracted
  data_source/nos.dbofs.fields.f001.YYYYMMDD.t00z.nc          -- daily files

Python dependencies
-------------------
  numpy, netCDF4, requests, scipy.spatial, tqdm,
  concurrent.futures (stdlib)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence, Optional

import numpy as np

try:
    import netCDF4 as nc4
except ImportError:
    raise ImportError("netCDF4 is required for cbofs_fetcher: pip install netCDF4")

try:
    import requests
except ImportError:
    raise ImportError("requests is required for cbofs_fetcher: pip install requests")

try:
    import httpx as _httpx
    try:
        import h2  # noqa: F401 — required for httpx HTTP/2 support
        _HAS_HTTPX = True
    except ImportError:
        _HAS_HTTPX = False  # httpx present but h2 not installed
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False

# curl.exe is built into Windows 10/11 (C:\Windows\System32\curl.exe) and
# uses the WinHTTP/Schannel stack — the same HTTP engine as Edge/Chrome.
# It is consistently faster than Python HTTP clients against Akamai CDN.
# Fall back to the hard-coded System32 path if not found on PATH (Jupyter
# kernels often have a stripped PATH that omits System32).
_CURL_EXE: str | None = shutil.which("curl") or shutil.which("curl.exe")
if _CURL_EXE is None and os.name == "nt":
    _win_curl = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "curl.exe"
    if _win_curl.is_file():
        _CURL_EXE = str(_win_curl)

# PowerShell Invoke-WebRequest — always present on Windows, uses WinHTTP
# (same HTTP/TLS stack as Edge/Chrome). CRITICAL: $ProgressPreference must
# be set to 'SilentlyContinue' — without it PowerShell writes a progress
# bar to stdout which throttles downloads to a crawl in non-interactive mode.
_PS_EXE: str | None = None
if os.name == "nt":
    _ps_candidate = (
        Path(os.environ.get("WINDIR", r"C:\Windows"))
        / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    if _ps_candidate.is_file():
        _PS_EXE = str(_ps_candidate)
    else:
        _PS_EXE = shutil.which("powershell") or shutil.which("powershell.exe")

try:
    from scipy.spatial import KDTree
except ImportError:
    raise ImportError("scipy is required for cbofs_fetcher: pip install scipy")

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# Persistent requests Session — reuses TCP/TLS connections across files
# (important for the mass fetch of ~4380 files in Cell 6).
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Connection": "keep-alive",
})


# ---------------------------------------------------------------------------
# Notebook progress display
# ---------------------------------------------------------------------------

class NotebookProgressDisplay:
    """Refreshing single-block progress display for Jupyter notebooks.

    Replaces the cell output on every update so the display never stacks.
    Falls back gracefully to plain ``print`` when IPython is not available
    (e.g. when called from a terminal or script).

    Parameters
    ----------
    total : int
        Total number of items to process.
    label : str
        Short title shown at the top of the block.
    bar_len : int
        Width of the ASCII progress bar in characters (default 40).
    n_recent : int
        Number of most-recent error / warning lines to keep visible.
    """

    def __init__(
        self,
        total: int,
        label: str = "Progress",
        bar_len: int = 40,
        n_recent: int = 10,
    ) -> None:
        self.total    = total
        self.label    = label
        self.bar_len  = bar_len
        self.n_recent = n_recent
        self._recent: list[str] = []
        self._closed  = False

        try:
            from IPython.display import clear_output as _co
            self._clear: Optional[object] = _co
        except ImportError:
            self._clear = None

    def _bar(self, done: int) -> str:
        filled = int(self.bar_len * done / max(self.total, 1))
        return "\u2588" * filled + "\u2591" * (self.bar_len - filled)

    def _render(
        self, done: int, current: str = "", status: str = "",
        extra: Optional[list[str]] = None,
    ) -> None:
        pct   = 100.0 * done / max(self.total, 1)
        lines = [
            f"{self.label}",
            f"  [{self._bar(done)}]  {done}/{self.total}  ({pct:.0f}%)",
        ]
        if current:
            lines.append(f"  Current : {current}  {status}")
        if self._recent:
            lines.append("")
            lines.append("  Recent errors / warnings:")
            for m in self._recent[-self.n_recent:]:
                lines.append(f"    {m}")
        if extra:
            lines.extend(extra)
        if self._clear is not None:
            self._clear(wait=True)   # type: ignore[call-arg]
        print("\n".join(lines), flush=True)

    def update(self, done: int, current: str = "", status: str = "") -> None:
        """Refresh the progress block.  Call once per item *before* the work."""
        if not self._closed:
            self._render(done, current=current, status=status)

    def log(self, msg: str) -> None:
        """Append *msg* to the rolling recent-messages list (no redraw)."""
        self._recent.append(msg)

    def close(self, summary: str = "") -> None:
        """Draw the final 100 % state with an optional summary line."""
        self._closed = True
        extra = ["", f"  {summary}"] if summary else []
        self._render(self.total, extra=extra)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Base URL of NCEI CBOFS archive (prod-model object store)
NCEI_CBOFS_BASE = (
    "https://www.ncei.noaa.gov/oa/prod-model/"
    "operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/"
    "access/chesapeake-bay-operational-forecast-system-cbofs/"
)

#: Available daily model cycles
ALL_CYCLES: tuple[str, ...] = ("t00z", "t06z", "t12z", "t18z")

#: Which ROMS C-grid each variable lives on
CBOFS_VARIABLE_GRIDS: dict[str, str] = {
    "salt": "rho",
    "temp": "rho",
    "zeta": "rho",
    "u"   : "u",
    "v"   : "v",
}

#: Whether each variable is 3-D (True) or 2-D (False)
CBOFS_VARIABLE_3D: dict[str, bool] = {
    "salt": True,
    "temp": True,
    "zeta": False,
    "u"   : True,
    "v"   : True,
}

#: Default C&D Canal / Chesapeake City target location
CANAL_LON_DEFAULT: float = -75.81   # degrees East
CANAL_LAT_DEFAULT: float =  39.53   # degrees North

# Legacy aliases kept for backward compatibility
CANAL_LON = CANAL_LON_DEFAULT
CANAL_LAT = CANAL_LAT_DEFAULT


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------

def _cbofs_url(date_str: str, cycle: str = "t00z", fhour: int = 0) -> str:
    """Build the NCEI prod-model download URL for one CBOFS file.

    Parameters
    ----------
    date_str : ``'YYYY-MM-DD'`` string
    cycle    : model run cycle, one of ``'t00z'``, ``'t06z'``, ``'t12z'``, ``'t18z'``
    fhour    : forecast hour (0 = nowcast/analysis)

    Returns
    -------
    Full HTTPS URL string.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy     = d.strftime("%Y")
    mm       = d.strftime("%m")
    yyyymmdd = d.strftime("%Y%m%d")
    fname    = f"nos.cbofs.fields.f{fhour:03d}.{yyyymmdd}.{cycle}.nc"
    return f"{NCEI_CBOFS_BASE}{yyyy}/{mm}/{fname}"


# ---------------------------------------------------------------------------
# HTTP downloader with exponential backoff
# ---------------------------------------------------------------------------

_NCEI_HEADERS: dict[str, str] = {
    # NCEI uses Akamai CDN which silently hangs connections from the default
    # 'python-requests/x.y' User-Agent.  A browser-like UA resolves this.
    # Accept-Encoding is intentionally omitted: NetCDF4/HDF5 files are already
    # internally compressed; gzip negotiation adds CPU overhead with no gain.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Connection": "keep-alive",
}


def _download_file(url: str,
                   local_path: Path,
                   timeout: "int | tuple[int, int]" = (30, 120),
                   max_retries: int = 3,
                   chunk_size: int = 1 << 20) -> None:
    """Download *url* to *local_path* with retry logic.

    Parameters
    ----------
    url         : HTTPS URL to download
    local_path  : destination Path (parent must exist)
    timeout     : ``(connect_timeout, read_timeout)`` tuple in seconds, or a
                  single int applied to both.  Default ``(30, 120)`` gives 30 s
                  to establish the TCP/TLS connection and 120 s per read chunk.
    max_retries : number of attempts before raising
    chunk_size  : streaming chunk size in bytes (default 1 MiB)

    Raises
    ------
    RuntimeError
        After all retries are exhausted.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    connect_t = timeout[0] if isinstance(timeout, tuple) else timeout
    read_t    = timeout[1] if isinstance(timeout, tuple) else timeout
    max_time  = connect_t + read_t

    for attempt in range(1, max_retries + 1):
        try:
            log.debug("Downloading (attempt %d/%d): %s", attempt, max_retries,
                      url)
            if _PS_EXE:
                # Priority 1: PowerShell Invoke-WebRequest.
                # $ProgressPreference='SilentlyContinue' is CRITICAL —
                # without it PS writes a progress bar which throttles
                # downloads to a crawl in non-interactive (Jupyter) mode.
                ua      = _NCEI_HEADERS["User-Agent"]
                out_str = str(local_path).replace('"', '`"')
                ps_cmd  = (
                    "$ProgressPreference='SilentlyContinue'; "
                    f'Invoke-WebRequest -Uri "{url}" '
                    f'-OutFile "{out_str}" '
                    f'-UserAgent "{ua}" '
                    "-UseBasicParsing"
                )
                result = subprocess.run(
                    [_PS_EXE, "-NoProfile", "-NonInteractive",
                     "-Command", ps_cmd],
                    capture_output=True, timeout=max_time + 60,
                )
                if result.returncode != 0:
                    err = result.stderr.decode(errors="replace").strip()
                    raise RuntimeError(
                        f"PowerShell exit {result.returncode}: {err}"
                    )
            elif _CURL_EXE:
                # Priority 2: curl — WinHTTP/Schannel on Windows
                cmd = [
                    _CURL_EXE,
                    "-L", "-s", "-f",
                    "-o", str(local_path),
                    "--connect-timeout", str(connect_t),
                    "--max-time",        str(max_time),
                    "-H", f"User-Agent: {_NCEI_HEADERS['User-Agent']}",
                    "-H", "Accept: */*",
                    url,
                ]
                result = subprocess.run(
                    cmd, capture_output=True, timeout=max_time + 30
                )
                if result.returncode != 0:
                    err = result.stderr.decode(errors="replace").strip()
                    raise RuntimeError(
                        f"curl exit {result.returncode}: {err}"
                    )
            elif _HAS_HTTPX:
                # Priority 3: httpx with HTTP/2
                _to = _httpx.Timeout(connect=connect_t, read=read_t,
                                     write=connect_t, pool=connect_t)
                with _httpx.Client(http2=True, headers=_NCEI_HEADERS,
                                   timeout=_to, follow_redirects=True) as cli:
                    resp = cli.get(url)
                    resp.raise_for_status()
                    local_path.write_bytes(resp.content)
            else:
                # Priority 4: persistent requests.Session (HTTP/1.1).
                # Session reuse avoids repeated TCP/TLS handshakes for the
                # mass fetch of many files.
                resp = _SESSION.get(url, stream=False, timeout=timeout)
                resp.raise_for_status()
                with open(local_path, "wb") as fh:
                    fh.write(resp.content)
            return  # success
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"CBOFS download failed after {max_retries} attempts: {url}\n"
                    f"  Last error: {exc}"
                ) from exc
            wait = 2 ** attempt
            log.warning("Download attempt %d/%d failed (%s); retrying in %ds",
                        attempt, max_retries, exc, wait)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# ROMS vertical coordinate: compute actual depths
# ---------------------------------------------------------------------------

def _roms_depths_at_node(ds: "nc4.Dataset", i_rho: int, j_rho: int) -> np.ndarray:
    """Return actual physical depths (negative-up, metres) at one rho-point.

    Reads ``s_rho``, ``Cs_r``, ``hc``, ``h``, ``zeta``, and ``Vtransform``
    directly from the open Dataset.  Both ROMS transformation equations are
    supported:

    **Vtransform = 1** (Song & Haidvogel 1994):

        z0 = (s − Cs) · hc + Cs · h
        z  = z0 + ζ · (1 + z0 / h)

    **Vtransform = 2** (Shchepetkin / UCLA-ROMS, default for CBOFS):

        z0 = (hc · s + h · Cs) / (hc + h)
        z  = ζ + (ζ + h) · z0

    Parameters
    ----------
    ds    : open ``netCDF4.Dataset`` for one CBOFS snapshot
    i_rho : xi_rho index of the target point
    j_rho : eta_rho index of the target point

    Returns
    -------
    z : ndarray, shape (N,), values ≤ 0 (negative = below surface).
        k=0 is deepest (near-bottom), k=N-1 is shallowest (near-surface).
    """
    s_rho = ds.variables["s_rho"][:]   # (N,)
    Cs_r  = ds.variables["Cs_r"][:]    # (N,)

    hc = float(ds.variables["hc"][0]) if "hc" in ds.variables else 2.0
    h  = float(ds.variables["h"][j_rho, i_rho])

    raw_zeta = ds.variables["zeta"][0, j_rho, i_rho]
    zeta = float(np.ma.filled(raw_zeta, 0.0))

    vtransform = 2  # default for modern CBOFS
    if "Vtransform" in ds.variables:
        vtransform = int(ds.variables["Vtransform"][0])
    elif hasattr(ds, "Vtransform"):
        vtransform = int(ds.Vtransform)

    if vtransform == 2:
        z0 = (hc * s_rho + h * Cs_r) / (hc + h)
        z  = zeta + (zeta + h) * z0
    else:  # vtransform == 1
        z0 = (s_rho - Cs_r) * hc + Cs_r * h
        z  = z0 + zeta * (1.0 + z0 / h)

    return z.astype(np.float64)


# ---------------------------------------------------------------------------
# Time reader
# ---------------------------------------------------------------------------

_CAL_ALIASES: dict[str, str] = {
    "gregorian_proleptic": "proleptic_gregorian",
}


def _norm_calendar(cal: str) -> str:
    """Normalise non-standard calendar strings to cftime-accepted equivalents."""
    return _CAL_ALIASES.get(cal.lower().strip(), cal)


def _get_ocean_time(ds: "nc4.Dataset") -> np.datetime64:
    """Return the single ocean_time of one CBOFS snapshot as numpy datetime64[s]."""
    ot_var  = ds.variables["ocean_time"]
    ot_val  = float(ot_var[0])
    ot_unit = getattr(ot_var, "units", "seconds since 2000-01-01")
    ot_cal  = _norm_calendar(getattr(ot_var, "calendar", "gregorian"))
    dt_cf   = nc4.num2date(ot_val, ot_unit, ot_cal)
    return np.datetime64(dt_cf.strftime("%Y-%m-%dT%H:%M:%S"), "s")


# ---------------------------------------------------------------------------
# Atomic per-variable extractors
# (all accept an open nc4.Dataset and precomputed node indices)
# ---------------------------------------------------------------------------

def _extract_salt(ds: "nc4.Dataset", i_rho: int, j_rho: int) -> np.ndarray:
    """Extract salinity profile [PSU] at rho-point (i_rho, j_rho).

    Returns ndarray (N,) where k=0 is near-bottom, k=N-1 is near-surface.
    """
    raw = ds.variables["salt"][0, :, j_rho, i_rho]
    return np.ma.filled(raw, np.nan).astype(np.float64)


def _extract_temp(ds: "nc4.Dataset", i_rho: int, j_rho: int) -> np.ndarray:
    """Extract temperature profile [°C] at rho-point (i_rho, j_rho).

    Returns ndarray (N,) where k=0 is near-bottom, k=N-1 is near-surface.
    """
    raw = ds.variables["temp"][0, :, j_rho, i_rho]
    return np.ma.filled(raw, np.nan).astype(np.float64)


def _extract_zeta(ds: "nc4.Dataset", i_rho: int, j_rho: int) -> float:
    """Extract surface elevation [m] at rho-point (i_rho, j_rho).

    Returns a scalar float.
    """
    raw = ds.variables["zeta"][0, j_rho, i_rho]
    return float(np.ma.filled(raw, np.nan))


def _extract_u(ds: "nc4.Dataset", i_u: int, j_u: int) -> np.ndarray:
    """Extract u-velocity profile [m/s] at u-grid point (i_u, j_u).

    Returns ndarray (N,).  Note: u-grid is staggered half a cell in the
    xi-direction relative to rho-grid.
    """
    raw = ds.variables["u"][0, :, j_u, i_u]
    return np.ma.filled(raw, np.nan).astype(np.float64)


def _extract_v(ds: "nc4.Dataset", i_v: int, j_v: int) -> np.ndarray:
    """Extract v-velocity profile [m/s] at v-grid point (i_v, j_v).

    Returns ndarray (N,).  Note: v-grid is staggered half a cell in the
    eta-direction relative to rho-grid.
    """
    raw = ds.variables["v"][0, :, j_v, i_v]
    return np.ma.filled(raw, np.nan).astype(np.float64)


# ---------------------------------------------------------------------------
# Canal node discovery (one-time, downloads then deletes one file)
# ---------------------------------------------------------------------------

def get_cbofs_canal_nodes(
        canal_lon: float = CANAL_LON_DEFAULT,
        canal_lat: float = CANAL_LAT_DEFAULT,
        sample_date: str = "2018-01-01",
        cycle: str = "t00z",
        work_dir: Optional[str | Path] = None) -> dict:
    """Find the CBOFS C-grid node indices nearest to the canal location.

    Downloads one sample file, searches all three ROMS C-grids (rho, u, v)
    for the nearest node to ``(canal_lon, canal_lat)``, then deletes the file.

    Call this **once** at the start of the notebook and cache the result.
    Pass the returned dict as ``canal_nodes`` to :func:`fetch_cbofs_canal`.

    Parameters
    ----------
    canal_lon, canal_lat : target location (Chesapeake City / C&D Canal)
    sample_date          : any YYYY-MM-DD for which a CBOFS file exists
    cycle                : model cycle for the sample file
    work_dir             : directory for the temporary download; defaults to
                           system temp folder

    Returns
    -------
    dict with keys:
      - ``i_rho``, ``j_rho``  : rho-grid indices (salt/temp/zeta)
      - ``i_u``,   ``j_u``    : u-grid indices
      - ``i_v``,   ``j_v``    : v-grid indices
      - ``lon_found``, ``lat_found`` : actual rho-point location
      - ``dist_deg``           : Euclidean distance in degrees
      - ``h_found``            : bathymetric depth at rho-point [m]
      - ``N``                  : number of ROMS sigma levels
    """
    if work_dir is None:
        import tempfile
        work_dir = Path(tempfile.gettempdir())
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    url = _cbofs_url(sample_date, cycle, fhour=0)
    tmp = work_dir / f"_cbofs_probe_{sample_date}_{cycle}.nc"

    print(f"Downloading probe file:\n  {url}")
    try:
        _download_file(url, tmp)
        print(f"  Size: {tmp.stat().st_size / 1e6:.1f} MB")

        ds = nc4.Dataset(tmp)

        # Number of sigma levels
        N = len(ds.dimensions["s_rho"]) if "s_rho" in ds.dimensions else 20

        lon_rho = ds.variables["lon_rho"][:]
        lat_rho = ds.variables["lat_rho"][:]
        h       = ds.variables["h"][:]

        # KDTree search on rho-grid
        rho_pts = np.column_stack([lon_rho.ravel(), lat_rho.ravel()])
        dist_rho, flat_rho = KDTree(rho_pts).query([canal_lon, canal_lat])
        j_rho, i_rho = np.unravel_index(flat_rho, lon_rho.shape)

        result: dict = {
            "i_rho"    : int(i_rho),
            "j_rho"    : int(j_rho),
            "lon_found": float(lon_rho[j_rho, i_rho]),
            "lat_found": float(lat_rho[j_rho, i_rho]),
            "dist_deg" : float(dist_rho),
            "h_found"  : float(h[j_rho, i_rho]),
            "N"        : N,
        }

        # u-grid
        if "lon_u" in ds.variables and "lat_u" in ds.variables:
            lon_u = ds.variables["lon_u"][:]
            lat_u = ds.variables["lat_u"][:]
            _, flat_u = KDTree(
                np.column_stack([lon_u.ravel(), lat_u.ravel()])
            ).query([canal_lon, canal_lat])
            j_u, i_u = np.unravel_index(flat_u, lon_u.shape)
        else:
            j_u, i_u = int(j_rho), max(0, int(i_rho) - 1)
        result.update({"i_u": int(i_u), "j_u": int(j_u)})

        # v-grid
        if "lon_v" in ds.variables and "lat_v" in ds.variables:
            lon_v = ds.variables["lon_v"][:]
            lat_v = ds.variables["lat_v"][:]
            _, flat_v = KDTree(
                np.column_stack([lon_v.ravel(), lat_v.ravel()])
            ).query([canal_lon, canal_lat])
            j_v, i_v = np.unravel_index(flat_v, lon_v.shape)
        else:
            j_v, i_v = max(0, int(j_rho) - 1), int(i_rho)
        result.update({"i_v": int(i_v), "j_v": int(j_v)})

        ds.close()

    finally:
        if tmp.exists():
            tmp.unlink()
            print(f"  Deleted: {tmp.name}")

    print(f"\nCanal rho-node:")
    print(f"  i_rho={result['i_rho']}, j_rho={result['j_rho']}")
    print(f"  lon={result['lon_found']:.4f}°, lat={result['lat_found']:.4f}°N")
    print(f"  dist={result['dist_deg']:.4f}°, depth={result['h_found']:.2f} m, N={N}")
    return result


# ---------------------------------------------------------------------------
# Per-snapshot extraction worker (serial + parallel paths share this)
# ---------------------------------------------------------------------------

def _extract_snapshot(url: str,
                       tmp_path: Path,
                       canal_nodes: dict,
                       variables: list[str]) -> dict | None:
    """Download one CBOFS file, extract requested variables, delete file.

    Returns a dict with ``time_dt64`` and one key per requested variable,
    or ``None`` if the file could not be fetched.

    The temporary file is always deleted in the ``finally`` block, even on
    error.
    """
    try:
        _download_file(url, tmp_path, timeout=180, max_retries=3)
    except RuntimeError as exc:
        log.warning("Skip (download failed): %s\n  %s", url, exc)
        return None

    try:
        ds = nc4.Dataset(tmp_path)
        snap: dict = {"time_dt64": _get_ocean_time(ds)}

        i_rho = canal_nodes["i_rho"]
        j_rho = canal_nodes["j_rho"]

        # Compute actual depths once if any 3-D variable is requested
        if any(CBOFS_VARIABLE_3D.get(v, False) for v in variables):
            snap["z_levels"] = _roms_depths_at_node(ds, i_rho, j_rho)

        for var in variables:
            if var == "salt":
                snap["salt"] = _extract_salt(ds, i_rho, j_rho)
            elif var == "temp":
                snap["temp"] = _extract_temp(ds, i_rho, j_rho)
            elif var == "zeta":
                snap["zeta"] = _extract_zeta(ds, i_rho, j_rho)
            elif var == "u":
                snap["u"] = _extract_u(ds,
                    canal_nodes.get("i_u", i_rho),
                    canal_nodes.get("j_u", j_rho))
            elif var == "v":
                snap["v"] = _extract_v(ds,
                    canal_nodes.get("i_v", i_rho),
                    canal_nodes.get("j_v", j_rho))

        ds.close()
        return snap

    except Exception as exc:
        log.warning("Extraction failed for %s: %s", tmp_path.name, exc)
        return None

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Main streaming fetcher
# ---------------------------------------------------------------------------

def fetch_cbofs_canal(
        t_start: str,
        t_end: str,
        canal_nodes: dict,
        variables: Sequence[str] = ("salt", "temp"),
        cycles: Sequence[str] = ALL_CYCLES,
        fhour: int = 0,
        work_dir: Optional[str | Path] = None,
        parallel: bool = False,
        n_workers: int = 4,
        progress: bool = True) -> dict:
    """Fetch CBOFS fields at the canal node for a date range.

    Each CBOFS file is downloaded, the requested fields are sampled at the
    ``canal_nodes`` location, and the file is deleted immediately.  No
    permanent copies of raw CBOFS files are kept on disk.

    Parameters
    ----------
    t_start, t_end : ISO date strings ``'YYYY-MM-DD'``
    canal_nodes    : dict returned by :func:`get_cbofs_canal_nodes`
    variables      : subset of ``['salt', 'temp', 'zeta', 'u', 'v']``.
                     Only the listed variables are extracted; the file is
                     downloaded once per snapshot regardless of count.
    cycles         : model cycles per day (default all 4 = 6-hourly output)
    fhour          : forecast hour (0 = nowcast analysis)
    work_dir       : directory for temporary files; auto-cleaned per file
    parallel       : enable concurrent downloads via ThreadPoolExecutor
    n_workers      : number of download threads (parallel mode)
    progress       : show tqdm progress bar

    Returns
    -------
    dict with keys:

    ``time_dt64`` : ndarray (ntime,)  numpy datetime64[s]
    ``z_levels``  : ndarray (ntime, N) actual ROMS depths [m, negative-up]
                    (included when any 3-D variable is requested)
    ``salt``      : ndarray (ntime, N) [PSU]   (if requested)
    ``temp``      : ndarray (ntime, N) [°C]    (if requested)
    ``zeta``      : ndarray (ntime,)   [m]     (if requested)
    ``u``         : ndarray (ntime, N) [m/s]   (if requested, u-grid)
    ``v``         : ndarray (ntime, N) [m/s]   (if requested, v-grid)

    Failed/missing snapshots are stored as NaN.  The output is sorted by
    time.

    Examples
    --------
    Salt + temp only (single download pass)::

        data = fetch_cbofs_canal(
            '2018-01-01', '2018-01-31', canal_nodes,
            variables=['salt', 'temp'])

    Zeta only (demonstrates modular single-variable use)::

        zeta_data = fetch_cbofs_canal(
            '2018-01-01', '2018-01-31', canal_nodes,
            variables=['zeta'])

    Combined (all vars, one download per file)::

        all_data = fetch_cbofs_canal(
            '2018-01-01', '2018-01-05', canal_nodes,
            variables=['salt', 'temp', 'zeta'])

    Parallel download (I/O-bound, safe with unique temp filenames)::

        data = fetch_cbofs_canal(
            '2018-01-01', '2020-12-31', canal_nodes,
            variables=['salt', 'temp'],
            parallel=True, n_workers=6)
    """
    if work_dir is None:
        import tempfile
        work_dir = Path(tempfile.gettempdir()) / "cbofs_tmp"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    variables = list(variables)
    for v in variables:
        if v not in CBOFS_VARIABLE_GRIDS:
            raise ValueError(
                f"Unknown variable '{v}'. "
                f"Choose from: {list(CBOFS_VARIABLE_GRIDS)}"
            )

    # Build list of (date_str, cycle) jobs
    d0   = datetime.strptime(t_start, "%Y-%m-%d")
    d1   = datetime.strptime(t_end,   "%Y-%m-%d")
    jobs: list[tuple[str, str]] = []
    d = d0
    while d <= d1:
        for cyc in cycles:
            jobs.append((d.strftime("%Y-%m-%d"), cyc))
        d += timedelta(days=1)

    n_jobs = len(jobs)
    N      = canal_nodes.get("N", 20)
    has_3d = any(CBOFS_VARIABLE_3D.get(v, False) for v in variables)

    print(f"CBOFS fetch: {n_jobs} snapshots  ({t_start} → {t_end}, "
          f"{len(cycles)} cycles/day)")
    print(f"  variables : {variables}")
    print(f"  mode      : {'parallel ×' + str(n_workers) if parallel else 'serial'}")

    # Pre-allocate output arrays (NaN / NaT for missing)
    out: dict = {"time_dt64": np.empty(n_jobs, dtype="datetime64[s]")}
    out["time_dt64"][:] = np.datetime64("NaT")
    if has_3d:
        out["z_levels"] = np.full((n_jobs, N), np.nan)
    for v in variables:
        if CBOFS_VARIABLE_3D.get(v, False):
            out[v] = np.full((n_jobs, N), np.nan)
        else:
            out[v] = np.full(n_jobs, np.nan)

    def _store(idx: int, snap: dict | None) -> None:
        """Write one snapshot result into the pre-allocated arrays."""
        if snap is None:
            return
        out["time_dt64"][idx] = snap["time_dt64"]
        if has_3d and "z_levels" in snap:
            out["z_levels"][idx] = snap["z_levels"]
        for v in variables:
            if v in snap:
                out[v][idx] = snap[v]

    # ── Serial path ──────────────────────────────────────────────────────────
    if not parallel:
        _iter = enumerate(jobs)
        if progress and _HAS_TQDM:
            _iter = enumerate(_tqdm(jobs, desc="CBOFS", unit="file"))

        for idx, (date_str, cyc) in _iter:
            url = _cbofs_url(date_str, cyc, fhour)
            tmp = work_dir / f"_cbofs_{date_str}_{cyc}.nc"
            _store(idx, _extract_snapshot(url, tmp, canal_nodes, variables))

    # ── Parallel path ────────────────────────────────────────────────────────
    else:
        idx_map: dict = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for idx, (date_str, cyc) in enumerate(jobs):
                url = _cbofs_url(date_str, cyc, fhour)
                # Unique temp name per job to avoid worker collisions
                uid = hashlib.md5(f"{date_str}_{cyc}".encode()).hexdigest()[:8]
                tmp = work_dir / f"_cbofs_{uid}.nc"
                fut = pool.submit(_extract_snapshot, url, tmp,
                                  canal_nodes, variables)
                idx_map[fut] = idx

            bar = (_tqdm(total=n_jobs, desc="CBOFS", unit="file")
                   if (progress and _HAS_TQDM) else None)
            for fut in as_completed(idx_map):
                _store(idx_map[fut], fut.result())
                if bar is not None:
                    bar.update(1)
            if bar is not None:
                bar.close()

    # Sort by time (parallel path returns in completion order)
    sort_idx = np.argsort(out["time_dt64"])
    for key in out:
        out[key] = out[key][sort_idx]

    n_ok = int(np.sum(out["time_dt64"] != np.datetime64("NaT")))
    print(f"\nFetch complete: {n_ok}/{n_jobs} OK  "
          f"({n_jobs - n_ok} missing/failed)")
    return out


# ---------------------------------------------------------------------------
# Station-file support (F05 v2 — CBOFS station nowcast files)
# ---------------------------------------------------------------------------
# Station nowcast files (~5–8 MB) contain pre-extracted T/S profiles at the
# ~58 CBOFS monitoring stations, including Chesapeake City (CO-OPS 8573927).
# They are far smaller than full 3-D field files (~61 MB) and available at
# the same NCEI prod-model archive (NCEI_CBOFS_BASE):
#
#   {YYYY}/{MM}/nos.cbofs.stations.nowcast.{YYYYMMDD}.{cycle}.nc
#   Confirmed available: 2018/01 – 2020/12, all 4 cycles/day.
#
# Dead-end routes documented for reference:
#   NCEI data/ path (...hydrodynamic.../cbofs/2019/)  → README only
#   S3 noaa-nos-ofs-pds/cbofs/                        → starts 2022, no T/S
#   CO-OPS THREDDS                                     → 404
#   CO-OPS API application=CBOFS                       → "No data found"
# ---------------------------------------------------------------------------

def _cbofs_station_url(date_str: str,
                       cycle: str = "t00z",
                       product: str = "nowcast") -> str:
    """Build NCEI prod-model URL for a CBOFS station file.

    Station files (~5–8 MB) hold T/S profiles at the ~58 predefined CBOFS
    monitoring stations, vs ~61 MB for full 3-D field files.

    Parameters
    ----------
    date_str : ``'YYYY-MM-DD'``
    cycle    : one of ``'t00z'``, ``'t06z'``, ``'t12z'``, ``'t18z'``
    product  : ``'nowcast'`` (default) or ``'forecast'``

    Returns
    -------
    Full HTTPS URL string.
    """
    d        = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy     = d.strftime("%Y")
    mm       = d.strftime("%m")
    yyyymmdd = d.strftime("%Y%m%d")
    fname    = f"nos.cbofs.stations.{product}.{yyyymmdd}.{cycle}.nc"
    return f"{NCEI_CBOFS_BASE}{yyyy}/{mm}/{fname}"


def roms_station_depths(Cs_r: np.ndarray,
                        s_rho: np.ndarray,
                        hc: float,
                        h_station: float,
                        zeta: float = 0.0,
                        vtransform: int = 2) -> np.ndarray:
    """Compute physical z-depths at a ROMS station point.

    Supports Vtransform 1 and 2 (CBOFS uses Vtransform 2).

    Parameters
    ----------
    Cs_r      : (s_rho,) non-dimensional depth parameter at rho-layers
    s_rho     : (s_rho,) sigma coordinate values in [-1, 0]
    hc        : critical depth [m] (surface-refinement parameter)
    h_station : undisturbed water depth at the station [m, positive]
    zeta      : sea-surface elevation [m] (default 0)
    vtransform: ROMS Vtransform (1 or 2)

    Returns
    -------
    z : (s_rho,) physical depths [m, negative-up, monotonically increasing
        from index 0 (near-bottom) to index -1 (near-surface)]
    """
    Cs_r  = np.asarray(Cs_r,  dtype=np.float64)
    s_rho = np.asarray(s_rho, dtype=np.float64)
    if vtransform == 2:
        z0 = (hc * s_rho + h_station * Cs_r) / (hc + h_station)
        z  = zeta + (zeta + h_station) * z0
    else:
        z0 = (s_rho - Cs_r) * hc + Cs_r * h_station
        z  = z0 + zeta * (1.0 + z0 / h_station)
    return z


def probe_cbofs_station_file(
        date_str: str = "2019-01-01",
        cycle: str = "t00z",
        station_match: str = "8573927",
        work_dir: Optional[str | Path] = None) -> dict:
    """Download one CBOFS station file, print its structure, return metadata.

    Call this once at the start of a notebook to verify archive access,
    discover the station index for *station_match*, and retrieve the sigma
    coordinate parameters needed for vertical depth calculations.

    Parameters
    ----------
    date_str      : ISO date to probe (any date in 2018–2020)
    cycle         : model cycle
    station_match : substring searched in station names (``'8573927'``)
    work_dir      : temp directory; defaults to system temp

    Returns
    -------
    dict with keys: ``station_idx``, ``station_name``, ``n_sigma``,
    ``n_stations``, ``n_time``, ``dt_minutes``, ``time_start``,
    ``time_end``, ``variables``, ``Cs_r``, ``s_rho``, ``hc``,
    ``vtransform``, ``h_station``, ``probe_url``
    """
    if work_dir is None:
        import tempfile
        work_dir = Path(tempfile.gettempdir())
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    url = _cbofs_station_url(date_str, cycle)
    tmp = work_dir / f"_cbofs_sta_probe_{date_str}_{cycle}.nc"

    print(f"Probe URL:\n  {url}")
    print("  Downloading (~5–8 MB) …", flush=True)

    try:
        _download_file(url, tmp)
        print(f"  Downloaded: {tmp.stat().st_size / 1e6:.1f} MB")

        with nc4.Dataset(tmp, "r") as ds:
            # ── dimensions ───────────────────────────────────────────────
            print("\nDimensions:")
            for k, v in ds.dimensions.items():
                print(f"  {k}: {len(v)}")

            # ── variables ────────────────────────────────────────────────
            print("\nVariables:")
            for name, var in ds.variables.items():
                print(f"  {name}: {var.dimensions} {var.shape}")

            # ── station count ────────────────────────────────────────────
            sta_dim_name = next(
                (k for k in ds.dimensions if "stat" in k.lower()), None)
            n_stations = (len(ds.dimensions[sta_dim_name])
                          if sta_dim_name else 0)

            # ── locate target station ────────────────────────────────────
            station_idx        = None
            station_name_found = "?"

            if "station_name" in ds.variables:
                snames = nc4.chartostring(ds.variables["station_name"][:])
                print(f"\nStation names (first 10):")
                for i, n in enumerate(snames[:10]):
                    print(f"  [{i:3d}] {n.strip()!r}")
                if n_stations > 10:
                    print(f"  ... ({n_stations} total)")
                for i, n in enumerate(snames):
                    if station_match in n:
                        station_idx        = i
                        station_name_found = n.strip()
                        break

            # fallback: nearest lat/lon
            if station_idx is None:
                lon_key = next((k for k in ("station_lon", "lon_rho", "lon")
                                if k in ds.variables), None)
                lat_key = next((k for k in ("station_lat", "lat_rho", "lat")
                                if k in ds.variables), None)
                if lon_key and lat_key:
                    slon  = ds.variables[lon_key][:].ravel()
                    slat  = ds.variables[lat_key][:].ravel()
                    dists = np.hypot(slon - CANAL_LON_DEFAULT,
                                     slat - CANAL_LAT_DEFAULT)
                    station_idx        = int(np.argmin(dists))
                    station_name_found = (
                        f"nearest lat/lon [idx={station_idx}  "
                        f"lon={float(slon[station_idx]):.4f}  "
                        f"lat={float(slat[station_idx]):.4f}]"
                    )
                    print(f"\nFallback station: {station_name_found}")

            if station_idx is None:
                raise RuntimeError(
                    f"Cannot find station '{station_match}' in CBOFS station "
                    "file. Check station_name variable."
                )

            print(f"\nTarget station: idx={station_idx}  "
                  f"name={station_name_found!r}")

            # ── sigma coordinate ─────────────────────────────────────────
            Cs_r_arr = ds.variables["Cs_r"][:].astype(np.float64)
            n_sigma  = len(Cs_r_arr)

            if "s_rho" in ds.variables:
                s_rho_arr = ds.variables["s_rho"][:].astype(np.float64)
            else:
                s_rho_arr = np.linspace(
                    -1.0 + 0.5 / n_sigma, -0.5 / n_sigma, n_sigma)

            hc_val = float(getattr(ds, "hc", 2.0))
            if "hc" in ds.variables:
                hc_val = float(ds.variables["hc"][:])
            vtransform_val = int(getattr(ds, "Vtransform", 2))
            if "Vtransform" in ds.variables:
                vtransform_val = int(ds.variables["Vtransform"][:])

            h_station = np.nan
            if "h" in ds.variables:
                h_arr = ds.variables["h"][:]
                h_station = float(h_arr[station_idx] if h_arr.ndim == 1
                                  else h_arr.ravel()[station_idx])

            print(f"\nSigma: n={n_sigma}  hc={hc_val}  "
                  f"Vtransform={vtransform_val}")
            print(f"  Cs_r[0]={Cs_r_arr[0]:.4f} (near-bottom)  "
                  f"Cs_r[-1]={Cs_r_arr[-1]:.4f} (near-surface)")
            print(f"  Water depth at station: h={h_station:.2f} m")

            # ── time axis ────────────────────────────────────────────────
            ot_var   = ds.variables["ocean_time"]
            ot_vals  = ot_var[:].astype(np.float64)
            ot_units = getattr(ot_var, "units", "seconds since 2000-01-01")
            ot_cal   = _norm_calendar(getattr(ot_var, "calendar", "gregorian"))
            dts      = nc4.num2date(ot_vals, ot_units, ot_cal)
            times    = np.array(
                [np.datetime64(d.strftime("%Y-%m-%dT%H:%M:%S"), "s")
                 for d in dts],
                dtype="datetime64[s]",
            )
            n_time = len(times)
            dt_min = (int(round((float(ot_vals[1]) - float(ot_vals[0])) / 60))
                      if n_time > 1 else 0)
            print(f"\nTime: n={n_time}  dt={dt_min} min  "
                  f"{times[0]} → {times[-1]}")

            # ── sample T/S ───────────────────────────────────────────────
            if "salt" in ds.variables and "temp" in ds.variables:
                s_arr = np.ma.filled(
                    ds.variables["salt"][:, :, station_idx].astype(
                        np.float64), np.nan)
                t_arr = np.ma.filled(
                    ds.variables["temp"][:, :, station_idx].astype(
                        np.float64), np.nan)
                print(f"\nSample at station {station_name_found!r}:")
                print(f"  salt: shape={s_arr.shape}  "
                      f"range=[{np.nanmin(s_arr):.2f}, "
                      f"{np.nanmax(s_arr):.2f}] PSU")
                print(f"  temp: shape={t_arr.shape}  "
                      f"range=[{np.nanmin(t_arr):.2f}, "
                      f"{np.nanmax(t_arr):.2f}] °C")

            result = dict(
                station_idx   = station_idx,
                station_name  = station_name_found,
                n_sigma       = n_sigma,
                n_stations    = n_stations,
                n_time        = n_time,
                dt_minutes    = dt_min,
                time_start    = times[0],
                time_end      = times[-1],
                variables     = list(ds.variables.keys()),
                Cs_r          = Cs_r_arr,
                s_rho         = s_rho_arr,
                hc            = hc_val,
                vtransform    = vtransform_val,
                h_station     = h_station,
                probe_url     = url,
            )

        print("\nProbe complete.")
        return result

    finally:
        if tmp.exists():
            tmp.unlink()
            print("  Temp file deleted.")


def fetch_cbofs_station_ts(
        station_match: str,
        t_start: str,
        t_end: str,
        cache_dir: str | Path,
        cycles: Sequence[str] = ALL_CYCLES,
        work_dir: Optional[str | Path] = None) -> dict:
    """Stream CBOFS station nowcast files and extract T/S at one station.

    For each day in [*t_start*, *t_end*] and each *cycle*, downloads the
    ~5–8 MB station nowcast file, extracts temperature and salinity
    profiles at the station identified by *station_match*, and deletes
    the file immediately.  Results are cached as a compressed NPZ.

    Parameters
    ----------
    station_match : substring to search in CBOFS ``station_name`` variable
                    (e.g. ``'8573927'`` for Chesapeake City)
    t_start, t_end: ``'YYYY-MM-DD'`` range (inclusive)
    cache_dir     : directory for the NPZ checkpoint file
    cycles        : subset of daily cycles; default all 4
    work_dir      : temp directory for downloads; defaults to system temp

    Returns
    -------
    dict with keys:

    ``time_dt64``  : ndarray (N,)        numpy datetime64[s], sorted unique
    ``salt``       : ndarray (N, s_rho)  salinity [PSU]
    ``temp``       : ndarray (N, s_rho)  temperature [°C]
    ``zeta``       : ndarray (N,)        sea-surface elevation [m]
    ``Cs_r``       : ndarray (s_rho,)    sigma non-dim depth parameter
    ``s_rho``      : ndarray (s_rho,)    sigma coordinate values
    ``hc``         : float               critical depth [m]
    ``vtransform`` : int                 ROMS Vtransform (1 or 2)
    ``h_station``  : float               station water depth [m]
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_start = t_start.replace("-", "")
    safe_end   = t_end.replace("-", "")
    npz_path   = (cache_dir
                  / f"cbofs_sta_{station_match}_{safe_start}_{safe_end}.npz")

    if npz_path.exists():
        print(f"Loading cached CBOFS station T/S:\n  {npz_path}")
        d = np.load(npz_path, allow_pickle=True)
        return {
            "time_dt64"  : d["time_dt64"].astype("datetime64[s]"),
            "salt"       : d["salt"].astype(np.float64),
            "temp"       : d["temp"].astype(np.float64),
            "zeta"       : d["zeta"].astype(np.float64),
            "Cs_r"       : d["Cs_r"].astype(np.float64),
            "s_rho"      : d["s_rho"].astype(np.float64),
            "hc"         : float(d["hc"]),
            "vtransform" : int(d["vtransform"]),
            "h_station"  : float(d["h_station"]),
        }

    if work_dir is None:
        import tempfile
        work_dir = Path(tempfile.gettempdir()) / "cbofs_sta_tmp"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    d0     = datetime.strptime(t_start, "%Y-%m-%d")
    d1     = datetime.strptime(t_end,   "%Y-%m-%d")
    dates  = [(d0 + timedelta(days=i)).strftime("%Y-%m-%d")
              for i in range((d1 - d0).days + 1)]
    cycles = list(cycles)
    n_total = len(dates) * len(cycles)

    # accumulators
    all_times: list[np.ndarray] = []
    all_salt : list[np.ndarray] = []
    all_temp : list[np.ndarray] = []
    all_zeta : list[np.ndarray] = []

    station_idx_cache : Optional[int]        = None
    Cs_r_cached       : Optional[np.ndarray] = None
    s_rho_cached      : Optional[np.ndarray] = None
    hc_cached         : float = 2.0
    vtransform_cached : int   = 2
    h_station_cached  : float = np.nan

    _SAVE_EVERY = 100   # save checkpoint every N successful files
    _ckpt_path  = (
        cache_dir
        / f"_ckpt_{station_match}_{safe_start}_{safe_end}.npz"
    )

    n_ok = n_miss = n_err = n_done = 0
    done_key_set: set[str] = set()

    # ── Resume from checkpoint if available ──────────────────────────────────
    if _ckpt_path.exists():
        try:
            _ck  = np.load(_ckpt_path, allow_pickle=False)
            _dks = _ck["done_keys"]
            done_key_set = set(str(k) for k in _dks) if len(_dks) > 0 else set()
            if done_key_set:
                all_times = [_ck["all_times"].astype("datetime64[s]")]
                all_salt  = [_ck["all_salt"].astype(np.float64)]
                all_temp  = [_ck["all_temp"].astype(np.float64)]
                all_zeta  = [_ck["all_zeta"].astype(np.float64)]
                station_idx_cache = int(_ck["station_idx"])
                Cs_r_cached       = _ck["Cs_r"].astype(np.float64)
                s_rho_cached      = _ck["s_rho"].astype(np.float64)
                hc_cached         = float(_ck["hc"])
                vtransform_cached = int(_ck["vtransform"])
                h_station_cached  = float(_ck["h_station"])
                n_ok   = int(_ck["n_ok"])
                n_miss = int(_ck["n_miss"])
                n_err  = int(_ck["n_err"])
                n_done = n_ok + n_miss + n_err
        except Exception as _ck_err:
            log.warning("Checkpoint unreadable (%s) — starting fresh", _ck_err)
            _ckpt_path.unlink(missing_ok=True)
            done_key_set = set()

    prog = NotebookProgressDisplay(
        total = n_total,
        label = (f"CBOFS station fetch  {t_start} \u2192 {t_end}  "
                 f"({len(cycles)} cycles/day, {n_total} files)"),
    )
    if done_key_set:
        prog.log(f"Checkpoint loaded \u2014 resuming at "
                 f"{len(done_key_set)}/{n_total} files")

    for di, date_str in enumerate(dates):
        for cyc in cycles:
            _key = f"{date_str}_{cyc}"
            if _key in done_key_set:
                n_done += 1
                continue
            prog.update(n_done, current=f"{date_str} {cyc}",
                        status="downloading \u2026")
            url = _cbofs_station_url(date_str, cyc)
            tmp = work_dir / f"_cbofs_sta_{date_str}_{cyc}.nc"
            try:
                _download_file(url, tmp, max_retries=2)

                with nc4.Dataset(tmp, "r") as ds:
                    # discover station index on first successful file
                    if station_idx_cache is None:
                        if "station_name" in ds.variables:
                            for i, n in enumerate(
                                    nc4.chartostring(
                                        ds.variables["station_name"][:])):
                                if station_match in n:
                                    station_idx_cache = i
                                    prog.log(f"Station idx={i}  "
                                             f"name={n.strip()!r}")
                                    break
                        if station_idx_cache is None:
                            lon_key = next(
                                (k for k in ("station_lon", "lon_rho", "lon")
                                 if k in ds.variables), None)
                            lat_key = next(
                                (k for k in ("station_lat", "lat_rho", "lat")
                                 if k in ds.variables), None)
                            if lon_key and lat_key:
                                slon  = ds.variables[lon_key][:].ravel()
                                slat  = ds.variables[lat_key][:].ravel()
                                dists = np.hypot(slon - CANAL_LON_DEFAULT,
                                                 slat - CANAL_LAT_DEFAULT)
                                station_idx_cache = int(np.argmin(dists))
                                prog.log(f"Station idx={station_idx_cache} "
                                         f"(nearest lat/lon fallback)")
                        if station_idx_cache is None:
                            raise RuntimeError(
                                f"Station '{station_match}' not found.")

                    si = station_idx_cache

                    # time
                    ot_var   = ds.variables["ocean_time"]
                    ot_vals  = ot_var[:].astype(np.float64)
                    ot_units = getattr(ot_var, "units",
                                       "seconds since 2000-01-01")
                    ot_cal   = _norm_calendar(getattr(ot_var, "calendar", "gregorian"))
                    dts      = nc4.num2date(ot_vals, ot_units, ot_cal)
                    times_arr = np.array(
                        [np.datetime64(d.strftime("%Y-%m-%dT%H:%M:%S"), "s")
                         for d in dts],
                        dtype="datetime64[s]",
                    )

                    # T/S/zeta at target station — handle both dimension orders:
                    #   (ocean_time, s_rho, station)  [most CBOFS files]
                    #   (ocean_time, station, s_rho)  [some older files]
                    def _sta_slice(var, sta_idx: int) -> np.ndarray:
                        """Extract the (time, sigma) slice for one station,
                        regardless of whether the station axis is 1 or 2."""
                        dims = [d.lower() for d in var.dimensions]
                        sta_ax = next(
                            (i for i, d in enumerate(dims) if "stat" in d), 2)
                        idx: list = [slice(None)] * var.ndim
                        idx[sta_ax] = sta_idx
                        return np.ma.filled(
                            var[tuple(idx)].astype(np.float64), np.nan)

                    salt_arr = _sta_slice(ds.variables["salt"], si)
                    temp_arr = _sta_slice(ds.variables["temp"], si)
                    zeta_arr = np.ma.filled(
                        ds.variables["zeta"][:, si].astype(np.float64),
                        np.nan)

                    # sigma metadata — cache on first hit
                    if Cs_r_cached is None:
                        Cs_r_cached = ds.variables["Cs_r"][:].astype(
                            np.float64)
                        n_sig = len(Cs_r_cached)
                        if "s_rho" in ds.variables:
                            s_rho_cached = ds.variables["s_rho"][:].astype(
                                np.float64)
                        else:
                            s_rho_cached = np.linspace(
                                -1.0 + 0.5/n_sig, -0.5/n_sig, n_sig)
                        hc_cached = float(getattr(ds, "hc", 2.0))
                        if "hc" in ds.variables:
                            hc_cached = float(ds.variables["hc"][:])
                        vtransform_cached = int(getattr(ds, "Vtransform", 2))
                        if "Vtransform" in ds.variables:
                            vtransform_cached = int(
                                ds.variables["Vtransform"][:])

                    if np.isnan(h_station_cached) and "h" in ds.variables:
                        h_arr = ds.variables["h"][:]
                        h_station_cached = float(
                            h_arr[si] if h_arr.ndim == 1
                            else h_arr.ravel()[si])

                all_times.append(times_arr)
                all_salt.append(salt_arr)
                all_temp.append(temp_arr)
                all_zeta.append(zeta_arr)
                done_key_set.add(_key)
                n_ok += 1

                # ── Periodic checkpoint ───────────────────────────────────
                if n_ok % _SAVE_EVERY == 0:
                    _t_c = np.concatenate(all_times)
                    _s_c = np.concatenate(all_salt,  axis=0)
                    _p_c = np.concatenate(all_temp,  axis=0)
                    _z_c = np.concatenate(all_zeta)
                    np.savez_compressed(
                        _ckpt_path,
                        done_keys   = np.array(sorted(done_key_set)),
                        all_times   = _t_c.astype("datetime64[s]"),
                        all_salt    = _s_c.astype(np.float32),
                        all_temp    = _p_c.astype(np.float32),
                        all_zeta    = _z_c.astype(np.float32),
                        station_idx = np.array([
                            station_idx_cache if station_idx_cache is not None
                            else -1]),
                        Cs_r        = (Cs_r_cached if Cs_r_cached is not None
                                       else np.array([])),
                        s_rho       = (s_rho_cached if s_rho_cached is not None
                                       else np.array([])),
                        hc          = np.array([hc_cached]),
                        vtransform  = np.array([vtransform_cached]),
                        h_station   = np.array([h_station_cached]),
                        n_ok        = np.array([n_ok]),
                        n_miss      = np.array([n_miss]),
                        n_err       = np.array([n_err]),
                    )
                    prog.log(f"Checkpoint saved ({n_ok}/{n_total} done)")

            except RuntimeError:
                n_miss += 1
                log.debug("Missing %s %s", date_str, cyc)
            except Exception as exc:
                n_err += 1
                log.warning("Error %s %s: %s", date_str, cyc, exc)
                prog.log(f"ERR {date_str} {cyc}: {str(exc)[:80]}")
            finally:
                if tmp.exists():
                    tmp.unlink()
                n_done += 1

    prog.close(
        summary=(
            f"Done: {n_ok} ok / {n_miss} missing / {n_err} errors  "
            f"({n_total} total) ✓"
        )
    )

    if not all_times:
        raise RuntimeError(
            f"No CBOFS station files successfully downloaded for "
            f"{t_start}–{t_end}.\n"
            f"Verify archive at: {NCEI_CBOFS_BASE}"
        )

    # concatenate + deduplicate by time
    time_cat = np.concatenate(all_times)
    salt_cat = np.concatenate(all_salt,  axis=0)
    temp_cat = np.concatenate(all_temp,  axis=0)
    zeta_cat = np.concatenate(all_zeta,  axis=0)

    _, uniq_idx = np.unique(time_cat, return_index=True)
    time_out    = time_cat[uniq_idx]
    salt_out    = salt_cat[uniq_idx]
    temp_out    = temp_cat[uniq_idx]
    zeta_out    = zeta_cat[uniq_idx]

    print(f"\nExtraction complete: {len(time_out)} unique time steps")
    print(f"  Time  : {time_out[0]} → {time_out[-1]}")
    print(f"  Salt  : {np.nanmin(salt_out):.2f} – {np.nanmax(salt_out):.2f} PSU")
    print(f"  Temp  : {np.nanmin(temp_out):.2f} – {np.nanmax(temp_out):.2f} °C")
    print(f"  Downloads: ok={n_ok}  missing={n_miss}  errors={n_err}")

    result = dict(
        time_dt64  = time_out,
        salt       = salt_out,
        temp       = temp_out,
        zeta       = zeta_out,
        Cs_r       = Cs_r_cached if Cs_r_cached is not None else np.array([]),
        s_rho      = (s_rho_cached if s_rho_cached is not None
                      else np.array([])),
        hc         = hc_cached,
        vtransform = vtransform_cached,
        h_station  = h_station_cached,
    )

    np.savez_compressed(
        npz_path,
        time_dt64  = time_out.astype("datetime64[s]"),
        salt       = salt_out.astype(np.float32),
        temp       = temp_out.astype(np.float32),
        zeta       = zeta_out.astype(np.float32),
        Cs_r       = result["Cs_r"].astype(np.float32),
        s_rho      = result["s_rho"].astype(np.float32),
        hc         = np.array([hc_cached]),
        vtransform = np.array([vtransform_cached]),
        h_station  = np.array([h_station_cached]),
    )
    print(f"  Cached → {npz_path}")
    # Checkpoint no longer needed now that the full NPZ is saved
    if _ckpt_path.exists():
        _ckpt_path.unlink()
    return result
