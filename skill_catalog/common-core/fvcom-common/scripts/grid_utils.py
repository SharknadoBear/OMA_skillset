"""
grid_utils.py
=============
FVCOM mesh I/O, spatial interpolation helpers, sigma-coordinate utilities,
and time conversion functions shared across all forcing modules.

Functions
---------
read_fvcom_mesh_dat     : read grid, bathymetry, and sigma levels from .dat files
read_obc_nodes_dat      : read open boundary node list from .dat file
interp_regular_to_nodes : bilinear/nearest interpolation from regular grid to nodes
interp_z_to_sigma       : vertical remapping from z-levels to FVCOM sigma levels
datetime64_to_mjd       : numpy datetime64 → Modified Julian Day
mjd_to_datetime64       : Modified Julian Day → numpy datetime64
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------

def read_fvcom_mesh_dat(grid_dat: str | Path,
                        dep_dat: str | Path | None = None,
                        sig_dat: str | Path | None = None) -> dict:
    """
    Read FVCOM unstructured mesh from ASCII .dat files.

    Parameters
    ----------
    grid_dat : path to waterPACT_grd.dat
        Format:
          Line 1: "Node Number = NVERT"
          Line 2: "Element Number = NELEM"
          NELEM lines: "elem_id  n1  n2  n3  type"
          NVERT lines: "node_id  lon  lat  [unused...]"
    dep_dat  : path to waterPACT_dep.dat  (optional)
        Format:
          Line 1: "Node Number = NVERT"
          NVERT lines: "?  ?  depth"   (depth is 3rd column)
    sig_dat  : path to waterPACT_sig.dat  (optional)
        Format: one sigma level value per non-comment line.

    Returns
    -------
    dict with keys: lon, lat, tri (0-based), nvert, nelem,
                    h (if dep_dat), siglev, siglay (if sig_dat)
    """
    grid_dat = Path(grid_dat)
    mesh = {}

    with open(grid_dat) as f:
        line1 = f.readline()
        nvert = int(line1.split("=")[-1].strip())
        line2 = f.readline()
        nelem = int(line2.split("=")[-1].strip())

        tri = np.zeros((nelem, 3), dtype=np.int32)
        for i in range(nelem):
            parts = f.readline().split()
            tri[i] = [int(parts[1]) - 1,
                      int(parts[2]) - 1,
                      int(parts[3]) - 1]   # convert to 0-based

        lon = np.empty(nvert)
        lat = np.empty(nvert)
        for i in range(nvert):
            parts = f.readline().split()
            lon[i] = float(parts[1])
            lat[i] = float(parts[2])

    mesh.update({"lon": lon, "lat": lat, "tri": tri,
                 "nvert": nvert, "nelem": nelem})

    if dep_dat is not None:
        dep_dat = Path(dep_dat)
        h = np.empty(nvert)
        with open(dep_dat) as f:
            f.readline()   # skip "Node Number = N"
            for i in range(nvert):
                parts = f.readline().split()
                h[i] = float(parts[2])   # 3rd column is depth
        mesh["h"] = h

    if sig_dat is not None:
        sig_dat = Path(sig_dat)
        raw_vals = []
        with open(sig_dat) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("!"):
                    continue
                try:
                    raw_vals.append(float(line.split()[-1]))
                except ValueError:
                    continue
        if raw_vals:
            siglev = np.array(raw_vals)
            siglay = (siglev[:-1] + siglev[1:]) / 2.0
            mesh["siglev"] = siglev
            mesh["siglay"] = siglay

    return mesh


def read_obc_nodes_dat(obc_dat: str | Path) -> np.ndarray:
    """
    Read FVCOM open boundary node list from waterPACT_obc.dat.

    Format::

        OBC Node Number = N
        [seq_idx]  [node_id_1based]  [type_flag]
        ...

    Returns
    -------
    1-D int32 array of 1-based global node indices (column 2).
    """
    obc_dat = Path(obc_dat)
    nodes = []
    with open(obc_dat) as f:
        header = f.readline()
        n_obc = int(header.split("=")[-1].strip())
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                nodes.append(int(parts[1]))   # 1-based global node ID
    nodes = np.array(nodes, dtype=np.int32)
    if len(nodes) != n_obc:
        raise ValueError(
            f"OBC node count mismatch: header says {n_obc}, "
            f"read {len(nodes)} lines in {obc_dat}"
        )
    return nodes   # 1-based


# ---------------------------------------------------------------------------
# Spatial interpolation
# ---------------------------------------------------------------------------

def interp_regular_to_nodes(lon_reg: np.ndarray,
                             lat_reg: np.ndarray,
                             data: np.ndarray,
                             target_lon: np.ndarray,
                             target_lat: np.ndarray,
                             method: str = "linear") -> np.ndarray:
    """
    Bilinear (or nearest-neighbour) interpolation from a regular grid
    to a set of target (lon, lat) points.

    Parameters
    ----------
    lon_reg, lat_reg : 1-D coordinate arrays of the source regular grid
    data             : 2-D array (lat x lon) or 3-D (time x lat x lon)
    target_lon/lat   : 1-D arrays of destination point coordinates
    method           : 'linear' (default) or 'nearest'

    Returns
    -------
    Array of shape (target_lon.size,) or (ntime, target_lon.size)

    Notes
    -----
    * ``lon_reg`` and ``target_lon`` must use the **same** longitude
      convention (both [0, 360] or both [-180, 180]).  Convert before
      calling if needed.
    * Out-of-bounds target points are handled by extrapolation clamping
      (``fill_value=None`` in :class:`~scipy.interpolate.RegularGridInterpolator`).
    """
    lon_reg    = np.asarray(lon_reg,    dtype=np.float64).ravel()
    lat_reg    = np.asarray(lat_reg,    dtype=np.float64).ravel()
    target_lon = np.asarray(target_lon, dtype=np.float64).ravel()
    target_lat = np.asarray(target_lat, dtype=np.float64).ravel()
    data       = np.asarray(data,       dtype=np.float64)

    pts = np.column_stack([target_lat, target_lon])  # (npts, 2)

    if data.ndim == 2:
        # Single snapshot (lat × lon)
        interp = RegularGridInterpolator(
            (lat_reg, lon_reg), data,
            method=method, bounds_error=False, fill_value=None,
        )
        return interp(pts)

    elif data.ndim == 3:
        # Time-varying (ntime × lat × lon)
        ntime, npts = data.shape[0], len(target_lat)
        out = np.empty((ntime, npts), dtype=np.float64)
        for t in range(ntime):
            interp = RegularGridInterpolator(
                (lat_reg, lon_reg), data[t],
                method=method, bounds_error=False, fill_value=None,
            )
            out[t] = interp(pts)
        return out

    else:
        raise ValueError(
            f"data must be 2-D (lat×lon) or 3-D (time×lat×lon); got shape {data.shape}"
        )


# ---------------------------------------------------------------------------
# Vertical coordinate utilities
# ---------------------------------------------------------------------------

def interp_z_to_sigma(data_z: np.ndarray,
                      z_levels: np.ndarray,
                      depth: float | np.ndarray,
                      sigma_levels: np.ndarray) -> np.ndarray:
    """
    Vertically re-map data from z-levels (ROMS) to FVCOM sigma levels.

    This function handles the single-node case used in F05 (CBOFS canal
    extraction), where data comes from one ROMS rho-point and must be
    remapped onto the FVCOM model sigma grid at the corresponding OBC node.

    Parameters
    ----------
    data_z       : (ntime, nz) array of field values on ROMS sigma levels.
                   k=0 is deepest (near-bottom), k=nz-1 is shallowest
                   (near-surface).  Same convention as ``z_levels``.
    z_levels     : (ntime, nz) or (nz,) array of actual physical depths
                   [metres, negative-up].  Must be monotonically *increasing*
                   from index 0 (most negative / deepest) to index nz-1
                   (least negative / shallowest).  This is the natural output
                   of ``_roms_depths_at_node()`` in cbofs_fetcher.
    depth        : scalar or (ntime,) array giving the total water depth at
                   the node [metres, positive].  Used to compute the target
                   FVCOM sigma depths via z_sigma[k] = sigma_levels[k] * depth.
    sigma_levels : (nsiglay,) FVCOM siglay values in the range [-1, 0].
                   siglay[0] is just below the surface (≈ −0.017),
                   siglay[nsiglay-1] is just above the bed (≈ −0.983).

    Returns
    -------
    data_sigma : (ntime, nsiglay) array of field values at the FVCOM sigma
                 levels.  Values outside the source z_levels range are
                 extrapolated by clamping (boundary-value hold).

    Notes
    -----
    * ``np.interp`` is used per time step.  It requires *xp* to be
      increasing, which is satisfied by ``z_levels`` (bottom → surface).
    * Extrapolation at both ends is handled by ``np.interp``'s default
      clamping behaviour (left=fp[0], right=fp[-1]).
    * The function deliberately does not modify *data_z* in-place.
    """
    data_z   = np.asarray(data_z,   dtype=np.float64)
    z_levels = np.asarray(z_levels, dtype=np.float64)
    depth    = np.asarray(depth,    dtype=np.float64)

    ntime, nz = data_z.shape
    nsiglay   = len(sigma_levels)

    # Broadcast depth to (ntime,) if scalar
    if depth.ndim == 0:
        depth_arr = np.full(ntime, float(depth))
    else:
        depth_arr = depth.ravel()[:ntime]

    # z_levels may be (ntime, nz) or (nz,) — broadcast to (ntime, nz)
    if z_levels.ndim == 1:
        z_levels_2d = np.broadcast_to(z_levels[np.newaxis, :], (ntime, nz))
    else:
        z_levels_2d = z_levels

    data_sigma = np.empty((ntime, nsiglay), dtype=np.float64)

    for t in range(ntime):
        # Target depths in physical metres (negative-up)
        # sigma_levels[k] ∈ [-1, 0]  →  z_sigma[k] = sigma * depth  (≤ 0)
        z_sigma = sigma_levels * depth_arr[t]   # shape (nsiglay,)

        # Source levels for this time step
        zp = z_levels_2d[t]      # increasing: bottom → surface
        fp = data_z[t]

        # np.interp clamps automatically outside [zp[0], zp[-1]]
        data_sigma[t] = np.interp(z_sigma, zp, fp)

    return data_sigma


# ---------------------------------------------------------------------------
# Time conversion utilities
# ---------------------------------------------------------------------------

MJD_EPOCH = np.datetime64("1858-11-17T00:00:00", "s")


def datetime64_to_mjd(times: np.ndarray) -> np.ndarray:
    """Convert numpy datetime64 array to Modified Julian Day (float64)."""
    delta = (times.astype("datetime64[s]") - MJD_EPOCH)
    return delta / np.timedelta64(1, "D")


def mjd_to_datetime64(mjd: np.ndarray) -> np.ndarray:
    """Convert Modified Julian Day array to numpy datetime64[s]."""
    delta = (mjd * 86400).astype("int64") * np.timedelta64(1, "s")
    return MJD_EPOCH + delta
