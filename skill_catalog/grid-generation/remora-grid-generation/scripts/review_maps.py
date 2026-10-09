"""Consistent geographic evidence and feature-section diagnostics."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from remora_regions import setup_map
from grid_math import vertical_section


def diagnostic_maps(out, a, h, raw, mask, initial, land, features, vertical_parameters):
    out = Path(out)
    paths = []
    lo, la = a["lon_rho"], a["lat_rho"]
    full = [lo.min(), la.min(), lo.max(), la.max()]
    change = np.where(mask, h - raw, np.nan)
    percent = np.where(mask, 100 * (h / raw - 1), np.nan)
    maxchange = max(1., float(np.nanmax(np.abs(change))))
    maxpercent = max(1., float(np.nanmax(np.abs(percent))))
    min_depth = min(float(h[mask].min()), float(raw[mask].min()))
    max_depth = max(float(h[mask].max()), float(raw[mask].max()))
    depthnorm = LogNorm(min_depth, max_depth) if max_depth / min_depth > 100 else None

    def panel_file(name, panels, bounds=full, title=None):
        inside = (lo >= bounds[0]) & (lo <= bounds[2]) & (la >= bounds[1]) & (la <= bounds[3])
        js, ins = np.where(inside)
        if not len(js):
            raise ValueError("Feature review box misses the grid: " + name)
        sl = (slice(max(0, js.min()-1), min(lo.shape[0], js.max()+2)),
              slice(max(0, ins.min()-1), min(lo.shape[1], ins.max()+2)))
        fig, axs = plt.subplots(1, len(panels), figsize=(6 * len(panels), 7), squeeze=False, layout="constrained")
        for ax, (data, caption, cmap, lower, upper, norm) in zip(axs[0], panels):
            setup_map(ax, land, bounds)
            kw = {"norm": norm} if norm else {"vmin": lower, "vmax": upper}
            im = ax.pcolormesh(lo[sl], la[sl], data[sl], shading="nearest", cmap=cmap, zorder=2, **kw)
            local = land.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            if not local.empty:
                local.boundary.plot(ax=ax, color="black", linewidth=.35, zorder=3)
            ax.set_title(caption)
            ax.set_xlabel("Longitude (degrees east)")
            ax.set_ylabel("Latitude (degrees north)")
            fig.colorbar(im, ax=ax, shrink=.7)
        fig.suptitle(title or name.replace("_", " "))
        path = out / (name + ".png")
        fig.savefig(path, dpi=125)
        plt.close(fig)
        paths.append(path)

    panel_file("mask_changes", [
        (initial.astype(float), "Initial wet mask (1=water)", "Blues", 0, 1, None),
        (mask.astype(float), "Final wet mask (1=water)", "Blues", 0, 1, None),
        ((initial != mask).astype(float), "Corrected cells (1=changed)", "RdPu", 0, 1, None)])
    depthpanels = [(np.where(mask, raw, np.nan), "Before smoothing (m)", "viridis", min_depth, max_depth, depthnorm),
                   (np.where(mask, h, np.nan), "After log smoothing (m)", "viridis", min_depth, max_depth, depthnorm)]
    panel_file("bathymetry_comparison", depthpanels, title="Bathymetry comparison — identical depth scales")
    panel_file("smoothing_changes", [
        (change, "Depth change (m)", "RdBu_r", -maxchange, maxchange, None),
        (percent, "Relative depth change (%)", "RdBu_r", -maxpercent, maxpercent, None)])
    panel_file("geometry_quality", [
        (1/a["pm"], "Cell width in xi (m)", "viridis", None, None, None),
        (a["orthogonality_error_deg"], "Orthogonality deviation (degrees)", "viridis", None, None, None)])
    for k, f in enumerate(features):
        x, y = f["point_lonlat"]
        dy = f.get("review_radius_km", 12)/111.32
        dx = dy/np.cos(np.deg2rad(y))
        panel_file(f"feature_{k}_mask_bathy", [
            (mask.astype(float), "Final wet mask (1=water)", "Blues", 0, 1, None),
            depthpanels[1],
            (change, "Depth change (m)", "RdBu_r", -maxchange, maxchange, None)],
            [x-dx, y-dy, x+dx, y+dy], title=f["name"])

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    j, i = int(np.argmax(mask.sum(axis=1))), int(np.argmax(mask.sum(axis=0)))
    p = vertical_parameters
    row = vertical_section(h, 0, j, p["N"], p["theta_s"], p["theta_b"], p["hc_m"])
    column = vertical_section(h, 1, i, p["N"], p["theta_s"], p["theta_b"], p["hc_m"])
    for ax, section, wet, x, title in [
        (axs[0], row, mask[j], a["x_rho"][j]/1000, f"eta={j}"),
        (axs[1], column, mask[:,i], a["y_rho"][:,i]/1000, f"xi={i}")]:
        for k in range(0, len(section), max(1, (len(section)-1)//20)):
            ax.plot(x, np.where(wet, section[k], np.nan), color="#13688c", lw=.6)
        ax.set(title="Vertical interfaces: "+title, xlabel="Logical distance (km)", ylabel="z (m)")
        ax.grid(alpha=.2)
    path=out/"vertical_sections.png"
    fig.savefig(path,dpi=140); plt.close(fig); paths.append(path)
    return paths


def feature_sections(out, a, raw, h, mask, anchors, features):
    """Grid-axis sections through each protected feature's contiguous wet reach.

    These are explicitly orientation-dependent diagnostics, not a claim that
    an automatically selected line is normal to the physical channel.
    """
    records, paths = [], []
    for k, ((j,i), feature) in enumerate(zip(anchors, features)):
        fig, axs = plt.subplots(2,1,figsize=(11,7),layout="constrained")
        for axis, ax in enumerate(axs):
            wet = mask[j,:] if axis == 0 else mask[:,i]
            center = i if axis == 0 else j
            width = 1/a["pm"][j,:] if axis == 0 else 1/a["pn"][:,i]
            before = raw[j,:] if axis == 0 else raw[:,i]
            after = h[j,:] if axis == 0 else h[:,i]
            radius = feature.get("review_radius_km",12)*1000
            distance = np.cumsum(width) - width/2
            distance -= distance[center]
            left=right=center
            while left>0 and wet[left-1] and abs(distance[left-1]) <= radius: left-=1
            while right<len(wet)-1 and wet[right+1] and abs(distance[right+1]) <= radius: right+=1
            sl=slice(left,right+1)
            old=float(np.sum(before[sl]*width[sl])); new=float(np.sum(after[sl]*width[sl]))
            tag="xi" if axis==0 else "eta"
            records.append(dict(name=feature["name"],axis=tag,j=int(j),i=int(i),wet_cells=right-left+1,
                                length_m=float(width[sl].sum()),raw_area_m2=old,smoothed_area_m2=new,
                                relative_area_change=(new-old)/old,
                                interpretation="contiguous grid-axis transect limited to review radius"))
            ax.plot(distance[sl]/1000,-before[sl],label="Before smoothing",color="#777777")
            ax.plot(distance[sl]/1000,-after[sl],label="After log smoothing",color="#087d82")
            ax.set(title=f"{feature['name']} — {tag} section; area change {(new/old-1)*100:.2f}%",
                   xlabel="Distance from feature (km)",ylabel="Bottom elevation (m)")
            ax.grid(alpha=.2); ax.legend()
        path=Path(out)/f"feature_{k}_sections.png"
        fig.savefig(path,dpi=140); plt.close(fig); paths.append(path)
    return records, paths
