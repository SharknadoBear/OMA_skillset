"""Source-grid maps and wet-cell profiles for a validated five-field GLORYS run."""
from __future__ import annotations
import argparse
import csv
import html
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import glorys_core as g


def mesh_outline(path):
    with Path(path).open() as f:
        f.readline(); ne, nn = map(int,f.readline().split()[:2])
        nodes = np.loadtxt(f,max_rows=nn,usecols=(1,2))
        triangles = np.loadtxt(f,max_rows=ne,usecols=(2,3,4),dtype=np.int64)-1
    nodes[:,0]=(nodes[:,0]+180)%360-180
    edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    return nodes[unique[counts==1]],nn,ne


def render(run_dir, output_dir, mesh=None):
    run_dir=Path(run_dir); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    h=g.health(run_dir)
    if not h["pass"]: raise g.ValidationError("Visual review requires a complete healthy run")
    plan=g.read_json(run_dir/"download_plan.json")
    if mesh is None and (run_dir/"mesh_reference.json").exists():
        m=g.read_json(run_dir/"mesh_reference.json"); mesh=run_dir/m["path"]
    outline,nn,ne=mesh_outline(mesh) if mesh else (None,0,0)
    with xr.open_dataset(run_dir/"subset.nc",engine="h5netcdf") as source:
        ds=source.load()
    for n in ("zos","thetao","so","uo","vo"):
        if n not in ds: raise g.RequestError("Visual pilot requires field: "+n)
    wet=np.isfinite(ds.zos.values).all(axis=0)
    for n in ("thetao","so","uo","vo"): wet &= np.isfinite(ds[n].values).all(axis=(0,1))
    candidates=np.argwhere(wet)
    if not candidates.size: raise g.ValidationError("No wet source cell covers every requested depth and record")
    # Offshore = westernmost complete wet column, nearest domain-centre latitude.
    western=candidates[candidates[:,1]==candidates[:,1].min()]
    iy,ix=western[np.argmin(np.abs(ds.latitude.values[western[:,0]]-np.mean(plan["request"]["bbox"][1::2])))]
    lon,lat=float(ds.longitude.values[ix]),float(ds.latitude.values[iy])
    clocks=[g.iso(v) for v in ds.time.values]
    point={"selection":"Westernmost column wet at all requested depths/times, closest to domain-centre latitude",
        "latitude":lat,"longitude":lon,"latitude_index":int(iy),"longitude_index":int(ix),
        "surface_depth_m":float(ds.depth.values[0]),"deepest_selected_depth_m":float(ds.depth.values[-1]),
        "source_clocks":clocks}
    g.write_json(out/"wet_cell.json",point)
    x,y=np.meshgrid(ds.longitude.values,ds.latitude.values)
    surface={n:ds[n].isel(time=0,depth=0).values for n in ("thetao","so","uo","vo")}
    speed=np.hypot(surface["uo"],surface["vo"])
    fig,axs=plt.subplots(2,2,figsize=(11,8.8),layout="constrained")
    panels=[(ds.zos.isel(time=0).values,"Sea-surface height above source geoid","m","viridis"),
        (surface["thetao"],"Potential temperature at shallowest source level",ds.thetao.attrs.get("units",""),"plasma"),
        (surface["so"],"Salinity at shallowest source level",ds.so.attrs.get("units",""),"viridis"),
        (speed,"Horizontal current speed and vectors",ds.uo.attrs.get("units",""),"magma")]
    for i,(ax,(z,title,unit,cmap)) in enumerate(zip(axs.flat,panels)):
        ax.set_facecolor("#dedede")
        im=ax.pcolormesh(x,y,np.ma.masked_invalid(z),shading="nearest",cmap=cmap)
        fig.colorbar(im,ax=ax,label=unit,shrink=.85)
        if outline is not None: ax.add_collection(LineCollection(outline,colors="#243c52",linewidths=.65))
        ax.plot(lon,lat,"o",color="cyan",markeredgecolor="black",markersize=6)
        ax.set(xlabel="Longitude (degrees east)",ylabel="Latitude (degrees north)",title=title)
        ax.set_aspect(1/np.cos(np.deg2rad(lat)))
        if i==3:
            q=ax.quiver(x,y,surface["uo"],surface["vo"],scale=2.5,color="#37daf5",width=.004)
            ax.quiverkey(q,.77,.94,.2,"0.2 m/s",coordinates="axes",labelpos="S",labelcolor="white")
    fig.suptitle(f"GLORYS12 source subset • {clocks[0]}\nMesh v5/r04 outline • masks retained • surface level {point['surface_depth_m']:.3f} m",fontsize=13)
    fig.savefig(out/"coverage_maps.png",dpi=155); plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(12,5.1),sharey=True,layout="constrained")
    for ax,n,label in zip(axs,("thetao","so","uo","vo"),("Potential temperature","Salinity","Eastward velocity","Northward velocity")):
        for t,clock in enumerate(clocks): ax.plot(ds[n].values[t,:,iy,ix],ds.depth.values,".-",label=clock[:10],markersize=3)
        ax.set(xlabel=ds[n].attrs.get("units",""),title=label); ax.grid(alpha=.2)
    axs[0].set_ylabel("Source depth (m, positive down)"); axs[0].invert_yaxis(); axs[-1].legend(fontsize=8)
    fig.suptitle(f"Wet source-cell profiles • {lon:.5f}°, {lat:.5f}° • no depth interpolation")
    fig.savefig(out/"wet_cell_profiles.png",dpi=155); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.3),layout="constrained")
    ax.plot(ds.time.values,ds.zos.values[:,iy,ix],"o-"); ax.grid(alpha=.25)
    ax.set(ylabel="SSH above source geoid (m)",xlabel="Stored source time (UTC)",title="Four-day SSH at documented offshore source cell")
    fig.savefig(out/"wet_cell_ssh.png",dpi=155); plt.close(fig)
    with (out/"wet_cell_profiles.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["time_utc","longitude","latitude","depth_m","thetao","so","uo","vo"])
        for t,clock in enumerate(clocks):
            for k,depth in enumerate(ds.depth.values): w.writerow([clock,lon,lat,float(depth)]+[float(ds[n].values[t,k,iy,ix]) for n in ("thetao","so","uo","vo")])
    with (out/"wet_cell_ssh.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["time_utc","longitude","latitude","zos_m"])
        for t,clock in enumerate(clocks): w.writerow([clock,lon,lat,float(ds.zos.values[t,iy,ix])])
    inventory=g.read_json(run_dir/"inventory.json")
    rows="".join(f"<tr><td>{html.escape(n)}</td><td>{html.escape(str(s['attributes'].get('units','')))}</td><td>{html.escape(', '.join(s['dimensions']))}</td></tr>" for n,s in plan["variables"].items())
    stages={}
    for name in ("smoke","august","resume","repeat"):
        path=run_dir.parent/("stage_"+name+".json")
        if path.exists(): stages[name]=g.read_json(path)
    summary={"health":h,"wet_cell":point,"mesh_sha256":g.file_hash(mesh) if mesh else None,
        "mesh_nodes":nn,"mesh_elements":ne,"available_variables":list(inventory["variables"]),"stages":stages}
    g.write_json(out/"visual_evidence.json",summary)
    offline_path=run_dir.parent/"offline_tests.json"
    offline=g.read_json(offline_path) if offline_path.exists() else {}
    prefix=html.escape(os.path.relpath(run_dir.resolve(),out.resolve()).replace(os.sep,"/"))
    phase="All stored records are at 00:00 UTC. The producer manual describes noon-centered daily means; no timestamp shift was applied." if all(c.endswith("T00:00:00Z") for c in clocks) else "Published timestamps were preserved without shifting."
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Cal Poly GLORYS12 verification</title>
<style>body{{font:16px system-ui;color:#1e3346;background:#f5f7fa;max-width:1100px;margin:40px auto;padding:0 24px}}h1{{font-size:30px}}section{{background:white;padding:22px;margin:18px 0;border-radius:12px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}code{{overflow-wrap:anywhere}}.pass{{color:#177345;font-weight:bold}}</style>
<h1>Cal Poly GLORYS12 verification</h1><p>Daily source acquisition • frozen v5/r04 domain • August 30–September 2, 2025</p>
<section><p class="pass">Authenticated live acquisition and independent health: PASS</p>
<p>Offline checks: {offline.get('tests_run','—')} tests; pass: {offline.get('pass','—')}. Source timestamp difference is documented below.</p>
<p>{len(clocks)} records; {ds.sizes['longitude']} × {ds.sizes['latitude']} source cells; {ds.sizes['depth']} existing depth levels, {point['surface_depth_m']:.3f}–{point['deepest_selected_depth_m']:.3f} m. Requested depth interval: 0–200 m.</p>
<p>{phase}</p><p>Version: <code>{plan['source']['dataset_version']}</code>. Source: <code>{plan['source']['dataset_id']}</code>.</p>
<p>Mesh: {nn:,} nodes, {ne:,} elements. SHA256: <code>{summary['mesh_sha256']}</code>.</p>
<p>August pause: {stages.get('august',{}).get('completed_chunks','—')} chunk. Resume new/reused: {stages.get('resume',{}).get('new_chunks','—')}/{stages.get('resume',{}).get('reused_chunks','—')}. Completed repeat new/reused: {stages.get('repeat',{}).get('new_chunks','—')}/{stages.get('repeat',{}).get('reused_chunks','—')}.</p>
<table><tr><th>Field</th><th>Published units</th><th>Dimensions</th></tr>{rows}</table></section>
<section><h2>Coverage and source fields</h2><p>First record. Grey cells are source missing values. Navy lines trace the complete latest mesh exterior; cyan marker identifies the profile cell. The regular published source grid is preserved.</p><img src="coverage_maps.png" alt="SSH, surface temperature, salinity and currents with mesh outline"></section>
<section><h2>Offshore profiles</h2><p>Source cell: {lon:.5f}° longitude, {lat:.5f}° latitude (indices longitude {ix}, latitude {iy}). Wet at every selected level and record. {point['selection']}.</p><img src="wet_cell_profiles.png" alt="Temperature, salinity and horizontal velocity profiles"><img src="wet_cell_ssh.png" alt="Four daily SSH values"></section>
<section><h2>Interpretation and evidence</h2><p>Temperature remains potential temperature and salinity retains the published convention. SSH remains above the source geoid. Masks and decoded values match the committed source-subset chunks. No datum conversion, tidal addition or model forcing was generated. This four-day pilot verifies acquisition, not the longer variability needed for boundary physics.</p>
<p>Plausibility excursions: {len(h['plausibility_excursions'])}. U/V alignment and missing-pattern checks are retained in chunk receipts.</p>
<p><a href="{prefix}/health_report.json">Independent health</a> · <a href="{prefix}/manifest.json">Provenance</a> · <a href="{prefix}/download_plan.json">Plan</a> · <a href="wet_cell.json">Wet cell</a> · <a href="wet_cell_profiles.csv">Profiles CSV</a> · <a href="wet_cell_ssh.csv">SSH CSV</a></p>
<p><a href="https://documentation.marine.copernicus.eu/PUM/CMEMS-GLO-PUM-001-030.pdf">Product manual</a> · <a href="https://toolbox-docs.marine.copernicus.eu/en/stable/usage/subset-usage.html">Official Toolbox subset documentation</a></p></section></html>'''
    (out/"report.html").write_text(doc,encoding="utf-8")
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--mesh")
    a=p.parse_args(); result=render(a.run_dir,a.output_dir,a.mesh)
    print("Visual report saved; wet cell:",result["wet_cell"])
