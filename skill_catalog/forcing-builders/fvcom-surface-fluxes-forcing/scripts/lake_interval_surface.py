"""Encode end-anchored ERA5 and CFS heat and freshwater intervals for FVCOM."""

import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4 as nc
import numpy as np
from pyproj import CRS

from acquire_wind_pressure import digest, read, source_file
from lake_surface import (LATENT_HEAT, WATER_DENSITY, create_file, interpolate, native_geometry,
                          precipitation_shape, unit_fluxes)
from prepare_wind_pressure import (NAMES, checked_fields, coordinates, geographic_points, hour_axis,
                                   model_basis, psimple_drag, require, source_law, source_time, stamp, utc)

ERA5 = {"short": "ssr", "long": "str", "sensible": "sshf", "latent": "slhf", "rain": "tp"}
CFS = {"down_sw": "surface_downwelling_shortwave_flux", "up_sw": "surface_upwelling_shortwave_flux",
       "down_lw": "surface_downwelling_longwave_flux", "up_lw": "surface_upwelling_longwave_flux",
       "sensible": "sensible_heat_net_flux", "latent": "latent_heat_net_flux", "rain": "precipitation_rate"}
CFS_CODES = dict(down_sw={"dswrf","sdswrf"}, up_sw={"uswrf","suswrf"}, down_lw={"dlwrf","sdlwrf"},
                 up_lw={"ulwrf","sulwrf"}, sensible={"shtfl","ishf","avg_ishf"},
                 latent={"lhtfl","slhtf","avg_slhtf"}, rain={"prate"})
ENERGY_UNITS = {"J m**-2", "J m-2", "J m^-2", "J/m^2"}
KNOT_SECONDS = np.array([0, 60, 3540, 3600])
HOUR = timedelta(hours=1)
ATM = [("air_pressure", "node", "Pa"), ("uwind_stress", "nele", "Pa"), ("vwind_stress", "nele", "Pa"),
       ("short_wave", "node", "W m-2"), ("net_heat_flux", "node", "W m-2")]
PE = [("precip", "node", "m s-1"), ("evap", "node", "m s-1")]


def native_interval(row, ending):
    """Return native [start,end] and cycle, rejecting an inconsistent GRIB valid time."""
    require(row["step_type"] in ("avg", "average"), "CFS lake rate requires a native GRIB interval mean")
    seconds = {"0":60, "m":60, "min":60, "1":3600, "h":3600, "hour":3600, "2":86400, "d":86400,
               "10":10800, "11":21600, "12":43200, "13":1, "s":1}.get(str(row["step_units"]).lower())
    match = re.fullmatch(r"(\d+)-(\d+)", str(row["step_range"]))
    require(seconds and match, "Unknown GRIB step units or interval bounds")
    cycle = datetime.strptime(f"{int(row['data_date']):08d}{int(row['data_time']):04d}",
                              "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    lo, hi = (cycle + timedelta(seconds=seconds*int(n)) for n in match.groups())
    require(lo < hi and hi == utc(row["valid_time"]) == ending, "CFS valid time and native bounds disagree")
    return cycle, lo, hi


def one_hour_mean(mean, bounds, previous=None):
    cycle, lo, hi = bounds
    hours = (hi-lo)/HOUR
    require(hours >= 1 and hours.is_integer(), "CFS mean does not cover the preceding hour")
    if hours == 1:
        return mean
    require(previous is not None and previous[1] == (cycle, lo, hi-HOUR),
            "Missing matching prior cumulative CFS mean from the same cycle and lower bound")
    return hours*mean - (hours-1)*previous[0]


def shared_boundary(first, second):
    """Keep common edges within both adjacent hourly means, including sign crossings."""
    return np.where(np.sign(first) == np.sign(second),
                    np.sign(first)*np.minimum(np.abs(first), np.abs(second)), 0.)


def nonnegative_rain(mean, error):
    """Record the water adjustment when GRIB quantization produces a small negative interval."""
    require(np.all(np.isfinite(error)) and np.all(error>=0), "Invalid native precipitation packing bound")
    require(np.all(mean >= -error-1e-13), "Negative native hourly precipitation beyond GRIB packing precision")
    correction = np.maximum(-mean,0)
    return mean+correction,correction


def candidate_files(root, provider):
    """Prove data and health bytes against their connector's independently published hashes."""
    if provider == "era5":
        health = root/"health_check.json"
        archive = read(root/"run_manifest.json")
        report = read(health)
        require(report["request_hash"]==archive["request_hash"] and report["plan_hash"]==archive["plan_hash"],
                "ERA5 health and archive request or plan differ")
        result = []
        for key,row in archive["chunks"].items():
            if row["state"]=="split" and key in archive.get("splits",{}):
                continue
            require(row["state"] == "complete", "ERA5 has an incomplete chunk")
            path = root/row["output_path"].replace("\\","/")
            result.append(source_file(path, health, row["output_sha256"], root))
        require(len(result)==report["completed_chunks"]==report["expected_chunks"], "ERA5 leaf chunk count differs")
        return result
    manifest = read(root/"source_provider_lock.json")
    health = read(root/"health_check.json")
    require(manifest["provider"] == health["source_provider"] == "ncei" and not manifest["mixed_providers"],
            "CFS lake bundle requires native NCEI flux and MSL pressure")
    path = root/Path(str(health["path"]).replace("\\","/")).name
    return [source_file(path, root/"health_check.json", health["sha256"], root)]


class Source:
    def __init__(self, root, provider, files, mesh, clock):
        self.provider = provider
        self.rows = {}
        self.bounds = {}
        self.rain_errors = {}
        self.corrections = {}
        self.mesh = mesh
        manifest = read(root/("run_manifest.json" if provider=="era5" else "source_provider_lock.json"))
        axes_reference = None
        wanted = set(clock)
        for item in files:
            path = root/item["path"]
            with nc.Dataset(path) as ds:
                times = source_time(ds)
                checked_fields(ds, provider, times, load_values=False)
                if provider=="era5":
                    require(ds.request_hash==manifest["request_hash"], "Wrong ERA5 source request")
                else:
                    split = utc("2011-04-01T00:00:00Z")
                    require((provider=="cfsr" and utc(stamp(clock[0]))>=utc("1979-01-01T00:00:00Z") and utc(stamp(clock[-1]))<split)
                            or (provider=="cfsv2" and utc(stamp(clock[0]))>=split), "Outside the requested NCEI era")
                    require(ds.request_hash==manifest["request_hash"], "CFS request hash disagrees with source lock")
                require(ds["latitude"].units=="degrees_north" and ds["longitude"].units=="degrees_east",
                        "Expected geographical atmospheric coordinates")
                axes, self.cells, paxes, self.pressure_nodes = coordinates(ds, provider, mesh)
                require(all(np.all(np.diff(a)>0) or np.all(np.diff(a)<0) for a in (*axes,*paxes)),
                        "Nonmonotonic atmospheric axes")
                if axes_reference is None:
                    axes_reference = (*axes, *paxes)
                else:
                    require(all(np.array_equal(a,b) for a,b in zip(axes_reference, (*axes,*paxes))),
                            "Atmospheric grid changed between source records")
                self.axes, self.paxes = axes, paxes
                self.nodes = geographic_points(mesh.lon, mesh.lat, axes[1])
                for i, ts in enumerate(times):
                    if ts in wanted:
                        require(ts not in self.rows, "Duplicate hourly source record")
                        self.rows[ts] = path,i
                for kind, name in (ERA5 if provider=="era5" else CFS).items():
                    v = ds[name]
                    require(v.dimensions == ("time","latitude","longitude"), f"Unexpected grid for {name}")
                    if provider=="era5":
                        require(v.units in ({"m"} if kind=="rain" else ENERGY_UNITS) and
                                v.interval_anchor=="end" and int(v.interval_seconds)==3600 and
                                getattr(v,"GRIB_stepType","accum")=="accum",
                                f"Expected preceding-hour ERA5 accumulation for {name}")
                    else:
                        meta = json.loads(v.source_grib_metadata)
                        original_units = meta["units"].lower().replace(" ","").replace("**","").replace("^","").replace("{","").replace("}","")
                        accepted_units = {"kgm-2s-1","kg/m2/s","kg/(m2s)"} if kind=="rain" else {"wm-2","w/m2"}
                        require(v.units==("kg m-2 s-1" if kind=="rain" else "W m-2") and
                                original_units in accepted_units and
                                meta["type_of_level"].lower() in ("surface","sfc") and
                                meta["short_name"].lower() in CFS_CODES[kind], f"Unexpected CFS field {name}")
                        rows = meta["intervals"]
                        require(len(rows)==len(times), f"Missing native bounds for {name}")
                        for ts, row in zip(times, rows):
                            if ts in wanted:
                                if kind=="rain":
                                    self.rain_errors[ts] = float(row.get("packing_error",0.))
                                # First endpoint can be an instantaneous analysis placeholder.
                                # It is admissible only if no subsequent cumulative interval needs it.
                                if ts==clock[0] and str(row["step_range"])=="0" and row["step_type"] in ("instant","avg"):
                                    require(utc(row["valid_time"])==utc(stamp(ts)), "CFS analysis epoch differs")
                                    self.bounds[ts,name] = None
                                else:
                                    self.bounds[ts,name] = native_interval(row, utc(stamp(ts)))
        require(set(self.rows)==wanted, "Incomplete inclusive hourly source coverage")

    def wind_pressure(self, ts, basis):
        path,i = self.rows[ts]
        with nc.Dataset(path) as ds:
            uname,vname,pname = NAMES[self.provider]
            u,v = (interpolate(ds,n,i,self.cells,self.axes) for n in (uname,vname))
            p = interpolate(ds,pname,i,self.pressure_nodes,self.paxes)
        require(np.all((p>75000)&(p<115000)), "Unphysical absolute MSLP")
        c,s = basis
        tx,ty = psimple_drag(c*u-s*v, s*u+c*v)
        return p,tx,ty

    def hourly_flux(self, prev, ts):
        path,i = self.rows[ts]
        with nc.Dataset(path) as ds:
            if self.provider=="era5":
                a = {k:interpolate(ds,n,i,self.nodes,self.axes)/3600 for k,n in ERA5.items()}
                sw = a["short"]
                heat = sw+a["long"]+a["sensible"]+a["latent"]
                vapour = -a["latent"]/(LATENT_HEAT*WATER_DENSITY)
                rain = a["rain"]
            else:
                fields = {}
                for key,name in CFS.items():
                    mean = interpolate(ds,name,i,self.nodes,self.axes)
                    bounds = self.bounds[ts,name]
                    prior = None
                    if bounds[2]-bounds[1] > HOUR:
                        priorpath,pidx = self.rows[prev]
                        with nc.Dataset(priorpath) as prefix:
                            prior = interpolate(prefix,name,pidx,self.nodes,self.axes), self.bounds[prev,name]
                    fields[key] = one_hour_mean(mean,bounds,prior)
                    if key=="rain":
                        hours = (bounds[2]-bounds[1])/HOUR
                        error = hours*self.rain_errors[ts] + (hours-1)*self.rain_errors[prev]
                        # The connector serializes native means as float32 before interpolation.
                        previous_magnitude = 0 if prior is None else np.abs(prior[0])
                        error += np.finfo("f4").eps*(hours*np.abs(mean)+(hours-1)*previous_magnitude)
                sw,heat,vapour = unit_fluxes(fields)
                rain,correction = nonnegative_rain(fields["rain"]/WATER_DENSITY,error/WATER_DENSITY)
                self.corrections[ts] = correction
        require(np.min(rain)>=-1e-13, "Negative native hourly precipitation")
        return (sw,heat,np.maximum(rain,0),np.maximum(vapour,0),np.maximum(-vapour,0))


def verify_interval(var, offset, target):
    values = np.ma.asarray(var[offset:offset+4])
    require(not np.ma.getmaskarray(values).any() and np.isfinite(values).all(), f"Invalid stored {var.name}")
    actual = np.trapezoid(np.asarray(values,dtype=float), x=KNOT_SECONDS, axis=0)/3600
    # Floating-point storage error scales with the absolute magnitude of the knots.
    tol = 3e-7*np.maximum(np.max(np.abs(values),axis=0),1e-14)
    require(np.all(np.abs(actual-target)<=tol), f"Stored hourly integral differs for {var.name}")


def build(args, root, provider):
    clock = hour_axis(args.start,args.end)
    require(len(clock)>=2, "At least one forcing interval required")
    files = candidate_files(root,provider)
    mesh,areas = native_geometry(args.mesh)
    source_law(args.mod_bulk)
    src = Source(root,provider,files,mesh,clock)
    basis = model_basis(CRS(args.projected_crs),mesh.lonc,mesh.latc) if args.projected_crs else (1.,0.)
    frame = "model_xy" if args.projected_crs else "earth_east_north"
    knots = np.r_[np.column_stack((clock[:-1],clock[:-1]+60000,clock[1:]-60000)).ravel(),clock[-1]]
    args.output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output) as tmp:
        temp = Path(tmp)
        atm,pe,budget = (temp/n for n in ("superior_atmosphere.nc","superior_pe.nc","surface_budget.csv"))
        with create_file(atm,mesh,knots,ATM,frame,provider) as a, create_file(pe,mesh,knots,PE,frame,provider) as p, budget.open("w",newline="") as out:
            a.pressure_reference = "absolute_mean_sea_level"
            writer = csv.writer(out)
            header = ["start_utc","end_utc","rain_m3","evaporated_m3","condensed_m3","net_heat_j"]
            if provider!="era5":
                header.extend(["unadjusted_grib_rain_m3","rain_quantization_adjustment_m3","rain_adjusted_nodes"])
            writer.writerow(header)
            adjusted_volume = 0.
            current = left = src.hourly_flux(clock[0],clock[1])
            wp = src.wind_pressure(clock[0],basis)
            for k in range(len(clock)-1):
                nxt = src.hourly_flux(clock[k+1],clock[k+2]) if k+2<len(clock) else current
                right = tuple(shared_boundary(u,v) for u,v in zip(current,nxt))
                shaped = [precipitation_shape(m,l,r) for m,l,r in zip(current,left,right)]
                wpnext = src.wind_pressure(clock[k+1],basis)
                for j,sec in enumerate(KNOT_SECONDS):
                    idx = 3*k+j
                    for (name,_,_),v0,v1 in zip(ATM[:3],wp,wpnext):
                        a[name][idx] = v0 + (v1-v0)*(sec/3600.)
                    sw,heat,rain,loss,dew = [v[j] for v in shaped]
                    a["short_wave"][idx],a["net_heat_flux"][idx] = sw,heat
                    p["precip"][idx],p["evap"][idx] = rain+dew,-loss
                sw,heat,rain,loss,dew = current
                for var,expected in ((a["short_wave"],sw),(a["net_heat_flux"],heat),
                                     (p["precip"],rain+dew),(p["evap"],-loss)):
                    verify_interval(var,3*k,expected)
                row = [stamp(clock[k]),stamp(clock[k+1]),*(np.sum(areas*v)*3600 for v in (rain,loss,dew,heat))]
                if provider!="era5":
                    correction = src.corrections.pop(clock[k+1])
                    adjustment = float(np.sum(areas*correction)*3600)
                    adjusted_volume += adjustment
                    row.extend([row[2]-adjustment,adjustment,int(np.count_nonzero(correction))])
                writer.writerow(row)
                current,left,wp = nxt,right,wpnext
        for file in (atm,pe):
            with nc.Dataset(file) as d:
                actual = d["Itime"][:].astype("i8")*86400000 + d["Itime2"][:]
                require(np.array_equal(actual,knots+40587*86400000), "Native FVCOM UTC readback differs")
                for name in ("air_pressure","uwind_stress","vwind_stress") if file==atm else ():
                    for k in range(0,len(knots),4):
                        values = np.ma.asarray(d[name][k:k+4])
                        require(not np.ma.getmaskarray(values).any() and np.isfinite(values).all(),
                                f"Nonfinite forcing in {name}")
        products = [atm,pe,budget]
        manifest = dict(schema="fvcom_lake_surface_v1",provider=provider,start_utc=stamp(clock[0]),end_utc=stamp(clock[-1]),
                        wind_frame=frame,area_m2=float(areas.sum()),mesh_sha256=digest(args.mesh),mod_bulk_sha256=digest(args.mod_bulk),
                        latent_heat_j_kg=LATENT_HEAT,water_density_kg_m3=WATER_DENSITY,source_root=str(root.resolve()),
                        source_files=files,interval_encoding="end-anchored native hourly integrals; 60-second conservative transitions",
                        verified_integral_intervals=len(clock)-1,files={f.name:digest(f) for f in products})
        if provider!="era5":
            manifest["rain_quantization_adjustment_m3"] = adjusted_volume
        receipt = temp/"surface_manifest.json"
        receipt.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf8")
        for f in products+[receipt]:
            dst = args.output/f.name
            require(not dst.exists() or digest(dst)==digest(f), f"Different existing lake forcing {dst}")
        for f in products+[receipt]:
            if not (args.output/f.name).exists():
                f.rename(args.output/f.name)
    return manifest
