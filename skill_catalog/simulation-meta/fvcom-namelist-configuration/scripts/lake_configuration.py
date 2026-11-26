"""Render the closed, thermally active freshwater lake from the native executable's blank."""
import argparse
from datetime import timedelta
import hashlib
import math
from pathlib import Path
import re

import fvcom_namelist as n


def lake_values(args):
    start = n.parse_utc(args.start, "start")
    end = n.parse_utc(args.end, "end")
    dt = args.extstep * args.isplit
    if not 0 < dt <= 3600 or not math.isfinite(dt) or args.isplit < 1 or end <= start:
        raise ValueError("invalid integration window or time step")
    for seconds in ((end-start).total_seconds(), 3600, 21600, 86400):
        if abs(seconds/dt-round(seconds/dt)) > 1e-7:
            raise ValueError("internal time step must divide station, grid, restart and integration windows")
    bulk = getattr(args, "coare_bulk", None)
    filenames = [args.startup, args.surface, args.pe, args.river_info, args.station]
    if bulk is not None:
        filenames.append(bulk)
    for name in filenames:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
            raise ValueError("active native input names must be flat basenames")
    group = {}
    def put(name, **values):
        group["NML_"+name] = {key: ("T" if v else "F") if isinstance(v,bool) else
                              n.q(v) if isinstance(v,str) else str(v) for key,v in values.items()}

    first = n.nml_date(start)
    title = "Lake Superior COARE26Z diagnostic" if bulk else "Lake Superior freshwater baroclinic"
    put("CASE", CASE_TITLE=title, TIMEZONE="UTC", DATE_FORMAT="YMD",
        DATE_REFERENCE="2018-01-01 00:00:00", START_DATE=first, END_DATE=n.nml_date(end))
    put("STARTUP", STARTUP_TYPE="hotstart", STARTUP_FILE=args.startup, STARTUP_UV_TYPE="set values",
        STARTUP_TS_TYPE="set values", STARTUP_TURB_TYPE="set values")
    put("IO", INPUT_DIR=args.input_dir, OUTPUT_DIR=args.output_dir, IREPORT=100, VISIT_ALL_VARS=False,
        WAIT_FOR_VISIT=False, USE_MPI_IO_MODE=False)
    put("INTEGRATION", EXTSTEP_SECONDS=args.extstep, ISPLIT=args.isplit, IRAMP=0, MIN_DEPTH=0.1, STATIC_SSH_ADJ=0.0)
    put("RESTART", RST_ON=True, RST_FIRST_OUT=n.nml_date(min(start+timedelta(days=1),end)),
        RST_OUT_INTERVAL="days=1.0", RST_OUTPUT_STACK=1)
    grid_interval = "seconds=600.0" if bulk and end-start == timedelta(seconds=600) else "seconds=21600.0"
    put("NETCDF", NC_ON=True, NC_FIRST_OUT=first, NC_OUT_INTERVAL=grid_interval, NC_OUTPUT_STACK=4,
        NC_SUBDOMAIN_FILES="FVCOM", NC_GRID_METRICS=True, NC_FILE_DATE=False, NC_VELOCITY=True,
        NC_SALT_TEMP=True, NC_TURBULENCE=True, NC_AVERAGE_VEL=True, NC_VERTICAL_VEL=True, NC_WIND_VEL=False,
        NC_WIND_STRESS=True, NC_ATM_PRESS=True, NC_EVAP_PRECIP=True, NC_SURFACE_HEAT=True,
        NC_GROUNDWATER=False, NC_BIO=False, NC_WQM=False, NC_VORTICITY=False)
    put("NETCDF_SURFACE", NCSF_ON=False)
    put("NETCDF_AV", NCAV_ON=False)
    put("SURFACE_FORCING", WIND_ON=True, WIND_TYPE="stress", WIND_KIND="variable", WIND_FILE=args.surface,
        HEATING_ON=not bulk, HEATING_TYPE="flux", HEATING_KIND="variable", HEATING_FILE=args.surface,
        HEATING_LONGWAVE_LENGTHSCALE=3.6, HEATING_LONGWAVE_PERCTAGE=0.55, HEATING_SHORTWAVE_LENGTHSCALE=0.35,
        AIRPRESSURE_ON=True, AIRPRESSURE_KIND="variable", AIRPRESSURE_FILE=args.surface,
        PRECIPITATION_ON=True, PRECIPITATION_KIND="variable", PRECIPITATION_FILE=args.pe, WAVE_ON=False)
    if bulk:
        put("HEATING_CALCULATED", HEATING_CALCULATE_ON=True, HEATING_FRESHWATER=True, COARE_VERSION="COARE26Z",
            HEATING_CALCULATE_TYPE="flux", HEATING_CALCULATE_KIND="variable", HEATING_CALCULATE_FILE=bulk,
            ZUU=10.0, ZTT=2.0, ZQQ=2.0, HEATING_LONGWAVE_LENGTHSCALE_IN_HEATFLUX=3.6,
            HEATING_LONGWAVE_PERCTAGE_IN_HEATFLUX=0.55, HEATING_SHORTWAVE_LENGTHSCALE_IN_HEATFLUX=0.35)
    limiter = getattr(args, "scalar_positivity_control", None)
    if limiter is None:
        limiter = not bulk
    put("PHYSICS", HORIZONTAL_MIXING_TYPE="closure", HORIZONTAL_MIXING_KIND="constant",
        HORIZONTAL_MIXING_COEFFICIENT=0.1, HORIZONTAL_PRANDTL_NUMBER=1.0, VERTICAL_MIXING_TYPE="closure",
        VERTICAL_MIXING_COEFFICIENT=1e-6, VERTICAL_PRANDTL_NUMBER=1.0, BOTTOM_ROUGHNESS_TYPE="orig",
        BOTTOM_ROUGHNESS_KIND="constant", BOTTOM_ROUGHNESS_LENGTHSCALE=0.002, BOTTOM_ROUGHNESS_MINIMUM=0.0025,
        CONVECTIVE_OVERTURNING=False, SCALAR_POSITIVITY_CONTROL=limiter,
        BAROTROPIC=False,
        BAROCLINIC_PRESSURE_GRADIENT="sigma levels", SEA_WATER_DENSITY_FUNCTION="dens1", TEMPERATURE_ACTIVE=True,
        SALINITY_ACTIVE=False, WETTING_DRYING_ON=True, SURFACE_WAVE_MIXING=False, FLUME_BOTTOM=False,
        FLUME_DRAG_COEF=0.0, ADCOR_ON=True, RECALCULATE_RHO_MEAN=False, INTERVAL_RHO_MEAN="days=1.0",
        BACKWARD_ADVECTION=False, BACKWARD_STEP=-1, EQUATOR_BETA_PLANE=False, NOFLUX_BOT_CONDITION=True)
    if args.river_number < 1:
        raise ValueError("positive count of signed edge sources required")
    put("RIVER_TYPE", RIVER_NUMBER=args.river_number, RIVER_TS_SETTING="calculated", RIVER_INFLOW_LOCATION="edge",
        RIVER_INFO_FILE=args.river_info, RIVER_KIND="variable")
    put("OPEN_BOUNDARY_CONTROL", OBC_ON=False, OBC_NODE_LIST_FILE="lsofs_obc.dat", OBC_ELEVATION_FORCING_ON=False,
        OBC_ELEVATION_FILE="none", OBC_TS_TYPE=1, OBC_TEMP_NUDGING=False, OBC_TEMP_FILE="none",
        OBC_SALT_NUDGING=False, OBC_SALT_FILE="none", OBC_MEANFLOW=False, OBC_MEANFLOW_FILE="none",
        OBC_LONGSHORE_FLOW_ON=False, OBC_LONGSHORE_FLOW_FILE="none", OBC_DEPTH_CONTROL_ON=False)
    put("GRID_COORDINATES", GRID_FILE="lsofs_grd.dat", GRID_FILE_UNITS="degrees", PROJECTION_REFERENCE="none",
        SIGMA_LEVELS_FILE="lsofs_sig.dat", DEPTH_FILE="lsofs_dep.dat", CORIOLIS_FILE="lsofs_cor.dat",
        SPONGE_FILE="lsofs_spg.dat")
    put("GROUNDWATER", GROUNDWATER_ON=False, GROUNDWATER_TEMP_ON=False, GROUNDWATER_SALT_ON=False)
    put("LAG", LAG_PARTICLES_ON=False)
    put("ADDITIONAL_MODELS", DATA_ASSIMILATION=False, BIOLOGICAL_MODEL=False, SEDIMENT_MODEL=False,
        ICING_MODEL=False, ICE_MODEL=False, HIGH_LATITUDE_WAVE=False, PLASTIC_MODEL=False)
    put("PROBES", PROBES_ON=False, PROBES_NUMBER=0)
    put("BOUNDSCHK", BOUNDSCHK_ON=False)
    put("NCNEST", NCNEST_ON=False)
    put("NESTING", NESTING_ON=False)
    put("STATION_TIMESERIES", OUT_STATION_TIMESERIES_ON=True, STATION_FILE=args.station,
        LOCATION_TYPE=args.station_location, OUT_ELEVATION=True, OUT_VELOCITY_3D=True, OUT_VELOCITY_2D=True,
        OUT_SALT_TEMP=True, OUT_WIND_VELOCITY=False, READ_GRID_EDGE_FROM_FILE=False,
        GRID_EDGE_FILE_NAME="superior_grid_edge.dat", OUT_INTERVAL="seconds=3600.0")
    return group


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for key in ("start","end","input-dir","output-dir","startup","surface","pe","river-info","station"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--station-location", choices=("node","cell"), default="node")
    p.add_argument("--river-number", type=int, required=True)
    p.add_argument("--extstep", type=float, required=True)
    p.add_argument("--isplit", type=int, required=True)
    p.add_argument("--coare-bulk", help="Native FVCOM-grid freshwater COARE26Z file. The lake build must support independent stress and signed vapor. Use raw rain for --pe.")
    p.add_argument("--scalar-positivity-control", action=argparse.BooleanOptionalAction, default=None,
                   help="Set the native scalar limiter for reproducible heat-budget sensitivity tests")
    p.add_argument("--input-root",type=Path,help="Verify the staged native input names, grid, signed rivers and hotstart")
    a=p.parse_args(argv)
    values=lake_values(a)
    rendered=n.render_template(a.template.read_text(encoding="utf-8", errors="replace"),values)
    if n.parse_values(rendered)!=values:
        raise ValueError("lake namelist did not round trip through the FVCOM parser")
    if a.input_root:
        check_native_inputs(a)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    if a.output.exists() and a.output.read_text()!=rendered:
        raise FileExistsError(a.output)
    a.output.write_text(rendered,encoding="utf-8",newline="\n")
    return 0


def check_startup_clock(dataset, start):
    from numpy import asarray

    day = dataset["Itime"]
    subday = dataset["Itime2"]
    if not day.units.startswith("days since ") or not subday.units.lower().startswith(("msec", "millisecond")):
        raise ValueError("unexpected units for native integer clock")
    origin = n.parse_utc(day.units.removeprefix("days since "), "native integer date reference")
    delta = n.parse_utc(start, "start") - origin
    millis = delta.seconds * 1000 + delta.microseconds // 1000
    if delta.microseconds % 1000 or not ((asarray(day[:]) == delta.days) & (asarray(subday[:]) == millis)).any():
        raise ValueError("no startup record at the requested UTC start")


def check_native_inputs(a):
    import netCDF4
    from numpy import asarray

    root=a.input_root
    for file in (a.startup,a.surface,a.pe,a.station,a.river_info,"lsofs_grd.dat","lsofs_dep.dat","lsofs_sig.dat",
                 "lsofs_spg.dat","lsofs_obc.dat","lsofs_cor.dat"):
        if not (root/file).is_file() or not (root/file).stat().st_size:
            raise ValueError("missing staged lake input "+file)
    river=(root/a.river_info).read_text()
    files=set(re.findall(r"(?mi)^\s*RIVER_FILE\s*=\s*['\"]([^'\"]+)",river))
    if (len(re.findall(r"(?mi)^\s*&NML_RIVER\b",river))!=a.river_number or len(files)!=1):
        raise ValueError("lake edge-river companion count or shared file disagrees")
    filename=files.pop()
    if Path(filename).name!=filename or not re.fullmatch(r"[A-Za-z0-9_.-]+",filename):
        raise ValueError("invalid shared river filename")
    with netCDF4.Dataset(root/filename) as dataset:
        flux=asarray(dataset["river_flux"][:])
        if flux.ndim!=2 or flux.shape[1]!=a.river_number or not (flux<0).any() or not (flux>0).any():
            raise ValueError("river input must contain positive inflows and negative withdrawals")
    with netCDF4.Dataset(root/a.startup) as dataset:
        if len(dataset.dimensions["siglay"])!=20 or len(dataset.dimensions["siglev"])!=21:
            raise ValueError("lake startup must retain the NOAA 20 layers and 21 interfaces")
        for key in ("zeta","et","u","v","ua","va","temp","salinity","rho1","tmean1","smean1","rmean1",
                    "q2","q2l","l","km","kq","kh","el_press","iint","wet_nodes","wet_cells","wet_nodes_prev_int",
                    "wet_cells_prev_int","wet_cells_prev_ext","cor","cc_sponge"):
            if key not in dataset.variables:
                raise ValueError("missing native hotstart field "+key)
        if not {"omega","w"} & dataset.variables.keys():
            raise ValueError("missing native vertical velocity omega or w")
        check_startup_clock(dataset, a.start)
    bulk = getattr(a, "coare_bulk", None)
    with netCDF4.Dataset(root/a.surface) as met:
        met_fields = [("uwind_stress","nele"),("vwind_stress","nele"),("air_pressure","node")]
        if not bulk:
            met_fields.extend([("short_wave","node"),("net_heat_flux","node")])
        for name,place in met_fields:
            if name not in met.variables or met[name].dimensions!=("time",place):
                raise ValueError("missing or mislocated surface variable "+name)
    with netCDF4.Dataset(root/a.pe) as pe:
        for name in ("precip","evap"):
            if name not in pe.variables or pe[name].dimensions!=("time","node"):
                raise ValueError("missing or mislocated freshwater variable "+name)
    if bulk:
        check_coare_inputs(a)


def check_coare_inputs(a):
    import netCDF4
    import numpy as np

    root = a.input_root
    def digest(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").digest()

    if digest(root/"superior_cor.dat") != digest(root/"lsofs_cor.dat"):
        raise ValueError("The COARE latitude file superior_cor.dat differs from the native grid Coriolis file")
    with (netCDF4.Dataset(root/a.coare_bulk) as bulk, netCDF4.Dataset(root/a.pe) as rain,
          netCDF4.Dataset(root/a.surface) as hydro):
        for ds, ids in ((bulk, ("node_id","element_id")), (rain, ("node_id",)), (hydro, ())):
            for key in ids:
                if not np.array_equal(ds[key][:], hydro[key][:]):
                    raise ValueError("COARE forcing and hydrodynamic grid disagree on "+key)
            day, millis = ds["Itime"], ds["Itime2"]
            if not day.units.startswith("days since ") or not millis.units.lower().startswith(("msec","millisecond")):
                raise ValueError("Unexpected native COARE time units")
            origin = n.parse_utc(day.units.removeprefix("days since "), "COARE time origin")
            clock = np.asarray(day[:], dtype=np.int64)*86400000 + np.asarray(millis[:], dtype=np.int64)
            left = (n.parse_utc(a.start,"start")-origin) / timedelta(milliseconds=1)
            right = (n.parse_utc(a.end,"end")-origin) / timedelta(milliseconds=1)
            if len(clock)<2 or np.any(np.diff(clock)<=0) or clock[0]>left or clock[-1]<right:
                raise ValueError("COARE forcing clocks must increase and cover the integration window")
        wind_units = {"ms-1","m/s"}
        radiation_units = {"wm-2","w/m2"}
        for name, place, units in (
            ("U10","nele",wind_units), ("V10","nele",wind_units),
            ("air_temperature","node",{"c","degc","degreecelsius","degreescelsius","celsius"}),
            ("relative_humidity","node",{"percent","%"}), ("air_pressure","node",{"pa","pascal","pascals"}),
            ("short_wave","node",radiation_units), ("long_wave","node",radiation_units),
        ):
            var=bulk.variables.get(name)
            unit=re.sub(r"[\s_*^]","",getattr(var,"units","").lower())
            if var is None or var.dimensions != ("time",place) or unit not in units:
                raise ValueError("Missing, mislocated or incorrectly labeled COARE field "+name)
        for i in range(0,len(rain.dimensions["time"]),8):
            p,e=(np.ma.asarray(rain[k][i:i+8]) for k in ("precip","evap"))
            if (np.ma.getmaskarray(p).any() or np.ma.getmaskarray(e).any() or not np.isfinite(p.data).all()
                    or np.any(p.data<0) or np.any(e.data!=0)):
                raise ValueError("Coupled heat requires nonnegative raw rainfall and a zero prescribed vapor field")


if __name__=="__main__":
    raise SystemExit(cli())
