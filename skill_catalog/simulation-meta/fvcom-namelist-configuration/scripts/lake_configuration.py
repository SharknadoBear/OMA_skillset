"""Render the closed, thermally active freshwater lake from the native executable's blank."""
import argparse
from datetime import timedelta
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
    for name in (args.startup, args.surface, args.pe, args.river_info, args.station):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
            raise ValueError("active native input names must be flat basenames")
    group = {}
    def put(name, **values):
        group["NML_"+name] = {key: ("T" if v else "F") if isinstance(v,bool) else
                              n.q(v) if isinstance(v,str) else str(v) for key,v in values.items()}

    first = n.nml_date(start)
    put("CASE", CASE_TITLE="Lake Superior freshwater baroclinic", TIMEZONE="UTC", DATE_FORMAT="YMD",
        DATE_REFERENCE="2018-01-01 00:00:00", START_DATE=first, END_DATE=n.nml_date(end))
    put("STARTUP", STARTUP_TYPE="hotstart", STARTUP_FILE=args.startup, STARTUP_UV_TYPE="set values",
        STARTUP_TS_TYPE="set values", STARTUP_TURB_TYPE="set values")
    put("IO", INPUT_DIR=args.input_dir, OUTPUT_DIR=args.output_dir, IREPORT=100, VISIT_ALL_VARS=False,
        WAIT_FOR_VISIT=False, USE_MPI_IO_MODE=False)
    put("INTEGRATION", EXTSTEP_SECONDS=args.extstep, ISPLIT=args.isplit, IRAMP=0, MIN_DEPTH=0.1, STATIC_SSH_ADJ=0.0)
    put("RESTART", RST_ON=True, RST_FIRST_OUT=n.nml_date(min(start+timedelta(days=1),end)),
        RST_OUT_INTERVAL="days=1.0", RST_OUTPUT_STACK=1)
    put("NETCDF", NC_ON=True, NC_FIRST_OUT=first, NC_OUT_INTERVAL="seconds=21600.0", NC_OUTPUT_STACK=4,
        NC_SUBDOMAIN_FILES="FVCOM", NC_GRID_METRICS=True, NC_FILE_DATE=False, NC_VELOCITY=True,
        NC_SALT_TEMP=True, NC_TURBULENCE=True, NC_AVERAGE_VEL=True, NC_VERTICAL_VEL=True, NC_WIND_VEL=False,
        NC_WIND_STRESS=True, NC_ATM_PRESS=True, NC_EVAP_PRECIP=True, NC_SURFACE_HEAT=True,
        NC_GROUNDWATER=False, NC_BIO=False, NC_WQM=False, NC_VORTICITY=False)
    put("NETCDF_SURFACE", NCSF_ON=False)
    put("NETCDF_AV", NCAV_ON=False)
    put("SURFACE_FORCING", WIND_ON=True, WIND_TYPE="stress", WIND_KIND="variable", WIND_FILE=args.surface,
        HEATING_ON=True, HEATING_TYPE="flux", HEATING_KIND="variable", HEATING_FILE=args.surface,
        HEATING_LONGWAVE_LENGTHSCALE=3.6, HEATING_LONGWAVE_PERCTAGE=0.55, HEATING_SHORTWAVE_LENGTHSCALE=0.35,
        AIRPRESSURE_ON=True, AIRPRESSURE_KIND="variable", AIRPRESSURE_FILE=args.surface,
        PRECIPITATION_ON=True, PRECIPITATION_KIND="variable", PRECIPITATION_FILE=args.pe, WAVE_ON=False)
    put("PHYSICS", HORIZONTAL_MIXING_TYPE="closure", HORIZONTAL_MIXING_KIND="constant",
        HORIZONTAL_MIXING_COEFFICIENT=0.1, HORIZONTAL_PRANDTL_NUMBER=1.0, VERTICAL_MIXING_TYPE="closure",
        VERTICAL_MIXING_COEFFICIENT=1e-6, VERTICAL_PRANDTL_NUMBER=1.0, BOTTOM_ROUGHNESS_TYPE="orig",
        BOTTOM_ROUGHNESS_KIND="constant", BOTTOM_ROUGHNESS_LENGTHSCALE=0.002, BOTTOM_ROUGHNESS_MINIMUM=0.0025,
        CONVECTIVE_OVERTURNING=False, SCALAR_POSITIVITY_CONTROL=True, BAROTROPIC=False,
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
                    "wet_cells_prev_int","wet_cells_prev_ext"):
            if key not in dataset.variables:
                raise ValueError("missing native hotstart field "+key)
        if not {"omega","w"} & dataset.variables.keys():
            raise ValueError("missing native vertical velocity omega or w")
        start=n.parse_utc(a.start,"start")
        epoch=n.parse_utc("1858-11-17T00:00:00Z","MJD epoch")
        delta=start-epoch
        mjd=delta.days
        millis=delta.seconds*1000+delta.microseconds//1000
        if not ((asarray(dataset["Itime"][:])==mjd)&(asarray(dataset["Itime2"][:])==millis)).any():
            raise ValueError("no startup record at the requested UTC start")
    with netCDF4.Dataset(root/a.surface) as met:
        for name,place in (("uwind_stress","nele"),("vwind_stress","nele"),("air_pressure","node"),
                           ("short_wave","node"),("net_heat_flux","node")):
            if name not in met.variables or met[name].dimensions!=("time",place):
                raise ValueError("missing or mislocated surface variable "+name)
    with netCDF4.Dataset(root/a.pe) as pe:
        for name in ("precip","evap"):
            if name not in pe.variables or pe[name].dimensions!=("time","node"):
                raise ValueError("missing or mislocated freshwater variable "+name)


if __name__=="__main__":
    raise SystemExit(cli())
