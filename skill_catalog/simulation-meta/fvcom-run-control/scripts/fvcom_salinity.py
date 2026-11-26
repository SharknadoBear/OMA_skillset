"""Resolve and freeze a bounded Cartesian Columbia active-salinity reader or pilot."""

from copy import deepcopy
from pathlib import Path, PurePosixPath
from argparse import Namespace
import hashlib, json, re
from tempfile import TemporaryDirectory

MODE = "baroclinic-salinity-tide-wind-river"
PROFILE = "columbia_baroclinic_salinity_pilot"
START = "2025-03-18T00:00:00Z"
STAGES = {"reader": 720, "bottom-sensitivity": 720, "pilot": 86400, "bottom-sensitivity-day": 86400}
DAY_READERS = {"pilot": "reader", "bottom-sensitivity-day": "bottom-sensitivity"}
SOURCE = "f8e4b50d4fd51409febcefc763cbc5dca11870e2"


def resolve(request):
    r = deepcopy(request)
    if (
        r.get("mode") != MODE
        or r.get("profile") != PROFILE
        or r.get("execution", {}).get("site") != "perlmutter"
    ):
        raise ValueError(
            "Columbia settings require explicit profile=columbia_baroclinic_salinity_pilot, its salinity mode and execution.site=perlmutter"
        )
    required = {
        "profile": PROFILE,
        "grid_policy": "retained_only",
        "physics": {
            "formulation": "three_dimensional_baroclinic",
            "equation_of_state": "dens2",
            "temperature_active": False,
            "salinity_active": True,
            "sigma_levels": 10,
        },
        "period": {
            "analysis_start": START,
            "analysis_end": "2025-03-19T00:00:00Z",
            "spinup_days": 0,
        },
        "execution": {"site": "perlmutter", "stages": ["reader", "smoke"]},
        "perlmutter": {
            "constraint": "cpu",
            "qos": "debug",
            "nodes": 1,
            "ranks": 52,
            "threads_per_rank": 1,
            "source_commit": SOURCE,
        },
    }

    def fill(target, settings):
        for k, v in settings.items():
            if isinstance(v, dict):
                if not isinstance(target.setdefault(k, {}), dict):
                    raise ValueError(k + " must be an object")
                fill(target[k], v)
            elif k in target and target[k] != v:
                raise ValueError("Incompatible salinity pilot setting " + k)
            else:
                target[k] = deepcopy(v)

    fill(r, required)
    cfg = r["perlmutter"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", cfg.get("account", "")):
        raise ValueError("supply perlmutter.account")
    p = cfg.get("project_dir", "")
    if (
        not re.fullmatch(r"/[A-Za-z0-9._/+\-]+", p)
        or ".." in PurePosixPath(p).parts
        or p == "/"
    ):
        raise ValueError("supply an absolute perlmutter.project_dir")
    return r


def digest(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def verify_native_inputs(inputs, files, ocean_manifest=None):
    """Check the actual reader arrays and the producing geometry/forcing hashes."""
    import netCDF4 as nc
    import numpy as np

    river = load(inputs / "river_manifest.json")
    startup = load(inputs / "initialization_manifest.json")
    for key, name in (("mesh", "grd"), ("depth", "dep"), ("obc", "obc")):
        if river[key + "_sha256"] != files["columbia_" + name + ".dat"]:
            raise ValueError("River geometry differs: " + key)
    for key, value in startup["required_runtime_grid_sha256"].items():
        if files["columbia_" + key + ".dat"] != value:
            raise ValueError("Startup geometry differs: " + key)
    atmosphere = load(inputs / "atmosphere_manifest.json")
    boundary = load(inputs / "wind_boundary_manifest.json")
    if (
        boundary["atmosphere_manifest_sha256"] != files["atmosphere_manifest.json"]
        or atmosphere["provider"] != "hrrr"
        or atmosphere["pressure_reference"] != "absolute_mean_sea_level"
        or atmosphere["wind_vector_frame"] != "model_xy"
    ):
        raise ValueError("Wind and elevation provenance disagree")
    for item in [*atmosphere["files"].values(), *boundary["files"].values()]:
        if files.get(item["path"]) != item["sha256"]:
            raise ValueError("Forcing companion differs: " + item["path"])
    if ocean_manifest:
        ocean = load(ocean_manifest)
        if (
            ocean["status"] != "ready"
            or ocean["files"]["netcdf"]["sha256"] != files["columbia_tsobc.nc"]
            or ocean["startup_sha256"] != files["columbia_start.nc"]
            or any(
                ocean[key + "_sha256"] != files["columbia_" + name + ".dat"]
                for key, name in (
                    ("mesh", "grd"),
                    ("depth", "dep"),
                    ("obc", "obc"),
                    ("sigma", "sig"),
                )
            )
        ):
            raise ValueError("Wrong ocean source, geometry, or startup")
    rows = np.loadtxt(inputs / "columbia_obc.dat", skiprows=1, dtype=int)
    if rows.shape != (124, 3) or not np.all(rows[:, 2] == 1):
        raise ValueError("Expected 124 type-1 ocean nodes")
    with (
        nc.Dataset(inputs / "columbia_start.nc") as start,
        nc.Dataset(inputs / "columbia_tsobc.nc") as ts,
        nc.Dataset(inputs / "river.nc") as rivers,
    ):
        ids = rows[:, 1]
        salt = np.ma.asarray(ts["obc_salinity"][:])
        temp = np.ma.asarray(ts["obc_temp"][:])
        times = np.array(ts["Itime"][:], dtype=np.int64) * 86400000 + np.array(
            ts["Itime2"][:], dtype=np.int64
        )
        if (
            times[0] > 60752 * 86400000
            or times[-1] < 60753 * 86400000
            or not np.all(np.diff(times) == 360000)
            or not np.array_equal(ts["obc_nodes"][:], ids)
            or salt.shape != (len(times), 9, 124)
            or temp.shape != salt.shape
            or np.any(np.ma.getmaskarray(salt))
            or np.any(np.ma.getmaskarray(temp))
            or not np.isfinite(salt).all()
            or salt.min() < 0
            or salt.max() > 42
            or not np.all(temp == np.asarray(start["temp"][0])[:, ids - 1][None, :, :])
        ):
            raise ValueError(
                "OBC clock, layer, temperature, or active salinity mismatch"
            )
        r = np.ma.asarray(rivers["river_salt"][:])
        if (
            r.ndim != 2
            or r.shape[1] != 19
            or np.any(np.ma.getmaskarray(r))
            or not np.all(r == 0)
        ):
            raise ValueError("Expected native zero-salinity river input")


def prepare(args):
    from fvcom_run_control import attempt_sbatch

    if args.stage == "smoke":
        args.stage = "pilot"
    project = Path(args.project)
    req = resolve(load(project / "request.json"))
    cfg = req["perlmutter"]
    freeze = load(args.build)
    if (
        freeze.get("schema") != "fvcom_estuary_salinity_build_v1"
        or freeze.get("status") != "frozen"
        or freeze.get("source_commit") != SOURCE
        or not re.fullmatch(r"[0-9a-f]{64}", freeze.get("instrument_sha256", ""))
    ):
        raise ValueError("Expected a frozen native active-salinity budget build")
    exe = freeze["native_executable"]
    if (
        not re.fullmatch(r"/[a-zA-Z0-9_./+\-]+", exe)
        or ".." in PurePosixPath(exe).parts
        or not PurePosixPath(exe).is_relative_to(cfg["project_dir"])
    ):
        raise ValueError("Expected a native executable under this immutable study")
    values = {}
    for line in args.namelist.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("!"):
            key, val = line.strip().split("=", 1)
            values[key.strip().upper()] = val.strip().rstrip(",").strip("'").lower()
    daily = args.stage in DAY_READERS
    end = "2025-03-19 00:00:00" if daily else "2025-03-18 00:12:00"
    expected = {
        "BAROTROPIC": "f",
        "IRAMP": "0",
        "ISPLIT": "1",
        "EXTSTEP_SECONDS": "1.2",
        "TEMPERATURE_ACTIVE": "f",
        "SALINITY_ACTIVE": "t",
        "SEA_WATER_DENSITY_FUNCTION": "dens2",
        "STARTUP_FILE": "columbia_start.nc",
        "STARTUP_TYPE": "hotstart",
        "START_DATE": "2025-03-18 00:00:00",
        "END_DATE": end,
        "RST_ON": "t",
        "RST_FIRST_OUT": end,
        "INPUT_DIR": "./input/",
        "OUTPUT_DIR": "./",
        "OBC_SALT_NUDGING": "t",
        "OBC_SALT_NUDGING_TIMESCALE": "1.0",
        "OBC_SALT_FILE": "columbia_tsobc.nc",
        "RIVER_TS_SETTING": "calculated",
        "OUT_SALT_TEMP": "t",
        "NC_SALT_TEMP": "t",
        "OUT_INTERVAL": "seconds=360.0",
        "NC_OUT_INTERVAL": "seconds="
        + ("3600.0" if daily else "360.0"),
        "NOFLUX_BOT_CONDITION": "f" if args.stage.startswith("bottom-sensitivity") else "t",
    }
    for k, v in expected.items():
        if values.get(k) != v:
            raise ValueError("Unexpected salinity stage setting " + k)
    inputs = args.inputs
    files = {
        p.name: digest(p)
        for p in inputs.iterdir()
        if p.is_file() and re.fullmatch(r"[A-Za-z0-9_.-]+", p.name)
    }
    for name in (
        "columbia_start.nc",
        "initialization_manifest.json",
        "river.nc",
        "river.nml",
        "river_manifest.json",
        "columbia_tsobc.nc",
        "columbia_station.dat",
        "station_mapping.json",
        "columbia_grd.dat",
        "columbia_dep.dat",
        "columbia_cor.dat",
        "columbia_obc.dat",
        "columbia_sig.dat",
        "columbia_spg.dat",
        "columbia_elevtide.nc",
        "columbia_surface.nc",
        "columbia_grid_edge.dat",
        "wind_boundary_manifest.json",
        "atmosphere_manifest.json",
    ):
        if name not in files:
            raise ValueError("Missing frozen input " + name)
    startup = load(inputs / "initialization_manifest.json")
    river = load(inputs / "river_manifest.json")
    if (
        startup["products"]["columbia_start.nc"] != files["columbia_start.nc"]
        or river["files"]["netcdf"]["sha256"] != files["river.nc"]
        or river["files"]["namelist"]["sha256"] != files["river.nml"]
    ):
        raise ValueError("Native tracer products disagree with producer manifests")
    if (
        river["tracers"]["mode"] != "active_salinity_freshwater"
        or river["tracers"]["salinity_psu"] != 0.0
        or river["tracers"]["startup_sha256"] != files["columbia_start.nc"]
    ):
        raise ValueError(
            "Expected verified freshwater salinity and startup-bound inactive river temperature"
        )
    ocean = getattr(args, "ocean_manifest", None)
    verify_native_inputs(
        inputs,
        files,
        ocean
        or (
            inputs / "ocean_ts_manifest.json"
            if "ocean_ts_manifest.json" in files
            else None
        ),
    )
    if daily:
        if not args.reader:
            raise ValueError("Pilot requires successful native reader audit")
        reader = load(args.reader)
        if (
            reader.get("status") != "verified"
            or reader.get("stage") != DAY_READERS[args.stage]
            or reader.get("executable_sha256") != freeze["executable_sha256"]
            or reader.get("inputs") != files
        ):
            raise ValueError("Reader did not qualify these pilot inputs and executable")
    data = {
        "schema": "fvcom_estuary_salinity_attempt_v1",
        "stage": args.stage,
        "execution_site": "perlmutter",
        "ranks": 52,
        "account": cfg["account"],
        "nodes": 1,
        "mode": MODE,
        "executable_sha256": freeze["executable_sha256"],
        "native_executable": exe,
        "build_sha256": digest(args.build),
        "inputs": files,
        "namelist_sha256": digest(args.namelist),
        "request_sha256": digest(project / "request.json"),
        "reader_sha256": digest(args.reader) if args.reader else None,
    }
    if ocean:
        data["ocean_manifest_sha256"] = digest(ocean)
    identity = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[
        :14
    ]
    aid = "attempt_" + args.stage + "_" + identity
    dest = project / "run/retained_columbia/attempts" / aid
    remote = str(
        PurePosixPath(cfg["project_dir"]) / "run/retained_columbia/attempts" / aid
    )
    data.update(attempt_id=aid, remote_directory=remote)
    manifest = json.dumps(data, indent=2, sort_keys=True) + "\n"
    params = Namespace(
        execution_site="perlmutter",
        site_configuration=cfg,
        job_prefix="col_salt",
        grid_case="retained_columbia",
        walltime="00:30:00" if daily else "00:05:00",
        account=cfg["account"],
        nodes=1,
        ranks=52,
        case_name="columbia",
        remote_executable=exe,
        module=freeze["build_environment"]["modules"],
    )
    checks = {
        **{"input/" + k: v for k, v in files.items()},
        "columbia_run.nml": data["namelist_sha256"],
        exe: data["executable_sha256"],
    }
    artifacts = {
        "runtime_checksums.sha256": "".join(value + "  " + k + "\n" for k, value in sorted(checks.items())).encode(),
        "columbia_run.nml": args.namelist.read_bytes(),
        "run.sbatch": attempt_sbatch(params, aid, args.stage).encode(),
        "attempt_manifest.json": manifest.encode(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with TemporaryDirectory(dir=dest.parent) as temporary:
            pending = Path(temporary) / "complete"
            pending.mkdir()
            for name, content in artifacts.items():
                (pending / name).write_bytes(content)
            try:
                pending.rename(dest)
            except OSError:
                if not dest.is_dir():
                    raise
    for name, content in artifacts.items():
        if (dest / name).read_bytes() != content:
            raise ValueError("Conflicting immutable attempt file: " + name)
    return data
