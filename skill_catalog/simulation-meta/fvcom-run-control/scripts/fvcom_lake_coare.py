"""Bind the optional Perlmutter freshwater COARE26Z reader and short diagnostic windows."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def require(test, reason):
    if not test:
        raise ValueError(reason)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def build_assets(directory=None):
    root = Path(directory) if directory else Path(__file__).resolve().parents[2] / "fvcom-build"
    assets = root / "assets" / "freshwater_coare"
    if directory and (root/"source_pin.json").is_file():
        assets = root  # flat archive emitted by freshwater_coare.py prepare
    require((assets / "source_pin.json").is_file() and (assets / "check_fields.py").is_file(),
            "Supply the installed fvcom-build directory or a prepared recipe with --build-skill")
    return assets


def utc(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def value(text, key):
    # Native archived groups may contain multiple comma-separated assignments per line.
    tokens = re.findall(r"(?im)(?:^|,)\s*" + key + r"\s*=\s*('[^']*'|\"[^\"]*\"|[^,!\r\n]+)",
                        re.sub(r"(?m)!.*$", "", text))
    require(len(tokens) == 1, "Expected one native parameter " + key)
    return tokens[0].strip().strip("'\x22")


def enabled(text):
    return bool(re.search(r"(?im)(?:^|,)\s*HEATING_CALCULATE_ON\s*=\s*\.?(?:TRUE|T)\.?(?=\s*[,!\r\n]|$)", text))


def configuration(nml, name):
    require(name in {"coare_reader", "coare_cold", "coare_warm", "coare_continuation"},
            "Use a separately named COARE reader, cold, warm or continuation attempt")
    for k, expected in {"HEATING_ON":"F", "HEATING_CALCULATE_ON":"T", "HEATING_FRESHWATER":"T",
                        "COARE_VERSION":"COARE26Z", "HEATING_CALCULATE_TYPE":"FLUX", "HEATING_CALCULATE_KIND":"VARIABLE",
                        "WIND_ON":"T", "WIND_TYPE":"STRESS", "WIND_KIND":"VARIABLE", "AIRPRESSURE_ON":"T",
                        "AIRPRESSURE_KIND":"VARIABLE", "PRECIPITATION_ON":"T", "PRECIPITATION_KIND":"VARIABLE",
                        "STARTUP_TYPE":"HOTSTART", "SCALAR_POSITIVITY_CONTROL":"F"}.items():
        actual = value(nml, k).upper().strip(".")
        require({"TRUE":"T","FALSE":"F"}.get(actual, actual) == expected, "Unexpected COARE setting " + k)
    for k, expected in {"ZUU":10, "ZTT":2, "ZQQ":2, "IRAMP":0, "EXTSTEP_SECONDS":2.5, "ISPLIT":2}.items():
        require(float(value(nml, k)) == expected, "Unexpected native setting " + k)
    start, end = [utc(value(nml, k)) for k in ("START_DATE", "END_DATE")]
    seconds = (end-start).total_seconds()
    reader = name == "coare_reader"
    require(seconds == 600 if reader else seconds in {86400,172800}, "Expected 600-second reader or 24/48-hour diagnostic")
    def interval(k):
        units, number = value(nml, k).lower().split("=")
        return {"days":86400, "seconds":1}[units] * float(number)
    require(utc(value(nml,"NC_FIRST_OUT")) == start and interval("NC_OUT_INTERVAL") == (600 if reader else 21600)
            and interval("OUT_INTERVAL") == 3600 and interval("RST_OUT_INTERVAL") == 86400
            and (utc(value(nml,"RST_FIRST_OUT"))-start).total_seconds() == (600 if reader else 86400),
            "Grid, station or restart clock differs from the diagnostic")
    return dict(name=name, start_utc=start.isoformat().replace("+00:00","Z"),
                end_utc=end.isoformat().replace("+00:00","Z"))


def verify_build(build, assets):
    pin = read(assets/"source_pin.json")
    flags = set(build["cpp"])
    require(build.get("schema") == "fvcom_coare_diagnostic_build_v1" and build.get("scope") == "diagnostic"
            and build.get("status") == "frozen" and build["source_commit"] == pin["source_commit"]
            and build["changed"] == pin["changed"] and {"FRESHWATER_LAKE","HEATING_CALCULATED","LAKE_COARE_DIAG",
            "MULTIPROCESSOR","SPHERICAL","AIR_PRESSURE","WET_DRY"}.issubset(flags) and "TWO_D_MODEL" not in flags
            and re.fullmatch("[a-f0-9]{64}",build["sha256"]) and build["sha256"] != build["baseline_executable_sha256"],
            "Expected the pinned freshwater COARE diagnostic derivative")


def verify_reader(reader, build):
    require(reader.get("status") == "verified" and reader.get("scope") == "diagnostic"
            and reader["executable_sha256"] == build["sha256"] and reader.get("records")
            and re.fullmatch("[a-f0-9]{64}", reader["history_sha256"])
            and all({"coare_wind_speed","short_wave","latent_mass_Wm2","heat_addition_Wm2","raw_rain",
            "rain_and_dew","evap","hydro_mslp","uwind_stress","vwind_stress"} <= set(row["max_abs_error"])
                    for row in reader["records"].values()), "Missing independent reader test for this executable")


def verify_forcing(nml, files, forcing, stage):
    if forcing.get("schema") == "fvcom_lake_hrrr_coare26z_diagnostic_v1":
        require(forcing["status"] == "prepared_for_native_qualification", "Incomplete HRRR COARE preparation")
        dates = [forcing["model_start_utc"],forcing["model_end_utc"]]
        hydro = forcing["hydrodynamic_reference"]
        native = forcing["files"]
    else:
        require(forcing.get("schema") == "fvcom_lsofs_coare_diagnostic_v1" and forcing.get("diagnostic_only")
                and forcing["status"] == "ready_for_native_probe", "Unknown native COARE forcing package")
        dates = forcing["main_utc"]
        hydro = forcing["unmodified_meteorology"]
        native = {k:v["sha256"] for k,v in forcing["files"].items()}
    require(utc(dates[0]) <= utc(stage["start_utc"]) < utc(stage["end_utc"]) <= utc(dates[1]),
            "COARE source does not cover requested integration")
    wind, pressure, rain, bulk, cor = (value(nml, k) for k in ("WIND_FILE","AIRPRESSURE_FILE","PRECIPITATION_FILE",
                                        "HEATING_CALCULATE_FILE","CORIOLIS_FILE"))
    require(wind == pressure and files[wind] == hydro["sha256"] and
            set(hydro["used_fields"]) == {"uwind_stress","vwind_stress","air_pressure"},
            "Wrong unchanged wind stress or hydrodynamic pressure")
    require(len({wind,rain,bulk}) == 3 and all(files[k] == native[k] for k in (rain,bulk,"superior_cor.dat"))
            and files[cor] == files["superior_cor.dat"], "Wrong bulk, raw rain or native COARE latitude")


def checked_sources(inputs, root):
    sources = {}
    for name, record in inputs.items():
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name), "Unsafe native input filename")
        if isinstance(record, str):
            record = dict(path=str(root/"data"/record),sha256=record)
        path = PurePosixPath(record["path"])
        require(re.fullmatch("[a-f0-9]{64}",record["sha256"]) and path.is_absolute() and path.is_relative_to(root)
                and ".." not in path.parts and re.fullmatch(r"/[A-Za-z0-9_./+-]+",str(path)), "Invalid input binding "+name)
        sources[name] = dict(path=str(path),sha256=record["sha256"])
    return sources
