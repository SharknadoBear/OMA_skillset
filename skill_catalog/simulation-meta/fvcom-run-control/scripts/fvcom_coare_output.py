"""Read the native calculated-heat fields, per-rank integrations and wet phase."""
import importlib.util
from pathlib import Path

import numpy as np

import fvcom_lake_audit as native
import fvcom_lake_coare as coare


def bind(args, folder, attempt, launch, nml):
    case = native.read(folder/"case.json")
    stage = attempt["stage"]
    label = stage["name"].removeprefix("coare_")
    expected = coare.configuration(nml, "coare_"+label)
    native.check(label != "reader" and all(native.clock(stage[k]) == native.clock(expected[k]) ==
                 native.clock(case[k]) for k in ("start_utc","end_utc")), "Diagnostic clocks differ")
    native.check(native.sha(folder/"case.json") == launch["case_sha256"] and case["kind"].removeprefix("coare_") == label
                 and case["nml_sha256"] == attempt["run_namelist_sha256"]
                 and case["mpi_ranks"] == attempt["mpi_ranks"], "Diagnostic launch differs from immutable case")
    assets = coare.build_assets(args.build_skill)
    qualified = attempt.get("qualification", {})
    evidence = {}
    for key in ("build","reader","forcing"):
        name = "coare_"+key+".json"
        path = getattr(args,"coare_"+key,None) or folder/name
        if qualified:
            native.check(native.sha(path) == qualified[name], "Changed frozen evidence: "+key)
        evidence[key] = native.read(path)
        if key == "forcing":
            native.check(native.sha(path) == case["forcing_manifest_sha256"], "Changed native forcing manifest")
    build = evidence["build"]
    coare.verify_build(build, assets)
    coare.verify_build(case["executable"], assets)
    native.check(native.sha(Path(attempt["executable"])) == case["executable"]["sha256"] == build["sha256"] ==
                 launch["executable_sha256"] and case["executable"]["exe"] == attempt["executable"],
                 "Runtime executable differs from the build and launch")
    coare.verify_reader(evidence["reader"],build)
    coare.verify_forcing(nml,attempt["files"],evidence["forcing"],stage)
    native.check(set(attempt["files"]) == set(case["inputs"]), "Changed input inventory")
    for name, digest in attempt["files"].items():
        native.check(Path(name).name == name and native.sha(folder/"input"/name) == digest == case["inputs"][name]["sha256"],
                     "Changed diagnostic input: "+name)
    module = importlib.util.spec_from_file_location("fvcom_native_coare_fields", assets/"check_fields.py")
    fields = importlib.util.module_from_spec(module)
    module.loader.exec_module(fields)
    return fields


def ledgers(folder, ranks, hours, i0):
    hashes, points = {}, {}
    # The pinned recipe writes one-based Fortran file ranks and counts local steps since startup.
    for tag, columns, ticks in (("heat",17,720),("ext",7,1440)):
        paths = sorted(folder.glob(f"coare_{tag}_rank_*.dat"))
        expected = [f"coare_{tag}_rank_{rank:05d}.dat" for rank in range(1,ranks+1)]
        native.check([p.name for p in paths] == expected, "Missing or unexpected native "+tag+" rank")
        for path in paths:
            rows = np.loadtxt(path,ndmin=2)
            hours_axis = np.arange(1,hours+1)
            native.check(rows.shape == (hours,columns) and np.isfinite(rows).all()
                         and np.array_equal(rows[:,0],ticks*hours_axis)
                         and np.array_equal(rows[:,1],i0+720*hours_axis) and np.all(np.diff(rows[:,2]) >= 0),
                         "Incorrect cumulative ledger clocks or wet area in "+path.name)
            hashes[path.name] = native.sha(path)
    for path in sorted(folder.glob("coare_point_rank_*.dat")):
        rows = np.loadtxt(path,ndmin=2)
        native.check(rows.shape[1] == 18 and np.isfinite(rows).all(), "Incomplete point trace "+path.name)
        for node in np.unique(rows[:,3]):
            subset = rows[rows[:,3] == node]
            native.check(np.array_equal(subset[:,0],i0+360*np.arange(2*hours+1)), "Incorrect half-hour point trace")
            points.setdefault(str(int(node)),[]).append(dict(file=path.name,count=len(subset),sha256=native.sha(path)))
    native.check(set(points) == {"694","39652","69552"}, "Missing point in the pinned source instrumentation")
    return hashes, points


def phase_table():
    assets = Path(__file__).resolve().parents[1]/"assets"
    policy = native.read(assets/"lake_thermal_policy.json")
    lookup = assets/"freshwater_freezing.csv"
    native.check(native.sha(lookup) == policy["freezing_lookup_sha256"], "Changed TEOS-10 freshwater phase table")
    return np.loadtxt(lookup,delimiter=",",skiprows=1), policy


def wet_phase(ds, i, lookup, pressure, previous):
    table, policy = lookup
    wet = np.asarray(native.values(ds,"wet_nodes",i),bool)
    depth = np.asarray(ds["h"][:],float) + np.asarray(native.values(ds,"zeta",i),float)
    thickness = -np.diff(np.asarray(ds["siglev"][:],float),axis=0)*depth
    native.check(np.all(thickness[:,wet]>0), "Nonpositive wet-layer thickness")
    mid = np.cumsum(thickness,axis=0)-thickness/2
    lat = np.deg2rad(np.asarray(ds["lat"][:],dtype=float))
    gravity = 9.780327*(1+.0053024*np.sin(lat)**2-.0000058*np.sin(2*lat)**2)
    sea_pressure = (pressure-101325+1000*gravity*mid)/10000
    native.check(sea_pressure[:,wet].min() >= table[0,0] and sea_pressure[:,wet].max() <= table[-1,0],
                 "Wet pressure outside freezing table")
    temp = np.asarray(native.values(ds,"temp",i),float)
    margin = temp-np.interp(sea_pressure,table[:,0],table[:,2])
    volume = np.asarray(ds["art1"][:],float)*thickness
    def extreme(field, low):
        eligible = np.where(wet,field,np.inf if low else -np.inf)
        k,n = np.unravel_index((np.argmin if low else np.argmax)(eligible),field.shape)
        return dict(node=int(n+1),layer=int(k+1),temp_c=float(temp[k,n]),margin_c=float(margin[k,n]),
                    depth_m=float(depth[n]),midlayer_m=float(mid[k,n]),lon=float((ds["lon"][n]+180)%360-180),lat=float(ds["lat"][n]))
    cold = wet & (margin < -policy["cold_conservative_margin_C"])
    persistent = np.zeros_like(cold) if previous is None else cold & previous
    result = dict(minimum=extreme(temp,True),maximum=extreme(temp,False),minimum_margin=extreme(margin,True))
    for label,mask in (("subzero",wet&(temp<0)),("conservative",cold),("persistent",persistent),
                       ("substantial",wet&(margin < -policy["cold_substantial_margin_C"])),("gross_hot",wet&(temp>40))):
        result[label] = dict(wet_layers=int(mask.sum()),volume_m3=float(volume[mask].sum()))
    return result,cold
