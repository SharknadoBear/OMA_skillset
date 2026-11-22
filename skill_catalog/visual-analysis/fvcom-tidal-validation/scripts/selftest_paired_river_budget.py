"""Analytic two-cell test for direction, inflow ramp, storage and level offset."""
import json
import tempfile
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

import paired_river_budget as budget


def main():
    start = pd.Timestamp("2025-03-29", tz="UTC")
    grid = np.array([[0, 1, 2], [1, 3, 2]])
    x = np.array([0., 1., 0., 1.])
    y = np.array([0., 0., 1., 1.])
    geometry = {"upstream_element_ids": [1], "section": [
        {"cell_id": 1, "edge_nodes": [3, 2], "station_id": "cut"}]}
    cells, faces = budget.section_geometry(grid, x, y, geometry)
    np.testing.assert_array_equal(cells, [0])
    np.testing.assert_allclose(faces[0][-1], np.array([1., 1.])/np.sqrt(2))
    try:
        budget.section_geometry(grid, x, y, {**geometry, "section": []})
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete section accepted")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        def write(name, payload):
            (root/name).write_text(json.dumps(payload))
        with nc.Dataset(root/"grid.nc", "w") as ds:
            for dim, n in (("node", 4), ("nele", 2), ("three", 3), ("time", 2)):
                ds.createDimension(dim, n)
            for key, a in (("x", x), ("y", y), ("h", np.full(4, 5.))):
                ds.createVariable(key, "f8", ("node",))[:] = a
            ds.createVariable("nv", "i4", ("three", "nele"))[:] = grid.T + 1
            ds.createVariable("iint", "i4", ("time",))[:] = [0, 360]
            ds.createVariable("zeta", "f8", ("time", "node"))[:] = [[0.]*4, [0.2]*4]
        with nc.Dataset(root/"station.nc", "w") as ds:
            for dim,n in (("station", 1), ("time", 2), ("namelen", 3)):
                ds.createDimension(dim,n)
            ds.createVariable("iint","i4",("time",))[:] = [0,360]
            ds.createVariable("name_station","S1",("station","namelen"))[0,:] = list("cut")
            for key, values in (("zeta",[0.,.2]),("ua",[2.,2.]),("va",[0.,0.])):
                ds.createVariable(key,"f8",("time","station"))[:,0] = values
        write("audit.json", {"full_grid_netcdf":[{"path":str(root/"grid.nc"),
                                                  "sha256":budget.sha256(root/"grid.nc")}]})
        write("mapping.json", {"stations":[{"station_id":"cut","cell_id":1}]})
        (root/"run.nml").write_text("GRID_FILE='grid.dat'\nSTART_DATE='2025-03-29'\nEND_DATE='2025-04-15'\nEXTSTEP_SECONDS=1\nISPLIT=1\nIRAMP=360\n")
        case = {"namelist":root/"run.nml","run_start":start,"start":start,"end":start+pd.Timedelta(minutes=6),
                "mapping_path":root/"mapping.json","audit_path":root/"audit.json","station_paths":[root/"station.nc"],
                "freeze":{"files":{"grid.dat":"a"*64}},"attempt":root,"input_dir":root}
        geometry["mesh_sha256"] = "a"*64
        q, v, z, names = budget.sample_member(case,geometry)
        np.testing.assert_allclose(q.values,[10.,10.4])
        np.testing.assert_allclose(v.values,[2.5,2.6])
        with nc.Dataset(root/"river.nc","w") as ds:
            ds.createDimension("time",2)
            ds.createDimension("rivers",2)
            day = (start-pd.Timestamp("1858-11-17",tz="UTC")).days
            ds.createVariable("Itime","i4",("time",))[:] = [day,day]
            ds.createVariable("Itime2","i4",("time",))[:] = [0,360000]
            ds.createVariable("river_flux","f4",("time","rivers"))[:] = [[1,3],[2,4]]
        write("river_manifest.json",{"files":{"netcdf":{"path":"river.nc"}}})
        np.testing.assert_allclose(budget.applied_flux(case,q.index),[0,6*np.tanh(1.)])
        np.testing.assert_allclose(budget.integral(q.values,np.array([0,360])),3672)
        (root/"water_level").mkdir()
        pd.DataFrame({"time":q.index,"observed":[1.,2.],"predicted":[1.,2.]}).to_csv(root/"water_level/cut_noaa_waterlevel.csv",index=False)
        case["observation_path"] = root/"obs.json"
        metrics = budget.water_level_metrics(case,case,names,q.index,names,z,z+.3)
        np.testing.assert_allclose(metrics["samples"],[1,1,1,1])
        np.testing.assert_allclose(metrics["mean_difference_m"],[-1,-.7,-1,-.7])
    print("PASS analytic section, storage, native tanh ramp and common NOAA support")


if __name__ == "__main__":
    main()
