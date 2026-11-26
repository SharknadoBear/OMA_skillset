"""Run the complete nine-output reduction on known wet/dry and field differences."""
import unittest
import netCDF4 as nc
import numpy as np
import fvcom_benchmark as benchmark
from fvcom_benchmark_fields import FIELDS,evaluate
import selftest_perlmutter_benchmark as evidence


class NativeDifferences(unittest.TestCase):
    def test_nine_histories(self):
        fixture=evidence.PerlmutterCPU()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        rows=[]
        inv={"members":{}}
        for j,slot in enumerate(fixture.plan["runs"]):
            row=fixture.collect(slot,50,60)
            rows.append(row)
            folder=fixture.root/row["attempt"]
            path=folder/"superior_0001.nc"
            with nc.Dataset(path,"w") as ds:
                for n,k in (("node",3),("nele",1),("siglay",2),("siglev",3),("three",3),("time",5)):
                    ds.createDimension(n,k)
                for name in ("lon","lat","h"):
                    ds.createVariable(name,"f4",("node",))[:]=[1,2,3]
                ds.createVariable("partition","i4",("nele",))[:]=1
                ds.createVariable("nprocs","i4",())[:]=slot["ranks"]
                ds.createVariable("nv","i4",("three","nele"))[:]=np.array([[1],[2],[3]])
                ds.createVariable("siglev","f4",("siglev","node"))[:]=np.array([0,-.5,-1])[:,None]
                day=ds.createVariable("Itime","i4",("time",))
                day.units="days since 2025-07-01 00:00:00"
                day[:]=[0,0,0,0,1]
                ds.createVariable("Itime2","i4",("time",))[:]=[0,21600000,43200000,64800000,0]
                ds.createVariable("wet_nodes","i4",("time","node"))[:]=[1,1,int(j==0)]
                ds.createVariable("wet_cells","i4",("time","nele"))[:]=1
                for name,(d,_) in FIELDS.items():
                    dims=("time",d) if name in {"zeta","ua","va"} else ("time","siglay",d)
                    var=ds.createVariable(name,"f4",dims)
                    var[:]=float(j)
                    if d=="node" and j:
                        var[...,-1]=100000.
            audit=folder/"native_audit.json"
            benchmark.write_json(audit,dict(status="verified",attempt=row["attempt"],job_id=row["job_id"],
                attempt_sha256=row["evidence_sha256"]["attempt"],executable_sha256=row["executable_sha256"],
                startup_sha256=row["restart_sha256"],output_files={path.name:{"sha256":benchmark.sha256(path)}}))
            inv["members"][slot["stage"]]={"attempt":folder.name,"audit":str(audit)}
        inventory=fixture.root/"inventory.json"
        benchmark.write_json(inventory,inv)
        result=evaluate(fixture.plan_path,rows,inventory)
        self.assertEqual((result["pair_count"],result["timestamps"]),(36,5))
        pair=next(p for p in result["comparisons"] if p["left"]=="benchmark_064_r1" and p["right"]=="benchmark_256_r3")
        self.assertEqual(pair["pooled"]["temp"],dict(count=20,max_abs=8.,rms=8.))
        self.assertEqual(pair["mask_disagreements"]["node"],5)
        self.assertTrue(pair["identical_element_partition"])
        self.assertEqual(result["distinct_partitions_per_layout"],{"64":1,"128":1,"256":1})
        path.write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError,"history file changed"):
            evaluate(fixture.plan_path,rows,inventory)


if __name__=="__main__":
    unittest.main()
