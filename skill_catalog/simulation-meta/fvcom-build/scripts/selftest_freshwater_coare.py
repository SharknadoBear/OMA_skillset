"""Exercise recipe preparation, source lock and the parallel wind diagnostic regression."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest

import freshwater_coare as recipe


class RecipeTests(unittest.TestCase):
    def test_missing_halo_exchange(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)
            bcond=src/"bcond_gcn.F"
            kernel=src/"coare26z.F"
            kernel.write_text("qualified bulk kernel\n")
            bcond.write_text("IF(PAR)CALL AEXCHANGE(NC,MYID,NPROCS,COARE_WIND_SPEED)\n")
            pin={"changed":{bcond.name:{"after":recipe.digest(bcond)}},"coare26z_sha256":recipe.digest(kernel)}
            recipe.check_changes(src,pin)
            bcond.write_text("! lost wind exchange during output\n")
            pin["changed"][bcond.name]["after"]=recipe.digest(bcond)
            with self.assertRaisesRegex(ValueError,"halo exchange"):
                recipe.check_changes(src,pin)

    def test_mismatched_source_and_stale_operator(self):
        assets=recipe.asset_root()
        sys.path.insert(0,str(assets))
        import patch_coare
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)
            (src/"mod_force.F").write_text("wrong baseline\n")
            with self.assertRaisesRegex(ValueError,"unrecognized frozen source"):
                patch_coare.run(src,src/"unexpected.json")
            self.assertFalse((src/"unexpected.json").exists())

    def test_payload_and_baseline_binding(self):
        pin=recipe.read(recipe.asset_root()/"source_pin.json")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binding=root/"binding.json"
            baseline=dict(status="frozen", execution_site="perlmutter",source_commit=pin["source_commit"],
                feature_defines=["SPHERICAL","WET_DRY","SINGLE_OUTPUT","AIR_PRESSURE","DOUBLE_PRECISION","FRESHWATER_LAKE","MULTIPROCESSOR"],
                native_executable="/scratch/base/source/fvcom_plased",executable_sha256="1"*64,native_freeze_sha256="2"*64,
                lake_physics_probe_sha256="3"*64,
                build_environment={"compiler":"ftn","programming_environment":"PrgEnv-gnu","modules":["cpu/1","PrgEnv-gnu/8"]})
            binding.write_text(json.dumps(baseline))
            args=argparse.Namespace(binding=binding,output=root/"payload.tgz",account="testcpu",qos="debug",
                baseline_source="/scratch/base/source/build_1_gnu",baseline_evidence="/scratch/base/run/build/evidence",
                remote_workdir="/scratch/new_study/diagnostics/bulk")
            result=recipe.package(args)
            self.assertEqual(result["archive_sha256"],recipe.digest(args.output))
            with tarfile.open(args.output) as tf:
                self.assertEqual(set(tf.getnames()),{"inputs.sha256","base_binding.json","build.sbatch","ledger.F","check_fields.py",
                                                     "patch_coare.py","freshwater_coare.py","source_pin.json","perlmutter_dependencies.py"})
                batch=tf.extractfile("build.sbatch").read().decode()
                self.assertIn("#SBATCH --account=testcpu",batch)
                self.assertIn("srun --exact --ntasks=1",batch)
                self.assertIn("payload=/scratch/new_study/diagnostics/bulk",batch)
            with self.assertRaises(FileExistsError):recipe.package(args)
            baseline["source_commit"]="0"*40;binding.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError,"pinned"):recipe.package(args)


def actual_replay(baseline, original_kernel):
    assets=recipe.asset_root()
    sys.path.insert(0,str(assets))
    import patch_coare
    pin=recipe.read(assets/"source_pin.json")
    with tempfile.TemporaryDirectory() as tmp:
        src=Path(tmp)
        for name in pin["changed"]:
            shutil.copyfile(baseline/name,src/name)
        (src/"coare26z.F").write_bytes(original_kernel.read_bytes().replace(b"\r\n",b"\n"))
        patch_coare.run(src,src/"changes.json")
        recipe.check_changes(src,pin)
        if recipe.read(src/"changes.json")!=pin["changed"]:
            raise ValueError("native source replay differs from qualified build")
    print("All five transformed sources reproduce the compiled reference hashes.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-source",type=Path)
    p.add_argument("--original-kernel",type=Path)
    a=p.parse_args()
    if not unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(RecipeTests)).wasSuccessful():
        raise SystemExit(1)
    if a.baseline_source:
        actual_replay(a.baseline_source,a.original_kernel or a.baseline_source/"coare26z.F")
