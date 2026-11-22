"""Exercise real river writer, both profiles, final configuration, and immutable attempts."""
import json
import shutil
import sys
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fvcom_profiles import PAIR_PROFILE, RIVER_MODE, resolve_request
from fvcom_run_control import create_attempt, init_project, sha256
from freeze_attempt_inputs import freeze_attempt

siblings = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(siblings / "fvcom-namelist-configuration/scripts"))
river = siblings / "fvcom-river-forcing/scripts"
if not river.is_dir():
    river = siblings.parent / "forcing-builders/fvcom-river-forcing/scripts"
sys.path.insert(0, str(river))
from selftest_barotropic_river import RiverTest
from selftest_fvcom_namelist import BINDINGS, template
from fvcom_namelist import configuration_values, render_template


def main():
    fixture = RiverTest()
    fixture.setUp()
    try:
        start = datetime(2025, 3, 29, tzinfo=timezone.utc)
        fixture.rows = [[(start + timedelta(hours=i)).isoformat(), 120, 120*.028316846592]
                        for i in range(17*24 + 1)]
        fixture.source["records"] = len(fixture.rows)
        fixture.end = int((start + timedelta(days=17)).timestamp())
        fixture.run_builder()
        root = fixture.root
        def write(name, data):
            path = root/name
            path.write_text(json.dumps(data))
            return path
        receipt = write("installation.json", {"commit":"0"*40, "installed_root":str(root),
                                              "files":{"mapping.json":sha256(root/"mapping.json")}})
        source = {"provider":"usgs", "station_id":"14246900", "parameter_code":"72137"}
        numerics = {"extstep_seconds":1.2, "isplit":4}
        controls = write("controls.json",numerics)
        for name, mode in (("control","barotropic_tide"),("river",RIVER_MODE)):
            req = {"schema":"simulation_request_v1","case_id":name,"mode":mode,"profile":PAIR_PROFILE}
            if mode == RIVER_MODE:
                req["river"] = source
            project = root/name
            init_project(write(name+".json",req), project)
            binding = project/"run/build/executable_reuse_binding.json"
            binding.parent.mkdir(parents=True, exist_ok=True)
            binding.write_text(json.dumps({"status":"frozen","executable_sha256":"1"*64}))
            inputs = project/"input/fresh_reproduction/base"
            (inputs/"station_mapping.json").write_text(json.dumps({"status":"ready","stations":[]}))
            bindings = dict(BINDINGS)
            for src,k in ((fixture.mesh,"grid_file"),(fixture.dep,"depth_file"),(fixture.obc,"obc_file")):
                shutil.copyfile(src,inputs/bindings[k])
            (inputs/bindings["sponge_file"]).write_text("Sponge Node Number = 1\n1 100.0 0.0025\n")
            if mode == RIVER_MODE:
                shutil.copytree(root/"product",inputs,dirs_exist_ok=True)
                bindings["river_manifest"] = str(inputs/"river_manifest.json")
            for stage in ("smoke","production"):
                values,_ = configuration_values(resolve_request(req),bindings,numerics,stage)
                nml = project/f"{stage}.nml"
                nml.write_text(render_template(template(values), values))
                args = Namespace(project=project,grid_case="fresh_reproduction",paper_stage=stage,
                    skill_installation=receipt, index=0,controls=controls,plan=None,
                    partition="shared" if stage=="smoke" else "auto",nodes=1 if stage=="smoke" else 2,
                    ranks=52 if stage=="smoke" else 208,base_input=inputs,base_namelist=nml,
                    stage_namelist=[],case_name="columbia",module=[],remote_executable="/sandbox/fvcom",
                    account="test",walltime="01:00:00")
                attempt = create_attempt(args)
                assert attempt["profile"] == PAIR_PROFILE and attempt["stage"] == stage
                assert attempt["skill_commit"] == "0"*40
                assert attempt["stage_attempt_limit"] == 3
                run = project/attempt["run_dir"]
                frozen = freeze_attempt(project, run)
                assert frozen == freeze_attempt(project, run)
                assert frozen["files"]["station_mapping.json"] == sha256(inputs/"station_mapping.json")
                staged_mapping = project/attempt["input_dir"]/"station_mapping.json"
                staged_mapping.write_text("{}")
                try:
                    freeze_attempt(project, run)
                except ValueError as e:
                    assert "existing freeze" in str(e)
                else:
                    raise AssertionError("changed input was silently refrozen")
            if mode == RIVER_MODE:
                with (inputs/"river.nc").open("ab") as f:
                    f.write(b"changed after configuration")
                args.index = 1
                try:
                    create_attempt(args)
                except ValueError as e:
                    assert "hash" in str(e), e
                else:
                    raise AssertionError("mutated staged forcing accepted")
    finally:
        fixture.doCleanups()
    print("PASS paired initialization, writer, configuration, four attempts, provenance and tamper check")


if __name__ == "__main__":
    main()
