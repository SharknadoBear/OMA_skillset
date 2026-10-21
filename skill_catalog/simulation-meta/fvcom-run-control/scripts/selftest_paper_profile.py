"""Offline profile regressions: defaults, stages, attempt caps and native builds."""
from __future__ import annotations
import copy
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

from fvcom_profiles import resolve_request
from fvcom_run_control import init_project, create_attempt
SKILLS = Path(__file__).resolve().parents[2]
for skill in ("fvcom-namelist-configuration", "fvcom-build"):
    sys.path.insert(0, str(SKILLS / skill / "scripts"))
import fvcom_namelist as fn
import fvcom_build as fb
from selftest_fvcom_namelist import BINDINGS

def rejects(call):
    try:
        call()
    except (ValueError, FileExistsError):
        return
    raise AssertionError("expected rejection")

def main():
    raw = {"schema": "simulation_request_v1", "case_id": "paper_test", "profile": "paper_barotropic",
           "kestrel": {"project_dir": "/scratch/test/paper_test", "source_dir": "/scratch/test/paper_test/source"}}
    request = resolve_request(raw)
    assert "period" not in raw
    assert request["period"]["analysis_end"] == "2025-04-15T00:00:00Z"
    assert request["execution"]["max_smoke_attempts"] == request["execution"]["max_production_attempts"] == 3
    assert request["benchmark"]["enabled"] is False
    for patch in ({"profile": "paper_barotropic_v1"}, {"mode": "benchmark"},
                  {"grid_policy": "accepted_then_fresh"}, {"execution": {"max_smoke_attempts": 4}},
                  {"benchmark": {"enabled": True}}, {"period": {"spinup_days": 7}},
                  {"physics": {"sigma_layers": 9}}):
        rejects(lambda: resolve_request(dict(raw, **patch)))
    legacy = {"mode": "benchmark", "period": {"analysis_start": "2024-06-01T00:00:00Z"}}
    assert resolve_request(legacy) == legacy
    numerics = {"extstep_seconds": 1.2, "isplit": 4}
    generated = {}
    for stage, days in (("smoke", 1), ("production", 17)):
        start, end, _, hot = fn.stage_times(raw, stage)
        assert (end-start).days == days and not hot and start.isoformat().startswith("2025-03-29")
        values, timing = fn.configuration_values(raw, BINDINGS, numerics, stage)
        assert fn.audit_values(values) == []
        assert fn.audit_stage_values(values, stage, "paper_barotropic") == []
        assert values["NML_STARTUP"]["STARTUP_FILE"] == "'none'"
        assert values["NML_RESTART"]["RST_ON"] == "T"
        assert timing["ramp_internal_steps"] == 36000
        generated[stage] = "\n".join(f"{key} = {value}," for group in values.values() for key, value in group.items())
    for stage in ("canary", "spinup", "benchmark", "full"):
        rejects(lambda: fn.stage_times(raw, stage))
    with tempfile.TemporaryDirectory(prefix="paper_profile_test_") as td:
        root = Path(td)
        req = root/'request.json'; req.write_text(json.dumps(raw))
        project = root/'project'
        manifest = init_project(req, project)
        assert manifest['grid_order'] == ['fresh_reproduction']
        assert manifest['kestrel_root'] == raw['kestrel']['project_dir']
        assert not (project/'gridding/accepted_t6v6').exists()
        inputs = root/'input'; inputs.mkdir()
        (inputs/'case_spg.dat').write_text('Sponge Node Number = 1\n1 100.0 0.0025\n')
        controls = root/'controls.json'; controls.write_text(json.dumps(numerics))
        for stage in ('smoke', 'production'):
            nml = root/f'{stage}.nml'; nml.write_text(generated[stage])
            args = Namespace(project=str(project), plan=None, controls=str(controls), paper_stage=stage,
                 index=0, grid_case='fresh_reproduction', base_input=str(inputs), base_namelist=str(nml),
                 stage_namelist=[], revision=0, module=[], remote_executable='/scratch/test/fvcom',
                 case_name='paper_test', account='test', partition='shared' if stage=='smoke' else 'auto',
                 nodes=1 if stage=='smoke' else 2, ranks=52 if stage=='smoke' else 208, walltime='04:00:00')
            for index in range(3):
                args.index = index
                result = create_attempt(args)
                assert result['stage_attempt'] == index+1 and result['stage_attempt_limit'] == 3
                batch = (project/result['run_dir']/'job.sbatch').read_text()
                if stage == 'production':
                    assert '#SBATCH --ntasks-per-node=104' in batch and '#SBATCH --exclusive' in batch
                    assert '#SBATCH --partition=' not in batch
                    assert 'srun --cpu-bind=cores --ntasks=208' in batch
                rejects(lambda: create_attempt(args))
            args.index = 3; rejects(lambda: create_attempt(args))
            init_project(req, project)
            args.index = 0; rejects(lambda: create_attempt(args))
            args.revision = 1; rejects(lambda: create_attempt(args))
            args.revision = 0; args.ranks = 416; rejects(lambda: create_attempt(args))
        rejects(lambda: fb.build_plan(raw, root/'build', '/scratch/test/build'))
        env = {'os': 'rh9', 'compiler': 'mpiifx', 'compiler_options': ['-O3'], 'modules': ['intel/native', 'netcdf/native']}
        native = copy.deepcopy(raw); native['kestrel']['build_environment'] = env
        plan = fb.build_plan(native, root/'build', '/scratch/test/build')
        assert plan['modules'] == env['modules'] and plan['compiler'] == 'mpiifx'
        rh8 = copy.deepcopy(native); rh8['kestrel']['build_environment']['os'] = 'rh8'
        rejects(lambda: fb.build_plan(rh8, root/'build', '/scratch/test/build'))
        evidence = root/'evidence'; evidence.mkdir()
        for name in fb.MANDATORY_EVIDENCE:
            (evidence/name).write_text('fixture')
        (evidence/'make_inc_audit.json').write_text(json.dumps({'status':'pass','sha256':'b'*64}))
        (evidence/'executable.sha256').write_text('a'*64+'  fvcom')
        (evidence/'freeze.json').write_text(json.dumps({'executable_sha256':'a'*64, 'source_edits_after_freeze_allowed':False, 'build_environment':env}))
        (evidence/'smoke.log').write_text('FVCOM_INPUT_SMOKE_PASS')
        (evidence/'blank_namelist.nml').write_text('&NML_CASE')
        (evidence/'os-release.txt').write_text('VERSION_ID="9.6"\n')
        (evidence/'modules.txt').write_text('intel/native netcdf/native')
        (evidence/'build_commands.txt').write_text('mpiifx -O3')
        result = fb.finalize(native, evidence, root/'build')
        assert result['status'] == 'ready', result
        (evidence/'os-release.txt').write_text('VERSION_ID="8.8"\n')
        assert fb.finalize(native, evidence, root/'bad')['status'] == 'blocked_compile'
    print('paper_barotropic profile selftest: PASS')

if __name__ == '__main__':
    main()
