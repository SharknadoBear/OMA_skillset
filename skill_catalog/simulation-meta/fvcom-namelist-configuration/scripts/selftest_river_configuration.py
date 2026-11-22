"""Exercise the renderer, control regression, and frozen river bindings offline."""
import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import fvcom_namelist as fn
from selftest_fvcom_namelist import BINDINGS, template


def run(args, expected):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        status = fn.cli([str(arg) for arg in args])
    assert status == expected, output.getvalue()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs, source = root / "inputs", root / "forcing"
        inputs.mkdir()
        source.mkdir()
        gauge = {"provider": "usgs", "station_id": "14246900", "parameter_code": "72137"}
        request = {"case_id": "columbia_river", "profile": "columbia_river_pair",
                   "mode": fn.RIVER_MODE, "river": gauge}
        req = root / "request.json"
        fn.write_json(req, request)
        for key in ("grid_file", "obc_file", "elevation_file", "depth_file", "sigma_file",
                    "sponge_file", "coriolis_file", "station_file"):
            (inputs / BINDINGS[key]).write_text(key)
        entries = "".join(
            f"&NML_RIVER\n RIVER_NAME = 'r{i}',\n RIVER_FILE = 'rivers.nc',\n"
            f" RIVER_GRID_LOCATION = {i},\n RIVER_VERTICAL_DISTRIBUTION = 'uniform',\n/\n"
            for i in (1, 2)
        )
        (source / "rivers.nml").write_text(entries)
        (source / "rivers.nc").write_bytes(b"opaque forcing payload certified by forcing specialist")
        manifest = {"schema": "fvcom_river_forcing_v1", "status": "ready", "source": gauge,
                    "start_utc": "2025-03-29T00:00:00Z", "end_utc": "2025-04-15T00:00:00Z",
                    "river_number": 2, "mesh_sha256": fn.sha256(inputs / BINDINGS["grid_file"]),
                    "depth_sha256": fn.sha256(inputs / BINDINGS["depth_file"]),
                    "nodes": [{"name": "r1", "node_id": 1, "weight": .4},
                              {"name": "r2", "node_id": 2, "weight": .6}],
                    "obc_sha256": fn.sha256(inputs / BINDINGS["obc_file"]), "files": {}}
        for key, filename in (("namelist", "rivers.nml"), ("netcdf", "rivers.nc")):
            manifest["files"][key] = {"path": filename, "sha256": fn.sha256(source / filename)}
            shutil.copyfile(source / filename, inputs / filename)
        manifest_path = source / "river_manifest.json"
        fn.write_json(manifest_path, manifest)
        bindings = dict(BINDINGS, river_manifest="forcing/river_manifest.json")
        bindfile, nums, stations, blank, pre = [root / f for f in (
            "bindings.json", "numerics.json", "stations.json", "blank.nml", "pre.json")]
        fn.write_json(bindfile, bindings)
        numeric = {"extstep_seconds": 0.3, "isplit": 6}
        fn.write_json(nums, numeric)
        fn.write_json(stations, {"stations": [{"station_id": "test", "longitude": -123.7,
                      "latitude": 46.2, "depth_m": 15, "cell_id": 3}]})
        fn.write_json(pre, {"sigma_levels": 10})
        for stage, days in (("smoke", 1), ("production", 17)):
            actual, _ = fn.configuration_values(
                request, dict(bindings, river_manifest=str(manifest_path)), numeric, stage)
            base, _ = fn.configuration_values(dict(mode="barotropic_tide", case_id="control",
                profile="columbia_river_pair"), BINDINGS, numeric, stage)
            for group in fn.GROUP_ORDER:
                if group not in {"NML_RIVER_TYPE", "NML_CASE"}:
                    assert actual[group] == base[group], group
            assert actual["NML_STARTUP"]["STARTUP_TYPE"] == "'coldstart'"
            assert actual["NML_RESTART"]["RST_ON"] == "T"
            assert actual["NML_RIVER_TYPE"]["RIVER_NUMBER"] == "2"
            s, e, _, _ = fn.stage_times(request, stage)
            assert (e - s).days == days
            blank.write_text(template(actual))
            out = root / stage
            args = ["render", "--request", req, "--bindings", bindfile, "--numerics", nums,
                    "--stage", stage, "--artifact-root", out, "--stations", stations,
                    "--input-root", inputs, "--blank-template", blank, "--preconfiguration-manifest", pre]
            run(args, 0)
            nml = out / f"columbia_river_{stage}_run.nml"
            verify = ["validate", "--namelist", nml, "--request", req, "--river-manifest",
                      manifest_path, "--input-root", inputs]
            run(verify, 0)
            run(["validate", "--namelist", nml], 2)
            assert fn.load_json(out / "configuration_manifest.json")["hashes"]["river_manifest_sha256"] == fn.sha256(manifest_path)
        for name in ("rivers.nc", "rivers.nml", BINDINGS["grid_file"], BINDINGS["depth_file"], BINDINGS["obc_file"]):
            path = inputs / name
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            run(verify, 2)
            path.write_bytes(original)
        for changed in ({"river_number": 3}, {"end_utc": "2025-04-14T00:00:00Z"},
                        {"source": dict(gauge, station_id="999")}, {"status": "failed"},
                        {"nodes": manifest["nodes"][::-1]},
                        {"nodes": [dict(node, weight=-1) for node in manifest["nodes"]]}):
            fn.write_json(manifest_path, dict(manifest, **changed))
            run(verify, 2)
        fn.write_json(manifest_path, manifest)
        fn.write_json(bindfile, BINDINGS)
        run(args, 2)
        provisional, _ = fn.configuration_values(request, BINDINGS, numeric, "smoke", provisional=True)
        assert "NML_RIVER_TYPE" not in provisional
        assert fn.audit_values(provisional, fn.RIVER_MODE, provisional=True) == []
        assert fn.audit_values(provisional, fn.RIVER_MODE)
    print("fvcom paired river configuration selftest: PASS")


if __name__ == "__main__":
    main()
