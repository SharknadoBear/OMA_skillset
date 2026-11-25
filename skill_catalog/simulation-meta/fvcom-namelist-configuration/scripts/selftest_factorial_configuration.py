"""Check independent river/wind switches and matched windows in all four factorial cells."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import tempfile

import fvcom_namelist as fn
from selftest_fvcom_namelist import BINDINGS

GAUGE = {"provider": "usgs", "station_id": "14246900", "parameter_code": "72137"}


def main():
    settings = {}
    river = {"river_number": 1, "files": {"namelist": {"path": "river.nml"}}}
    for mode in ("barotropic_tide", fn.WIND_MODE, fn.RIVER_MODE, fn.COMBINED_MODE):
        req = {"case_id": "test", "profile": fn.FACTORIAL_PROFILE, "mode": mode}
        bindings = dict(BINDINGS)
        if fn.has_river(req):
            req["river"] = GAUGE
            bindings["river_manifest"] = "river_manifest.json"
        for stage, duration in (("smoke", 1), ("production", 28)):
            v, timing = fn.configuration_values(req, bindings, {"extstep_seconds":1.2, "isplit":1}, stage,
                                                river=river if fn.has_river(req) else None)
            start, end, _, hot = fn.stage_times(req, stage)
            assert timing["start_utc"] == "2025-03-18T00:00:00Z" and not hot and end-start == timedelta(days=duration)
            assert not fn.audit_stage_values(v, stage, fn.FACTORIAL_PROFILE)
            assert not fn.audit_values(v, mode, river if fn.has_river(req) else None, profile=fn.FACTORIAL_PROFILE)
            settings[mode] = v
            if mode == fn.COMBINED_MODE:
                broken = deepcopy(v)
                broken["NML_SURFACE_FORCING"]["AIRPRESSURE_KIND"] = "'constant'"
                assert fn.audit_values(broken, mode, river, profile=fn.FACTORIAL_PROFILE)
                broken = deepcopy(v)
                broken["NML_RIVER_TYPE"]["RIVER_NUMBER"] = "0"
                assert fn.audit_values(broken, mode, river, profile=fn.FACTORIAL_PROFILE)
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    nml = root/"case.nml"
                    fn.write_overlay(nml,v)
                    try:
                        fn.verify_wind_stage(req,nml,root)
                    except FileNotFoundError as exc:
                        assert "river_manifest.json" in str(exc)
                    else:
                        raise AssertionError("combined stage ignored missing discharge")
    assert settings[fn.COMBINED_MODE]["NML_SURFACE_FORCING"] == settings[fn.WIND_MODE]["NML_SURFACE_FORCING"]
    assert settings[fn.RIVER_MODE]["NML_SURFACE_FORCING"] == settings["barotropic_tide"]["NML_SURFACE_FORCING"]
    assert settings[fn.COMBINED_MODE]["NML_RIVER_TYPE"] == settings[fn.RIVER_MODE]["NML_RIVER_TYPE"]
    assert fn.resolve_request({"mode":fn.COMBINED_MODE,"river":GAUGE})["period"]["spinup_days"] == 14
    for wrong in ({"profile":fn.FACTORIAL_PROFILE,"mode":fn.COMBINED_MODE},
                  {"profile":fn.FACTORIAL_PROFILE,"mode":fn.WIND_MODE,"river":GAUGE},
                  {"profile":fn.WIND_PROFILE,"mode":fn.COMBINED_MODE,"river":GAUGE},
                  {"profile":fn.FACTORIAL_PROFILE,"mode":fn.RIVER_MODE,"river":dict(GAUGE, station_id="999")}):
        try:
            fn.resolve_request(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError(wrong)
    print("factorial configuration, paired switches, source/profile validation and cold dates: PASS")


if __name__ == "__main__":
    main()
