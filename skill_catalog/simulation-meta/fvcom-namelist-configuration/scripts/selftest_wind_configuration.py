"""Regress the matched native surface settings, cold windows and specialist dispatch."""
from pathlib import Path
import tempfile
import fvcom_namelist as fn
from selftest_fvcom_namelist import BINDINGS


def main():
    for mode in ("barotropic_tide", fn.WIND_MODE):
        request = {"case_id": "wind_pair", "profile": fn.WIND_PROFILE, "mode": mode}
        for stage, expected_end in (("smoke", "2025-03-19"), ("production", "2025-04-15")):
            values, timing = fn.configuration_values(request, BINDINGS, {"extstep_seconds": 1.2, "isplit": 1}, stage)
            assert timing["start_utc"] == "2025-03-18T00:00:00Z"
            assert timing["end_utc"] == expected_end + "T00:00:00Z"
            assert fn.audit_stage_values(values, stage, fn.WIND_PROFILE) == []
            assert fn.audit_values(values, mode, profile=fn.WIND_PROFILE) == []
            assert values["NML_RIVER_TYPE"]["RIVER_NUMBER"] == "0"
            surface = values["NML_SURFACE_FORCING"]
            assert surface["AIRPRESSURE_ON"] == "T"
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                nml = root/"case.nml"
                fn.write_overlay(nml, values)
                if mode == fn.WIND_MODE:
                    assert surface["WIND_ON"] == "T" and surface["WIND_TYPE"] == "'stress'"
                    # Verify both specialist imports through the normal source or installed layout.
                    try:
                        fn.verify_wind_stage(request, nml, root)
                    except FileNotFoundError as exc:
                        assert "atmosphere_manifest.json" in str(exc), str(exc)
                    else:
                        raise AssertionError("wind accepted missing atmospheric data")
                    surface["AIRPRESSURE_FILE"] = "'mismatch.nc'"
                    assert fn.audit_values(values, mode, profile=fn.WIND_PROFILE)
                else:
                    assert fn.verify_wind_stage(request, nml, root) is None
                    assert surface["WIND_ON"] == "F" and float(surface["AIRPRESSURE_VALUE"]) == 101325
                    surface["AIRPRESSURE_VALUE"] = "0"
                    assert fn.audit_values(values, mode, profile=fn.WIND_PROFILE)
    print("fvcom wind and constant-control configuration selftest: PASS")


if __name__ == "__main__":
    main()
