"""Exercise common NOAA support, signed response, UTC peaks, and the retained-grid request."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from paired_wind_validation import station_scores
from fvcom_profiles import WIND_PROFILE, WIND_MODE, resolve_request
from fvcom_run_control import init_project


def main():
    clock = pd.date_range("2025-04-01", periods=7, freq="6min", tz="UTC")
    tide = np.array([0., 1., 2., 1., 0., -1., 0.])
    observed_residual = np.array([0., 1., 2., 0., 0., 0., 0.])
    levels = pd.DataFrame({"control":tide, "wind":tide+np.roll(observed_residual, 1),
                           "observed":tide+observed_residual, "predicted":tide}, index=clock)
    levels.loc[clock[-1],"observed"] = np.nan
    stats = station_scores(levels)
    assert stats["residual_samples"] == 6 and stats["wind"]["peak_time_error_hours"] == .1
    assert np.isclose(stats["signed_mean_response_m"], 3/7)
    assert np.isclose(stats["control"]["total"]["rmse_m"], np.sqrt(5/6))
    assert stats["control"]["residual"]["correlation"] is None
    levels["observed"] = np.nan
    assert station_scores(levels)["residual_samples"] == 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for mode in ("barotropic_tide", WIND_MODE):
            request = {"schema":"simulation_request_v1","case_id":mode,"profile":WIND_PROFILE,"mode":mode}
            reqfile = root/(mode+".json")
            reqfile.write_text(json.dumps(request))
            manifest = init_project(reqfile, root/mode)
            assert manifest["grid_order"] == ["retained_columbia"]
            assert resolve_request(request)["period"]["spinup_days"] == 14
        for changed in ({"period":{"spinup_days":3}}, {"grid_policy":"fresh_only"},
                        {"mode":"barotropic-tide-river"}, {"river":{"provider":"usgs"}}):
            try:
                resolve_request({**request, **changed})
            except ValueError:
                continue
            raise AssertionError(f"contradictory wind profile accepted: {changed}")
    print("PASS matched wind profile, observed residuals, timing and signed response")


if __name__ == "__main__":
    main()
