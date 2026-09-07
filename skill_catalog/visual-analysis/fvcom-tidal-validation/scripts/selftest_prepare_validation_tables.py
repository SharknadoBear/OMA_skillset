from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from prepare_validation_tables import prepare


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_dir = root / "model"; observation_root = root / "observations"
        times = pd.date_range("2025-04-01", periods=4, freq="6min", tz="UTC")
        model_dir.mkdir()
        pd.DataFrame({"time": times, "model": [1.0, 2.0, 3.0, 4.0]}).to_csv(model_dir / "wl.csv", index=False)
        pd.DataFrame({"time": times, "model_u": [0.0, 1.0, 2.0, 3.0], "model_v": [3.0, 2.0, 1.0, 0.0]}).to_csv(model_dir / "cur.csv", index=False)
        (observation_root / "water_level").mkdir(parents=True)
        pd.DataFrame({"time": times, "observed": [1.1, 2.1, 3.1, 4.1], "predicted": [1, 2, 3, 4]}).to_csv(
            observation_root / "water_level" / "8771341_noaa_waterlevel.csv", index=False)
        (observation_root / "currents" / "g06010").mkdir(parents=True)
        current_times = times[:3] + pd.Timedelta(minutes=3)
        pd.DataFrame({"time": current_times, "east_m_s": [0.5, 1.5, 2.5], "north_m_s": [2.5, 1.5, 0.5]}).to_csv(
            observation_root / "currents" / "g06010" / "depth_mean.csv", index=False)
        condensation = root / "condensation.json"
        condensation.write_text(json.dumps({"status": "ready", "products": [
            {"station_id": "8771341", "role": "water_level", "path": str(model_dir / "wl.csv")},
            {"station_id": "g06010", "role": "current", "path": str(model_dir / "cur.csv")},
        ]}), encoding="utf-8")
        result = prepare(condensation, observation_root, root / "tables", root / "manifest.json")
        assert result["status"] == "ready" and len(result["products"]) == 2
        current = pd.read_csv(root / "tables" / "g06010_current.csv")
        assert current["model_u"].tolist() == [0.5, 1.5, 2.5]
        assert current["model_v"].tolist() == [2.5, 1.5, 0.5]
    print("FVCOM/NOAA validation table selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
