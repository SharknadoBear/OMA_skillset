from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import netCDF4 as nc4
import numpy as np

from condense_fvcom_station import condense


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        mapping = root / "mapping.json"
        mapping.write_text(json.dumps({
            "status": "ready",
            "stations": [
                {"station_id": "g06010", "role": "current", "cell_id": 8},
                {"station_id": "8771341", "role": "water_level", "cell_id": 9},
            ],
        }), encoding="utf-8")
        namelist = root / "run.nml"
        namelist.write_text(
            "&NML_CASE\n START_DATE = '2025-04-01 00:00:00',\n END_DATE = '2025-04-01 00:12:00',\n/\n"
            "&NML_INTEGRATION\n EXTSTEP_SECONDS = 1.2,\n ISPLIT = 2,\n/\n",
            encoding="utf-8",
        )
        with namelist.open("a", encoding="utf-8") as stream:
            stream.write("STARTUP_TYPE = 'hotstart',\nSTARTUP_FILE = 'startup.nc',\nINPUT_DIR = '.',\n")
        with nc4.Dataset(root / "startup.nc", "w") as ds:
            ds.createDimension("time", 1); ds.createDimension("DateStrLen", 26)
            ds.createVariable("iint", "i4", ("time",))[:] = [500000]
            ds.createVariable("Itime", "i4", ("time",))[:] = [60766]
            ds.createVariable("Itime2", "i4", ("time",))[:] = [0]
            ds.createVariable("Times", "S1", ("time", "DateStrLen"))[0] = np.asarray(list("2025-04-01T00:00:00.000000"), dtype="S1")
        station = root / "station.nc"
        with nc4.Dataset(station, "w") as ds:
            ds.createDimension("time", 3); ds.createDimension("station", 2); ds.createDimension("namelen", 20)
            names = ds.createVariable("name_station", "S1", ("station", "namelen"))
            for index, value in enumerate(("g06010", "8771341")):
                names[index, :] = np.asarray(list(value.ljust(20)), dtype="S1")
            iint = ds.createVariable("iint", "i4", ("time",)); iint[:] = [500000, 500150, 500300]
            time = ds.createVariable("time", "f4", ("time",)); time.units = "days since 1858-11-17 00:00:00"
            time[:] = np.asarray([60766.0, 60766.0 + 360 / 86400, 60766.0 + 720 / 86400], dtype="f4")
            for field, values in {
                "zeta": [[0, 1], [0, 2], [0, 3]],
                "ua": [[1, 0], [2, 0], [3, 0]],
                "va": [[-1, 0], [-2, 0], [-3, 0]],
            }.items():
                variable = ds.createVariable(field, "f4", ("time", "station")); variable[:] = values
        manifest = condense([station], mapping, namelist, root / "out", root / "manifest.json")
        assert manifest["status"] == "ready" and manifest["time_reconstruction"]["iint_origin"] == 500000
        assert manifest["time_reconstruction"]["isplit"] == 2
        assert manifest["time_reconstruction"]["internal_step_seconds"] == 2.4
        assert manifest["time_reconstruction"]["end_utc"].startswith("2025-04-01T00:12:00")
        water = next(item for item in manifest["products"] if item["role"] == "water_level")
        assert [item["station_id"] for item in manifest["products"]] == ["g06010", "8771341"]
        with Path(water["path"]).open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert rows[-1]["time"].startswith("2025-04-01T00:12:00") and rows[-1]["model"] == "3"
        assert manifest["warnings"], "float32 MJD quantization should be disclosed"
        mixed = json.loads(mapping.read_text())
        mixed['stations'][0].update(role='model_diagnostic', validation_eligible=False)
        mixed['stations'][1]['spatial_mapping'] = {'method':'nearest_wet_cell_centroid','distance_m':123.0}
        mapping.write_text(json.dumps(mixed))
        diagnostic = condense([station], mapping, namelist, root/'mixed', root/'mixed_manifest.json')
        assert [p['station_id'] for p in diagnostic['products']] == ['8771341']
        assert [p['station_id'] for p in diagnostic['diagnostic_products']] == ['g06010']
        assert diagnostic['products'][0]['spatial_mapping']['distance_m'] == 123.0
        mixed['stations'][0]['validation_eligible'] = True
        mapping.write_text(json.dumps(mixed))
        try: condense([station], mapping, namelist, root/'invalid_diagnostic', root/'invalid_manifest.json')
        except ValueError as error: assert 'unsupported station role' in str(error)
        else: raise AssertionError('A diagnostic marked eligible for comparison was accepted')
        mapping.write_text(json.dumps({
            "status": "ready",
            "stations": [
                {"station_id": "8771341", "role": "water_level", "cell_id": 9},
                {"station_id": "g06010", "role": "current", "cell_id": 8},
            ],
        }), encoding="utf-8")
        try:
            condense([station], mapping, namelist, root / "reordered", root / "reordered_manifest.json")
        except ValueError as error:
            assert "IDs/order" in str(error)
        else:
            raise AssertionError("a permuted station mapping must be rejected")
    print("fvcom station condensation selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
