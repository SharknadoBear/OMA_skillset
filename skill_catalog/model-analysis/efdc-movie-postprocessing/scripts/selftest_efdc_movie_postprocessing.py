#!/usr/bin/env python3
"""Offline EFDC fixed-scale GIF regression tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np
from netCDF4 import Dataset
from PIL import Image, ImageChops

import efdc_movie_postprocessing as movie_module
from efdc_movie_postprocessing import main as movie_main


def validate_schema_instance(schema, value, path="$"):
    if "const" in schema:
        assert value == schema["const"], f"{path}: const mismatch"
    if "enum" in schema:
        assert value in schema["enum"], f"{path}: enum mismatch"
    kind = schema.get("type")
    types = kind if isinstance(kind, list) else [kind] if kind else []
    checks = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
              "string": lambda v: isinstance(v, str), "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool), "null": lambda v: v is None}
    if types:
        assert any(checks[item](value) for item in types), f"{path}: wrong type"
    if isinstance(value, dict):
        for required in schema.get("required", []):
            assert required in value, f"{path}: missing {required}"
        for name, child in schema.get("properties", {}).items():
            if name in value:
                validate_schema_instance(child, value[name], f"{path}.{name}")
    if isinstance(value, list):
        assert len(value) >= schema.get("minItems", 0), f"{path}: too few items"
        if "items" in schema:
            for index, child_value in enumerate(value):
                validate_schema_instance(schema["items"], child_value, f"{path}[{index}]")
    if isinstance(value, (int, float)) and "minimum" in schema:
        assert value >= schema["minimum"], f"{path}: below minimum"
    if isinstance(value, str) and "pattern" in schema:
        assert re.search(schema["pattern"], value), f"{path}: pattern mismatch"


def write_raw(path: Path, start_hour: int, count: int):
    ny, nx = 5, 7
    y, x = np.mgrid[:ny, :nx]
    mask = np.zeros((ny, nx), dtype=np.float32)
    mask[1:4, 1:6] = 5
    mask[0, 0] = -85.48875
    lon = np.where(mask == 5, -81.7 + .02*x + .002*y*y, 0)
    lat = np.where(mask == 5, 29.6 + .025*y + .001*x*y, .1)
    sigma = np.array([.75, 0, .25], dtype=np.float32)
    with Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        for name, size in (("time", count), ("sigma", 3), ("ny", ny), ("nx", nx)):
            ds.createDimension(name, size)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "seconds since 1970-01-01 00:00:00 UTC"
        t[:] = [(start_hour+i)*3600 + 1800 for i in range(count)]
        for name, value in (("lon", lon), ("lat", lat), ("mask", mask), ("depth", np.where(mask == 5, 8, .01))):
            ds.createVariable(name, "f4", ("ny", "nx"))[:] = value
        ds.createVariable("sigma", "f4", ("sigma",))[:] = sigma
        shape = (count, 3, ny, nx)
        for name, standard, factor in (
            ("salt", "sea_water_practical_salinity", 1),
            ("u", "eastward_sea_water_velocity", .1),
            ("v", "northward_sea_water_velocity", .2),
        ):
            values = np.full(shape, -99999, dtype=np.float32)
            for ti in range(count):
                for k in range(3):
                    values[ti, k, mask == 5] = factor * (1 + start_hour + ti + k + x[mask == 5]/10)
            var = ds.createVariable(name, "f4", ("time", "sigma", "ny", "nx"), fill_value=-99999)
            var.missing_value = -99999.0
            var.standard_name = standard
            var[:] = values


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="efdc_movie_selftest_") as td:
        root = Path(td)
        movie_scripts = Path(__file__).resolve().parent
        expected_shared = {
            "efdc_output.py": "442e1071d5c8f8e6f473b4b545d3c08f8b36b85d4f59cc6d9f752dc3c3d0fa10",
            "efdc_map_tools.py": "50d4a9b6a8c64990918cb0428909d54277122eadb427905ea4e55a3f7b70a6d9",
        }
        for shared, expected in expected_shared.items():
            assert hashlib.sha256((movie_scripts / shared).read_bytes()).hexdigest() == expected
        package_root = movie_scripts.parent
        map_candidates = [
            package_root.parent / "efdc-map-postprocessing" / "scripts",
            package_root.parents[2] / "efdc-map-postprocessing" / "prototype" / "efdc-map-postprocessing" / "scripts",
        ]
        map_scripts = next((candidate for candidate in map_candidates if candidate.is_dir()), None)
        if map_scripts is not None:
            for shared in expected_shared:
                assert (map_scripts / shared).read_bytes() == (movie_scripts / shared).read_bytes()
        first, second = root / "b.nc", root / "a.nc"
        write_raw(first, 0, 12)
        write_raw(second, 12, 12)
        gif, report = root / "movie.gif", root / "movie.json"
        args = [
            "gif", "--input", str(second), "--input", str(first), "--variable", "salinity",
            "--layer", "depth_average", "--start", "1970-01-01T00:30:00Z",
            "--end-exclusive", "1970-01-02T00:30:00Z", "--fps", "4",
            "--output", str(gif), "--report", str(report),
        ]
        assert movie_main(args) == 0
        manifest = json.loads(report.read_text())
        schema_path = Path(__file__).resolve().parent.parent / "references" / "movie_manifest.schema.json"
        schema = json.loads(schema_path.read_text())
        assert schema["$schema"].endswith("2020-12/schema")
        validate_schema_instance(schema, manifest)
        assert manifest["selection"]["frame_count"] == 24
        assert manifest["rendering"]["style"] == "wet_cells"
        assert manifest["fixed_color_limits"]["method"] == "full_series_quantiles"
        assert manifest["coverage"]["minimum_finite_wet_fraction"] == 1.0
        assert manifest["output"]["sha256"] == hashlib.sha256(gif.read_bytes()).hexdigest()
        with Image.open(gif) as image:
            assert image.n_frames == 24
            frames = []
            for i in range(image.n_frames):
                image.seek(i)
                frames.append(image.convert("RGB").copy())
            assert all(ImageChops.difference(frames[i-1], frames[i]).getbbox() for i in range(1, len(frames)))
        explicit, explicit_report = root / "explicit.gif", root / "explicit.json"
        assert movie_main([
            "gif", "--input", str(first), "--variable", "current_speed", "--layer", "surface",
            "--vmin", "0", "--vmax", "10", "--output", str(explicit), "--report", str(explicit_report),
        ]) == 0
        assert json.loads(explicit_report.read_text())["fixed_color_limits"]["vmax"] == 10
        try:
            movie_main(["inspect", "--input", str(first), "--output", str(first)])
        except ValueError as exc:
            assert "overwrite" in str(exc)
        else:
            raise AssertionError("inspect collision must fail before writing")
        with Dataset(first):
            pass
        protected = root / "protected.gif"
        protected.write_bytes(b"preexisting-gif")
        original_save = Image.Image.save
        def collapse_save(self, fp, format=None, **params):
            if Path(fp).name == "protected.tmp.gif":
                return original_save(self, fp, format="GIF", save_all=False)
            return original_save(self, fp, format=format, **params)
        Image.Image.save = collapse_save
        try:
            try:
                movie_module.create_gif(
                    [first], variable="salinity", output=protected,
                    report=root / "protected.json", fps=4,
                )
            except RuntimeError as exc:
                assert "Staged GIF contains" in str(exc)
            else:
                raise AssertionError("collapsed staged GIF must fail validation")
        finally:
            Image.Image.save = original_save
        assert protected.read_bytes() == b"preexisting-gif"
        assert not protected.with_name("protected.tmp.gif").exists()
        assert not list(root.glob("efdc_movie_*"))
        print(json.dumps({"status": "pass", "checks": 16, "frames": 24}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
