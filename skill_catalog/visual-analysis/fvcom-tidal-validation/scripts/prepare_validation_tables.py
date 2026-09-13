from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from water_level_support import water_support


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def read_time_csv(path: Path, required: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in ["time", *required] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time").drop_duplicates("time")
    if not frame["time"].is_monotonic_increasing:
        raise ValueError(f"{path} has non-monotonic timestamps")
    return frame


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    output["time"] = output["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, lineterminator="\n")


def prepare(condensation_manifest: Path, observation_root: Path, output_dir: Path,
            manifest_path: Path, station_inventory_path: Path | None = None) -> dict[str, Any]:
    condensation = json.loads(condensation_manifest.read_text(encoding="utf-8-sig"))
    if condensation.get("status") != "ready":
        raise ValueError("condensation manifest is not ready")
    output_dir.mkdir(parents=True, exist_ok=True)
    products: list[dict[str, Any]] = []
    warnings: list[str] = []
    for model_product in condensation.get("products", []):
        station_id = str(model_product["station_id"])
        role = str(model_product["role"])
        model_path = Path(model_product["path"])
        support = None
        if role == "water_level":
            observation_path = observation_root / "water_level" / f"{station_id}_noaa_waterlevel.csv"
            model = read_time_csv(model_path, ["model"])
            observation = read_time_csv(observation_path, ["observed", "predicted"])
            combined = model.merge(observation[["time", "observed", "predicted"]], on="time", how="inner")
            # Preserve the two reference supports independently. Missing diagnostic
            # observations must not erase valid astronomical prediction samples.
            support = water_support(combined, station_id, station_inventory_path)
            combined = combined.loc[np.isfinite(combined['model']) &
                (np.isfinite(combined['predicted']) | np.isfinite(combined['observed']))].copy()
            if support['observed']['status'] != 'available':
                warnings.append(f"{station_id}: total-water observation diagnostic {support['observed']['status']}; astronomical predictions remain primary.")
            output_path = output_dir / f"{station_id}_water.csv"
            method = "exact_utc_timestamp_inner_join"
        elif role == "current":
            observation_path = observation_root / "currents" / station_id / "depth_mean.csv"
            model = read_time_csv(model_path, ["model_u", "model_v"]).dropna(subset=["model_u", "model_v"])
            observation = read_time_csv(observation_path, ["east_m_s", "north_m_s"]).dropna(subset=["east_m_s", "north_m_s"])
            model_ns = model["time"].astype("int64").to_numpy()
            gaps = np.diff(model_ns) / 1.0e9
            if len(gaps) and float(np.max(gaps)) > 720.0:
                raise ValueError(f"{station_id} model current has a gap longer than 720 seconds")
            obs_ns = observation["time"].astype("int64").to_numpy()
            inside = (obs_ns >= model_ns[0]) & (obs_ns <= model_ns[-1])
            observation = observation.loc[inside].copy()
            obs_ns = obs_ns[inside]
            if len(observation) < 3:
                raise ValueError(f"{station_id} has fewer than three current observations inside the model period")
            combined = pd.DataFrame({
                "time": observation["time"].to_numpy(),
                "model_u": np.interp(obs_ns, model_ns, model["model_u"].to_numpy(dtype=float)),
                "model_v": np.interp(obs_ns, model_ns, model["model_v"].to_numpy(dtype=float)),
                "observed_u": observation["east_m_s"].to_numpy(dtype=float),
                "observed_v": observation["north_m_s"].to_numpy(dtype=float),
            })
            output_path = output_dir / f"{station_id}_current.csv"
            method = "linear_model_interpolation_to_noaa_profile_midpoints"
            warnings.append(f"{station_id}: FVCOM six-minute vectors were linearly interpolated to NOAA profile timestamps (normally minute 03/09/15...).")
        else:
            raise ValueError(f"unsupported condensed role {role!r}")
        write_frame(combined, output_path)
        products.append({
            "station_id": station_id, "role": role, "path": str(output_path), "sha256": sha256(output_path),
            "spatial_mapping": model_product.get("spatial_mapping"),
            "reference_support": support,
            "rows": int(len(combined)), "coverage_start": combined["time"].iloc[0].isoformat(),
            "coverage_end": combined["time"].iloc[-1].isoformat(), "alignment_method": method,
            "model_source": str(model_path), "model_source_sha256": sha256(model_path),
            "observation_source": str(observation_path), "observation_source_sha256": sha256(observation_path),
        })
    if not products:
        raise ValueError("condensation manifest contains no products")
    result = {
        "schema": "fvcom_noaa_validation_tables_v1", "status": "ready", "generated_at": utcnow(),
        "condensation_manifest": str(condensation_manifest), "condensation_manifest_sha256": sha256(condensation_manifest),
        "observation_root": str(observation_root), "products": products, "warnings": warnings,
        "station_inventory": str(station_inventory_path) if station_inventory_path else None,
        "station_inventory_sha256": sha256(station_inventory_path) if station_inventory_path else None,
        "blocking_reasons": [], "resume_token": f"validation_tables:{sha256(condensation_manifest)}",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Join condensed FVCOM station series with NOAA validation records")
    parser.add_argument("--condensation-manifest", required=True)
    parser.add_argument("--observation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--station-inventory")
    args = parser.parse_args()
    result = prepare(Path(args.condensation_manifest), Path(args.observation_root), Path(args.output_dir), Path(args.manifest), Path(args.station_inventory) if args.station_inventory else None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
