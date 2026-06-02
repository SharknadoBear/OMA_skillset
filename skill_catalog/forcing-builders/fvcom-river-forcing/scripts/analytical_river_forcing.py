"""
Analytical river-forcing helpers for FVCOM preprocessing.

The functions here are intended to be reusable toolkit pieces, consistent with
the v001 goal of a transferable agent-compatible preprocessing library.  They
combine river discharge, analytical or literature-derived tracer formulas, node
splitting, and FVCOM NetCDF/NML writing.  The Delaware P0-P7 definitions below
are a ready-to-run configuration, not a limitation of the module design.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fvcom_writer import write_river_nc, write_river_nml
from grid_utils import datetime64_to_mjd
from usgs_rivers_fetcher import (
    DELAWARE_NODE_WEIGHTS,
    MP_CONC_FLOC_MP,
    RIVER_NAMES,
    RIVER_NODES,
    SCHUYLKILL_NODE_WEIGHTS,
    compute_floc_concentration,
    compute_nash_sediment_load,
    discharge_to_hourly,
    distribute_flow_to_nodes,
    fetch_usgs_discharge,
)
from microplastic_forcing import (
    build_case_microfiber_summary,
    river_mp_array_from_boundary_values,
)


T_START = "2018-01-01"
T_END = "2020-12-31"
SITE_DELAWARE = "01463500"
SITE_SCHUYLKILL = "01474500"
N_SIGMA = 30
T_RIVER = 25.0
S_RIVER = 0.0
PLASTIC_SOURCE_DRBC = "drbc_microfiber"
PLASTIC_SOURCE_LEGACY = "legacy_constant"


@dataclass(frozen=True)
class CaseSpec:
    """Tracer-class setup for one river-forcing scenario."""

    case: str
    diameters_mm: tuple[float, ...]
    fractions: tuple[float, ...]
    trenton_fractions: tuple[float, ...] | None = None
    schuylkill_fractions: tuple[float, ...] | None = None
    floc_scenario: str = "none"

    @property
    def nsed(self) -> int:
        return len(self.fractions)

    @property
    def has_floc(self) -> bool:
        return self.nsed > 0

    def fractions_for_source(self, source: str) -> tuple[float, ...]:
        """Return class mass fractions for a named upstream river source."""

        source_key = source.lower()
        if source_key in {"trenton", "delaware", "dr", "trenton_dr"}:
            return self.trenton_fractions or self.fractions
        if source_key in {"schuylkill", "sr", "schuylkill_sr"}:
            return self.schuylkill_fractions or self.fractions
        raise ValueError(f"Unknown floc source {source!r}")


DEFAULT_CASE_SPECS: dict[str, CaseSpec] = {
    "P0": CaseSpec("P0", (), ()),
    "P1": CaseSpec("P1", (), ()),
    "P2": CaseSpec(
        "P2",
        (0.030268,),
        (1.00,),
        trenton_fractions=(1.00,),
        schuylkill_fractions=(1.00,),
        floc_scenario="R2 baseline coarse-grained to 1 class",
    ),
    "P3": CaseSpec(
        "P3",
        (0.030268,),
        (1.00,),
        trenton_fractions=(1.00,),
        schuylkill_fractions=(1.00,),
        floc_scenario="R2 baseline coarse-grained to 1 class",
    ),
    "P4": CaseSpec(
        "P4",
        (0.015000, 0.081233),
        (0.769476, 0.230524),
        trenton_fractions=(0.78, 0.22),
        schuylkill_fractions=(0.68, 0.32),
        floc_scenario="R2 baseline coarse-grained to 2 classes",
    ),
    "P5": CaseSpec(
        "P5",
        (0.015000, 0.081233),
        (0.769476, 0.230524),
        trenton_fractions=(0.78, 0.22),
        schuylkill_fractions=(0.68, 0.32),
        floc_scenario="R2 baseline coarse-grained to 2 classes",
    ),
    "P6": CaseSpec(
        "P6",
        (0.015000, 0.070000, 0.130000),
        (0.769476, 0.187367, 0.043157),
        trenton_fractions=(0.78, 0.18, 0.04),
        schuylkill_fractions=(0.68, 0.25, 0.07),
        floc_scenario="R2 baseline 3 classes",
    ),
    "P7": CaseSpec(
        "P7",
        (0.015000, 0.070000, 0.130000),
        (0.769476, 0.187367, 0.043157),
        trenton_fractions=(0.78, 0.18, 0.04),
        schuylkill_fractions=(0.68, 0.25, 0.07),
        floc_scenario="R2 baseline 3 classes",
    ),
}

# Alias used by the Delaware example notebooks.
CASE_SPECS = DEFAULT_CASE_SPECS


def preprocessing_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def case_output_names(case: str) -> tuple[str, str]:
    return (
        f"waterPACT_riv_{case}_2018_2020.nc",
        f"waterPACT_riv_{case}_2018_2020.nml",
    )


def fetch_hourly_discharge(
    data_raw: Path,
    t_start: str = T_START,
    t_end: str = T_END,
) -> tuple[pd.Series, pd.Series]:
    """Fetch/cache and hourly-interpolate the default Delaware river gauges."""

    cache = data_raw / "usgs_discharge"
    cache.mkdir(parents=True, exist_ok=True)

    df_dr = fetch_usgs_discharge(SITE_DELAWARE, t_start, t_end, cache)
    df_sr = fetch_usgs_discharge(SITE_SCHUYLKILL, t_start, t_end, cache)

    hourly_end = (pd.Timestamp(t_end) + pd.Timedelta("23h")).strftime(
        "%Y-%m-%d %H:%M"
    )
    q_dr = discharge_to_hourly(df_dr, t_start, hourly_end)
    q_sr = discharge_to_hourly(df_sr, t_start, hourly_end)
    if not q_dr.index.equals(q_sr.index):
        raise ValueError("Delaware and Schuylkill hourly discharge grids differ")
    return q_dr, q_sr


def build_case_arrays(
    spec: CaseSpec,
    q_dr_hourly: pd.Series,
    q_sr_hourly: pd.Series,
    plastic_source: str = PLASTIC_SOURCE_DRBC,
    drbc_workbook: Path | None = None,
    plastic_config: Path | None = None,
) -> dict[str, object]:
    """Build FVCOM river arrays from hourly gauge discharge and a tracer spec."""

    q_dr = q_dr_hourly.values.astype(np.float64)
    q_sr = q_sr_hourly.values.astype(np.float64)

    flux_dr = distribute_flow_to_nodes(q_dr, DELAWARE_NODE_WEIGHTS)
    flux_sr = distribute_flow_to_nodes(q_sr, SCHUYLKILL_NODE_WEIGHTS)
    flux = np.concatenate([flux_dr, flux_sr], axis=1).astype(np.float32)

    temp = np.full_like(flux, T_RIVER, dtype=np.float32)
    salt = np.full_like(flux, S_RIVER, dtype=np.float32)

    plastic_source = plastic_source.lower()
    plastic_summary = None
    if plastic_source == PLASTIC_SOURCE_LEGACY:
        plastic = np.full_like(flux, MP_CONC_FLOC_MP, dtype=np.float32)
    elif plastic_source == PLASTIC_SOURCE_DRBC:
        plastic_summary = build_case_microfiber_summary(
            spec.case,
            workbook=drbc_workbook,
            plastic_config=plastic_config,
        )
        boundary_values = dict(
            zip(plastic_summary["boundary"], plastic_summary["mp1_g_l"])
        )
        plastic = river_mp_array_from_boundary_values(
            template=flux,
            boundary_values_g_l=boundary_values,
            n_delaware_nodes=len(DELAWARE_NODE_WEIGHTS),
            n_schuylkill_nodes=len(SCHUYLKILL_NODE_WEIGHTS),
        )
    else:
        raise ValueError(
            f"Unknown plastic_source={plastic_source!r}; expected "
            f"{PLASTIC_SOURCE_DRBC!r} or {PLASTIC_SOURCE_LEGACY!r}"
        )

    floc = None
    if spec.has_floc:
        c_dr = compute_floc_concentration(q_dr)
        c_sr = compute_floc_concentration(q_sr)
        frac_dr = np.asarray(spec.fractions_for_source("trenton"), dtype=np.float64)
        frac_sr = np.asarray(spec.fractions_for_source("schuylkill"), dtype=np.float64)
        for label, fractions in (("Trenton", frac_dr), ("Schuylkill", frac_sr)):
            if fractions.size != spec.nsed:
                raise ValueError(
                    f"{spec.case} {label} fractions have {fractions.size} "
                    f"classes, expected {spec.nsed}"
                )
            if not np.isclose(fractions.sum(), 1.0, atol=1e-8):
                raise ValueError(
                    f"{spec.case} {label} fractions sum to {fractions.sum()}"
                )

        floc_dr = c_dr[:, np.newaxis, np.newaxis] * frac_dr[np.newaxis, np.newaxis, :]
        floc_sr = c_sr[:, np.newaxis, np.newaxis] * frac_sr[np.newaxis, np.newaxis, :]
        floc = np.concatenate(
            [
                np.tile(floc_dr, (1, len(DELAWARE_NODE_WEIGHTS), 1)),
                np.tile(floc_sr, (1, len(SCHUYLKILL_NODE_WEIGHTS), 1)),
            ],
            axis=1,
        ).astype(np.float32)

    time_np = q_dr_hourly.index.to_numpy().astype("datetime64[s]")
    time_mjd = datetime64_to_mjd(time_np)
    return {
        "time_mjd": time_mjd,
        "flux": flux,
        "temp": temp,
        "salt": salt,
        "floc": floc,
        "plastic": plastic,
        "plastic_summary": plastic_summary,
        "plastic_source": plastic_source,
    }


def write_case_forcing(
    case: str,
    data_raw: Path | None = None,
    data_processed: Path | None = None,
    plastic_source: str = PLASTIC_SOURCE_DRBC,
    drbc_workbook: Path | None = None,
    plastic_config: Path | None = None,
) -> dict[str, object]:
    """Write one configured river-forcing scenario to FVCOM NetCDF and NML."""

    case = case.upper()
    if case not in DEFAULT_CASE_SPECS:
        raise KeyError(
            f"Unknown case {case}; expected one of {sorted(DEFAULT_CASE_SPECS)}"
        )

    base = preprocessing_dir()
    data_raw = data_raw or base / "data_raw"
    data_processed = data_processed or base / "data_processed"
    data_processed.mkdir(parents=True, exist_ok=True)

    spec = DEFAULT_CASE_SPECS[case]
    q_dr, q_sr = fetch_hourly_discharge(data_raw)
    arrays = build_case_arrays(
        spec,
        q_dr,
        q_sr,
        plastic_source=plastic_source,
        drbc_workbook=drbc_workbook,
        plastic_config=plastic_config,
    )

    nc_name, nml_name = case_output_names(case)
    nc_out = data_processed / nc_name
    nml_out = data_processed / nml_name

    write_river_nc(
        out_path=nc_out,
        river_names=RIVER_NAMES,
        time_mjd=arrays["time_mjd"],
        flux=arrays["flux"],
        temp=arrays["temp"],
        salt=arrays["salt"],
        floc=arrays["floc"],
        plastic=arrays["plastic"],
        info1="Delaware River Estuary",
        info2=f"Analytical river forcing {case}",
        casename=f"waterPACT_{case}",
    )
    write_river_nml(
        out_path=nml_out,
        river_names=RIVER_NAMES,
        river_nodes=RIVER_NODES,
        nc_file=nc_name,
        n_sigma=N_SIGMA,
    )

    floc = arrays["floc"]
    plastic_summary = arrays["plastic_summary"]
    q_dr_vals = q_dr.values.astype(np.float64)
    q_sr_vals = q_sr.values.astype(np.float64)
    summary = {
        "case": case,
        "nc": nc_out,
        "nml": nml_out,
        "ntime": int(len(arrays["time_mjd"])),
        "floc_classes": 0 if floc is None else int(floc.shape[2]),
        "floc_scenario": spec.floc_scenario,
        "floc_diameters_mm": spec.diameters_mm,
        "floc_fractions_global": spec.fractions,
        "floc_fractions_trenton": spec.fractions_for_source("trenton")
        if spec.has_floc
        else (),
        "floc_fractions_schuylkill": spec.fractions_for_source("schuylkill")
        if spec.has_floc
        else (),
        "mp_classes": 1,
        "plastic_source": arrays["plastic_source"],
        "nash_load_dr_t_day": (
            float(np.nanmin(compute_nash_sediment_load(q_dr_vals))),
            float(np.nanmean(compute_nash_sediment_load(q_dr_vals))),
            float(np.nanmax(compute_nash_sediment_load(q_dr_vals))),
        ),
        "nash_load_sr_t_day": (
            float(np.nanmin(compute_nash_sediment_load(q_sr_vals))),
            float(np.nanmean(compute_nash_sediment_load(q_sr_vals))),
            float(np.nanmax(compute_nash_sediment_load(q_sr_vals))),
        ),
    }
    if plastic_summary is not None:
        by_boundary = plastic_summary.set_index("boundary")
        summary.update(
            {
                "mp1_trenton_dr_g_l": float(by_boundary.loc["Trenton_DR", "mp1_g_l"]),
                "mp1_schuylkill_sr_g_l": float(
                    by_boundary.loc["Schuylkill_SR", "mp1_g_l"]
                ),
                "drbc_extrapolation_factor": float(
                    plastic_summary["sr_extrapolation_factor"].iloc[0]
                ),
                "particle_mass_kg": float(plastic_summary["particle_mass_kg"].iloc[0]),
                "density_kg_m3": float(plastic_summary["density_kg_m3"].iloc[0]),
            }
        )
    else:
        summary.update(
            {
                "mp1_trenton_dr_g_l": MP_CONC_FLOC_MP,
                "mp1_schuylkill_sr_g_l": MP_CONC_FLOC_MP,
                "drbc_extrapolation_factor": np.nan,
                "particle_mass_kg": np.nan,
                "density_kg_m3": np.nan,
            }
        )
    return summary


def write_all_cases(
    cases: list[str] | None = None,
    plastic_source: str = PLASTIC_SOURCE_DRBC,
) -> list[dict[str, object]]:
    """Write all requested scenarios from the default case registry."""

    cases = cases or list(DEFAULT_CASE_SPECS)
    return [write_case_forcing(case, plastic_source=plastic_source) for case in cases]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="*",
        default=list(DEFAULT_CASE_SPECS),
        help="Default-registry cases to write, for example P0 P1 P2.",
    )
    parser.add_argument(
        "--plastic-source",
        choices=[PLASTIC_SOURCE_DRBC, PLASTIC_SOURCE_LEGACY],
        default=PLASTIC_SOURCE_DRBC,
        help="Microplastic forcing source.",
    )
    args = parser.parse_args()
    for summary in write_all_cases(
        [case.upper() for case in args.cases],
        plastic_source=args.plastic_source,
    ):
        print(
            "{case}: {ntime} steps, floc classes={floc_classes}, "
            "mp classes={mp_classes}, mp DR={mp1_trenton_dr_g_l:.3e}, "
            "mp SR={mp1_schuylkill_sr_g_l:.3e}, nc={nc}".format(**summary)
        )


if __name__ == "__main__":
    main()
