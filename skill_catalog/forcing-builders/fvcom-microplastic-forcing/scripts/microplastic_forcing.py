"""
Microplastic concentration helpers for river-forcing setup.

This module keeps the DRBC measurement parsing, SR supplementary size-spectrum
extrapolation, and particle number-to-mass conversion separate from any one
FVCOM case.  The Delaware P0-P7 notebooks use the defaults here, but the helper
functions are written so other river-boundary studies can pass their own
workbooks, particle settings, and source/target size ranges.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FT3_TO_M3 = 0.028316846592
PARTICLES_PER_FT3_TO_M3 = 1.0 / FT3_TO_M3
LEGACY_MP_CONC_FLOC_MP = 100.0 * 35.31 * 4.97e-10

DRBC_TRENTON_NAME = "Delaware River - Zone 1E Surface"
DRBC_SCHUYLKILL_NAME = "Schuylkill River Surface"

SR_PDF_A = 0.0016
SR_PDF_BETA1 = 1.42
SR_PDF_BETA2 = -3.02
SR_PDF_X0_UM = 15.0
DEFAULT_SOURCE_RANGE_UM = (50.0, 2000.0)
DEFAULT_TARGET_RANGE_UM = (0.1, 100.0)


DRBC_COLUMNS = [
    "no",
    "datetime",
    "name",
    "total_concentration_particles_ft3",
    "shape_fiber",
    "shape_fiber_bundle",
    "shape_film",
    "shape_fragment",
    "shape_sphere",
    "size_1000_2000_um",
    "size_500_1000_um",
    "size_250_500_um",
    "size_100_250_um",
    "size_50_100_um",
    "type_acr",
    "type_azl",
    "type_ply",
    "type_cel",
    "type_kev",
    "type_nyl",
    "type_pbt",
    "type_pe",
    "type_phx",
    "type_pp",
    "type_ps",
    "type_ptfe",
    "type_pur",
    "type_pvc",
    "type_pvs",
    "type_ray",
    "sampling_location_unk",
    "longitude",
    "latitude",
]


@dataclass(frozen=True)
class PlasticParticle:
    """Single model particle geometry and density."""

    name: str
    a_axis_mm: float
    b_axis_mm: float
    c_axis_mm: float
    density_kg_m3: float

    @property
    def volume_m3(self) -> float:
        """Ellipsoid volume using configured full axis lengths."""

        a = self.a_axis_mm * 1.0e-3 / 2.0
        b = self.b_axis_mm * 1.0e-3 / 2.0
        c = self.c_axis_mm * 1.0e-3 / 2.0
        return 4.0 / 3.0 * math.pi * a * b * c

    @property
    def mass_kg(self) -> float:
        return self.volume_m3 * self.density_kg_m3


def project_root() -> Path:
    """Return the WaterPACT project root for this repository layout."""

    return Path(__file__).resolve().parents[3]


def default_drbc_workbook() -> Path:
    return project_root() / "Resources" / "DRBC_data.xlsx"


def default_plastic_config(case: str) -> Path:
    return project_root() / "Workspace" / "INPUT" / case.upper() / "generic_plastic.inp"


def load_drbc_microplastic_data(workbook: str | Path) -> pd.DataFrame:
    """
    Load the DRBC microfiber workbook and normalize its two-row header.

    The workbook stores group labels in row 1 and detailed labels in row 2.
    This helper returns one tidy row per observation with numeric fractions
    converted where possible.
    """

    workbook = Path(workbook)
    raw = pd.read_excel(workbook, sheet_name=0, header=None)
    if raw.shape[1] != len(DRBC_COLUMNS):
        raise ValueError(
            f"Expected {len(DRBC_COLUMNS)} columns in {workbook}, got {raw.shape[1]}"
        )

    df = raw.iloc[2:].copy()
    df.columns = DRBC_COLUMNS
    df = df.dropna(how="all").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c not in {"datetime", "name"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["name"] = df["name"].astype(str).str.strip()
    return df


def _normalize_site_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("-Zone", "- Zone")).strip().lower()


def select_drbc_boundary_rows(
    df: pd.DataFrame,
    trenton_name: str = DRBC_TRENTON_NAME,
    schuylkill_name: str = DRBC_SCHUYLKILL_NAME,
) -> pd.DataFrame:
    """Select direct DRBC observations for the Delaware and Schuylkill boundaries."""

    wanted = {
        "Trenton_DR": _normalize_site_name(trenton_name),
        "Schuylkill_SR": _normalize_site_name(schuylkill_name),
    }
    normalized = df["name"].map(_normalize_site_name)
    rows = []
    for boundary, site_name in wanted.items():
        match = df.loc[normalized == site_name].copy()
        if match.empty:
            raise KeyError(f"Could not find DRBC row for {boundary}: {site_name}")
        if len(match) > 1:
            match = match.sort_values(["datetime", "no"]).head(1)
        match = match.head(1).copy()
        match.insert(0, "boundary", boundary)
        rows.append(match)
    return pd.concat(rows, ignore_index=True)


def sr_size_pdf_um(
    x_um: np.ndarray | float,
    a: float = SR_PDF_A,
    beta1: float = SR_PDF_BETA1,
    beta2: float = SR_PDF_BETA2,
    x0_um: float = SR_PDF_X0_UM,
) -> np.ndarray:
    """Evaluate SR Supplement Eq. S1/S2 for particle size in micrometers."""

    x = np.asarray(x_um, dtype=np.float64)
    if np.any(x <= 0.0):
        raise ValueError("Particle size must be positive")
    x_star = np.log(x)
    x0_star = math.log(x0_um)
    exponent = (
        (beta1 + beta2) * x_star
        - beta2 * x0_star
        + beta2 * np.log1p(np.exp(-(x_star - x0_star)))
        - beta2
    )
    return a * np.exp(exponent)


def integrate_sr_size_pdf(size_range_um: tuple[float, float]) -> float:
    """Integrate the SR size PDF over a size range in micrometers."""

    lo, hi = size_range_um
    if lo <= 0.0 or hi <= lo:
        raise ValueError(f"Invalid size range: {size_range_um}")
    try:
        from scipy.integrate import quad

        value, _ = quad(
            lambda x: float(sr_size_pdf_um(x)),
            lo,
            hi,
            epsabs=1.0e-12,
            epsrel=1.0e-9,
            limit=200,
        )
        return float(value)
    except Exception:
        grid = np.geomspace(lo, hi, 20000)
        return float(np.trapz(sr_size_pdf_um(grid), grid))


def sr_size_extrapolation_factor(
    source_range_um: tuple[float, float] = DEFAULT_SOURCE_RANGE_UM,
    target_range_um: tuple[float, float] = DEFAULT_TARGET_RANGE_UM,
) -> float:
    """Return target/source number-concentration scaling from SR Eq. S1."""

    source = integrate_sr_size_pdf(source_range_um)
    target = integrate_sr_size_pdf(target_range_um)
    if source <= 0.0:
        raise ValueError("Source size integral must be positive")
    return target / source


def microfiber_number_concentration_particles_m3(
    row: pd.Series,
    extrapolation_factor: float,
    shape_column: str = "shape_fiber",
) -> tuple[float, float]:
    """
    Convert one DRBC row to source and extrapolated microfiber number concentration.

    Returns
    -------
    source_particles_m3, target_particles_m3
    """

    total_ft3 = float(row["total_concentration_particles_ft3"])
    shape_fraction = float(row[shape_column])
    source = total_ft3 * shape_fraction * PARTICLES_PER_FT3_TO_M3
    target = source * extrapolation_factor
    return source, target


def _parse_scalar_config(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("!", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip().strip(",")
    return values


def _as_float(value: str) -> float:
    first = re.split(r"[\s,]+", str(value).strip())[0]
    return float(first)


def parse_plastic_particle_config(path: str | Path) -> PlasticParticle:
    """Parse the single-particle FVCOM-MP plastic configuration used by P0-P7."""

    values = _parse_scalar_config(path)
    required = ["PLAST_AAXI", "PLAST_BAXI", "PLAST_CAXI", "PLAST_PRHO"]
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError(f"Missing plastic config keys in {path}: {missing}")
    return PlasticParticle(
        name=values.get("PLAST_NAME", "mp1").split()[0],
        a_axis_mm=_as_float(values["PLAST_AAXI"]),
        b_axis_mm=_as_float(values["PLAST_BAXI"]),
        c_axis_mm=_as_float(values["PLAST_CAXI"]),
        density_kg_m3=_as_float(values["PLAST_PRHO"]),
    )


def number_to_mass_concentration_g_l(
    number_concentration_particles_m3: float | np.ndarray,
    particle_mass_kg: float,
) -> np.ndarray:
    """Convert number/m3 to g/L; numerically this is kg/m3."""

    return np.asarray(number_concentration_particles_m3, dtype=np.float64) * particle_mass_kg


def build_drbc_microfiber_summary(
    workbook: str | Path,
    plastic_particle: PlasticParticle,
    source_range_um: tuple[float, float] = DEFAULT_SOURCE_RANGE_UM,
    target_range_um: tuple[float, float] = DEFAULT_TARGET_RANGE_UM,
    shape_column: str = "shape_fiber",
) -> pd.DataFrame:
    """Compute boundary-specific microfiber number and mass concentrations."""

    df = load_drbc_microplastic_data(workbook)
    anchors = select_drbc_boundary_rows(df)
    factor = sr_size_extrapolation_factor(source_range_um, target_range_um)

    rows = []
    for _, row in anchors.iterrows():
        source_n, target_n = microfiber_number_concentration_particles_m3(
            row, factor, shape_column=shape_column
        )
        rows.append(
            {
                "boundary": row["boundary"],
                "drbc_name": row["name"],
                "datetime": row["datetime"],
                "total_concentration_particles_ft3": row[
                    "total_concentration_particles_ft3"
                ],
                "fiber_fraction": row[shape_column],
                "source_range_um": f"{source_range_um[0]:g}-{source_range_um[1]:g}",
                "target_range_um": f"{target_range_um[0]:g}-{target_range_um[1]:g}",
                "sr_extrapolation_factor": factor,
                "source_microfiber_particles_m3": source_n,
                "target_microfiber_particles_m3": target_n,
                "particle_name": plastic_particle.name,
                "a_axis_mm": plastic_particle.a_axis_mm,
                "b_axis_mm": plastic_particle.b_axis_mm,
                "c_axis_mm": plastic_particle.c_axis_mm,
                "density_kg_m3": plastic_particle.density_kg_m3,
                "particle_mass_kg": plastic_particle.mass_kg,
                "mp1_g_l": float(
                    number_to_mass_concentration_g_l(target_n, plastic_particle.mass_kg)
                ),
                "legacy_mp1_g_l": LEGACY_MP_CONC_FLOC_MP,
            }
        )
    return pd.DataFrame(rows)


def build_case_microfiber_summary(
    case: str,
    workbook: str | Path | None = None,
    plastic_config: str | Path | None = None,
) -> pd.DataFrame:
    """Compute DRBC microfiber forcing summary for one configured model case."""

    case = case.upper()
    workbook = Path(workbook) if workbook is not None else default_drbc_workbook()
    plastic_config = (
        Path(plastic_config) if plastic_config is not None else default_plastic_config(case)
    )
    particle = parse_plastic_particle_config(plastic_config)
    summary = build_drbc_microfiber_summary(workbook, particle)
    summary.insert(0, "case", case)
    summary.insert(1, "plastic_config", str(plastic_config))
    return summary


def boundary_mp_concentrations_for_case(
    case: str,
    workbook: str | Path | None = None,
    plastic_config: str | Path | None = None,
) -> dict[str, float]:
    """Return ``{'Trenton_DR': value, 'Schuylkill_SR': value}`` in g/L."""

    summary = build_case_microfiber_summary(
        case=case, workbook=workbook, plastic_config=plastic_config
    )
    return dict(zip(summary["boundary"], summary["mp1_g_l"]))


def river_mp_array_from_boundary_values(
    template: np.ndarray,
    boundary_values_g_l: dict[str, float],
    n_delaware_nodes: int,
    n_schuylkill_nodes: int,
) -> np.ndarray:
    """Build an FVCOM ``mp1`` array from Trenton and Schuylkill boundary values."""

    ntime = template.shape[0]
    dr = np.full(
        (ntime, n_delaware_nodes),
        float(boundary_values_g_l["Trenton_DR"]),
        dtype=np.float32,
    )
    sr = np.full(
        (ntime, n_schuylkill_nodes),
        float(boundary_values_g_l["Schuylkill_SR"]),
        dtype=np.float32,
    )
    return np.concatenate([dr, sr], axis=1)


def combine_case_summaries(cases: Iterable[str]) -> pd.DataFrame:
    """Compute microfiber summaries for multiple case names."""

    return pd.concat(
        [build_case_microfiber_summary(case) for case in cases],
        ignore_index=True,
    )
