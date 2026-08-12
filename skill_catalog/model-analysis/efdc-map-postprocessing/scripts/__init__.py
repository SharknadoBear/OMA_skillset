"""Reusable EFDC map postprocessing package."""

from .efdc_map_tools import (
    EFDCPlotResult,
    WetCellFootprints,
    build_wet_cell_footprints,
    colormap_for_variable,
    plot_efdc_scalar,
    quantile_limits,
    save_efdc_scalar_map,
)
from .efdc_output import (
    EFDCGrid,
    ScalarSeries,
    inspect_inputs,
    load_scalar_series,
    normalize_times,
    read_grid,
    resolve_layer_index,
    sigma_weights,
    validate_vector_components,
)

__all__ = [
    "EFDCGrid",
    "EFDCPlotResult",
    "WetCellFootprints",
    "build_wet_cell_footprints",
    "ScalarSeries",
    "colormap_for_variable",
    "inspect_inputs",
    "load_scalar_series",
    "normalize_times",
    "plot_efdc_scalar",
    "quantile_limits",
    "read_grid",
    "resolve_layer_index",
    "save_efdc_scalar_map",
    "sigma_weights",
    "validate_vector_components",
]
