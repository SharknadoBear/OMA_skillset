"""Staged FVCOM map postprocessing helpers."""

from __future__ import annotations

from .fvcom_map_tools import (
    ContourGrid,
    ScalarMapResult,
    boundary_rings_from_triangles,
    colormap_for_variable,
    plot_fvcom_scalar_map,
    quantile_limits,
    scatter_to_contour_grid,
)
from .fvcom_mesh_tools import (
    apply_zoom,
    auto_zoom_boxes,
    build_triangulation,
    element_to_node_average,
    load_case_mesh,
    mesh_extent,
    mesh_from_output,
    read_fvcom_mesh_dat,
    resolve_zoom,
)
from .fvcom_output import (
    decode_fvcom_time,
    discover_output_stacks,
    parse_time_like,
)
