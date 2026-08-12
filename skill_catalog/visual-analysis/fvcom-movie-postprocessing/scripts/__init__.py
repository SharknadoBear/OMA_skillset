"""Staged FVCOM movie postprocessing helpers."""

from __future__ import annotations

from .fvcom_map_tools import (
    ContourGrid,
    ScalarMapResult,
    plot_fvcom_scalar_map,
    quantile_limits,
    scatter_to_contour_grid,
)
from .fvcom_mesh_tools import (
    load_case_mesh,
    mesh_from_output,
    read_fvcom_mesh_dat,
)
from .fvcom_movie_tools import (
    default_scale,
    make_scalar_gif,
    make_scalar_gif_from_frames,
    plot_scalar_map,
    read_scalar_frame,
    resolve_variable_names,
    sigma_layer_index,
    transform_values,
)
from .fvcom_output import (
    decode_fvcom_time,
    discover_output_stacks,
    parse_time_like,
)
