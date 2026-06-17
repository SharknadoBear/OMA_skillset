"""FVCOM grid generation helpers for local bathymetry to SMS 2DM meshes."""

from .bathymetry import BathymetryGrid, load_bathymetry
from .coastline_domain import DomainPrepareConfig, PreparedDomain, prepare_coastline_domain
from .domain import DomainBoundary, build_elliptical_domain, infer_offshore_side
from .gmsh_builder import generate_coastline_mesh, require_gmsh
from .mesh_builder import MeshBuildConfig, TriMesh, build_mesh
from .mesh_quality import QualityThresholds, evaluate_mesh_quality
from .open_boundary_designer import OpenBoundaryDesignResult, design_open_boundary
from .sms_2dm import Mesh2DM, read_2dm, write_2dm

__all__ = [
    "BathymetryGrid",
    "DomainBoundary",
    "DomainPrepareConfig",
    "Mesh2DM",
    "MeshBuildConfig",
    "OpenBoundaryDesignResult",
    "PreparedDomain",
    "QualityThresholds",
    "TriMesh",
    "build_elliptical_domain",
    "build_mesh",
    "evaluate_mesh_quality",
    "generate_coastline_mesh",
    "design_open_boundary",
    "infer_offshore_side",
    "load_bathymetry",
    "prepare_coastline_domain",
    "read_2dm",
    "require_gmsh",
    "write_2dm",
]
