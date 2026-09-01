"""Generator-neutral, fail-closed conditioning for raw FVCOM 2DM meshes.

The portfolio wrapper deliberately owns the generator-to-FVCOM handoff.  It
uses only an immutable raw 2DM, a canonical regular-grid v4 size field, and an
immutable bathymetry grid.  Every topological boundary node is fixed, so the
same policy can be applied to meshes produced by unrelated generators without
requiring generator-specific boundary-node metadata.

SMS 2DM ``NS`` records preserve ordered nodestrings and integer identifiers but
do not encode whether a nodestring is cyclic.  Accept an optional external
boundary contract to retain that fact as a sidecar, and never infer cyclicity
from the 2DM alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from .bathymetry import load_bathymetry
from .edge_size_audit import audit_edge_target_sizes
from .local_topology import (
    AggressiveConditioningConfig,
    LocalTopologyResult,
    condition_mesh_aggressive,
)
from .metrics import (
    build_edge_topology,
    chain_edges,
    compute_mesh_metrics,
    constraint_integrity,
    triangle_geometry,
)
from .postprocess import boundary_chains_from_mesh
from .projection import (
    LocalProjection,
    local_utm_projection,
    project_points,
    unproject_points,
)
from .quality import evaluate_mesh_quality
from .quality_policy import (
    apply_quality_policy,
    classify_failure_codes,
    load_quality_policy,
)
from .regional_conditioning import (
    AreaTransitionRelaxConfig,
    relax_mesh_area_transitions,
)
from .size_field import recorded_size_interpolator
from .sms_2dm import Mesh2DM, read_2dm, write_2dm


class UnsupportedCyclicOpenBoundaryError(ValueError):
    """Raised when a 2DM contains an explicit cyclic-OBC indicator."""


@dataclass(frozen=True)
class PortfolioConditioningConfig:
    """One fixed, bounded policy shared by every generator candidate."""

    conditioning_profile: str = "auto"
    primary_rounds: int = 4
    terminal_rounds: int = 1
    max_prunes_per_round: int = 500
    max_valence_repairs_per_round: int = 500
    max_valence_flip_batch: int = 64
    max_valence_cluster_merges_per_round: int = 25
    max_valence_l_over_h_count_increase: int = 0
    micro_relax_cycles: int = 3
    area_transition_max_patches: int = 12
    area_transition_raw_threshold: float = 0.50
    area_transition_target_gradient_threshold: float = 0.10
    wall_time_s: float = 3_600.0

    def validate(self) -> None:
        if str(self.conditioning_profile) not in {
            "auto",
            "minimal-topology-v1",
            "guarded-v1",
            "aggressive-local-v2",
            "none",
        }:
            raise ValueError(
                "conditioning_profile must be auto, minimal-topology-v1, "
                "guarded-v1, aggressive-local-v2, or none"
            )
        integer_positive = {
            "primary_rounds": self.primary_rounds,
            "terminal_rounds": self.terminal_rounds,
            "max_valence_flip_batch": self.max_valence_flip_batch,
        }
        for name, value in integer_positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        integer_nonnegative = {
            "max_prunes_per_round": self.max_prunes_per_round,
            "max_valence_repairs_per_round": self.max_valence_repairs_per_round,
            "max_valence_cluster_merges_per_round": (
                self.max_valence_cluster_merges_per_round
            ),
            "max_valence_l_over_h_count_increase": (
                self.max_valence_l_over_h_count_increase
            ),
            "area_transition_max_patches": self.area_transition_max_patches,
            "micro_relax_cycles": self.micro_relax_cycles,
        }
        for name, value in integer_nonnegative.items():
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 < float(self.area_transition_raw_threshold) < 1.0:
            raise ValueError("area_transition_raw_threshold must be in (0, 1)")
        if float(self.area_transition_target_gradient_threshold) < 0.0:
            raise ValueError(
                "area_transition_target_gradient_threshold must be nonnegative"
            )
        if float(self.wall_time_s) <= 0.0:
            raise ValueError("wall_time_s must be positive")


@dataclass
class _ConditioningState:
    points: np.ndarray
    triangles: np.ndarray
    fixed: np.ndarray
    constraint_chains: list[list[int]]
    boundary_kinds: list[str]
    hard: np.ndarray
    targets: np.ndarray
    raw_lineage: np.ndarray

    def clone(self) -> "_ConditioningState":
        return _ConditioningState(
            points=self.points.copy(),
            triangles=self.triangles.copy(),
            fixed=self.fixed.copy(),
            constraint_chains=[chain.copy() for chain in self.constraint_chains],
            boundary_kinds=self.boundary_kinds.copy(),
            hard=self.hard.copy(),
            targets=self.targets.copy(),
            raw_lineage=self.raw_lineage.copy(),
        )


@dataclass(frozen=True)
class _RegularGrid:
    lon: np.ndarray
    lat: np.ndarray
    values: np.ndarray
    coverage: np.ndarray
    value_name: str
    source_path: Path
    source_sha256: str
    schema_version: str | None
    domain_mask: np.ndarray | None = None
    sampling_interface_schema_version: str | None = None

    def sample(self, lonlat: np.ndarray) -> np.ndarray:
        locations = np.asarray(lonlat, dtype=float)
        if locations.ndim != 2 or locations.shape[1] != 2:
            raise ValueError("Regular-grid samples require an N x 2 lon/lat array")
        query = np.column_stack((locations[:, 1], locations[:, 0]))
        if self.sampling_interface_schema_version is not None:
            sampled = recorded_size_interpolator(
                self.lat,
                self.lon,
                self.values,
                np.asarray(self.coverage, dtype=bool),
                (
                    np.asarray(self.domain_mask, dtype=bool)
                    if self.domain_mask is not None
                    else np.asarray(self.coverage, dtype=bool)
                ),
                self.sampling_interface_schema_version,
            ).sample(query)
            invalid = ~np.isfinite(sampled) | (sampled <= 0.0)
            if np.any(invalid):
                first = int(np.flatnonzero(invalid)[0])
                raise ValueError(
                    f"{self.value_name} sampling is outside finite positive "
                    f"coverage for {int(np.count_nonzero(invalid))} point(s); "
                    f"first lon/lat=({locations[first, 0]:.10f}, "
                    f"{locations[first, 1]:.10f})"
                )
            return np.asarray(sampled, dtype=float)
        interpolation_values = np.where(
            np.asarray(self.coverage, dtype=bool),
            np.asarray(self.values, dtype=float),
            np.nan,
        )
        interpolator = RegularGridInterpolator(
            (self.lat, self.lon),
            interpolation_values,
            bounds_error=False,
            fill_value=np.nan,
        )
        coverage_interpolator = RegularGridInterpolator(
            (self.lat, self.lon),
            np.asarray(self.coverage, dtype=np.uint8),
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )
        sampled = np.asarray(interpolator(query), dtype=float)
        covered = np.asarray(coverage_interpolator(query), dtype=float) >= 0.5
        invalid = ~covered | ~np.isfinite(sampled) | (sampled <= 0.0)
        if np.any(invalid):
            first = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"{self.value_name} sampling is outside finite positive coverage "
                f"for {int(np.count_nonzero(invalid))} point(s); first lon/lat="
                f"({locations[first, 0]:.10f}, {locations[first, 1]:.10f})"
            )
        return sampled


def condition_portfolio_mesh(
    mesh_path: str | Path,
    size_field_nc: str | Path,
    bathymetry_nc: str | Path,
    output_dir: str | Path,
    *,
    name: str | None = None,
    config: PortfolioConditioningConfig | None = None,
    boundary_contract: dict[str, Any] | None = None,
    source_boundary_metadata: dict[str, Any] | None = None,
    scientific_input_valid: bool = True,
    scientific_input_note: str | None = None,
) -> dict[str, Any]:
    """Condition one raw 2DM under the common portfolio policy.

    The output directory must not already exist.  A delivered mesh can be
    structurally valid but still receive ``needs_review`` when any existing
    FVCOM quality gate remains open.
    """

    policy = config or PortfolioConditioningConfig()
    policy.validate()
    effective_profile = (
        "minimal-topology-v1"
        if str(policy.conditioning_profile) == "auto"
        else str(policy.conditioning_profile)
    )
    mesh_path = Path(mesh_path).resolve()
    size_path = Path(size_field_nc).resolve()
    bathy_path = Path(bathymetry_nc).resolve()
    output = Path(output_dir).resolve()
    for path in (mesh_path, size_path, bathy_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(
            f"Portfolio output directory must be fresh and non-existing: {output}"
        )

    raw_mesh = read_2dm(mesh_path)
    _validate_mesh_arrays(raw_mesh)
    raw_obc_chains, raw_obc_ids = _input_open_boundaries(raw_mesh)
    cyclicity = _cyclicity_contract(
        raw_obc_chains,
        raw_obc_ids,
        boundary_contract,
    )
    open_boundary_cyclic = [
        bool(item["cyclic"])
        for item in cyclicity["chains"]
    ]

    projection = local_utm_projection(_bbox(raw_mesh.nodes_lonlat))
    raw_points = project_points(raw_mesh.nodes_lonlat, projection)
    raw_triangles, reoriented_count = _orient_ccw(
        raw_points,
        np.asarray(raw_mesh.triangles, dtype=int) - 1,
    )
    raw_boundary_chains = boundary_chains_from_mesh(raw_triangles + 1)
    raw_topology = build_edge_topology(len(raw_points), raw_triangles)
    raw_boundary_nodes = sorted(
        {int(node) for edge in raw_topology.boundary_edges for node in edge}
    )
    fixed = np.zeros(len(raw_points), dtype=bool)
    fixed[np.asarray(raw_boundary_nodes, dtype=int)] = True
    if not raw_boundary_nodes:
        raise ValueError("The raw mesh has no topological boundary")

    size_field = _load_canonical_size_field(size_path)
    bathymetry = _load_bathymetry_grid(bathy_path)

    def target_sampler_xy(locations: np.ndarray) -> np.ndarray:
        return size_field.sample(
            unproject_points(np.asarray(locations, dtype=float), projection)
        )

    raw_targets = target_sampler_xy(raw_points)
    raw_state = _ConditioningState(
        points=raw_points.copy(),
        triangles=raw_triangles.copy(),
        fixed=fixed.copy(),
        constraint_chains=[chain.copy() for chain in raw_boundary_chains],
        boundary_kinds=[
            "fixed_boundary" if fixed[index] else "interior"
            for index in range(len(raw_points))
        ],
        hard=fixed.copy(),
        targets=raw_targets.copy(),
        raw_lineage=np.arange(len(raw_points), dtype=int),
    )
    raw_audit = _global_structural_audit(
        raw_state,
        raw_points,
        raw_boundary_chains,
        raw_obc_chains,
        raw_obc_ids,
        open_boundary_cyclic,
        target_sampler_xy,
    )
    if not raw_audit["core_passed"]:
        raise ValueError(
            "Raw mesh fails the structural precondition for transactional "
            "conditioning: "
            + ", ".join(raw_audit["core_failures"])
        )
    raw_depths = bathymetry.sample(raw_mesh.nodes_lonlat)
    raw_triangle_targets = _triangle_targets(
        raw_points,
        raw_triangles,
        target_sampler_xy,
    )
    raw_quality = evaluate_mesh_quality(
        raw_points,
        raw_depths,
        raw_mesh.triangles,
        raw_mesh.open_boundary_nodes,
        {
            "boundary_constraint_recovered": bool(
                raw_audit["boundary_edge_set_exact"]
            ),
            "source": "raw_2dm_preconditioning_audit",
        },
        constraint_chains=raw_boundary_chains,
        open_boundary_chains=[
            np.asarray(chain, dtype=int).tolist()
            for chain in raw_mesh.open_boundary_chains
        ],
        open_boundary_cyclic=open_boundary_cyclic,
        require_open_boundary=bool(raw_obc_chains),
        expected_open_boundary_count=len(raw_obc_chains),
        enforce_size_error=True,
        enforce_no_unused_nodes=True,
        target_size_by_triangle=raw_triangle_targets,
    )
    raw_edge_size = _edge_size_continuity_audit(
        raw_points,
        raw_triangles,
        raw_boundary_chains,
        target_sampler_xy,
    )

    primary_result = condition_mesh_aggressive(
        raw_state.points,
        raw_state.triangles,
        raw_state.fixed,
        raw_state.constraint_chains,
        # The current core has a legacy single-flat-OBC interface.  Passing
        # plural chains would invent cross-chain pairs.  Every boundary is
        # fixed here, and plural OBCs are independently remapped/audited below.
        np.empty(0, dtype=int),
        target_spacing_m=raw_state.targets,
        boundary_kinds=raw_state.boundary_kinds,
        hard_anchor_mask=raw_state.hard,
        target_spacing_sampler=target_sampler_xy,
        config=_primary_topology_config(
            policy,
            effective_profile=effective_profile,
        ),
    )
    primary_candidate = _state_from_result(primary_result)
    primary_audit = _global_structural_audit(
        primary_candidate,
        raw_points,
        raw_boundary_chains,
        raw_obc_chains,
        raw_obc_ids,
        open_boundary_cyclic,
        target_sampler_xy,
        boundary_refinement_ledger=primary_result.edit_ledger,
    )
    primary_regressions = _stage_regressions(
        raw_audit,
        primary_audit,
        minimal_policy=(effective_profile == "minimal-topology-v1"),
    )
    primary_report_only_deltas = (
        _minimal_report_only_deltas(raw_audit, primary_audit)
        if effective_profile == "minimal-topology-v1"
        else {}
    )
    primary_rollback = bool(primary_regressions)
    state = raw_state.clone() if primary_rollback else primary_candidate
    accepted_primary_audit = raw_audit if primary_rollback else primary_audit

    pre_area_state = state.clone()
    pre_area_audit = accepted_primary_audit
    if effective_profile == "minimal-topology-v1":
        area_report: dict[str, Any] = {
            "enabled": False,
            "reason": "disabled_by_minimal_topology_v1",
            "accepted_patch_count": 0,
        }
        terminal_report: dict[str, Any] = {
            "enabled": False,
            "reason": "terminal_valence_scan_is_internal_to_minimal_profile",
        }
        terminal_edit_ledger: list[dict[str, Any]] = []
        terminal_audit = pre_area_audit
        terminal_regressions: list[str] = []
        terminal_rollback = False
        final_state = pre_area_state
        final_audit = pre_area_audit
    elif effective_profile == "none":
        area_report = {
            "enabled": False,
            "reason": "conditioning_profile_none",
            "accepted_patch_count": 0,
        }
        terminal_report = {
            "enabled": False,
            "reason": "conditioning_profile_none",
        }
        terminal_edit_ledger = []
        terminal_audit = pre_area_audit
        terminal_regressions = []
        terminal_rollback = False
        final_state = pre_area_state
        final_audit = pre_area_audit
    else:
        area_result = relax_mesh_area_transitions(
            state.points,
            state.triangles,
            state.fixed,
            target_spacing_sampler=target_sampler_xy,
            constraint_chains=state.constraint_chains,
            open_boundary_nodes_zero_based=np.empty(0, dtype=int),
            config=AreaTransitionRelaxConfig(
                enabled=True,
                max_patches=int(policy.area_transition_max_patches),
                raw_area_change_threshold=float(
                    policy.area_transition_raw_threshold
                ),
                target_gradient_threshold=float(
                    policy.area_transition_target_gradient_threshold
                ),
            ),
        )
        area_report = area_result.report
        area_state = state.clone()
        area_state.points = np.asarray(area_result.nodes_xy, dtype=float)
        area_state.targets = np.asarray(
            area_result.target_spacing_m,
            dtype=float,
        )
        terminal_result = condition_mesh_aggressive(
            area_state.points,
            area_state.triangles,
            area_state.fixed,
            area_state.constraint_chains,
            np.empty(0, dtype=int),
            target_spacing_m=area_state.targets,
            boundary_kinds=area_state.boundary_kinds,
            hard_anchor_mask=area_state.hard,
            target_spacing_sampler=target_sampler_xy,
            config=_terminal_topology_config(policy),
        )
        terminal_report = terminal_result.report
        terminal_edit_ledger = terminal_result.edit_ledger
        terminal_candidate = _state_from_result(
            terminal_result,
            previous_raw_lineage=area_state.raw_lineage,
        )
        terminal_audit = _global_structural_audit(
            terminal_candidate,
            raw_points,
            raw_boundary_chains,
            raw_obc_chains,
            raw_obc_ids,
            open_boundary_cyclic,
            target_sampler_xy,
        )
        terminal_regressions = _stage_regressions(
            pre_area_audit,
            terminal_audit,
        )
        terminal_rollback = bool(terminal_regressions)
        final_state = (
            pre_area_state.clone()
            if terminal_rollback
            else terminal_candidate
        )
        final_audit = (
            pre_area_audit if terminal_rollback else terminal_audit
        )

    final_boundary_chains = [
        list(map(int, chain))
        for chain in final_audit["_mapped_boundary_chains"]
    ]
    final_obc_chains = [
        list(map(int, chain)) for chain in final_audit["_mapped_obc_chains"]
    ]
    final_lonlat = unproject_points(final_state.points, projection)
    final_depths = bathymetry.sample(final_lonlat)
    if np.any(~np.isfinite(final_depths)) or np.any(final_depths <= 0.0):
        raise ValueError(
            "Immutable bathymetry returned nonfinite or nonpositive final depths"
        )

    output.mkdir(parents=True, exist_ok=False)
    mesh_output = output / "conditioned.2dm"
    report_output = output / "conditioning_report.json"
    boundary_output = output / "delivered_boundary_nodes.geojson"
    obc_output = output / "obc_remap_manifest.json"
    quality_output = output / "mesh_quality.json"
    output_name = (
        str(name).strip()
        if name is not None and str(name).strip()
        else f"{raw_mesh.mesh_name}_portfolio_conditioned"
    )
    rejected_primary_evidence = None
    if primary_rollback:
        rejected_primary_evidence = _write_rejected_primary_candidate(
            output / "rejected_primary_candidate",
            state=primary_candidate,
            audit=primary_audit,
            projection=projection,
            bathymetry=bathymetry,
            target_sampler_xy=target_sampler_xy,
            raw_obc_ids=raw_obc_ids,
            open_boundary_cyclic=open_boundary_cyclic,
            rollback_reasons=primary_regressions,
            report_only_deltas=primary_report_only_deltas,
            edit_ledger=primary_result.edit_ledger,
            mesh_name=f"{output_name}_rejected_primary_candidate",
        )
    write_2dm(
        mesh_output,
        final_lonlat,
        final_depths,
        final_state.triangles + 1,
        np.empty(0, dtype=int),
        mesh_name=output_name,
        open_boundary_chains=[
            np.asarray(chain, dtype=int) + 1 for chain in final_obc_chains
        ],
        open_boundary_ids=raw_obc_ids,
    )

    roundtrip = _roundtrip_audit(
        mesh_output,
        final_state,
        projection,
        final_obc_chains,
        raw_obc_ids,
    )
    serialized = read_2dm(mesh_output)
    serialized_points = project_points(serialized.nodes_lonlat, projection)
    serialized_triangles = np.asarray(serialized.triangles, dtype=int) - 1
    serialized_triangle_targets = _triangle_targets(
        serialized_points,
        serialized_triangles,
        target_sampler_xy,
    )
    serialized_obc_chains = [
        np.asarray(chain, dtype=int).tolist()
        for chain in serialized.open_boundary_chains
    ]
    quality = evaluate_mesh_quality(
        serialized_points,
        serialized.depths,
        serialized.triangles,
        serialized.open_boundary_nodes,
        {
            "boundary_constraint_recovered": bool(
                final_audit["boundary_edge_set_exact"]
            ),
            "source": "raw_2dm_topological_boundary_fixed_and_lineage_mapped",
        },
        constraint_chains=final_boundary_chains,
        open_boundary_chains=serialized_obc_chains,
        open_boundary_cyclic=open_boundary_cyclic,
        require_open_boundary=bool(raw_obc_chains),
        expected_open_boundary_count=len(raw_obc_chains),
        enforce_size_error=True,
        enforce_no_unused_nodes=True,
        target_size_by_triangle=serialized_triangle_targets,
    )
    quality["canonical_inputs"] = {
        "size_field_nc": str(size_path),
        "size_field_sha256": size_field.source_sha256,
        "size_field_schema_version": size_field.schema_version,
        "sampling_interface_schema_version": (
            size_field.sampling_interface_schema_version
        ),
        "bathymetry_netcdf": str(bathy_path),
        "bathymetry_sha256": bathymetry.source_sha256,
        "depth_sampling": (
            "direct_regular_grid_interpolation_from_immutable_bathymetry;"
            "input_2dm_depths_ignored"
        ),
    }
    quality["open_boundary_cyclicity_contract"] = cyclicity
    quality["serialized_roundtrip"] = roundtrip
    edge_size = _edge_size_continuity_audit(
        serialized_points,
        serialized_triangles,
        final_boundary_chains,
        target_sampler_xy,
    )
    quality["edge_size_continuity"] = edge_size

    boundary_document = _boundary_geojson(
        final_lonlat,
        final_boundary_chains,
        final_obc_chains,
        raw_obc_ids,
        final_state.raw_lineage,
        target_sampler_xy(final_state.points),
        open_boundary_cyclic,
    )
    obc_manifest = _obc_remap_manifest(
        mesh_path,
        raw_mesh,
        raw_obc_chains,
        raw_obc_ids,
        final_obc_chains,
        final_state.raw_lineage,
        cyclicity,
        roundtrip,
        source_boundary_metadata,
    )
    _write_json(boundary_output, boundary_document)
    _write_json(obc_output, obc_manifest)

    minimal_local_debt_closed = bool(
        not primary_rollback
        and primary_result.report.get("minimal_local_debt_closed", False)
        and int(final_audit["superthin_triangle_count"]) == 0
        and int(final_audit["count_valence_above_8"]) == 0
        and roundtrip["passed"]
        and final_audit["core_passed"]
    )
    policy_document = load_quality_policy()
    all_findings = list(map(str, quality.get("all_quality_findings", [])))
    if not bool(roundtrip["passed"]):
        all_findings.append("sms_2dm_roundtrip_failed")
    if not bool(final_audit["core_passed"]):
        all_findings.extend(
            f"terminal_core:{value}"
            for value in final_audit["core_failures"]
        )
    if not bool(obc_manifest["forcing_compatible"]):
        all_findings.append("open_boundary_forcing_incompatible")
    if not bool(edge_size["passed"]):
        all_findings.extend(
            f"edge_size_continuity:{value}"
            for value in edge_size["failure_taxonomy"]
        )
    if raw_obc_chains and not bool(cyclicity["supported"]):
        all_findings.append("open_boundary_cyclicity_unknown")
    if any(open_boundary_cyclic):
        all_findings.append(
            "cyclic_obc_not_self_describing_in_sms_2dm"
        )
    if not bool(scientific_input_valid):
        all_findings.append("scientific_input_invalid")
    all_findings = sorted(set(all_findings))
    apply_quality_policy(
        quality,
        all_findings,
        advisories=quality.get("quality_advisories", {}),
        policy=policy_document,
    )
    classified = classify_failure_codes(all_findings, policy_document)
    baseline_failures = list(quality["baseline_failure_taxonomy"])
    submission_failures = list(classified["submission_preconditions"])
    benchmark_ready = bool(quality["benchmark_grid_baseline_ready"])
    submission_eligible = bool(benchmark_ready and not submission_failures)
    quality["raw_stage"] = False
    quality["common_conditioning_applied"] = bool(
        effective_profile != "none"
    )
    quality["conditioning_profile_requested"] = str(
        policy.conditioning_profile
    )
    quality["conditioning_profile_effective"] = effective_profile
    quality["minimal_local_debt_closed"] = minimal_local_debt_closed
    quality["submission_failure_taxonomy"] = submission_failures
    quality["submission_eligible"] = submission_eligible
    quality["all_quality_findings"] = all_findings
    _write_json(quality_output, quality)

    status = "pass" if benchmark_ready else "needs_review"
    report = {
        "schema_version": "fvcom_portfolio_conditioning_v3",
        "status": status,
        "minimal_local_debt_closed": minimal_local_debt_closed,
        "benchmark_grid_baseline_ready": benchmark_ready,
        "fvcom_ready": benchmark_ready,
        "submission_eligible": submission_eligible,
        "fvcom_readiness_failure_taxonomy": baseline_failures,
        "regional_refinement_debt": quality["regional_refinement_debt"],
        "quality_advisories": quality["quality_advisories"],
        "submission_failure_taxonomy": submission_failures,
        "quality_policy": quality["quality_policy"],
        "policy": {
            "name": effective_profile,
            "requested_profile": str(policy.conditioning_profile),
            "effective_profile": effective_profile,
            "settings": asdict(policy),
            "stage_order": (
                [
                    "raw-global-audit",
                    "valence-first-local-transactions",
                    "immediate-post-valence-superthin-cleanup",
                    "residual-superthin-component-repair",
                    "terminal-valence-and-superthin-scan",
                    "terminal-global-audit-or-rollback",
                    "immutable-bathymetry-resampling",
                    "serialized-quality-audit",
                ]
                if effective_profile == "minimal-topology-v1"
                else [
                    "raw-global-audit",
                    "aggressive-local-v2-guarded-primary",
                    "target-aware-area-transition-relax-v1",
                    "guarded-terminal-thin-valence",
                    "terminal-global-audit-or-rollback",
                    "immutable-bathymetry-resampling",
                    "serialized-quality-audit",
                ]
            ),
            "boundary_edit_policy": (
                "one_ledger_authorized_non_obc_source_arc_midpoint_per_"
                "fixed_hard_fan_transaction"
                if effective_profile == "minimal-topology-v1"
                else "none"
            ),
            "all_topological_boundary_nodes_fixed": True,
            "internal_legacy_flat_obc_disabled": True,
            "plural_obc_policy": (
                "independent lineage mapping and terminal boundary-edge audit"
            ),
            "outer_stage_acceptance": {
                "benchmark_first_priority": [
                    "absolute_structural_invariants",
                    "valence_debt",
                    "superthin_debt",
                ],
                "regional_refinement_debt_never_rolls_back_minimal_repairs": True,
                "legacy_profiles_unchanged": True,
            },
        },
        "inputs": {
            "mesh": {"path": str(mesh_path), "sha256": _sha256(mesh_path)},
            "canonical_size_field": {
                "path": str(size_path),
                "sha256": size_field.source_sha256,
                "schema_version": size_field.schema_version,
                "sampling_interface_schema_version": (
                    size_field.sampling_interface_schema_version
                ),
            },
            "immutable_bathymetry": {
                "path": str(bathy_path),
                "sha256": bathymetry.source_sha256,
            },
            "boundary_contract": {
                "supplied": bool(boundary_contract is not None),
                "canonical_sha256": (
                    hashlib.sha256(
                        json.dumps(
                            boundary_contract,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    if boundary_contract is not None
                    else None
                ),
            },
            "source_boundary_metadata": {
                "supplied": bool(source_boundary_metadata is not None),
                "canonical_sha256": (
                    hashlib.sha256(
                        json.dumps(
                            source_boundary_metadata,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    if source_boundary_metadata is not None
                    else None
                ),
            },
        },
        "projection_epsg": int(projection.epsg),
        "input_triangle_reorientation_count": int(reoriented_count),
        "open_boundary_cyclicity_contract": cyclicity,
        "scientific_input": {
            "valid": bool(scientific_input_valid),
            "note": scientific_input_note,
        },
        "raw_global_audit": _public_audit(raw_audit),
        "raw_quality": raw_quality,
        "raw_edge_size_continuity": raw_edge_size,
        "primary_topology": {
            "report": primary_result.report,
            "edit_ledger": primary_result.edit_ledger,
            "candidate_global_audit": _public_audit(primary_audit),
            "rollback_applied": primary_rollback,
            "rollback_reasons": primary_regressions,
            "report_only_deltas": primary_report_only_deltas,
            "rejected_candidate_evidence": rejected_primary_evidence,
        },
        "area_transition": area_report,
        "terminal_topology": {
            "report": terminal_report,
            "edit_ledger": terminal_edit_ledger,
            "candidate_global_audit": _public_audit(terminal_audit),
            "rollback_to_pre_area_applied": terminal_rollback,
            "rollback_reasons": terminal_regressions,
        },
        "final_global_audit": _public_audit(final_audit),
        "final_quality": quality,
        "final_edge_size_continuity": edge_size,
        "depth_sampling": {
            "source": str(bathy_path),
            "source_sha256": bathymetry.source_sha256,
            "input_mesh_depths_used": False,
            "minimum_delivered_depth_m": float(np.min(final_depths)),
            "maximum_delivered_depth_m": float(np.max(final_depths)),
        },
        "roundtrip": roundtrip,
        "quality_accepted": benchmark_ready,
        "quality_failure_taxonomy": baseline_failures,
        "outputs": {
            "conditioned_2dm": _artifact(mesh_output),
            "delivered_boundary_nodes_geojson": _artifact(boundary_output),
            "obc_remap_manifest": _artifact(obc_output),
            "mesh_quality_json": _artifact(quality_output),
            "conditioning_report_json": str(report_output),
            "rejected_primary_candidate": rejected_primary_evidence,
        },
        "limitations": [
            (
                "SMS 2DM does not encode OBC cyclicity; an external contract "
                "can preserve it as evidence but a cyclic result is not "
                "self-describing or submission-eligible as a standalone 2DM."
            ),
            (
                "The current aggressive and area-transition cores expose a "
                "legacy flat-OBC argument.  The wrapper passes no internal OBC "
                "and instead fixes the complete topological boundary and audits "
                "each plural NS chain independently."
            ),
            (
                "A needs_review result is retained when the bounded policy "
                "cannot close the benchmark structural, valence, or "
                "superthin baseline. Regional refinement debt is reported "
                "separately and is never a baseline veto."
            ),
        ],
    }
    _write_json(report_output, report)
    return _json_safe(report)


def _write_rejected_primary_candidate(
    output: Path,
    *,
    state: _ConditioningState,
    audit: dict[str, Any],
    projection: LocalProjection,
    bathymetry: _RegularGrid,
    target_sampler_xy: Callable[[np.ndarray], np.ndarray],
    raw_obc_ids: list[int],
    open_boundary_cyclic: list[bool],
    rollback_reasons: list[str],
    report_only_deltas: dict[str, Any],
    edit_ledger: list[dict[str, Any]],
    mesh_name: str,
) -> dict[str, Any]:
    """Serialize a rejected whole-stage candidate as immutable evidence."""

    output.mkdir(parents=False, exist_ok=False)
    mesh_output = output / "candidate.2dm"
    quality_output = output / "mesh_quality.json"
    boundary_output = output / "boundary_nodes.geojson"
    ledger_output = output / "edit_ledger.json"
    manifest_output = output / "rollback_manifest.json"

    boundary_chains = [
        list(map(int, chain))
        for chain in audit.get("_mapped_boundary_chains", [])
    ]
    obc_chains = [
        list(map(int, chain))
        for chain in audit.get("_mapped_obc_chains", [])
    ]
    obc_ids = (
        list(map(int, raw_obc_ids))
        if len(obc_chains) == len(raw_obc_ids)
        else []
    )
    if not obc_ids:
        obc_chains = []

    lonlat = unproject_points(state.points, projection)
    depths = bathymetry.sample(lonlat)
    if np.any(~np.isfinite(depths)) or np.any(depths <= 0.0):
        raise ValueError(
            "Rejected primary candidate cannot be serialized with finite "
            "positive-down depths"
        )
    write_2dm(
        mesh_output,
        lonlat,
        depths,
        state.triangles + 1,
        np.empty(0, dtype=int),
        mesh_name=mesh_name,
        open_boundary_chains=[
            np.asarray(chain, dtype=int) + 1 for chain in obc_chains
        ],
        open_boundary_ids=obc_ids,
    )
    roundtrip = _roundtrip_audit(
        mesh_output,
        state,
        projection,
        obc_chains,
        obc_ids,
    )
    serialized = read_2dm(mesh_output)
    serialized_points = project_points(serialized.nodes_lonlat, projection)
    serialized_triangles = np.asarray(serialized.triangles, dtype=int) - 1
    triangle_targets = _triangle_targets(
        serialized_points,
        serialized_triangles,
        target_sampler_xy,
    )
    quality = evaluate_mesh_quality(
        serialized_points,
        serialized.depths,
        serialized.triangles,
        serialized.open_boundary_nodes,
        {
            "boundary_constraint_recovered": bool(
                audit["boundary_edge_set_exact"]
            ),
            "source": "rejected_primary_candidate_evidence",
        },
        constraint_chains=boundary_chains,
        open_boundary_chains=[
            np.asarray(chain, dtype=int).tolist()
            for chain in serialized.open_boundary_chains
        ],
        open_boundary_cyclic=(
            list(map(bool, open_boundary_cyclic))
            if len(obc_chains) == len(open_boundary_cyclic)
            else [False] * len(obc_chains)
        ),
        require_open_boundary=bool(obc_chains),
        expected_open_boundary_count=len(obc_chains),
        enforce_size_error=True,
        enforce_no_unused_nodes=True,
        target_size_by_triangle=triangle_targets,
    )
    quality["artifact_role"] = "rejected_primary_candidate"
    quality["accepted_for_delivery"] = False
    quality["rollback_reasons"] = list(map(str, rollback_reasons))
    quality["candidate_global_audit"] = _public_audit(audit)
    quality["report_only_deltas"] = report_only_deltas
    quality["serialized_roundtrip"] = roundtrip
    _write_json(quality_output, quality)

    boundary_document = _boundary_geojson(
        lonlat,
        boundary_chains,
        obc_chains,
        obc_ids,
        state.raw_lineage,
        target_sampler_xy(state.points),
        (
            list(map(bool, open_boundary_cyclic))
            if len(obc_chains) == len(open_boundary_cyclic)
            else [False] * len(obc_chains)
        ),
    )
    _write_json(boundary_output, boundary_document)
    _write_json(
        ledger_output,
        {
            "schema_version": "fvcom_rejected_topology_edit_ledger_v1",
            "artifact_role": "rejected_primary_candidate",
            "accepted_for_delivery": False,
            "rollback_reasons": list(map(str, rollback_reasons)),
            "edit_ledger": edit_ledger,
        },
    )
    manifest = {
        "schema_version": "fvcom_rejected_primary_candidate_v1",
        "status": "rejected",
        "accepted_for_delivery": False,
        "rollback_reasons": list(map(str, rollback_reasons)),
        "report_only_deltas": report_only_deltas,
        "candidate_global_audit": _public_audit(audit),
        "roundtrip": roundtrip,
        "artifacts": {
            "candidate_2dm": _artifact(mesh_output),
            "mesh_quality_json": _artifact(quality_output),
            "boundary_nodes_geojson": _artifact(boundary_output),
            "edit_ledger_json": _artifact(ledger_output),
        },
    }
    _write_json(manifest_output, manifest)
    return {
        "status": "rejected",
        **manifest["artifacts"],
        "rollback_manifest_json": _artifact(manifest_output),
    }


def _primary_topology_config(
    config: PortfolioConditioningConfig,
    *,
    effective_profile: str,
) -> AggressiveConditioningConfig:
    minimal = effective_profile == "minimal-topology-v1"
    disabled = effective_profile == "none"
    return AggressiveConditioningConfig(
        enabled=not disabled,
        profile_name=effective_profile,
        stage_order=(
            "valence-before-thin" if minimal else "thin-before-valence"
        ),
        enable_pruning=not (minimal or disabled),
        enable_thin_repair=not disabled,
        enable_valence_repair=not disabled,
        thin_repair_profile="guarded-v1",
        max_rounds=int(config.primary_rounds),
        boundary_edit_policy="none",
        max_boundary_edits_per_round=0,
        enable_fixed_hard_fan_arc_refinement=minimal,
        max_fixed_hard_fan_arc_refinements_per_round=(8 if minimal else 0),
        max_boundary_welds_per_round=0,
        max_boundary_ear_removals_per_round=0,
        max_prunes_per_round=(
            0 if minimal or disabled else int(config.max_prunes_per_round)
        ),
        max_valence_removals_per_round=int(
            config.max_valence_repairs_per_round
        ),
        max_valence_flip_batch=int(config.max_valence_flip_batch),
        max_valence_cluster_merges_per_round=int(
            config.max_valence_cluster_merges_per_round
        ),
        max_valence_l_over_h_count_increase=int(
            config.max_valence_l_over_h_count_increase
        ),
        topology_escrow_enabled=minimal,
        topology_escrow_maximum_superthin_count=1_000_000,
        topology_escrow_maximum_superthin_severity=1.0e12,
        topology_escrow_maximum_valence=1_000_000,
        micro_relax_cycles=(
            0 if minimal or disabled else int(config.micro_relax_cycles)
        ),
        deadline_monotonic_s=(
            time.perf_counter() + float(config.wall_time_s)
        ),
    )


def _terminal_topology_config(
    config: PortfolioConditioningConfig,
) -> AggressiveConditioningConfig:
    return AggressiveConditioningConfig(
        enabled=True,
        profile_name="guarded-v1-terminal",
        stage_order="thin-before-valence",
        enable_pruning=False,
        enable_thin_repair=True,
        enable_valence_repair=True,
        thin_repair_profile="guarded-v1",
        max_rounds=int(config.terminal_rounds),
        boundary_edit_policy="none",
        max_boundary_edits_per_round=0,
        max_boundary_welds_per_round=0,
        max_boundary_ear_removals_per_round=0,
        max_prunes_per_round=0,
        max_valence_removals_per_round=int(
            config.max_valence_repairs_per_round
        ),
        max_valence_flip_batch=int(config.max_valence_flip_batch),
        max_valence_cluster_merges_per_round=int(
            config.max_valence_cluster_merges_per_round
        ),
        max_valence_l_over_h_count_increase=int(
            config.max_valence_l_over_h_count_increase
        ),
        micro_relax_cycles=int(config.micro_relax_cycles),
        deadline_monotonic_s=(
            time.perf_counter() + float(config.wall_time_s)
        ),
    )


def _state_from_result(
    result: LocalTopologyResult,
    *,
    previous_raw_lineage: np.ndarray | None = None,
) -> _ConditioningState:
    lineage = np.asarray(result.node_lineage, dtype=int)
    if previous_raw_lineage is not None:
        lineage = _compose_lineage(previous_raw_lineage, lineage)
    return _ConditioningState(
        points=np.asarray(result.nodes_xy, dtype=float).copy(),
        triangles=np.asarray(result.triangles, dtype=int).copy(),
        fixed=np.asarray(result.fixed_node_mask, dtype=bool).copy(),
        constraint_chains=[
            list(map(int, chain)) for chain in result.constraint_chains
        ],
        boundary_kinds=list(map(str, result.boundary_kinds)),
        hard=np.asarray(result.hard_anchor_mask, dtype=bool).copy(),
        targets=np.asarray(result.target_spacing_m, dtype=float).copy(),
        raw_lineage=lineage.copy(),
    )


def _compose_lineage(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_values = np.asarray(previous, dtype=int)
    current_values = np.asarray(current, dtype=int)
    next_inserted = min(
        int(np.min(previous_values)) if len(previous_values) else 0,
        0,
    ) - 1
    output = np.empty(len(current_values), dtype=int)
    for delivered, parent in enumerate(current_values):
        if 0 <= int(parent) < len(previous_values):
            output[delivered] = int(previous_values[int(parent)])
        else:
            output[delivered] = int(next_inserted)
            next_inserted -= 1
    return output


def _authorized_fixed_hard_fan_refinements(
    edit_ledger: Iterable[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract only complete, accepted fixed-hard-fan midpoint transactions."""

    entries = [dict(entry) for entry in (edit_ledger or [])]
    accepted_entries = [
        entry
        for entry in entries
        if entry.get("operation")
        == "minimal-fixed-hard-fan-transaction-accepted"
    ]
    reconstruction_entries = [
        entry
        for entry in entries
        if entry.get("operation")
        == "minimal-fixed-hard-fan-source-arc-refinement"
    ]
    authorizations: list[dict[str, Any]] = []
    failures: list[str] = []
    seen_components: set[str] = set()
    seen_edges: set[tuple[int, int]] = set()
    for accepted in accepted_entries:
        component_id = str(accepted.get("component_id", "")).strip()
        source_edge = accepted.get("source_edge_lineage")
        if (
            not component_id
            or not isinstance(source_edge, (list, tuple))
            or len(source_edge) != 2
        ):
            failures.append("malformed_accepted_refinement_record")
            continue
        try:
            source_pair = tuple(sorted(map(int, source_edge)))
        except (TypeError, ValueError):
            failures.append("malformed_accepted_refinement_source_edge")
            continue
        if source_pair[0] < 0 or source_pair[0] == source_pair[1]:
            failures.append("invalid_accepted_refinement_source_edge")
            continue
        matches = [
            entry
            for entry in reconstruction_entries
            if str(entry.get("component_id", "")) == component_id
            and bool(entry.get("automatic"))
            and not bool(entry.get("review_required", True))
            and int(entry.get("inserted_boundary_node_count", -1)) == 1
            and int(entry.get("inserted_support_node_count", -1)) == 0
            and int(entry.get("removed_movable_node_count", -1)) == 0
        ]
        if len(matches) != 1:
            failures.append("accepted_refinement_missing_unique_reconstruction")
            continue
        if component_id in seen_components or source_pair in seen_edges:
            failures.append("duplicate_accepted_refinement_authorization")
            continue
        seen_components.add(component_id)
        seen_edges.add(source_pair)
        authorizations.append(
            {
                "component_id": component_id,
                "source_edge_lineage": list(source_pair),
            }
        )
    if len(reconstruction_entries) != len(accepted_entries):
        failures.append("unpaired_fixed_hard_fan_reconstruction_record")
    return authorizations, sorted(set(failures))


def _audit_authorized_boundary_refinements(
    state: _ConditioningState,
    raw_points: np.ndarray,
    raw_boundary_chains: list[list[int]],
    raw_obc_chains: list[list[int]],
    open_boundary_cyclic: list[bool],
    mapped_boundary: list[list[int]],
    edit_ledger: Iterable[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Prove that delivered boundary changes are only logged midpoints."""

    authorizations, failures = _authorized_fixed_hard_fan_refinements(
        edit_ledger
    )
    delivered_chains = [
        list(map(int, chain)) for chain in state.constraint_chains
    ]
    lineage = np.asarray(state.raw_lineage, dtype=int)
    boundary_occurrences: dict[int, list[tuple[int, int]]] = {}
    for chain_index, chain in enumerate(delivered_chains):
        for position, delivered in enumerate(chain):
            if delivered < 0 or delivered >= len(lineage):
                failures.append("delivered_boundary_node_out_of_range")
                continue
            if int(lineage[delivered]) < 0:
                boundary_occurrences.setdefault(int(delivered), []).append(
                    (int(chain_index), int(position))
                )

    inserted_boundary_nodes = sorted(boundary_occurrences)
    if not inserted_boundary_nodes and not authorizations and not failures:
        return {
            "authorized": False,
            "passed": True,
            "failures": [],
            "inserted_boundary_nodes": [],
            "authorizations": [],
            "delivered_boundary_chains": mapped_boundary,
            "raw_boundary_lineage_topology_preserved": True,
        }

    if len(delivered_chains) != len(raw_boundary_chains):
        failures.append("boundary_chain_count_changed")
    else:
        for chain_index, (delivered, raw) in enumerate(
            zip(delivered_chains, raw_boundary_chains, strict=True)
        ):
            collapsed = [
                int(lineage[node])
                for node in delivered
                if 0 <= node < len(lineage) and int(lineage[node]) >= 0
            ]
            if collapsed != list(map(int, raw)):
                failures.append(
                    f"boundary_chain_{chain_index}_raw_order_changed"
                )

    authorized_edges = {
        tuple(map(int, record["source_edge_lineage"]))
        for record in authorizations
    }
    observed_edges: set[tuple[int, int]] = set()
    raw_obc_edges: set[tuple[int, int]] = set()
    for chain, cyclic in zip(
        raw_obc_chains,
        open_boundary_cyclic,
        strict=True,
    ):
        nodes = list(map(int, chain))
        for first, second in zip(nodes[:-1], nodes[1:]):
            raw_obc_edges.add(tuple(sorted((first, second))))
        if cyclic and len(nodes) > 1:
            raw_obc_edges.add(tuple(sorted((nodes[-1], nodes[0]))))
    points = np.asarray(state.points, dtype=float)
    raw_xy = np.asarray(raw_points, dtype=float)
    for delivered in inserted_boundary_nodes:
        occurrences = boundary_occurrences[delivered]
        if len(occurrences) != 1:
            failures.append("inserted_boundary_node_not_unique_to_one_chain")
            continue
        chain_index, position = occurrences[0]
        chain = delivered_chains[chain_index]
        if len(chain) < 3:
            failures.append("refined_boundary_chain_too_short")
            continue
        previous_node = int(chain[(position - 1) % len(chain)])
        next_node = int(chain[(position + 1) % len(chain)])
        previous_source = int(lineage[previous_node])
        next_source = int(lineage[next_node])
        if previous_source < 0 or next_source < 0:
            failures.append("adjacent_inserted_boundary_nodes_not_allowed")
            continue
        source_edge = tuple(sorted((previous_source, next_source)))
        observed_edges.add(source_edge)
        if source_edge not in authorized_edges:
            failures.append("inserted_boundary_node_has_no_authorized_parent_edge")
        if source_edge not in chain_edges([raw_boundary_chains[chain_index]]):
            failures.append("refinement_parent_is_not_a_raw_boundary_edge")
        if source_edge in raw_obc_edges:
            failures.append("open_boundary_edge_refinement_not_allowed")
        if (
            delivered >= len(state.fixed)
            or not bool(state.fixed[delivered])
        ):
            failures.append("inserted_boundary_midpoint_not_fixed")
        if (
            delivered < len(state.boundary_kinds)
            and str(state.boundary_kinds[delivered]) == "open_boundary"
        ):
            failures.append("inserted_boundary_midpoint_classified_as_obc")
        midpoint = 0.5 * (
            raw_xy[source_edge[0]] + raw_xy[source_edge[1]]
        )
        edge_length = float(
            np.linalg.norm(
                raw_xy[source_edge[0]] - raw_xy[source_edge[1]]
            )
        )
        tolerance = max(1.0e-9, edge_length * 1.0e-12)
        if float(np.linalg.norm(points[delivered] - midpoint)) > tolerance:
            failures.append("inserted_boundary_node_is_not_exact_midpoint")

    if observed_edges != authorized_edges:
        failures.append("accepted_refinement_authorizations_not_one_to_one")
    passed = bool(authorizations and not failures)
    return {
        "authorized": bool(authorizations),
        "passed": passed,
        "failures": sorted(set(failures)),
        "inserted_boundary_nodes": inserted_boundary_nodes,
        "authorizations": authorizations,
        "delivered_boundary_chains": (
            delivered_chains if passed else mapped_boundary
        ),
        "raw_boundary_lineage_topology_preserved": bool(
            not any("raw_order_changed" in item for item in failures)
            and "boundary_chain_count_changed" not in failures
        ),
    }


def _global_structural_audit(
    state: _ConditioningState,
    raw_points: np.ndarray,
    raw_boundary_chains: list[list[int]],
    raw_obc_chains: list[list[int]],
    raw_obc_ids: list[int],
    open_boundary_cyclic: list[bool],
    target_sampler_xy: Callable[[np.ndarray], np.ndarray],
    *,
    boundary_refinement_ledger: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    points = np.asarray(state.points, dtype=float)
    triangles = np.asarray(state.triangles, dtype=int)
    topology = build_edge_topology(len(points), triangles)
    geometry = triangle_geometry(points, triangles)
    mapped_boundary: list[list[int]] = []
    mapped_obc: list[list[int]] = []
    mapping_failures: list[str] = []
    try:
        mapped_boundary = _mapped_source_chains(
            raw_boundary_chains,
            state.raw_lineage,
            "boundary",
        )
    except ValueError as exc:
        mapping_failures.append(str(exc))
    try:
        mapped_obc = _mapped_source_chains(
            raw_obc_chains,
            state.raw_lineage,
            "open-boundary",
        )
    except ValueError as exc:
        mapping_failures.append(str(exc))

    refinement_audit = _audit_authorized_boundary_refinements(
        state,
        raw_points,
        raw_boundary_chains,
        raw_obc_chains,
        open_boundary_cyclic,
        mapped_boundary,
        boundary_refinement_ledger,
    )
    audited_boundary = [
        list(map(int, chain))
        for chain in refinement_audit["delivered_boundary_chains"]
    ]
    expected_boundary_edges = chain_edges(audited_boundary)
    actual_boundary_edges = set(topology.boundary_edges)
    expected_boundary_nodes = {
        int(node) for chain in audited_boundary for node in chain
    }
    actual_boundary_nodes = {
        int(node) for edge in topology.boundary_edges for node in edge
    }
    integrity = constraint_integrity(
        topology,
        audited_boundary,
        None,
        mapped_obc,
        open_boundary_cyclic,
    )
    reverse = _raw_to_delivered(state.raw_lineage)
    raw_boundary_nodes = sorted(
        {int(node) for chain in raw_boundary_chains for node in chain}
    )
    shifts: list[float] = []
    for source in raw_boundary_nodes:
        delivered = reverse.get(source)
        if delivered is None:
            continue
        shifts.append(
            float(np.linalg.norm(points[delivered] - raw_points[source]))
        )
    boundary_shift = max(shifts, default=float("inf") if raw_boundary_nodes else 0.0)
    targets_by_triangle = _triangle_targets(
        points,
        triangles,
        target_sampler_xy,
    )
    metrics = compute_mesh_metrics(
        points,
        triangles,
        constraint_chains=audited_boundary,
        open_boundary_chains_zero_based=mapped_obc,
        open_boundary_cyclic=open_boundary_cyclic,
        target_size_by_triangle=targets_by_triangle,
    )
    areas = np.asarray(geometry["area"], dtype=float)
    area_defect_count = 0
    for attached in topology.edge_to_triangles.values():
        if len(attached) != 2:
            continue
        first, second = map(int, attached)
        change = abs(float(areas[first]) - float(areas[second])) / max(
            float(areas[first]),
            float(areas[second]),
            1.0e-30,
        )
        area_defect_count += int(change > 0.50)
    valence = np.asarray(
        [len(neighbors) for neighbors in topology.node_neighbors],
        dtype=int,
    )
    valence_excess = int(np.sum(np.maximum(valence - 8, 0)))
    minimum_angles = (
        np.min(np.asarray(geometry["angles_deg"], dtype=float), axis=1)
        if len(triangles)
        else np.empty(0, dtype=float)
    )
    unique_superthin_count = int(
        np.count_nonzero(
            (np.asarray(geometry["quality"], dtype=float) < 0.10)
            | (minimum_angles < 5.0)
        )
    )
    boundary_edge_exact = bool(
        not mapping_failures
        and expected_boundary_edges == actual_boundary_edges
        and expected_boundary_nodes == actual_boundary_nodes
    )
    obc_count_and_ids = bool(
        len(mapped_obc) == len(raw_obc_chains)
        and len(raw_obc_ids) == len(raw_obc_chains)
    )
    core_failures: list[str] = []
    if mapping_failures:
        core_failures.append("source_lineage_mapping_failure")
    if not bool(refinement_audit["passed"]):
        core_failures.append("unauthorized_boundary_refinement")
    if int(metrics["topology"]["connected_component_count"]) != 1:
        core_failures.append("wet_component_count_not_one")
    if int(metrics["topology"]["nonmanifold_edge_count"]) != 0:
        core_failures.append("nonmanifold_edges")
    if int(metrics["topology"]["boundary_degree_anomaly_count"]) != 0:
        core_failures.append("boundary_degree_anomaly")
    if int(metrics["topology"]["nonpositive_signed_area_count"]) != 0:
        core_failures.append("nonpositive_signed_area")
    if int(metrics["topology"]["unused_node_count"]) != 0:
        core_failures.append("unused_nodes")
    if not boundary_edge_exact:
        core_failures.append("topological_boundary_changed")
    if not np.isfinite(boundary_shift) or boundary_shift > 1.0e-10:
        core_failures.append("fixed_boundary_coordinate_changed")
    if not bool(integrity["all_protected_edges_present"]):
        core_failures.append("protected_boundary_edge_missing")
    if not bool(integrity["open_boundary_ordered"]):
        core_failures.append("open_boundary_order_or_edge_failure")
    if not obc_count_and_ids:
        core_failures.append("open_boundary_count_or_id_failure")
    size_error = metrics["size_error_l_over_h"]
    return {
        "core_passed": bool(not core_failures),
        "core_failures": core_failures,
        "mapping_failures": mapping_failures,
        "connected_component_count": int(
            metrics["topology"]["connected_component_count"]
        ),
        "nonmanifold_edge_count": int(
            metrics["topology"]["nonmanifold_edge_count"]
        ),
        "boundary_degree_anomaly_count": int(
            metrics["topology"]["boundary_degree_anomaly_count"]
        ),
        "nonpositive_signed_area_count": int(
            metrics["topology"]["nonpositive_signed_area_count"]
        ),
        "unused_node_count": int(metrics["topology"]["unused_node_count"]),
        "singly_connected_triangle_count": int(
            metrics["topology"]["singly_connected_triangle_count"]
        ),
        "boundary_edge_set_exact": boundary_edge_exact,
        "raw_boundary_refinement_authorized": bool(
            refinement_audit["authorized"] and refinement_audit["passed"]
        ),
        "authorized_boundary_insertion_count": int(
            len(refinement_audit["inserted_boundary_nodes"])
            if refinement_audit["passed"]
            else 0
        ),
        "boundary_refinement_failures": list(
            refinement_audit["failures"]
        ),
        "raw_boundary_lineage_topology_preserved": bool(
            refinement_audit["raw_boundary_lineage_topology_preserved"]
        ),
        "fixed_boundary_coordinate_max_shift_m": float(boundary_shift),
        "open_boundary_chain_count": int(len(mapped_obc)),
        "open_boundary_ids": list(map(int, raw_obc_ids)),
        "open_boundary_ordered": bool(integrity["open_boundary_ordered"]),
        "superthin_triangle_count": unique_superthin_count,
        "count_valence_above_8": int(
            metrics["valence"]["count_valence_above_8"]
        ),
        "valence_excess_above_8": valence_excess,
        "maximum_valence": int(metrics["valence"]["max_node_valence"]),
        "q_min": float(metrics["oceanmesh_quality"]["q_min"]),
        "q_p01": float(
            metrics["oceanmesh_quality"]["q_quantiles"]["p01"]
        ),
        "q_l3_sigma": float(
            metrics["oceanmesh_quality"]["q_l3_sigma"]
        ),
        "minimum_angle_deg": float(metrics["angles"]["min_angle_deg"]),
        "maximum_adjacent_area_change": float(
            metrics["max_adjacent_area_change"]
        ),
        "area_transition_defect_count": int(area_defect_count),
        "l_over_h_count_above_1_55": int(
            size_error["count_above_1_55"]
        ),
        "l_over_h_p95": float(size_error["quantiles"]["p95"]),
        "l_over_h_maximum": float(size_error["maximum"]),
        "_mapped_boundary_chains": audited_boundary,
        "_mapped_obc_chains": mapped_obc,
    }


def _stage_regressions(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    minimal_policy: bool = False,
) -> list[str]:
    failures: list[str] = []
    if not bool(after["core_passed"]):
        failures.extend(
            f"terminal_core:{value}" for value in after["core_failures"]
        )
    nonincrease = [
        "superthin_triangle_count",
        "count_valence_above_8",
        "valence_excess_above_8",
        "l_over_h_count_above_1_55",
    ]
    if minimal_policy:
        before_valence = (
            int(before["count_valence_above_8"]),
            int(before["valence_excess_above_8"]),
            int(before["maximum_valence"]),
        )
        after_valence = (
            int(after["count_valence_above_8"]),
            int(after["valence_excess_above_8"]),
            int(after["maximum_valence"]),
        )
        if after_valence > before_valence:
            failures.append("valence_debt_regressed")
        elif after_valence == before_valence and int(
            after["superthin_triangle_count"]
        ) > int(before["superthin_triangle_count"]):
            failures.append("superthin_triangle_count_regressed")
        # All Class-2 metrics, including singly connected count, ordinary
        # quality tails, area transition, and L/h, are report-only here.
        return sorted(set(failures))
    if not minimal_policy:
        nonincrease.append("area_transition_defect_count")
    for key in nonincrease:
        if int(after[key]) > int(before[key]):
            failures.append(f"{key}_regressed")
    if not minimal_policy:
        nondecrease_float = (
            "q_min",
            "q_p01",
            "q_l3_sigma",
            "minimum_angle_deg",
        )
        for key in nondecrease_float:
            if float(after[key]) + 1.0e-12 < float(before[key]):
                failures.append(f"{key}_regressed")
    if (
        float(after["maximum_adjacent_area_change"])
        > float(before["maximum_adjacent_area_change"]) + 1.0e-12
    ):
        failures.append("maximum_adjacent_area_change_regressed")
    for key in ("l_over_h_p95", "l_over_h_maximum"):
        baseline = float(before[key])
        tolerance = 1.0e-3 * max(abs(baseline), 1.0e-12)
        if float(after[key]) > baseline + tolerance:
            failures.append(f"{key}_regressed")
    return sorted(set(failures))


def _minimal_report_only_deltas(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Expose benchmark-first Class-2 deltas without vetoing a repair."""

    q_before = float(before["q_p01"])
    q_after = float(after["q_p01"])
    area_before = int(before["area_transition_defect_count"])
    area_after = int(after["area_transition_defect_count"])
    return {
        "policy": "regional_refinement_debt_never_outer_stage_veto",
        "benchmark_first_policy": True,
        "q_p01": {
            "before": q_before,
            "after": q_after,
            "delta": q_after - q_before,
            "regressed_under_previous_gate": bool(q_after + 1.0e-9 < q_before),
        },
        "area_transition_defect_count_above_0_50": {
            "before": area_before,
            "after": area_after,
            "delta": area_after - area_before,
            "regressed_under_previous_gate": bool(area_after > area_before),
        },
        "q_min": {
            "before": float(before["q_min"]),
            "after": float(after["q_min"]),
            "delta": float(after["q_min"]) - float(before["q_min"]),
        },
        "q_l3_sigma": {
            "before": float(before["q_l3_sigma"]),
            "after": float(after["q_l3_sigma"]),
            "delta": float(after["q_l3_sigma"]) - float(before["q_l3_sigma"]),
        },
        "minimum_angle_deg": {
            "before": float(before["minimum_angle_deg"]),
            "after": float(after["minimum_angle_deg"]),
            "delta": float(after["minimum_angle_deg"]) - float(before["minimum_angle_deg"]),
        },
        "maximum_adjacent_area_change": {
            "before": float(before["maximum_adjacent_area_change"]),
            "after": float(after["maximum_adjacent_area_change"]),
            "delta": float(after["maximum_adjacent_area_change"]) - float(before["maximum_adjacent_area_change"]),
        },
        "singly_connected_triangle_count": {
            "before": int(before["singly_connected_triangle_count"]),
            "after": int(after["singly_connected_triangle_count"]),
            "delta": int(after["singly_connected_triangle_count"]) - int(before["singly_connected_triangle_count"]),
        },
        "l_over_h": {
            "p95_before": float(before["l_over_h_p95"]),
            "p95_after": float(after["l_over_h_p95"]),
            "maximum_before": float(before["l_over_h_maximum"]),
            "maximum_after": float(after["l_over_h_maximum"]),
        },
    }


def _mapped_source_chains(
    source_chains: Iterable[Iterable[int]],
    raw_lineage: np.ndarray,
    label: str,
) -> list[list[int]]:
    reverse = _raw_to_delivered(raw_lineage)
    mapped: list[list[int]] = []
    for chain_index, source_chain in enumerate(source_chains):
        delivered_chain: list[int] = []
        for source in map(int, source_chain):
            if source not in reverse:
                raise ValueError(
                    f"{label} chain {chain_index} lost source node {source + 1}"
                )
            delivered_chain.append(int(reverse[source]))
        if len(delivered_chain) != len(set(delivered_chain)):
            raise ValueError(
                f"{label} chain {chain_index} has duplicate delivered nodes"
            )
        mapped.append(delivered_chain)
    return mapped


def _raw_to_delivered(raw_lineage: np.ndarray) -> dict[int, int]:
    reverse: dict[int, int] = {}
    for delivered, source in enumerate(np.asarray(raw_lineage, dtype=int)):
        if int(source) < 0:
            continue
        if int(source) in reverse:
            raise ValueError(f"Raw source node {int(source) + 1} has duplicate lineage")
        reverse[int(source)] = int(delivered)
    return reverse


def _triangle_targets(
    points: np.ndarray,
    triangles: np.ndarray,
    target_sampler_xy: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    tris = np.asarray(triangles, dtype=int)
    if not len(tris):
        return np.empty(0, dtype=float)
    centroids = np.asarray(points, dtype=float)[tris].mean(axis=1)
    return np.asarray(target_sampler_xy(centroids), dtype=float)


def _edge_size_continuity_audit(
    points: np.ndarray,
    triangles: np.ndarray,
    boundary_chains: list[list[int]],
    target_sampler_xy: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    targets = np.asarray(target_sampler_xy(points), dtype=float)
    return audit_edge_target_sizes(
        points,
        triangles,
        [
            {
                "chain_id": f"constraint_{index:03d}",
                "nodes": chain,
                "cyclic": True,
            }
            for index, chain in enumerate(boundary_chains, start=1)
        ],
        target_sampler_xy,
        boundary_target_by_node=targets,
        transition_graph_rings=2,
        thresholds=(1.55, 2.0),
        boundary_gradation_limit=0.10,
        boundary_field_ratio_limit=1.50,
        boundary_first_ring_p95_limit=1.50,
        boundary_first_ring_maximum_limit=2.0,
    )


def _load_canonical_size_field(path: Path) -> _RegularGrid:
    with xr.open_dataset(path) as dataset:
        schema = str(dataset.attrs.get("schema_version", "")).strip()
        if schema != "fvcom_size_field_v4":
            raise ValueError(
                "Canonical portfolio conditioning requires "
                f"schema_version=fvcom_size_field_v4, received {schema!r}"
            )
        for name in ("lon", "lat", "mesh_size_m", "size_field_coverage_mask"):
            if name not in dataset:
                raise ValueError(f"Canonical size field is missing {name!r}")
        lon = np.asarray(dataset["lon"].values, dtype=float)
        lat = np.asarray(dataset["lat"].values, dtype=float)
        values = np.asarray(
            dataset["mesh_size_m"].transpose("lat", "lon").values,
            dtype=float,
        )
        coverage = np.asarray(
            dataset["size_field_coverage_mask"]
            .transpose("lat", "lon")
            .values,
            dtype=bool,
        )
        domain = (
            np.asarray(
                dataset["model_domain_mask"]
                .transpose("lat", "lon")
                .values,
                dtype=bool,
            )
            if "model_domain_mask" in dataset
            else coverage.copy()
        )
        sampling_interface_schema = str(
            dataset.attrs.get(
                "sampling_interface_schema_version",
                "legacy_unspecified",
            )
        ).strip()
    supported_sampling_interfaces = {
        "legacy_unspecified",
        "fvcom_size_sampling_halo_v1",
        "fvcom_wet_mask_sampling_v1",
        "fvcom_wet_mask_sampling_v2",
    }
    if sampling_interface_schema not in supported_sampling_interfaces:
        raise ValueError(
            "Unsupported canonical sampling interface schema "
            f"{sampling_interface_schema!r}"
        )
    reverse_lon = bool(np.all(np.diff(lon) < 0.0))
    reverse_lat = bool(np.all(np.diff(lat) < 0.0))
    if reverse_lon:
        domain = domain[:, ::-1]
    if reverse_lat:
        domain = domain[::-1, :]
    lon, lat, values, coverage = _normalize_regular_grid(
        lon,
        lat,
        values,
        coverage,
        "canonical size field",
    )
    if np.any(coverage & (~np.isfinite(values) | (values <= 0.0))):
        raise ValueError(
            "Canonical size field contains nonfinite or nonpositive covered cells"
        )
    return _RegularGrid(
        lon=lon,
        lat=lat,
        values=values,
        coverage=coverage,
        value_name="canonical mesh_size_m",
        source_path=path,
        source_sha256=_sha256(path),
        schema_version=schema,
        domain_mask=domain,
        sampling_interface_schema_version=sampling_interface_schema,
    )


def _load_bathymetry_grid(path: Path) -> _RegularGrid:
    bathymetry = load_bathymetry(path)
    values = np.asarray(bathymetry.depth, dtype=float)
    coverage = np.isfinite(values) & (values > 0.0)
    lon, lat, values, coverage = _normalize_regular_grid(
        np.asarray(bathymetry.lon, dtype=float),
        np.asarray(bathymetry.lat, dtype=float),
        values,
        coverage,
        "immutable bathymetry",
    )
    return _RegularGrid(
        lon=lon,
        lat=lat,
        values=values,
        coverage=coverage,
        value_name="positive-down bathymetry",
        source_path=path,
        source_sha256=_sha256(path),
        schema_version=None,
    )


def _normalize_regular_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    coverage: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    values = np.asarray(values, dtype=float)
    coverage = np.asarray(coverage, dtype=bool)
    if lon.ndim != 1 or lat.ndim != 1 or len(lon) < 2 or len(lat) < 2:
        raise ValueError(f"{label} requires one-dimensional lon/lat axes")
    if values.shape != (len(lat), len(lon)) or coverage.shape != values.shape:
        raise ValueError(f"{label} data do not match the lon/lat grid")
    if np.all(np.diff(lon) < 0.0):
        lon = lon[::-1]
        values = values[:, ::-1]
        coverage = coverage[:, ::-1]
    if np.all(np.diff(lat) < 0.0):
        lat = lat[::-1]
        values = values[::-1, :]
        coverage = coverage[::-1, :]
    if np.any(~np.isfinite(lon)) or np.any(np.diff(lon) <= 0.0):
        raise ValueError(f"{label} longitude axis must be finite and monotonic")
    if np.any(~np.isfinite(lat)) or np.any(np.diff(lat) <= 0.0):
        raise ValueError(f"{label} latitude axis must be finite and monotonic")
    return lon, lat, values, coverage


def _input_open_boundaries(mesh: Mesh2DM) -> tuple[list[list[int]], list[int]]:
    chains_1based = (
        [np.asarray(chain, dtype=int) for chain in mesh.open_boundary_chains]
        if mesh.open_boundary_chains
        else (
            [np.asarray(mesh.open_boundary_nodes, dtype=int)]
            if len(mesh.open_boundary_nodes)
            else []
        )
    )
    ids = (
        list(map(int, mesh.open_boundary_ids))
        if mesh.open_boundary_ids
        else list(range(1, len(chains_1based) + 1))
    )
    if (
        len(ids) != len(chains_1based)
        or any(value <= 0 for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Input 2DM has invalid or duplicate OBC nodestring IDs")
    chains: list[list[int]] = []
    membership: set[int] = set()
    for index, values in enumerate(chains_1based):
        chain = [int(value) - 1 for value in values]
        if len(chain) < 2:
            raise ValueError(f"OBC chain {index} has fewer than two nodes")
        if chain[0] == chain[-1]:
            raise UnsupportedCyclicOpenBoundaryError(
                f"OBC chain ID {ids[index]} repeats its first node; cyclic OBCs "
                "cannot be represented safely by the portfolio 2DM contract"
            )
        if len(chain) != len(set(chain)):
            raise ValueError(f"OBC chain ID {ids[index]} contains duplicate nodes")
        shared = membership.intersection(chain)
        if shared:
            raise ValueError(
                f"OBC chain ID {ids[index]} shares nodes with another OBC"
            )
        membership.update(chain)
        chains.append(chain)
    return chains, ids


def _cyclicity_contract(
    chains: list[list[int]],
    ids: list[int],
    boundary_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if boundary_contract is not None:
        declared_count = int(
            boundary_contract.get(
                "open_boundary_count",
                len(boundary_contract.get("open_boundary_cyclic", [])),
            )
        )
        cyclic = list(boundary_contract.get("open_boundary_cyclic", []))
        if declared_count != len(chains) or len(cyclic) != len(chains):
            raise ValueError(
                "Boundary contract OBC count/cyclicity does not match the "
                "input 2DM nodestring count"
            )
        declared_ids = list(
            boundary_contract.get("open_boundary_ids", ids)
        )
        if len(declared_ids) != len(chains):
            raise ValueError(
                "Boundary contract open_boundary_ids count does not match "
                "the input 2DM"
            )
        return {
            "supported": True,
            "input_format": "SMS_2DM_NS_plus_external_boundary_contract",
            "boundary_contract_sha256": hashlib.sha256(
                json.dumps(
                    boundary_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "reason": (
                "cyclicity supplied by an external hash-bound sidecar; "
                "SMS 2DM remains non-self-describing"
            ),
            "policy": "external_cyclicity_sidecar",
            "explicit_repeated_first_last_node_rejected": True,
            "chains": [
                {
                    "nodestring_id": int(nodestring_id),
                    "declared_open_boundary_id": str(declared_id),
                    "cyclic": bool(is_cyclic),
                    "cyclicity": (
                        "cyclic" if bool(is_cyclic) else "noncyclic"
                    ),
                    "processed_as": (
                        "ordered_cyclic_sidecar_chain"
                        if bool(is_cyclic)
                        else "ordered_noncyclic_arc"
                    ),
                    "node_count": int(len(chain)),
                }
                for nodestring_id, declared_id, chain, is_cyclic in zip(
                    ids,
                    declared_ids,
                    chains,
                    cyclic,
                )
            ],
        }
    return {
        "supported": False,
        "input_format": "SMS_2DM_NS",
        "reason": "SMS 2DM NS records do not encode OBC cyclicity",
        "policy": "ordered_noncyclic_arcs_only",
        "explicit_repeated_first_last_node_rejected": True,
        "chains": [
            {
                "nodestring_id": int(nodestring_id),
                "declared_open_boundary_id": str(nodestring_id),
                "cyclic": False,
                "cyclicity": "unknown_not_encoded",
                "processed_as": "ordered_noncyclic_arc",
                "node_count": int(len(chain)),
            }
            for nodestring_id, chain in zip(ids, chains)
        ],
    }


def _roundtrip_audit(
    path: Path,
    expected: _ConditioningState,
    projection: LocalProjection,
    expected_obc_chains: list[list[int]],
    expected_obc_ids: list[int],
) -> dict[str, Any]:
    written = read_2dm(path)
    expected_triangles = np.asarray(expected.triangles, dtype=int) + 1
    written_chains = [
        np.asarray(chain, dtype=int).tolist()
        for chain in written.open_boundary_chains
    ]
    expected_chains = [
        (np.asarray(chain, dtype=int) + 1).tolist()
        for chain in expected_obc_chains
    ]
    node_count_match = bool(len(written.nodes_lonlat) == len(expected.points))
    coordinate_shift = float("inf")
    if node_count_match:
        coordinate_shift = float(
            np.max(
                np.linalg.norm(
                    project_points(written.nodes_lonlat, projection)
                    - expected.points,
                    axis=1,
                )
            )
        )
    triangle_match = bool(
        np.array_equal(written.triangles, expected_triangles)
    )
    chain_match = bool(written_chains == expected_chains)
    id_match = bool(
        list(map(int, written.open_boundary_ids))
        == list(map(int, expected_obc_ids))
    )
    finite_positive_depths = bool(
        np.all(np.isfinite(written.depths)) and np.all(written.depths > 0.0)
    )
    positive_area = False
    minimum_signed_area = float("nan")
    if node_count_match and triangle_match and len(written.triangles):
        geometry = triangle_geometry(
            project_points(written.nodes_lonlat, projection),
            written.triangles - 1,
        )
        minimum_signed_area = float(np.min(geometry["signed_area"]))
        positive_area = bool(np.all(geometry["signed_area"] > 0.0))
    passed = bool(
        node_count_match
        and coordinate_shift <= 0.01
        and triangle_match
        and chain_match
        and id_match
        and finite_positive_depths
        and positive_area
    )
    return {
        "passed": passed,
        "node_count_match": node_count_match,
        "coordinate_max_shift_m": coordinate_shift,
        "coordinate_tolerance_m": 0.01,
        "triangle_connectivity_match": triangle_match,
        "open_boundary_chain_count_match": bool(
            len(written_chains) == len(expected_chains)
        ),
        "open_boundary_chain_order_match": chain_match,
        "open_boundary_id_match": id_match,
        "finite_positive_depths": finite_positive_depths,
        "positive_signed_areas": positive_area,
        "minimum_signed_area_m2": minimum_signed_area,
    }


def _obc_remap_manifest(
    mesh_path: Path,
    raw_mesh: Mesh2DM,
    source_chains: list[list[int]],
    source_ids: list[int],
    delivered_chains: list[list[int]],
    raw_lineage: np.ndarray,
    cyclicity: dict[str, Any],
    roundtrip: dict[str, Any],
    source_boundary_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = []
    exact = True
    for nodestring_id, source, delivered, cycle_record in zip(
        source_ids,
        source_chains,
        delivered_chains,
        cyclicity["chains"],
    ):
        source_1based = (np.asarray(source, dtype=int) + 1).tolist()
        delivered_1based = (np.asarray(delivered, dtype=int) + 1).tolist()
        retained = bool(len(source_1based) == len(delivered_1based))
        exact &= retained
        records.append(
            {
                "nodestring_id": int(nodestring_id),
                "source_node_ids_1based": source_1based,
                "delivered_node_ids_1based": delivered_1based,
                "source_node_count": int(len(source_1based)),
                "delivered_node_count": int(len(delivered_1based)),
                "source_sequence_retained_by_lineage": retained,
                "orientation_preserved": retained,
                "cyclicity": str(cycle_record["cyclicity"]),
                "cyclic": bool(cycle_record["cyclic"]),
            }
        )
    source_forcing_compatible = True
    if source_boundary_metadata is not None and source_chains:
        source_forcing_compatible = bool(
            source_boundary_metadata.get(
                "forcing_compatible",
                source_boundary_metadata.get(
                    "forcing_compatible_without_remap",
                    True,
                ),
            )
        )
    return {
        "schema_version": "fvcom_portfolio_obc_remap_v1",
        "input_mesh": str(mesh_path),
        "input_mesh_sha256": _sha256(mesh_path),
        "input_mesh_name": raw_mesh.mesh_name,
        "source_chain_count": int(len(source_chains)),
        "delivered_chain_count": int(len(delivered_chains)),
        "source_nodestring_ids": list(map(int, source_ids)),
        "delivered_nodestring_ids": list(map(int, source_ids)),
        "chains": records,
        "node_lineage_current_to_raw_zero_based": np.asarray(
            raw_lineage,
            dtype=int,
        ).tolist(),
        "cyclicity_contract": cyclicity,
        "source_boundary_metadata_supplied": bool(
            source_boundary_metadata is not None
        ),
        "source_boundary_forcing_compatible": (
            source_forcing_compatible
        ),
        "forcing_interpolation_performed": False,
        "forcing_compatible": bool(
            exact
            and roundtrip["open_boundary_chain_order_match"]
            and roundtrip["open_boundary_id_match"]
            and source_forcing_compatible
        ),
    }


def _boundary_geojson(
    lonlat: np.ndarray,
    boundary_chains: list[list[int]],
    obc_chains: list[list[int]],
    obc_ids: list[int],
    raw_lineage: np.ndarray,
    targets: np.ndarray,
    open_boundary_cyclic: list[bool],
) -> dict[str, Any]:
    memberships: dict[int, list[int]] = {}
    cyclic_memberships: dict[int, list[bool]] = {}
    for nodestring_id, chain, is_cyclic in zip(
        obc_ids,
        obc_chains,
        open_boundary_cyclic,
    ):
        for node in chain:
            memberships.setdefault(int(node), []).append(int(nodestring_id))
            cyclic_memberships.setdefault(int(node), []).append(
                bool(is_cyclic)
            )
    features: list[dict[str, Any]] = []
    for chain_index, chain in enumerate(boundary_chains):
        for position, node in enumerate(chain):
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "node_index_zero_based": int(node),
                        "node_id_1based": int(node) + 1,
                        "constraint_chain_id": int(chain_index),
                        "constraint_chain_position": int(position),
                        "source_node_index_zero_based": (
                            int(raw_lineage[node])
                            if int(raw_lineage[node]) >= 0
                            else None
                        ),
                        "source_node_id_1based": (
                            int(raw_lineage[node]) + 1
                            if int(raw_lineage[node]) >= 0
                            else None
                        ),
                        "is_open_boundary": bool(int(node) in memberships),
                        "open_boundary_nodestring_ids": memberships.get(
                            int(node),
                            [],
                        ),
                        "open_boundary_cyclic": cyclic_memberships.get(
                            int(node),
                            [],
                        ),
                        "boundary_kind": "fixed_topological_boundary",
                        "is_hard_anchor": True,
                        "target_spacing_m": float(targets[node]),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            float(lonlat[node, 0]),
                            float(lonlat[node, 1]),
                        ],
                    },
                }
            )
    return {
        "type": "FeatureCollection",
        "schema_version": "fvcom_portfolio_boundary_nodes_v1",
        "features": features,
    }


def _validate_mesh_arrays(mesh: Mesh2DM) -> None:
    nodes = np.asarray(mesh.nodes_lonlat, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=int)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) < 3:
        raise ValueError("Raw 2DM must contain at least three finite lon/lat nodes")
    if np.any(~np.isfinite(nodes)):
        raise ValueError("Raw 2DM contains nonfinite coordinates")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("Raw 2DM must contain triangular E3T elements")
    if int(np.min(triangles)) < 1 or int(np.max(triangles)) > len(nodes):
        raise ValueError(
            "Raw 2DM connectivity requires contiguous one-based node IDs"
        )
    if any(len(set(map(int, row))) != 3 for row in triangles):
        raise ValueError("Raw 2DM contains a repeated-node triangle")


def _orient_ccw(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, int]:
    output = np.asarray(triangles, dtype=int).copy()
    geometry = triangle_geometry(np.asarray(points, dtype=float), output)
    clockwise = np.where(geometry["signed_area"] < 0.0)[0]
    if len(clockwise):
        swap = output[clockwise, 1].copy()
        output[clockwise, 1] = output[clockwise, 2]
        output[clockwise, 2] = swap
    degenerate = np.where(
        triangle_geometry(np.asarray(points, dtype=float), output)[
            "signed_area"
        ]
        <= 0.0
    )[0]
    if len(degenerate):
        raise ValueError(
            f"Raw 2DM contains {len(degenerate)} degenerate triangle(s)"
        )
    return output, int(len(clockwise))


def _bbox(lonlat: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(lonlat, dtype=float)
    return (
        float(np.min(values[:, 0])),
        float(np.min(values[:, 1])),
        float(np.max(values[:, 0])),
        float(np.max(values[:, 1])),
    )


def _public_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in audit.items() if not str(key).startswith("_")
    }


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


__all__ = [
    "PortfolioConditioningConfig",
    "UnsupportedCyclicOpenBoundaryError",
    "condition_portfolio_mesh",
]
