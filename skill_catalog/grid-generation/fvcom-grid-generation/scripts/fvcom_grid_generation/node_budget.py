"""Shared default node-budget policy for FVCOM mesh generators."""

from __future__ import annotations


DEFAULT_HARD_NODE_LIMIT = 1_000_000
"""Maximum delivered node count for a default regional run."""

DEFAULT_PREFLIGHT_NODE_LIMIT = 900_000
"""Planning threshold that reserves ten percent below the hard cap."""

DEFAULT_NODE_BUDGET_STOP_FRACTION = (
    DEFAULT_PREFLIGHT_NODE_LIMIT / DEFAULT_HARD_NODE_LIMIT
)
"""Fraction of the hard cap available to the pre-triangulation estimate."""

DEFAULT_MAX_INTERIOR_POINTS = DEFAULT_PREFLIGHT_NODE_LIMIT
"""Default clean-room interior seed ceiling before boundary nodes are added."""

DEFAULT_SPACING_QUANTUM_M = 25.0
"""Numerical rounding/search quantum; not bathymetry raster resolution."""


def delivered_node_budget_report(
    node_count: int,
    maximum_total_nodes: int = DEFAULT_HARD_NODE_LIMIT,
) -> dict[str, object]:
    """Return the hard delivered-node audit shared by production workflows."""

    delivered = int(node_count)
    maximum = int(maximum_total_nodes)
    if delivered < 0:
        raise ValueError("node_count must be non-negative")
    if maximum < 1:
        raise ValueError("maximum_total_nodes must be positive")
    passed = bool(delivered <= maximum)
    return {
        "schema_version": "fvcom_delivered_node_budget_v1",
        "delivered_node_count": delivered,
        "maximum_total_nodes": maximum,
        "remaining_node_capacity": int(maximum - delivered),
        "passed": passed,
        "hard_gate_enforced": True,
        "failure_taxonomy": (
            [] if passed else ["hard_node_cap_exceeded"]
        ),
    }


__all__ = [
    "DEFAULT_HARD_NODE_LIMIT",
    "DEFAULT_MAX_INTERIOR_POINTS",
    "DEFAULT_NODE_BUDGET_STOP_FRACTION",
    "DEFAULT_PREFLIGHT_NODE_LIMIT",
    "DEFAULT_SPACING_QUANTUM_M",
    "delivered_node_budget_report",
]
