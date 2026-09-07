from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .mesh_io import ordered_nodes_sha256, parse_2dm, sha256_file, write_fvcom_dat
except ImportError:  # pragma: no cover - supports direct script execution
    from mesh_io import ordered_nodes_sha256, parse_2dm, sha256_file, write_fvcom_dat


CONTRACT_SCHEMA = "fvcom_grid_delivery_contract_v1"


def _flatten_int_args(values: list[str] | None) -> list[int]:
    out: list[int] = []
    for value in values or []:
        out.extend(int(item) for item in value.replace(",", " ").split())
    return out


def _resolve_contract_path(contract_path: Path, supplied: str) -> Path:
    path = Path(supplied)
    return path if path.is_absolute() else contract_path.parent / path


def load_grid_contract(path: str | Path) -> tuple[dict[str, object], Path]:
    contract_path = Path(path).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError(f"Grid contract must use schema_version {CONTRACT_SCHEMA}")
    mesh_spec = contract.get("mesh")
    if not isinstance(mesh_spec, dict):
        raise ValueError("Grid contract requires a mesh object")
    required_mesh = {
        "path",
        "sha256",
        "node_count",
        "triangle_count",
        "coordinate_crs",
        "horizontal_crs",
        "horizontal_units",
    }
    missing_mesh = sorted(required_mesh - set(mesh_spec))
    if missing_mesh:
        raise ValueError(f"Grid contract mesh is missing: {missing_mesh}")
    if str(mesh_spec["horizontal_units"]).lower() not in {
        "m",
        "meter",
        "metre",
        "meters",
        "metres",
    }:
        raise ValueError("Grid contract coordinates must be projected in meters")
    boundaries = contract.get("open_boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("Grid contract requires a non-empty open_boundaries list")
    required_boundary = {"obc_id", "nodestring_id", "node_ids", "node_order_sha256"}
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, dict):
            raise ValueError(f"open_boundaries[{index}] must be an object")
        missing = sorted(required_boundary - set(boundary))
        if missing:
            raise ValueError(f"open_boundaries[{index}] is missing: {missing}")
        nodes = [int(value) for value in boundary["node_ids"]]
        if not nodes:
            raise ValueError(f"open_boundaries[{index}] has no nodes")
        if ordered_nodes_sha256(nodes) != str(boundary["node_order_sha256"]).lower():
            raise ValueError(f"open_boundaries[{index}] node_order_sha256 is stale")
    return contract, contract_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a complete grid-delivery contract or SMS 2DM mesh to FVCOM DAT files."
    )
    parser.add_argument("--grid-contract", help="fvcom_grid_delivery_contract_v1 JSON (preferred)")
    parser.add_argument("--mesh", help="Input SMS .2dm mesh; required only on the legacy route")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--prefix", default="waterPACT", help="FVCOM case prefix")
    parser.add_argument(
        "--open-ns", action="append", help="Legacy OBC nodestring id; repeat or comma-separate ids"
    )
    parser.add_argument(
        "--river-ns", action="append", help="River nodestring id to exclude; repeat or comma-separate ids"
    )
    parser.add_argument("--obc-type", default="prescribed", help="prescribed/1 or radiation/3")
    parser.add_argument("--depth-mode", default="auto", choices=["auto", "negate-z", "positive-z"])
    parser.add_argument("--constant-depth", type=float, default=None, help="Override all mesh depths")
    parser.add_argument(
        "--coriolis-mode",
        default="auto",
        choices=["auto", "zero", "latitude", "constant-latitude", "y-coordinate", "node-y", "crs-latitude", "projected-latitude"],
        help="auto uses CRS-derived latitude for contracts and zero for the legacy route",
    )
    parser.add_argument("--source-crs", help="CRS serialized in the legacy source mesh")
    parser.add_argument("--target-crs", help="Projected metric output CRS for the legacy route")
    parser.add_argument("--latitude-deg", type=float, default=None)
    parser.add_argument("--sponge-mode", default="estimate", choices=["estimate", "constant"])
    parser.add_argument("--sponge-radius", type=float, default=None)
    parser.add_argument("--sponge-radius-scale", type=float, default=3.0)
    parser.add_argument("--sponge-coeff", type=float, default=0.0025)
    parser.add_argument("--sponge-profile", help="Attempt-specific fvcom_sponge_profile_v1 JSON")
    parser.add_argument("--attempt-id", help="Immutable attempt identifier")
    parser.add_argument("--sigma-levels", type=int, default=None, help="Default: contract value or 10")
    parser.add_argument("--sigma-type", default=None, help="Default: contract value or UNIFORM")
    parser.add_argument(
        "--tge-source",
        help="Exact FVCOM TGE.F used to verify and hash-bind the source-equivalent ISONB gate",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contract: dict[str, object] | None = None
    contract_path: Path | None = None
    contract_hash: str | None = None

    if args.grid_contract:
        contract, contract_path = load_grid_contract(args.grid_contract)
        contract_hash = sha256_file(contract_path)
        mesh_spec = contract["mesh"]
        assert isinstance(mesh_spec, dict)
        mesh_path = _resolve_contract_path(contract_path, str(mesh_spec["path"])).resolve()
        if args.mesh and Path(args.mesh).resolve() != mesh_path:
            raise ValueError("--mesh cannot override the grid contract mesh")
        if sha256_file(mesh_path) != str(mesh_spec["sha256"]).lower():
            raise ValueError("Grid contract mesh SHA-256 does not match the file")
        source_crs = str(mesh_spec["coordinate_crs"])
        target_crs = str(mesh_spec["horizontal_crs"])
        horizontal_units = str(mesh_spec["horizontal_units"])
        open_boundaries = contract["open_boundaries"]
        excluded = contract.get("excluded_nodestring_ids", [])
        river_ids = [int(value) for value in excluded]
        sigma_spec = contract.get("sigma", {})
        if not isinstance(sigma_spec, dict):
            raise ValueError("Grid contract sigma must be an object")
        sigma_levels = args.sigma_levels or int(sigma_spec.get("levels", 10))
        sigma_type = args.sigma_type or str(sigma_spec.get("type", "UNIFORM"))
        coriolis_mode = "crs-latitude" if args.coriolis_mode == "auto" else args.coriolis_mode
    else:
        if not args.mesh:
            raise ValueError("Provide --grid-contract or --mesh")
        mesh_path = Path(args.mesh).resolve()
        open_ids = _flatten_int_args(args.open_ns)
        if not open_ids:
            raise ValueError("Legacy --mesh route requires at least one --open-ns")
        river_ids = _flatten_int_args(args.river_ns)
        source_crs = args.source_crs
        target_crs = args.target_crs or args.source_crs
        horizontal_units = "m"
        sigma_levels = args.sigma_levels or 10
        sigma_type = args.sigma_type or "UNIFORM"
        coriolis_mode = "zero" if args.coriolis_mode == "auto" else args.coriolis_mode
        open_boundaries = [
            {"obc_id": f"obc_{index:03d}", "nodestring_id": ns_id, "obc_type": args.obc_type}
            for index, ns_id in enumerate(open_ids, start=1)
        ]

    mesh = parse_2dm(mesh_path)
    if contract:
        mesh_spec = contract["mesh"]
        assert isinstance(mesh_spec, dict)
        if len(mesh.nodes) != int(mesh_spec["node_count"]):
            raise ValueError("Grid contract node_count does not match the mesh")
        if len(mesh.elements) != int(mesh_spec["triangle_count"]):
            raise ValueError("Grid contract triangle_count does not match the mesh")

    sponge_profile = None
    sponge_profile_path = None
    if args.sponge_profile:
        sponge_profile_path = Path(args.sponge_profile).resolve()
        sponge_profile = json.loads(sponge_profile_path.read_text(encoding="utf-8"))

    manifest = write_fvcom_dat(
        mesh,
        out_dir=args.out_dir,
        prefix=args.prefix,
        river_ns=river_ids,
        obc_type=args.obc_type,
        depth_mode=args.depth_mode,
        constant_depth=args.constant_depth,
        coriolis_mode=coriolis_mode,
        latitude_deg=args.latitude_deg,
        source_crs=source_crs,
        target_crs=target_crs,
        horizontal_units=horizontal_units,
        sponge_mode=args.sponge_mode,
        sponge_coeff=args.sponge_coeff,
        sponge_radius=args.sponge_radius,
        sponge_radius_scale=args.sponge_radius_scale,
        sponge_profile=sponge_profile,
        sponge_profile_path=sponge_profile_path,
        open_boundaries=open_boundaries,
        sigma_levels=sigma_levels,
        sigma_type=sigma_type,
        grid_contract_path=contract_path,
        grid_contract_sha256=contract_hash,
        attempt_id=args.attempt_id,
        tge_source_path=Path(args.tge_source).resolve() if args.tge_source else None,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
