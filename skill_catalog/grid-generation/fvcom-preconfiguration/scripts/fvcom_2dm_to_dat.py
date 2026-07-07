from __future__ import annotations

import argparse
import json

try:
    from .mesh_io import parse_2dm, write_fvcom_dat
except ImportError:  # pragma: no cover - supports direct script execution
    from mesh_io import parse_2dm, write_fvcom_dat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an SMS 2DM mesh with nodestrings to FVCOM ASCII grid DAT files."
    )
    parser.add_argument("--mesh", required=True, help="Input SMS .2dm mesh")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--prefix", default="waterPACT", help="FVCOM case prefix")
    parser.add_argument("--open-ns", type=int, required=True, help="Nodestring id used as open boundary")
    parser.add_argument("--river-ns", type=int, default=None, help="River/inflow nodestring id to record as excluded")
    parser.add_argument("--obc-type", default="prescribed", help="prescribed/1 or radiation/3")
    parser.add_argument("--depth-mode", default="auto", choices=["auto", "negate-z", "positive-z"])
    parser.add_argument("--constant-depth", type=float, default=None, help="Override all mesh depths with this value")
    parser.add_argument(
        "--coriolis-mode",
        default="zero",
        choices=["zero", "latitude", "constant-latitude", "y-coordinate", "node-y"],
        help="Values written to _cor.dat before FVCOM internal conversion",
    )
    parser.add_argument("--latitude-deg", type=float, default=None, help="Constant latitude for coriolis-mode latitude")
    parser.add_argument("--sponge-mode", default="estimate", choices=["estimate", "constant"])
    parser.add_argument("--sponge-radius", type=float, default=None, help="Constant sponge radius when requested")
    parser.add_argument("--sponge-coeff", type=float, default=0.0025, help="Initial sponge damping coefficient")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mesh = parse_2dm(args.mesh)
    manifest = write_fvcom_dat(
        mesh,
        out_dir=args.out_dir,
        prefix=args.prefix,
        open_ns=args.open_ns,
        river_ns=args.river_ns,
        obc_type=args.obc_type,
        depth_mode=args.depth_mode,
        constant_depth=args.constant_depth,
        coriolis_mode=args.coriolis_mode,
        latitude_deg=args.latitude_deg,
        sponge_mode=args.sponge_mode,
        sponge_coeff=args.sponge_coeff,
        sponge_radius=args.sponge_radius,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

