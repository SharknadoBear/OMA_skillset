from __future__ import annotations

import argparse
import json
from pathlib import Path


SIGMA_TYPES = {"UNIFORM", "GEOMETRIC", "TANH", "GENERALIZED", "USER"}


def parse_float_list(value: str | None) -> list[float]:
    if value is None or value.strip() == "":
        return []
    return [float(item) for item in value.replace(",", " ").split()]


def sigma_text(
    levels: int,
    sigma_type: str,
    sigma_power: float = 1.0,
    du: float = 1.0,
    dl: float = 1.0,
    min_constant_depth: float = 5.9,
    ku: int = 0,
    kl: int = 0,
    zku: list[float] | None = None,
    zkl: list[float] | None = None,
) -> str:
    if levels < 2:
        raise ValueError("NUMBER OF SIGMA LEVELS must be at least 2")
    kind = sigma_type.upper()
    if kind not in SIGMA_TYPES:
        raise ValueError(f"Unsupported sigma type {sigma_type!r}; choose from {sorted(SIGMA_TYPES)}")

    lines = [
        f"NUMBER OF SIGMA LEVELS = {levels}",
        f"SIGMA COORDINATE TYPE = {kind}",
    ]

    if kind == "GEOMETRIC":
        lines.append(f"SIGMA POWER = {sigma_power}")
    elif kind == "TANH":
        lines.extend([f"DU = {du}", f"DL = {dl}"])
    elif kind == "GENERALIZED":
        zku = zku or []
        zkl = zkl or []
        if ku < 0 or kl < 0:
            raise ValueError("KU and KL must be non-negative")
        if ku and len(zku) != ku:
            raise ValueError("ZKU length must equal KU")
        if kl and len(zkl) != kl:
            raise ValueError("ZKL length must equal KL")
        lines.extend(
            [
                f"DU = {du}",
                f"DL = {dl}",
                f"MIN CONSTANT DEPTH = {min_constant_depth}",
                f"KU = {ku}",
                f"KL = {kl}",
            ]
        )
        if ku:
            lines.append("ZKU = " + " ".join(f"{value:g}" for value in zku))
        if kl:
            lines.append("ZKL = " + " ".join(f"{value:g}" for value in zkl))
    elif kind == "USER":
        lines.append("! USER sigma requires sigma_level_user.inp in INPUT_DIR.")

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an FVCOM sigma-coordinate configuration file.")
    parser.add_argument("--out", required=True, help="Output _sig.dat file")
    parser.add_argument("--levels", type=int, default=41, help="NUMBER OF SIGMA LEVELS")
    parser.add_argument("--type", default="UNIFORM", choices=sorted(SIGMA_TYPES), help="Sigma coordinate type")
    parser.add_argument("--sigma-power", type=float, default=1.0)
    parser.add_argument("--du", type=float, default=1.0)
    parser.add_argument("--dl", type=float, default=1.0)
    parser.add_argument("--min-constant-depth", type=float, default=5.9)
    parser.add_argument("--ku", type=int, default=0)
    parser.add_argument("--kl", type=int, default=0)
    parser.add_argument("--zku", default=None, help="Comma or space separated ZKU values")
    parser.add_argument("--zkl", default=None, help="Comma or space separated ZKL values")
    parser.add_argument("--manifest", default=None, help="Optional JSON manifest path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    text = sigma_text(
        levels=args.levels,
        sigma_type=args.type,
        sigma_power=args.sigma_power,
        du=args.du,
        dl=args.dl,
        min_constant_depth=args.min_constant_depth,
        ku=args.ku,
        kl=args.kl,
        zku=parse_float_list(args.zku),
        zkl=parse_float_list(args.zkl),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="ascii", newline="\n")

    manifest = {
        "sig_file": str(out),
        "levels": args.levels,
        "sigma_type": args.type.upper(),
        "notes": "FVCOM counts NUMBER OF SIGMA LEVELS as KB; this tool defaults to 41 for the flume branch.",
    }
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

