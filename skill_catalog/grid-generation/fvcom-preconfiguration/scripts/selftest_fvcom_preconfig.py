from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

try:
    from .mesh_io import parse_2dm
except ImportError:  # pragma: no cover
    from mesh_io import parse_2dm


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="ascii").splitlines()


def assert_match(pattern: str, text: str, label: str) -> None:
    if not re.fullmatch(pattern, text.strip()):
        raise AssertionError(f"{label} has unexpected format: {text!r}")


def parse_dep(path: Path) -> list[tuple[float, float, float]]:
    lines = read_lines(path)
    out: list[tuple[float, float, float]] = []
    for line in lines[1:]:
        x, y, h = map(float, line.split())
        out.append((x, y, h))
    return out


def parse_cor(path: Path) -> list[tuple[float, float, float]]:
    return parse_dep(path)


def parse_obc(path: Path) -> list[tuple[int, int, int]]:
    rows = []
    for line in read_lines(path)[1:]:
        counter, node_id, kind = map(int, line.split())
        rows.append((counter, node_id, kind))
    return rows


def parse_spg(path: Path) -> list[tuple[int, float, float]]:
    rows = []
    for line in read_lines(path)[1:]:
        node_id, radius, coeff = line.split()
        rows.append((int(node_id), float(radius), float(coeff)))
    return rows


def check_reference_shape(reference_dir: Path, generated_dir: Path, prefix: str) -> None:
    for suffix in ("grd", "dep", "cor", "obc", "spg"):
        ref = reference_dir / f"{prefix}_{suffix}.dat"
        gen = generated_dir / f"{prefix}_{suffix}.dat"
        if not ref.exists():
            continue
        ref_head = read_lines(ref)[:2]
        gen_head = read_lines(gen)[:2]
        if suffix in {"grd"}:
            if ref_head[0].split("=")[0] != gen_head[0].split("=")[0]:
                raise AssertionError(f"{suffix} first header key differs from reference")
            if ref_head[1].split("=")[0] != gen_head[1].split("=")[0]:
                raise AssertionError(f"{suffix} second header key differs from reference")
        else:
            if ref_head[0].split("=")[0] != gen_head[0].split("=")[0]:
                raise AssertionError(f"{suffix} header key differs from reference")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate FVCOM preconfiguration outputs.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="waterPACT")
    parser.add_argument("--expect-nodes", type=int, default=None)
    parser.add_argument("--expect-elements", type=int, default=None)
    parser.add_argument("--expect-depth", type=float, default=None)
    parser.add_argument("--expect-cor", type=float, default=None)
    parser.add_argument("--expect-obc-nodes", default=None, help="Comma separated expected OBC node ids")
    parser.add_argument("--expect-sig-levels", type=int, default=None)
    parser.add_argument("--reference-dir", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mesh = parse_2dm(args.mesh)
    out_dir = Path(args.out_dir)
    prefix = args.prefix

    if args.expect_nodes is not None and len(mesh.nodes) != args.expect_nodes:
        raise AssertionError(f"node count {len(mesh.nodes)} != {args.expect_nodes}")
    if args.expect_elements is not None and len(mesh.elements) != args.expect_elements:
        raise AssertionError(f"element count {len(mesh.elements)} != {args.expect_elements}")

    grd = read_lines(out_dir / f"{prefix}_grd.dat")
    assert grd[0] == f"Node Number = {len(mesh.nodes)}"
    assert grd[1] == f"Cell Number = {len(mesh.elements)}"
    assert_match(r"\d+ \d+ \d+ \d+", grd[2], "first grid element")
    assert_match(rf"\d+ {FLOAT_RE} {FLOAT_RE} {FLOAT_RE}", grd[2 + len(mesh.elements)], "first grid node")

    dep = parse_dep(out_dir / f"{prefix}_dep.dat")
    cor = parse_cor(out_dir / f"{prefix}_cor.dat")
    obc = parse_obc(out_dir / f"{prefix}_obc.dat")
    spg = parse_spg(out_dir / f"{prefix}_spg.dat")

    if len(dep) != len(mesh.nodes):
        raise AssertionError("dep row count does not match mesh node count")
    if len(cor) != len(mesh.nodes):
        raise AssertionError("cor row count does not match mesh node count")
    if any(row[2] <= 0 for row in dep):
        raise AssertionError("dep file contains non-positive depth")
    if args.expect_depth is not None:
        for _, _, depth in dep:
            if not math.isclose(depth, args.expect_depth, rel_tol=0.0, abs_tol=1e-6):
                raise AssertionError(f"unexpected depth {depth}")
    if args.expect_cor is not None:
        for _, _, value in cor:
            if not math.isclose(value, args.expect_cor, rel_tol=0.0, abs_tol=1e-6):
                raise AssertionError(f"unexpected coriolis value {value}")
    if args.expect_obc_nodes:
        expected = [int(item) for item in args.expect_obc_nodes.split(",")]
        actual = [node_id for _, node_id, _ in obc]
        if actual != expected:
            raise AssertionError(f"OBC nodes {actual} != {expected}")
    if len(spg) != len(obc):
        raise AssertionError("sponge node count does not match OBC node count")
    if any(radius <= 0 or coeff <= 0 for _, radius, coeff in spg):
        raise AssertionError("sponge radius/coefficient must be positive")

    sig_path = out_dir / f"{prefix}_sig.dat"
    if args.expect_sig_levels is not None:
        sig_lines = read_lines(sig_path)
        if sig_lines[0] != f"NUMBER OF SIGMA LEVELS = {args.expect_sig_levels}":
            raise AssertionError("unexpected sigma level count")
        if sig_lines[1] != "SIGMA COORDINATE TYPE = UNIFORM":
            raise AssertionError("expected active UNIFORM sigma type")

    if args.reference_dir:
        check_reference_shape(Path(args.reference_dir), out_dir, prefix)

    print("FVCOM preconfiguration selftest passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
