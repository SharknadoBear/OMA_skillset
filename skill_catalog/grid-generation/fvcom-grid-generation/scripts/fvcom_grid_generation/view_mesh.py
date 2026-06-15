"""Plot FVCOM 2DM meshes and quality failures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from .mesh_quality import evaluate_mesh_quality
from .sms_2dm import read_2dm


def plot_2dm(mesh_path: str | Path, output_png: str | Path | None = None, show: bool = False) -> Path | None:
    mesh = read_2dm(mesh_path)
    tri0 = mesh.triangles - 1
    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], tri0)
    quality = evaluate_mesh_quality(
        mesh.nodes,
        mesh.depths,
        mesh.triangles,
        mesh.open_boundaries[0] if mesh.open_boundaries else None,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    depth_plot = ax.tricontourf(triang, mesh.depths, levels=24, cmap="viridis")
    fig.colorbar(depth_plot, ax=ax, label="Depth positive down (m)")
    ax.triplot(triang, color="white", linewidth=0.2, alpha=0.45)
    if mesh.open_boundaries:
        ob = mesh.open_boundaries[0] - 1
        ax.plot(mesh.nodes[ob, 0], mesh.nodes[ob, 1], color="red", linewidth=2.0, label="Open boundary")
    failed = quality["failed_angle_elements"]
    if failed.size:
        centers = mesh.nodes[tri0[failed - 1]].mean(axis=1)
        ax.scatter(centers[:, 0], centers[:, 1], s=10, c="magenta", label="Angle failures")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{Path(mesh_path).name}: {quality['n_nodes']} nodes, {quality['n_triangles']} triangles"
    )
    ax.legend(loc="best")
    fig.tight_layout()

    result = Path(output_png) if output_png is not None else None
    if result is not None:
        fig.savefig(result, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot an FVCOM/SMS 2DM mesh.")
    parser.add_argument("mesh")
    parser.add_argument("--output-png", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    result = plot_2dm(args.mesh, args.output_png, args.show)
    if result is not None:
        print(f"Wrote {result}")


if __name__ == "__main__":
    main()
