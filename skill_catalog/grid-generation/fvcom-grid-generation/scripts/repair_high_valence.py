#!/usr/bin/env python3
"""Remove or repartition FVCOM nodes whose true valence exceeds eight."""

from condition_mesh_local import main_with_mode


if __name__ == "__main__":
    raise SystemExit(main_with_mode("valence"))
