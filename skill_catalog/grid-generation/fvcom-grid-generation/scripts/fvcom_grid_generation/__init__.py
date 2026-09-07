"""Clean-room Python FVCOM grid-generation helpers.

Keep the high-level workflow import lazy so focused topology and contract
audits do not require optional GIS drivers merely to import their submodule.
"""

from __future__ import annotations

from typing import Any


__all__ = ["GridConfig", "run_fvcom_grid"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .workflow import GridConfig, run_fvcom_grid

        return {"GridConfig": GridConfig, "run_fvcom_grid": run_fvcom_grid}[name]
    raise AttributeError(name)
