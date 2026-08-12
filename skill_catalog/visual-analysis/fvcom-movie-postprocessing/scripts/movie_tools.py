"""Compatibility shim for the renamed FVCOM movie toolkit."""

from __future__ import annotations

try:
    from .fvcom_movie_tools import *  # noqa: F401,F403
except ImportError:
    from fvcom_movie_tools import *  # noqa: F401,F403
