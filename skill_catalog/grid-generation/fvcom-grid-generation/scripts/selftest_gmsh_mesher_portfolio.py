#!/usr/bin/env python3
"""Focused tests for the generator-neutral Gmsh portfolio adapter.

The fake-Gmsh tests require only NumPy.  When the pinned Gmsh environment is
available, a tiny projected-coordinate fixture also exercises algorithms 1,
5, and 6 with the same canonical size callback.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from typing import Callable

import numpy as np


def _load_backend():
    backend_path = (
        Path(__file__).resolve().parent
        / "fvcom_grid_generation"
        / "gmsh_backend.py"
    )
    module_name = "_fvcom_grid_generation_gmsh_portfolio_selftest"
    spec = importlib.util.spec_from_file_location(module_name, backend_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Gmsh backend from {backend_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BACKEND = _load_backend()


def _expect_raises(
    error_type: type[BaseException],
    action: Callable[[], object],
    message_fragment: str | None = None,
) -> BaseException:
    try:
        action()
    except error_type as exc:
        if message_fragment is not None:
            assert message_fragment.lower() in str(exc).lower(), str(exc)
        return exc
    raise AssertionError(f"Expected {error_type.__name__} was not raised")


def _geometry(*, open_boundary: bool) -> object:
    kinds = ("open", "land", "land", "land") if open_boundary else (
        "land",
        "land",
        "land",
        "land",
    )
    exterior = BACKEND.SourceLoop(
        loop_id="exterior",
        xy=np.asarray(
            [
                [0.0, 0.0],
                [12_000.0, 0.0],
                [12_000.0, 8_000.0],
                [0.0, 8_000.0],
            ],
            dtype=float,
        ),
        segment_kinds=kinds,
        source_vertex_ids=("v0", "v1", "v2", "v3"),
        role="exterior",
    )
    chains = (
        (
            BACKEND.SourceOpenBoundary(
                chain_id="south",
                exterior_segment_indices=(0,),
            ),
        )
        if open_boundary
        else ()
    )
    return BACKEND.GmshGeometry(exterior=exterior, open_boundaries=chains)


class _OptionSpy:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def setNumber(self, name: str, value: float) -> None:
        self.values[str(name)] = float(value)


class _ForbiddenField:
    def add(self, field_kind: str) -> int:
        raise AssertionError(
            f"canonical callback must supersede legacy {field_kind!r} field"
        )


class _CallbackMeshSpy:
    def __init__(self) -> None:
        self.field = _ForbiddenField()
        self.callback = None

    def setSizeCallback(self, callback) -> None:
        self.callback = callback


class _LegacyFieldSpy:
    def __init__(self) -> None:
        self.added: list[tuple[int, str]] = []
        self.numbers: list[tuple[int, str, tuple[float, ...]]] = []
        self.strings: list[tuple[int, str, str]] = []
        self.background: int | None = None

    def add(self, field_kind: str) -> int:
        tag = len(self.added) + 1
        self.added.append((tag, str(field_kind)))
        return tag

    def setNumbers(self, tag: int, name: str, values) -> None:
        self.numbers.append(
            (int(tag), str(name), tuple(float(value) for value in values))
        )

    def setNumber(self, tag: int, name: str, value: float) -> None:
        self.numbers.append((int(tag), str(name), (float(value),)))

    def setString(self, tag: int, name: str, value: str) -> None:
        self.strings.append((int(tag), str(name), str(value)))

    def setAsBackgroundMesh(self, tag: int) -> None:
        self.background = int(tag)


def _fake_cad() -> object:
    return BACKEND._CadModel(
        surface_tag=1,
        point_tags_by_loop={"exterior": (1, 2, 3, 4)},
        line_tags_by_loop={"exterior": (11, 12, 13, 14)},
        physical_groups={},
    )


def test_algorithm_portfolio_validation_and_names() -> None:
    assert dict(BACKEND.GMSH_2D_ALGORITHM_NAMES) == {
        1: "MeshAdapt",
        5: "Delaunay",
        6: "Frontal-Delaunay",
    }
    default = BACKEND.GmshConfig(h_uniform_m=2_000.0)
    assert default.algorithm == 6
    assert default.algorithm_name == "Frontal-Delaunay"
    assert default.as_meshing_config().algorithm_name == "Frontal-Delaunay"

    for algorithm, expected_name in BACKEND.GMSH_2D_ALGORITHM_NAMES.items():
        compact = BACKEND.GmshConfig(
            h_uniform_m=2_000.0,
            algorithm=algorithm,
        )
        expanded = compact.as_meshing_config()
        assert expanded.algorithm == algorithm
        assert expanded.algorithm_name == expected_name
        assert BACKEND.gmsh_algorithm_name(algorithm) == expected_name

    for rejected in (0, 2, 7, 6.5, True, "6"):
        _expect_raises(
            ValueError,
            lambda value=rejected: BACKEND.GmshMeshingConfig(
                uniform_target_m=2_000.0,
                algorithm=value,
            ),
            "one of",
        )


def test_deterministic_options_follow_selected_algorithm() -> None:
    option = _OptionSpy()
    gmsh = SimpleNamespace(option=option)
    config = BACKEND.GmshMeshingConfig(
        uniform_target_m=2_000.0,
        algorithm=5,
    )
    BACKEND._set_deterministic_options(gmsh, config)
    assert option.values["General.NumThreads"] == 1.0
    assert option.values["Mesh.MaxNumThreads1D"] == 1.0
    assert option.values["Mesh.MaxNumThreads2D"] == 1.0
    assert option.values["Mesh.Algorithm"] == 5.0
    assert option.values["Mesh.AlgorithmSwitchOnFailure"] == 0.0
    assert option.values["Mesh.MeshSizeFromPoints"] == 0.0
    assert option.values["Mesh.MeshSizeFromCurvature"] == 0.0
    assert option.values["Mesh.MeshSizeExtendFromBoundary"] == 0.0


def test_canonical_callback_adapter_and_validation() -> None:
    sampled_xy: list[tuple[float, float]] = []

    def canonical_size(x: float, y: float) -> float:
        sampled_xy.append((x, y))
        return 1_000.0 + 0.1 * x + 0.05 * y

    mesh = _CallbackMeshSpy()
    gmsh = SimpleNamespace(model=SimpleNamespace(mesh=mesh))
    geometry = _geometry(open_boundary=True)
    config = BACKEND.GmshMeshingConfig(
        uniform_target_m=9_999.0,
        constant_field=True,
        canonical_size_callback=canonical_size,
    )
    callback = BACKEND._configure_size_field(
        gmsh,
        geometry,
        _fake_cad(),
        config,
    )
    assert callback is mesh.callback
    assert callback(1, 11, 2_000.0, 4_000.0, 0.0, 123.0) == 1_400.0
    assert callback(2, 1, 6_000.0, 2_000.0, 0.0, 456.0) == 1_700.0
    assert sampled_xy == [(2_000.0, 4_000.0), (6_000.0, 2_000.0)]
    assert BACKEND._size_field_mode(geometry, config) == (
        "canonical_projected_callback"
    )

    invalid = BACKEND._gmsh_size_callback(lambda x, y: np.nan)
    _expect_raises(
        BACKEND.GmshBackendError,
        lambda: invalid(2, 1, 1.0, 2.0, 0.0, 5.0),
        "non-finite or non-positive",
    )
    _expect_raises(
        BACKEND.GmshBackendError,
        lambda: callback(2, 1, np.inf, 2.0, 0.0, 5.0),
        "non-finite projected",
    )


def test_legacy_fields_remain_the_default() -> None:
    open_geometry = _geometry(open_boundary=True)
    open_fields = _LegacyFieldSpy()
    open_gmsh = SimpleNamespace(
        model=SimpleNamespace(mesh=SimpleNamespace(field=open_fields))
    )
    returned = BACKEND._configure_size_field(
        open_gmsh,
        open_geometry,
        _fake_cad(),
        BACKEND.GmshMeshingConfig(uniform_target_m=2_000.0),
    )
    assert returned is None
    assert open_fields.added == [(1, "Distance"), (2, "Threshold")]
    assert open_fields.background == 2
    assert BACKEND._size_field_mode(
        open_geometry,
        BACKEND.GmshMeshingConfig(uniform_target_m=2_000.0),
    ) == "gmsh_distance_threshold"

    closed_geometry = _geometry(open_boundary=False)
    closed_fields = _LegacyFieldSpy()
    closed_gmsh = SimpleNamespace(
        model=SimpleNamespace(mesh=SimpleNamespace(field=closed_fields))
    )
    returned = BACKEND._configure_size_field(
        closed_gmsh,
        closed_geometry,
        _fake_cad(),
        BACKEND.GmshMeshingConfig(
            uniform_target_m=2_000.0,
            constant_field=True,
        ),
    )
    assert returned is None
    assert closed_fields.added == [(1, "MathEval")]
    assert closed_fields.strings == [(1, "F", "2000")]
    assert closed_fields.background == 1


def test_real_gmsh_portfolio_uses_one_canonical_callback() -> None:
    try:
        BACKEND.load_pinned_gmsh()
    except BACKEND.GmshBackendError:
        print("SKIP real Gmsh fixture: pinned gmsh==4.15.2 is unavailable")
        return

    geometry = _geometry(open_boundary=False)
    with tempfile.TemporaryDirectory(prefix="gmsh_mesher_portfolio_") as temporary:
        root = Path(temporary)
        for algorithm, expected_name in BACKEND.GMSH_2D_ALGORITHM_NAMES.items():
            dimensions: set[int] = set()

            def canonical_size(
                x: float,
                y: float,
                *,
                seen: set[int] = dimensions,
            ) -> float:
                del y
                # Entity dimensions are captured by the adapter wrapper below;
                # this callback deliberately depends only on projected x.
                return 1_200.0 + 0.05 * x

            config = BACKEND.GmshConfig(
                h_uniform_m=2_000.0,
                algorithm=algorithm,
                canonical_size_callback=canonical_size,
                model_name=f"portfolio_algorithm_{algorithm}",
            )
            original_adapter = BACKEND._gmsh_size_callback

            def recording_adapter(source_callback):
                adapted = original_adapter(source_callback)

                def record(dim, tag, x, y, z, lc):
                    dimensions.add(int(dim))
                    return adapted(dim, tag, x, y, z, lc)

                return record

            BACKEND._gmsh_size_callback = recording_adapter
            try:
                result = BACKEND.run_gmsh_attempt(
                    geometry,
                    config,
                    root / f"algorithm_{algorithm}.msh",
                )
            finally:
                BACKEND._gmsh_size_callback = original_adapter

            assert result.algorithm == algorithm
            assert result.algorithm_name == expected_name
            assert result.size_field_mode == "canonical_projected_callback"
            assert result.nodes_xy.shape[0] > 4
            assert result.triangles_1based.shape[0] > 2
            assert 1 in dimensions, dimensions
            assert 2 in dimensions, dimensions


TESTS: tuple[Callable[[], None], ...] = (
    test_algorithm_portfolio_validation_and_names,
    test_deterministic_options_follow_selected_algorithm,
    test_canonical_callback_adapter_and_validation,
    test_legacy_fields_remain_the_default,
    test_real_gmsh_portfolio_uses_one_canonical_callback,
)


def main() -> int:
    failures: list[tuple[str, BaseException]] = []
    for test in TESTS:
        try:
            test()
        except BaseException as exc:
            failures.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    if failures:
        print(f"{len(failures)} of {len(TESTS)} Gmsh portfolio tests failed")
        return 1
    print(f"All {len(TESTS)} Gmsh portfolio tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
