#!/usr/bin/env python3
"""Independent numerical and source-evidence checks for v2 conditioning."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from conditioning import repair_source_artifacts, smooth_depth_vertical
from grid_math import depths, haney, vertical_diagnostics


def independent_rx1(h, mask, n, theta_s, theta_b, hc):
    """Scalar transcription of interface stretching and Haney's equation."""
    interfaces = {}
    for j, i in np.argwhere(mask):
        d = float(h[j, i])
        z = []
        for k in range(n + 1):
            s = -1.0 + k / n
            c = (
                (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
                if theta_s
                else -s * s
            )
            if theta_b:
                c = (np.exp(theta_b * c) - 1.0) / (1.0 - np.exp(-theta_b))
            z.append(d * (hc * s + d * c) / (hc + d))
        interfaces[int(j), int(i)] = z
    worst = 0.0
    for (j, i), a in interfaces.items():
        for key in ((j + 1, i), (j, i + 1)):
            if key not in interfaces:
                continue
            b = interfaces[key]
            for k in range(1, n + 1):
                numerator = abs(a[k] - b[k] + a[k - 1] - b[k - 1])
                denominator = a[k] + b[k] - a[k - 1] - b[k - 1]
                worst = max(worst, numerator / denominator)
    return worst


class ConditioningTests(unittest.TestCase):
    def smooth(self, h, **kwargs):
        h = np.asarray(h, float)
        options = dict(
            mask=np.ones_like(h, bool),
            area=np.ones_like(h),
            rx0_max=0.2,
            rx1_max=7.0,
            N=40,
            theta_s=6.0,
            theta_b=2.0,
            hc=20.0,
            hmin=2.0,
        )
        options.update(kwargs)
        return smooth_depth_vertical(h, **options)

    def test_analytic_natural_log_pair_equal_and_unequal_area(self):
        h = np.array([[2.0, 100.0]])
        for weights in ([1.0, 1.0], [1.0, 3.0]):
            area = np.array([weights])
            result, report = self.smooth(h, area=area, N=1, rx1_max=9.0)
            # With one layer Haney equals rx0, so the log-ratio constraint
            # has a closed-form answer independent of the implementation.
            mean = float(np.sum(area * np.log(h)) / area.sum())
            jump = np.log(
                (1.0 + report["internal_rx0_target"])
                / (1.0 - report["internal_rx0_target"])
            )
            expected = np.exp(
                [
                    [
                        mean - weights[1] / sum(weights) * jump,
                        mean + weights[0] / sum(weights) * jump,
                    ]
                ]
            )
            np.testing.assert_allclose(result, expected, rtol=3e-14)
            self.assertAlmostEqual(
                float(np.sum(area * np.log(h))),
                float(np.sum(area * np.log(result))),
                places=12,
            )
            self.assertLess(report["final_volume_m3"], report["initial_volume_m3"])
            self.assertEqual(report["space"], "natural_log_depth")
            self.assertEqual(report["volume_policy"], "report_only_no_rescaling")

    def test_vertical_constraint_drives_pair_with_acceptable_rx0(self):
        h = np.array([[700.0, 1000.0]])
        initial = independent_rx1(h, np.ones_like(h, bool), 60, 8.0, 2.0, 20.0)
        self.assertGreater(initial, 7.0)
        result, report = self.smooth(h, N=60, theta_s=8.0, rx1_max=7.0)
        self.assertFalse(np.array_equal(h, result))
        achieved = independent_rx1(result, np.ones_like(h, bool), 60, 8.0, 2.0, 20.0)
        self.assertLess(achieved, 7.0)
        self.assertLessEqual(report["rx0"], 0.2)
        self.assertAlmostEqual(
            float(np.log(h).sum()), float(np.log(result).sum()), places=12
        )
        self.assertLess(report["internal_rx1_target"], 7.0)
        self.assertTrue(report["strict_rx1_pass"])

    def test_rx1_limit_is_strict_at_exact_boundary(self):
        result, report = self.smooth([[8.0, 12.0]], N=1, rx0_max=0.4, rx1_max=0.2)
        self.assertGreater(report["iterations"], 0)
        self.assertLess(report["rx1"], 0.2)
        self.assertLess(
            independent_rx1(result, np.ones_like(result, bool), 1, 6.0, 2.0, 20.0), 0.2
        )

    def test_acceptable_values_and_isolated_cells_are_bitwise_unchanged(self):
        h = np.array([[20.0, 20.1, np.nan, 900.0]])
        mask = np.array([[1, 1, 0, 1]], bool)
        result, report = self.smooth(h, mask=mask)
        np.testing.assert_array_equal(result, h)
        self.assertEqual(report["iterations"], 0)
        self.assertEqual(report["modified_wet_cells"], 0)

    def test_land_barriers_floor_and_input_immutability(self):
        h = np.array([[2.0, 100.0, -999.0, 6.0, 6.1], [2.0, 200.0, np.nan, 6.0, 6.1]])
        mask = np.isfinite(h) & (h > 0)
        area = np.array([[1.0, 2.0, np.nan, 3.0, 3.0], [4.0, 2.0, np.nan, 2.0, 2.0]])
        saved_h, saved_mask, saved_area = h.copy(), mask.copy(), area.copy()
        result, report = self.smooth(h, mask=mask, area=area)
        np.testing.assert_array_equal(h, saved_h)
        np.testing.assert_array_equal(mask, saved_mask)
        np.testing.assert_array_equal(area, saved_area)
        np.testing.assert_array_equal(result[:, 2:], h[:, 2:])
        self.assertGreaterEqual(result[mask].min(), 2.0)
        self.assertLessEqual(result[mask].max(), h[mask].max())
        self.assertAlmostEqual(
            float(np.sum(area[mask] * np.log(h[mask]))),
            float(np.sum(area[mask] * np.log(result[mask]))),
            places=10,
        )
        self.assertEqual(report["hmin_m"], 2.0)

    def test_full_and_blocked_pairs_and_vertical_diagnostics(self):
        h = np.geomspace(2.0, 1200.0, 77).reshape(11, 7)
        h[3:6] *= 3.0
        mask = np.ones(h.shape, bool)
        mask[4:7, 2:4] = False
        area = np.linspace(1.0, 4.0, h.size).reshape(h.shape)
        results = []
        for size in (1, 9, 4096):
            result, report = self.smooth(
                h, mask=mask, area=area, N=12, rx1_max=3.0, pair_block_size=size
            )
            results.append(result)
            self.assertLess(report["haney_rx1_diagnostic"], 3.0)
            self.assertLess(independent_rx1(result, mask, 12, 6.0, 2.0, 20.0), 3.0)
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[0], results[2])
        _, zw, _ = depths(results[0], 12, 6.0, 2.0, 20.0)
        reference = haney(zw, mask)
        for rows in (1, 3, 32):
            diagnostic = vertical_diagnostics(
                results[0], 12, 6.0, 2.0, 20.0, mask=mask, block_rows=rows
            )
            self.assertEqual(diagnostic["haney_rx1_diagnostic"], reference)
        self.assertAlmostEqual(
            reference, independent_rx1(results[0], mask, 12, 6.0, 2.0, 20.0), places=11
        )

    def test_haney_analytic_single_layer_and_stretching_limits(self):
        h = np.array([[2.0, 20.0], [200.0, 2000.0]])
        mask = np.ones_like(h, bool)
        for n, ts, tb, hc in (
            (1, 0.0, 0.0, 0.0),
            (8, 0.0, 2.0, 20.0),
            (20, 6.0, 0.0, 20.0),
            (20, 6.0, 2.0, 0.0),
        ):
            result, report = self.smooth(h, N=n, theta_s=ts, theta_b=tb, hc=hc)
            independent = independent_rx1(result, mask, n, ts, tb, hc)
            self.assertAlmostEqual(
                report["haney_rx1_diagnostic"], independent, places=10
            )
            self.assertLess(independent, 7.0)
        result, report = self.smooth([[2.0, 20.0]], N=1)
        self.assertAlmostEqual(
            report["rx1"], abs(result[0, 0] - result[0, 1]) / result.sum()
        )

    def test_invalid_inputs_and_explicit_nonconvergence(self):
        h = np.array([[2.0, 100.0]])
        for kwargs in (
            {"rx0_max": 0.0},
            {"rx0_max": 1.0},
            {"rx0_max": np.nan},
            {"rx1_max": 0.0},
            {"rx1_max": np.inf},
            {"hc": -1.0},
            {"hmin": 0.0},
            {"hmin": 3.0},
            {"max_iter": -1},
            {"max_iter": 1.5},
            {"pair_block_size": 0},
            {"pair_block_size": True},
            {"N": 0},
            {"theta_s": -1.0},
            {"theta_b": 30.0},
            {"mask": np.zeros_like(h)},
            {"mask": np.array([[1, 2]])},
            {"area": np.array([[1.0, 0.0]])},
            {"area": np.ones((2, 2))},
            {"area": np.array([[1.0, np.nan]])},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.smooth(h, **kwargs)
        for invalid in (np.nan, np.inf, 0.0, -1.0):
            with self.assertRaises(ValueError):
                self.smooth([[invalid, 20.0]])
        with self.assertRaisesRegex(ValueError, "did not converge"):
            self.smooth(h, max_iter=0)
        # A zero iteration budget is sufficient when no correction is needed.
        result, report = self.smooth([[10.0, 10.0]], max_iter=0)
        self.assertEqual(report["iterations"], 0)


class SourceRepairTests(unittest.TestCase):
    def fixture(self):
        source = np.full((7, 7), -10.0)
        source[3, 3] = 80.0
        selected = np.zeros_like(source, bool)
        selected[3, 3] = True
        return source, selected

    def repair(self, source, selected, **kwargs):
        arguments = dict(
            mapped_land=np.zeros_like(source, bool),
            supporting_elevation=np.full_like(source, -12.0),
            supporting_source={
                "reference": "independent survey fixture",
                "rationale": "Verified spike artifact",
            },
        )
        arguments.update(kwargs)
        return repair_source_artifacts(source, suspect_cells=selected, **arguments)

    def test_only_explicit_source_artifact_changes(self):
        source, selected = self.fixture()
        source[0, 0] = np.nan
        source[1, 1] = 50.0
        original = source.copy()
        repaired, report = self.repair(source, selected)
        self.assertEqual(repaired[3, 3], -10.0)
        np.testing.assert_array_equal(repaired[~selected], original[~selected])
        np.testing.assert_array_equal(source, original)
        self.assertEqual(report["missing_coverage_filled"], 0)
        self.assertEqual(report["changed_cells"], 1)

    def test_repair_forbids_land_missing_and_absent_wet_evidence(self):
        source, selected = self.fixture()
        for kwargs in (
            {"mapped_land": selected},
            {"supporting_source": {}},
            {"supporting_source": {"reference": "survey"}},
            {"supporting_elevation": np.ones_like(source)},
            {"supporting_elevation": source * np.nan},
            {"radius_cells": 0},
            {"min_neighbors": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.repair(source, selected, **kwargs)
        source[3, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "missing primary coverage"):
            self.repair(source, selected)

    def test_repair_donors_do_not_cross_land(self):
        source, selected = self.fixture()
        source[:, 4:] = -1000.0
        land = np.zeros_like(source, bool)
        land[:, 4] = True
        repaired, report = self.repair(
            source, selected, mapped_land=land, radius_cells=3
        )
        self.assertEqual(repaired[3, 3], -10.0)
        self.assertTrue(all(i < 4 for j, i in report["cells"][0]["donor_indices"]))
        land[:] = True
        land[3, 3] = False
        with self.assertRaisesRegex(ValueError, "local wet source donors"):
            self.repair(source, selected, mapped_land=land)

    def test_source_cli_preserves_source_and_writes_reviewable_new_artifacts(self):
        source, selected = self.fixture()
        source[0, 0] = np.nan
        # Temporary outputs belong to the test runner's configured TEMP, never
        # the installed skill. The test runner may set TEMP to its validation root.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, values in (
                ("primary", source),
                ("support", np.full_like(source, -12.0)),
            ):
                with Dataset(root / f"{name}.nc", "w") as dataset:
                    dataset.createDimension("lat", 7)
                    dataset.createDimension("lon", 7)
                    dataset.createVariable("lat", "f8", ("lat",))[:] = (
                        np.arange(7) + 45.0
                    )
                    dataset.createVariable("lon", "f8", ("lon",))[:] = (
                        np.arange(7) - 130.0
                    )
                    variable = dataset.createVariable(
                        "elevation", "f8", ("lat", "lon"), fill_value=-9999.0
                    )
                    variable.units = "m"
                    variable.positive = "up"
                    variable[:] = np.ma.masked_invalid(values)
            np.savez(
                root / "selection.npz",
                suspect_cells=selected,
                mapped_land=np.zeros_like(selected),
            )
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("conditioning.py")),
                    "source-artifact-repair",
                    "--primary",
                    str(root / "primary.nc"),
                    "--variable",
                    "elevation",
                    "--support",
                    str(root / "support.nc"),
                    "--support-variable",
                    "elevation",
                    "--selection-npz",
                    str(root / "selection.npz"),
                    "--support-reference",
                    "independent fixture survey",
                    "--rationale",
                    "Evidence confirms isolated source spike",
                    "--output-dir",
                    str(root / "result"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            report = json.loads(
                (root / "result/source_artifact_repair.json").read_text()
            )
            self.assertTrue(report["readback_passed"])
            self.assertEqual(report["visual_review_status"], "pending")
            self.assertGreater(
                (root / "result/source_artifact_repair.png").stat().st_size, 1000
            )
            with Dataset(root / "primary.nc") as original, Dataset(
                root / "result/repaired_source.nc"
            ) as repaired:
                self.assertEqual(original["elevation"][3, 3], 80.0)
                self.assertEqual(repaired["elevation"][3, 3], -10.0)
                np.testing.assert_array_equal(
                    original["elevation"][:][~selected],
                    repaired["elevation"][:][~selected],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
