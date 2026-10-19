"""Offline regressions for source provenance labels and rendered categories."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import xarray as xr

from cudem_bathy.plot import plot_source_id_map


CURRENT = {0: "none", 1: "cudem", 2: "nbs_bluetopo", 3: "crm",
           4: "etopo", 5: "regional_candidate"}
LEGACY = {0: "none", 1: "cudem", 2: "crm", 3: "etopo"}


def dataset(values, legend=None, global_legend=None):
    values = np.asarray(values)
    ds = xr.Dataset({"source_id": (("lat", "lon"), values)},
                    coords={"lat": np.arange(values.shape[0]),
                            "lon": np.arange(values.shape[1])})
    if legend is not None:
        ds.source_id.attrs["legend"] = json.dumps(legend)
    if global_legend is not None:
        ds.attrs["source_legend"] = json.dumps(global_legend)
    return ds


class SourcePlotTests(unittest.TestCase):
    def render(self, ds):
        original = ds.copy(deep=True)
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "sources.png"
            # Inspect the actual rendered artists without changing their behavior.
            with patch("cudem_bathy.plot.plt.close"):
                plot_source_id_map(ds, output)
                fig = plt.gcf()
            try:
                self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                ax = fig.axes[0]
                legend = ax.get_legend()
                labels = [] if legend is None else [t.get_text() for t in legend.get_texts()]
                swatches = [] if legend is None else [h.get_facecolor() for h in legend.legend_handles]
                mesh = ax.collections[0]
                colors = mesh.get_facecolors().copy()
                masked = np.ma.getmaskarray(mesh.get_array()).copy()
                xr.testing.assert_identical(ds, original)
                return labels, swatches, colors, masked
            finally:
                plt.close(fig)

    def test_current_sparse_ids_preserve_crm_identity(self):
        labels, swatches, colors, _ = self.render(dataset([[1, 3], [3, 1]], CURRENT))
        self.assertEqual(labels, ["CUDEM", "CRM"])
        np.testing.assert_allclose(colors, [swatches[0], swatches[1], swatches[1], swatches[0]])

    def test_current_all_categories_have_distinct_colors(self):
        labels, swatches, colors, _ = self.render(dataset([[0, 1, 2], [3, 4, 5]], CURRENT))
        self.assertEqual(labels, ["none", "CUDEM", "NBS BlueTopo", "CRM", "ETOPO", "regional candidate"])
        self.assertEqual(len(set(map(tuple, swatches))), 6)
        np.testing.assert_allclose(colors, swatches)

    def test_explicit_legacy_encoding(self):
        labels, swatches, _, _ = self.render(dataset([[0, 1], [2, 3]], LEGACY))
        self.assertEqual(labels, ["none", "CUDEM", "CRM", "ETOPO"])
        np.testing.assert_allclose(swatches[2:], [to_rgba("#009e73"), to_rgba("#d55e00")])

    def test_dataset_global_legend_fallback(self):
        labels, _, _, _ = self.render(dataset([[3, 4], [4, 3]], global_legend=CURRENT))
        self.assertEqual(labels, ["CRM", "ETOPO"])

    def test_variable_legend_has_priority(self):
        labels, _, _, _ = self.render(dataset([[2, 3], [3, 2]], LEGACY, CURRENT))
        self.assertEqual(labels, ["CRM", "ETOPO"])

    def test_missing_legend_does_not_guess_legacy_or_current(self):
        labels, _, _, _ = self.render(dataset([[2, 3], [3, 2]]))
        self.assertEqual(labels, ["source ID 2 (undeclared)", "source ID 3 (undeclared)"])

    def test_custom_and_undeclared_sparse_ids(self):
        labels, swatches, colors, _ = self.render(dataset([[17, 101], [101, 17]], {17: "Survey 2025"}))
        self.assertEqual(labels, ["Survey 2025", "source ID 101 (undeclared)"])
        self.assertNotEqual(tuple(swatches[0]), tuple(swatches[1]))
        np.testing.assert_allclose(colors, [swatches[0], swatches[1], swatches[1], swatches[0]])

    def test_nan_is_masked_not_assigned_a_source(self):
        labels, _, colors, masked = self.render(dataset([[1, np.nan], [3, 1]], CURRENT))
        self.assertEqual(labels, ["CUDEM", "CRM"])
        self.assertEqual(int(masked.sum()), 1)
        self.assertEqual(colors[1, 3], 0)

    def test_all_missing_has_no_source_legend(self):
        labels, _, _, masked = self.render(dataset([[np.nan] * 2] * 2, CURRENT))
        self.assertEqual(labels, [])
        self.assertTrue(masked.all())

    def test_malformed_legend_is_not_silently_mislabeled(self):
        for bad in ("not JSON", "[]", '{"1.5": "CRM"}', '{"1": null}'):
            with self.subTest(metadata=bad), tempfile.TemporaryDirectory() as td:
                ds = dataset([[1, 3], [3, 1]])
                ds.source_id.attrs["legend"] = bad
                with self.assertRaises(ValueError):
                    plot_source_id_map(ds, Path(td) / "bad.png")

    def test_fractional_source_id_rejected(self):
        with tempfile.TemporaryDirectory() as td, self.assertRaises(ValueError):
            plot_source_id_map(dataset([[1, 1.5], [3, 1]], CURRENT), Path(td) / "bad.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
