import tempfile
import unittest
from pathlib import Path
from columbia_salinity_build import SOURCE_SHA256, instrument
from salinity_recipe import remote


class SalinityBuild(unittest.TestCase):
    def test_unqualified_source_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            for filename in SOURCE_SHA256:
                (source / filename).write_text("unqualified source\n")
            with self.assertRaises(ValueError):
                instrument(source, source / "unused.F", source / "receipt.json")
            self.assertFalse((source / "receipt.json").exists())
            self.assertEqual(
                (source / "internal_step.F").read_text(), "unqualified source\n"
            )

    def test_site_paths_are_resolved(self):
        for path in ("/", "relative", "/scratch/x/../y", "/scratch/x\ny"):
            with self.assertRaises(ValueError):
                remote(path)
        self.assertEqual(remote("/scratch/fvcom_source"), "/scratch/fvcom_source")


if __name__ == "__main__":
    unittest.main()
