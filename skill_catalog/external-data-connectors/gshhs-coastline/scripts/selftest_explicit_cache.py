"""An explicit scientific cache must not discover a different project's data."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from gshhs_coastline.sources import find_gshhs_cache, _candidate_dirs

class ExplicitCache(unittest.TestCase):
    def test_empty_explicit_cache_ignores_populated_legacy_cache(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            historical=root/'Workspace/Preprocessing/fvcom-gshhs-coastline/cache/gshhg/GSHHS_shp/f'
            historical.mkdir(parents=True)
            (historical/'GSHHS_f_L1.shp').write_bytes(b'not scientific data; discovery fixture')
            assigned=root/'trial/cache'
            with patch('gshhs_coastline.sources.workspace_roots',return_value=[root]):
                self.assertIsNotNone(find_gshhs_cache())
                self.assertIsNone(find_gshhs_cache(assigned))
                for candidate,_ in _candidate_dirs(assigned):
                    self.assertTrue(candidate.is_relative_to(assigned))
    def test_within_trial_reuse_remains_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            cache=Path(root)/'trial/cache'
            shp=cache/'GSHHS_shp/f/GSHHS_f_L1.shp'; shp.parent.mkdir(parents=True)
            shp.write_bytes(b'discovery fixture')
            self.assertEqual(find_gshhs_cache(cache).source_kind,'requested_cache')

if __name__=='__main__': unittest.main(verbosity=2)
