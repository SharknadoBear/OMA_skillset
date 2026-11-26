"""Exercise distributed hourly integration and rejection of an incorrect restart clock."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lake_coare_ledger import EXTERNAL, HEAT, reduce_ranks, steps_per_hour, validate_hour
from lake_physics import sha


class LedgerTest(unittest.TestCase):
    def test_multiple_ranks_and_absolute_restart_clock(self):
        absolute = np.array([35280, 36000])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            hashes = {}
            for rank in (1, 2):
                file = root / f"coare_ext_rank_{rank:05}.dat"
                totals = np.array([[7200, 1, -2, 1, -3], [14400, 4, -4, 2, -6]]) * rank
                np.savetxt(file, np.column_stack(([1440, 2880], absolute, totals)))
                hashes[file.name] = sha(file)
            hours = reduce_ranks(root, "ext", EXTERNAL, 2, 2, steps_per_hour(2.5), absolute, hashes)
            self.assertEqual([v["rain_m3"] for v in hours], [3, 9])
            with self.assertRaisesRegex(ValueError, "clock"):
                reduce_ranks(root, "ext", EXTERNAL, 2, 2, 1440, absolute + 1, hashes)
            with self.assertRaisesRegex(ValueError, "rank"):
                reduce_ranks(root, "ext", EXTERNAL, 3, 2, 1440, absolute, hashes)

    def test_latent_mass_and_condensation(self):
        h = dict.fromkeys(HEAT, 0.)
        e = dict.fromkeys(EXTERNAL, 0.)
        h.update(area_s=3600., sensible_J=10., latent_J=-100., latent_from_mass_J=-100., net_J=-90.)
        for r in (h, e):
            r.update(area_s=3600., rain_m3=5., vapor_m3=-2., condensation_m3=1., evap_m3=-3.)
        h["precip_m3"] = 6.
        validate_hour(h, e)
        h["precip_m3"] = 5.
        with self.assertRaisesRegex(ValueError, "condensation"):
            validate_hour(h, e)
        h["precip_m3"] = 6.
        h["latent_from_mass_J"] = 0.
        with self.assertRaisesRegex(ValueError, "latent"):
            validate_hour(h, e)


if __name__ == "__main__":
    unittest.main()
