import unittest
from fvcom_profiles import has_river, has_wind, resolve_request
from fvcom_run_control import validate_request, build_parser
from fvcom_salinity import DAY_READERS, MODE, PROFILE, STAGES


class SalinityPilotTests(unittest.TestCase):
    def setUp(self):
        self.req = dict(
            schema="simulation_request_v1",
            case_id="test",
            mode=MODE,
            profile=PROFILE,
            execution={"site": "perlmutter"},
            perlmutter={"account": "example_cpu", "project_dir": "/scratch/test"},
        )

    def test_explicit_baroclinic_route(self):
        r = resolve_request(self.req)
        validate_request(r)
        self.assertEqual(r["profile"], PROFILE)
        self.assertEqual(r["execution"]["stages"], ["reader", "smoke"])
        self.assertFalse(r["physics"]["temperature_active"])
        self.assertEqual(r["physics"]["equation_of_state"], "dens2")
        self.assertTrue(has_river(r) and has_wind(r))
        self.assertEqual(self.req["execution"], {"site": "perlmutter"})

    def test_reject_other_formulation(self):
        self.req["physics"] = {"temperature_active": True}
        with self.assertRaises(ValueError):
            resolve_request(self.req)

    def test_mode_alone_does_not_select_dated_columbia_settings(self):
        self.req.pop("profile")
        with self.assertRaisesRegex(ValueError, "explicit profile"):
            resolve_request(self.req)

    def test_traversing_path_rejected(self):
        self.req["perlmutter"]["project_dir"] = "/scratch/project/../other"
        with self.assertRaises(ValueError):
            resolve_request(self.req)

    def test_bounded_diagnostic_stage(self):
        args = build_parser().parse_args(
            "salinity-attempt --stage bottom-sensitivity --project study --build b --inputs i --namelist n".split()
        )
        self.assertEqual(args.stage, "bottom-sensitivity")

    def test_daily_sensitivity_requires_its_matching_reader(self):
        self.assertEqual(DAY_READERS["bottom-sensitivity-day"], "bottom-sensitivity")
        self.assertEqual(DAY_READERS["pilot"], "reader")
        self.assertEqual(STAGES["bottom-sensitivity-day"], STAGES["pilot"])


if __name__ == "__main__":
    unittest.main()
