import unittest
from pathlib import Path
from unittest.mock import patch
from salinity_configuration import render


class SalinityConfiguration(unittest.TestCase):
    def test_invalid_offset(self):
        with self.assertRaisesRegex(ValueError, "UTC"):
            render(
                Path("unused"), Path("unused"), "reader", "2025-03-18T00:00:00-07:00"
            )

    def test_stage_durations_are_bounded(self):
        with self.assertRaises(KeyError):
            render(Path("unused"), Path("unused"), "production")

    def test_reject_different_native_grid(self):
        with patch(
            "salinity_configuration.parse_values",
            return_value={"NML_GRID_COORDINATES": {"GRID_FILE_UNITS": "'degrees'"}},
        ):
            with patch.object(Path, "read_text", return_value=""):
                with self.assertRaisesRegex(ValueError, "GRID_FILE_UNITS"):
                    render(Path("unused"), Path("unused"), "reader")


if __name__ == "__main__":
    unittest.main()
