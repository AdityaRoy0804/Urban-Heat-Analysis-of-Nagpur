import unittest

import numpy as np

from nagpur_uhi.features.lulc import CLASS_LABELS

import sys
from pathlib import Path

# scripts/ isn't a package - import it directly by path so these tests don't
# need the repo installed in a particular way beyond the standard src layout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import importlib

synthetic_demo = importlib.import_module("00_generate_synthetic_demo_data")


class TestSyntheticDemoGenerator(unittest.TestCase):
    def test_water_pixels_are_cooler_than_built_up_pixels(self):
        data = synthetic_demo.generate_year(2020, [2014, 2017, 2020, 2023, 2026])
        water_mask = data["lulc_class"] == CLASS_LABELS.index("water")
        built_up_mask = data["lulc_class"] == CLASS_LABELS.index("built_up")
        self.assertGreater(water_mask.sum(), 0)
        self.assertGreater(built_up_mask.sum(), 0)
        self.assertLess(data["lst"][water_mask].mean(), data["lst"][built_up_mask].mean())

    def test_built_up_fraction_grows_over_study_years(self):
        years = [2014, 2017, 2020, 2023, 2026]
        built_up_fractions = []
        for year in years:
            data = synthetic_demo.generate_year(year, years)
            built_up_fractions.append(np.mean(data["lulc_class"] == CLASS_LABELS.index("built_up")))
        # Urban growth narrative should be monotonic (or very close to it) across the study years.
        self.assertLess(built_up_fractions[0], built_up_fractions[-1])
        self.assertGreaterEqual(sum(np.diff(built_up_fractions) >= -1e-9), len(years) - 2)

    def test_all_expected_arrays_present_with_consistent_shape(self):
        data = synthetic_demo.generate_year(2020, [2014, 2017, 2020, 2023, 2026])
        expected_keys = {"lst", "ndvi", "ndbi", "ndwi", "sar_vv_vh_ratio", "lulc_class"}
        self.assertEqual(set(data.keys()), expected_keys)
        shapes = {v.shape for v in data.values()}
        self.assertEqual(len(shapes), 1, "all feature arrays must share one shape")

    def test_terrain_context_has_low_relief_consistent_with_nagpur(self):
        terrain = synthetic_demo.generate_terrain_context()
        self.assertTrue(200 < terrain["elevation_m"].mean() < 400)
        self.assertLess(np.nanmean(terrain["slope_deg"]), 10.0)  # Nagpur is largely flat


if __name__ == "__main__":
    unittest.main()
