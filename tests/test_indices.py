import unittest

import numpy as np

from nagpur_uhi.features import indices


class TestNormalizedDifferenceIndices(unittest.TestCase):
    def test_ndvi_known_values(self):
        nir = np.array([[0.5, 0.2], [0.8, 0.0]])
        red = np.array([[0.1, 0.2], [0.0, 0.0]])
        result = indices.ndvi(nir, red)
        # (0.5-0.1)/(0.5+0.1) = 0.6667
        self.assertAlmostEqual(result[0, 0], 0.6667, places=3)
        # nir == red -> 0
        self.assertAlmostEqual(result[0, 1], 0.0, places=6)
        # nir == red == 0 -> division by zero handled as NaN, not a crash
        self.assertTrue(np.isnan(result[1, 1]))

    def test_ndvi_range_is_bounded(self):
        rng = np.random.default_rng(0)
        nir = rng.uniform(0, 1, size=(50, 50))
        red = rng.uniform(0, 1, size=(50, 50))
        result = indices.ndvi(nir, red)
        valid = ~np.isnan(result)
        self.assertTrue(np.all(result[valid] >= -1.0001))
        self.assertTrue(np.all(result[valid] <= 1.0001))

    def test_ndbi_higher_for_more_built_up_signature(self):
        # Built-up surfaces: higher SWIR than NIR -> positive NDBI
        swir = np.array([[0.4]])
        nir = np.array([[0.2]])
        self.assertGreater(indices.ndbi(swir, nir)[0, 0], 0)

    def test_ndwi_detects_water(self):
        green = np.array([[0.3]])
        nir = np.array([[0.05]])  # water absorbs strongly in NIR
        self.assertGreater(indices.ndwi(green, nir)[0, 0], 0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            indices.ndvi(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_sar_vv_vh_ratio(self):
        vv = np.array([[-8.0, -12.0]])
        vh = np.array([[-15.0, -12.0]])
        ratio = indices.sar_vv_vh_ratio_db(vv, vh)
        np.testing.assert_allclose(ratio, [[7.0, 0.0]])


class TestLandsatLST(unittest.TestCase):
    def test_known_dn_conversion(self):
        # DN=0 -> 149.0 K -> -124.15 C (sanity check on the official formula, not a realistic scene)
        result = indices.lst_celsius_from_landsat_c2(np.array([0]))
        self.assertAlmostEqual(result[0], 149.0 - 273.15, places=2)

    def test_typical_dn_gives_plausible_summer_temperature(self):
        # A DN in the range typically seen for a hot Indian summer surface (~45C = 318.15K)
        dn = (318.15 - 149.0) / 0.00341802
        result = indices.lst_celsius_from_landsat_c2(np.array([dn]))
        self.assertAlmostEqual(result[0], 45.0, places=1)


if __name__ == "__main__":
    unittest.main()
