import unittest

import numpy as np

from nagpur_uhi.analysis import hotspots


def make_raster_with_hot_cluster(size=30, hot_slice=slice(5, 10), background=20.0, hot_value=40.0):
    raster = np.full((size, size), background, dtype="float64")
    raster[hot_slice, hot_slice] = hot_value
    return raster


class TestGetisOrdGiStar(unittest.TestCase):
    def test_embedded_hot_cluster_gets_high_positive_z(self):
        raster = make_raster_with_hot_cluster()
        z = hotspots.getis_ord_gi_star(raster, window_size=3)
        # centre of the hot cluster should have a strongly positive z-score
        self.assertGreater(z[7, 7], 1.96)

    def test_background_far_from_cluster_is_not_significant(self):
        raster = make_raster_with_hot_cluster(size=40, hot_slice=slice(5, 9))
        z = hotspots.getis_ord_gi_star(raster, window_size=3)
        far_corner_z = z[35, 35]
        self.assertTrue(np.isnan(far_corner_z) or abs(far_corner_z) < 1.96)

    def test_rejects_even_window_size(self):
        with self.assertRaises(ValueError):
            hotspots.getis_ord_gi_star(np.ones((5, 5)), window_size=4)

    def test_constant_raster_raises(self):
        with self.assertRaises(ValueError):
            hotspots.getis_ord_gi_star(np.full((5, 5), 10.0), window_size=3)

    def test_nodata_propagates_as_nan(self):
        raster = make_raster_with_hot_cluster()
        raster[0, 0] = np.nan
        z = hotspots.getis_ord_gi_star(raster, window_size=3)
        self.assertTrue(np.isnan(z[0, 0]))


class TestClassifyHotspots(unittest.TestCase):
    def test_classification_thresholds(self):
        z = np.array([3.0, 2.0, 1.7, 0.5, -1.7, -2.0, -3.0, np.nan])
        cat = hotspots.classify_hotspots(z)
        expected = [
            hotspots.HOT_99, hotspots.HOT_95, hotspots.HOT_90, hotspots.NOT_SIGNIFICANT,
            hotspots.COLD_90, hotspots.COLD_95, hotspots.COLD_99, hotspots.NODATA,
        ]
        np.testing.assert_array_equal(cat, expected)


class TestPersistentHotspots(unittest.TestCase):
    def test_pixel_hot_every_year_is_persistent(self):
        hot = np.array([[hotspots.HOT_99]])
        cold = np.array([[hotspots.NOT_SIGNIFICANT]])
        years = [hot, hot, hot, cold]  # hot in 3/4 years
        frequency, persistent = hotspots.persistent_hotspots(years, persistent_threshold=0.6)
        self.assertAlmostEqual(frequency[0, 0], 0.75)
        self.assertTrue(persistent[0, 0])

    def test_pixel_rarely_hot_is_not_persistent(self):
        hot = np.array([[hotspots.HOT_99]])
        cold = np.array([[hotspots.NOT_SIGNIFICANT]])
        years = [hot, cold, cold, cold]
        frequency, persistent = hotspots.persistent_hotspots(years, persistent_threshold=0.6)
        self.assertAlmostEqual(frequency[0, 0], 0.25)
        self.assertFalse(persistent[0, 0])

    def test_nodata_years_excluded_from_denominator(self):
        hot = np.array([[hotspots.HOT_99]])
        nodata = np.array([[hotspots.NODATA]])
        years = [hot, hot, nodata]
        frequency, persistent = hotspots.persistent_hotspots(years, persistent_threshold=0.6)
        self.assertAlmostEqual(frequency[0, 0], 1.0)  # 2/2 valid years, not 2/3


if __name__ == "__main__":
    unittest.main()
