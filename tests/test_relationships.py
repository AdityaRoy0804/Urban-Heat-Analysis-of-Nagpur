import unittest

import numpy as np

from nagpur_uhi.analysis import relationships


class TestGlobalCorrelation(unittest.TestCase):
    def test_negative_correlation_between_lst_and_ndvi(self):
        rng = np.random.default_rng(1)
        ndvi = rng.uniform(0, 1, size=(40, 40))
        lst = 45 - 10 * ndvi + rng.normal(0, 0.5, size=(40, 40))  # more vegetation -> cooler
        result = relationships.global_correlation(lst, ndvi)
        self.assertLess(result["pearson_r"], -0.8)
        self.assertLess(result["p_value"], 0.01)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            relationships.global_correlation(np.zeros((3, 3)), np.zeros((4, 4)))

    def test_too_few_valid_pixels_raises(self):
        lst = np.array([[np.nan, np.nan], [1.0, np.nan]])
        ndvi = np.array([[np.nan, np.nan], [0.5, np.nan]])
        with self.assertRaises(ValueError):
            relationships.global_correlation(lst, ndvi)


class TestMultivariateLinearFit(unittest.TestCase):
    def test_recovers_known_linear_relationship(self):
        rng = np.random.default_rng(2)
        ndvi = rng.uniform(0, 1, size=(60, 60))
        ndbi = rng.uniform(-0.5, 0.5, size=(60, 60))
        lst = 30 - 8 * ndvi + 6 * ndbi + rng.normal(0, 0.1, size=(60, 60))

        fit = relationships.multivariate_linear_fit(lst, {"ndvi": ndvi, "ndbi": ndbi})
        self.assertAlmostEqual(fit["coefficients"]["ndvi"], -8, delta=0.5)
        self.assertAlmostEqual(fit["coefficients"]["ndbi"], 6, delta=0.5)
        self.assertGreater(fit["r_squared"], 0.95)


class TestLocalRegressionMap(unittest.TestCase):
    def test_local_slope_sign_matches_global_relationship(self):
        rng = np.random.default_rng(3)
        ndvi = rng.uniform(0, 1, size=(25, 25))
        lst = 40 - 10 * ndvi + rng.normal(0, 0.2, size=(25, 25))
        result = relationships.local_regression_map(lst, ndvi, window_size=9)
        valid_slopes = result["slope"][~np.isnan(result["slope"])]
        self.assertGreater(len(valid_slopes), 0)
        self.assertLess(np.median(valid_slopes), 0)  # vegetation cools, so slope should be negative


if __name__ == "__main__":
    unittest.main()
