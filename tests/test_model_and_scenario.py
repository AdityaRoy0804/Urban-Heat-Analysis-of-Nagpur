import unittest

import numpy as np
import pandas as pd

from nagpur_uhi.ml.dataset import spatial_block_split
from nagpur_uhi.ml.explain import ScenarioExplainer
from nagpur_uhi.ml.model import evaluate_directional_validity, evaluate_holdout, train_lst_model
from nagpur_uhi.ml.scenario import find_analog_zones, simulate_scenario

FEATURE_COLUMNS = ["ndvi", "ndbi", "elevation_m"]


def make_synthetic_zone_dataset(n_zones=200, seed=42) -> pd.DataFrame:
    """
    Zones with a known ground-truth relationship:
        LST = 32 - 9*ndvi + 6*ndbi - 0.01*elevation_m + noise
    so tests can check the model recovers the correct *direction* of effect.
    """
    rng = np.random.default_rng(seed)
    ndvi = rng.uniform(0, 1, n_zones)
    ndbi = rng.uniform(-0.3, 0.5, n_zones)
    elevation = rng.uniform(280, 320, n_zones)  # Nagpur is ~305m elevation
    noise = rng.normal(0, 0.3, n_zones)
    lst = 32 - 9 * ndvi + 6 * ndbi - 0.01 * elevation + noise

    lulc_class = np.where(ndvi > 0.5, "vegetation", np.where(ndbi > 0.1, "built_up", "bare_open"))
    rows = np.repeat(np.arange(10), n_zones // 10 + 1)[:n_zones]
    cols = np.tile(np.arange(n_zones // 10 + 1), 10)[:n_zones]

    return pd.DataFrame(
        {
            "zone_id": [f"z_{i}" for i in range(n_zones)],
            "row": rows,
            "col": cols,
            "ndvi": ndvi,
            "ndbi": ndbi,
            "elevation_m": elevation,
            "target": lst,
            "lulc_class": lulc_class,
        }
    )


class TestTrainLSTModel(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_zone_dataset()
        self.train_df, self.test_df = spatial_block_split(self.df, block_size_px=2, test_size=0.25, random_state=0)
        self.model = train_lst_model(
            self.train_df, FEATURE_COLUMNS, model_type="random_forest", n_estimators=100, random_state=0
        )

    def test_holdout_r_squared_is_reasonably_high(self):
        result = evaluate_holdout(self.model, self.test_df, FEATURE_COLUMNS)
        self.assertGreater(result.r_squared, 0.7)

    def test_model_learns_correct_direction_for_ndvi(self):
        # Increasing NDVI, holding everything else fixed, should lower predicted LST.
        base = pd.DataFrame({"ndvi": [0.2], "ndbi": [0.0], "elevation_m": [300.0]})
        high_ndvi = base.copy()
        high_ndvi["ndvi"] = 0.8
        pred_low = self.model.predict(base[FEATURE_COLUMNS].to_numpy())[0]
        pred_high = self.model.predict(high_ndvi[FEATURE_COLUMNS].to_numpy())[0]
        self.assertLess(pred_high, pred_low)

    def test_directional_validity_backtest(self):
        rng = np.random.default_rng(5)
        n = 30
        before = pd.DataFrame(
            {"ndvi": rng.uniform(0.4, 0.6, n), "ndbi": rng.uniform(-0.1, 0.1, n), "elevation_m": rng.uniform(295, 305, n)}
        )
        after = before.copy()
        after["ndvi"] = before["ndvi"] - 0.4  # vegetation cleared
        after["ndbi"] = before["ndbi"] + 0.3  # built-up increases

        lst_before = 32 - 9 * before["ndvi"] + 6 * before["ndbi"] - 0.01 * before["elevation_m"]
        lst_after = 32 - 9 * after["ndvi"] + 6 * after["ndbi"] - 0.01 * after["elevation_m"]

        result = evaluate_directional_validity(
            self.model, before, after, lst_before.to_numpy(), lst_after.to_numpy(), FEATURE_COLUMNS
        )
        # Vegetation loss + built-up gain should consistently predict warming (positive delta)
        self.assertGreater(result.directional_accuracy, 0.8)


class TestScenarioExplainer(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_zone_dataset()
        self.model = train_lst_model(self.df, FEATURE_COLUMNS, model_type="random_forest", n_estimators=100, random_state=0)
        self.explainer = ScenarioExplainer(self.model, FEATURE_COLUMNS, background_data=self.df[FEATURE_COLUMNS])

    def test_uses_occlusion_fallback_without_shap(self):
        # shap is not installed in this test environment - confirm graceful fallback, not a crash.
        instance = self.df[FEATURE_COLUMNS].iloc[0]
        explanation = self.explainer.explain_instance(instance)
        self.assertIn(explanation.method, ("shap", "occlusion"))
        self.assertEqual(set(explanation.contributions.keys()), set(FEATURE_COLUMNS))

    def test_contributions_are_additively_consistent(self):
        instance = self.df[FEATURE_COLUMNS].iloc[10]
        explanation = self.explainer.explain_instance(instance)
        total_contribution = sum(explanation.contributions.values())
        self.assertAlmostEqual(total_contribution, explanation.prediction - explanation.base_value, delta=0.05)

    def test_high_ndvi_gets_negative_contribution_relative_to_baseline(self):
        # An instance with much higher NDVI than the background mean should
        # show NDVI pulling the prediction down (cooling effect).
        background_mean_ndvi = self.df["ndvi"].mean()
        instance = pd.Series({"ndvi": min(background_mean_ndvi + 0.4, 0.99), "ndbi": 0.0, "elevation_m": 300.0})
        explanation = self.explainer.explain_instance(instance)
        self.assertLess(explanation.contributions["ndvi"], 0)


class TestScenarioSimulation(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_zone_dataset()
        self.model = train_lst_model(self.df, FEATURE_COLUMNS, model_type="random_forest", n_estimators=100, random_state=0)
        self.explainer = ScenarioExplainer(self.model, FEATURE_COLUMNS, background_data=self.df[FEATURE_COLUMNS])

    def test_find_analog_zones_returns_requested_k(self):
        reference = self.df.iloc[0]
        analogs = find_analog_zones(
            target_land_cover_class="vegetation",
            scenario_land_cover_class="built_up",
            candidate_zones=self.df,
            feature_columns=FEATURE_COLUMNS,
            reference_features=reference,
            k=5,
        )
        self.assertEqual(len(analogs), 5)
        self.assertTrue((analogs["lulc_class"] == "built_up").all())

    def test_find_analog_zones_raises_if_too_few_candidates(self):
        tiny_pool = self.df[self.df["lulc_class"] == "built_up"].head(2)
        with self.assertRaises(ValueError):
            find_analog_zones(
                target_land_cover_class="vegetation",
                scenario_land_cover_class="built_up",
                candidate_zones=tiny_pool,
                feature_columns=FEATURE_COLUMNS,
                reference_features=self.df.iloc[0],
                k=5,
            )

    def test_simulate_scenario_vegetation_to_built_up_warms(self):
        # A vegetated zone simulating a transition to built_up should warm up,
        # matching the synthetic ground truth (higher ndbi, lower ndvi -> hotter).
        vegetated_zone = self.df[self.df["lulc_class"] == "vegetation"].iloc[0]
        result = simulate_scenario(
            zone_id=vegetated_zone["zone_id"],
            baseline_features=vegetated_zone,
            scenario_land_cover_class="built_up",
            candidate_zones=self.df,
            model=self.model,
            explainer=self.explainer,
            feature_columns=FEATURE_COLUMNS,
            static_context_columns=["elevation_m"],
            hotspot_frequency=0.6,
            k_analogs=5,
        )
        self.assertGreater(result.estimated_shift_celsius, 0)
        self.assertEqual(result.explanation.method in ("shap", "occlusion"), True)
        self.assertIn("hotspot", result.historical_hotspot_context.lower())
        self.assertEqual(len(result.analog_zone_ids), 5)


if __name__ == "__main__":
    unittest.main()
