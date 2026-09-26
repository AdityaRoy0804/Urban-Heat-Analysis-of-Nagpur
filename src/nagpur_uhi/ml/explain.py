"""
Explainability layer for the LST scenario model.

Per the project priorities, explainability is the deliverable here, not raw
accuracy. `ScenarioExplainer` produces a per-feature attribution for a single
prediction (a "local explanation") so a user gets "this estimate is driven
60% by vegetation loss, 30% by built-up increase, 10% by terrain" rather than
just a number.

Uses `shap.TreeExplainer` when the `shap` package is installed (the intended
production path for tree-ensemble models - exact Shapley values, additive and
consistent). Falls back to a simple, dependency-free occlusion/leave-one-out
attribution otherwise, so this module - and its tests - work even where shap
isn't installed. The fallback is a legitimate (if cruder) local explanation
technique: it measures each feature's marginal contribution by replacing it
with a baseline (background mean) value and observing the change in
prediction, one feature at a time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    base_value: float
    prediction: float
    contributions: dict[str, float]  # feature_name -> signed contribution to (prediction - base_value)
    method: str  # "shap" or "occlusion"

    def as_sorted_list(self) -> list[tuple[str, float]]:
        """Feature contributions sorted by absolute magnitude, largest first."""
        return sorted(self.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

    def summary_text(self, top_n: int = 3) -> str:
        """A short, human-readable summary of the top contributing features."""
        total_abs = sum(abs(v) for v in self.contributions.values()) or 1e-9
        parts = []
        for name, value in self.as_sorted_list()[:top_n]:
            pct = 100 * abs(value) / total_abs
            direction = "raising" if value > 0 else "lowering"
            parts.append(f"{name} ({direction} the estimate, ~{pct:.0f}% of the explained change)")
        return "; ".join(parts)


class ScenarioExplainer:
    def __init__(self, model, feature_names: list[str], background_data: pd.DataFrame):
        """
        model: a fitted sklearn tree-ensemble regressor (see ml/model.py).
        feature_names: ordered list of feature column names the model expects.
        background_data: a representative sample of feature rows (e.g. the
            training set, or a random subsample of it) used as the SHAP
            background distribution / occlusion baseline.
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data[feature_names]
        self._baseline_values = self.background_data.mean().to_dict()
        self._shap_explainer = self._try_build_shap_explainer()

    def _try_build_shap_explainer(self):
        try:
            import shap  # type: ignore

            return shap.TreeExplainer(self.model, data=self.background_data.sample(
                min(100, len(self.background_data)), random_state=0
            ))
        except ImportError:
            logger.warning(
                "shap is not installed; ScenarioExplainer will use the deterministic "
                "occlusion-based fallback. Install `shap` for exact Shapley-value "
                "explanations in production: pip install shap"
            )
            return None
        except Exception as e:  # pragma: no cover - defensive; shap version quirks vary
            logger.warning("Failed to initialise shap.TreeExplainer (%s); using fallback.", e)
            return None

    def explain_instance(self, instance: pd.Series | dict) -> ExplanationResult:
        row = pd.Series(instance)[self.feature_names]
        X = row.to_numpy().reshape(1, -1)
        prediction = float(self.model.predict(X)[0])

        if self._shap_explainer is not None:
            return self._shap_explain(row, prediction)
        return self._occlusion_explain(row, prediction)

    def _shap_explain(self, row: pd.Series, prediction: float) -> ExplanationResult:
        import shap  # type: ignore

        shap_values = self._shap_explainer.shap_values(row.to_numpy().reshape(1, -1))
        shap_values = np.asarray(shap_values).reshape(-1)
        base_value = float(np.asarray(self._shap_explainer.expected_value).reshape(-1)[0])
        contributions = {name: float(v) for name, v in zip(self.feature_names, shap_values)}
        return ExplanationResult(base_value=base_value, prediction=prediction, contributions=contributions, method="shap")

    def _occlusion_explain(self, row: pd.Series, prediction: float) -> ExplanationResult:
        baseline_row = pd.Series(self._baseline_values)[self.feature_names]
        base_value = float(self.model.predict(baseline_row.to_numpy().reshape(1, -1))[0])

        contributions: dict[str, float] = {}
        for name in self.feature_names:
            perturbed = row.copy()
            perturbed[name] = baseline_row[name]
            perturbed_pred = float(self.model.predict(perturbed.to_numpy().reshape(1, -1))[0])
            # Contribution = how much removing this feature (reverting it to baseline)
            # would have changed the prediction, i.e. this feature's marginal effect.
            contributions[name] = prediction - perturbed_pred

        # Rescale so contributions sum to (prediction - base_value), matching the
        # additive property SHAP guarantees, for a consistent summary_text().
        raw_total = sum(contributions.values())
        target_total = prediction - base_value
        if abs(raw_total) > 1e-9:
            scale = target_total / raw_total
            contributions = {k: v * scale for k, v in contributions.items()}

        return ExplanationResult(base_value=base_value, prediction=prediction, contributions=contributions, method="occlusion")
