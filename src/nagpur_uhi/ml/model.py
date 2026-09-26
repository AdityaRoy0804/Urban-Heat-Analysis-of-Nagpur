"""
Surrogate ML model: LST as a function of land-cover and terrain features.

Framing (important, per project scope): this model is a *statistical
surrogate* learned from historical spatial correlations between land cover
and LST. It is not a physical climate simulator, and it is not a forecaster -
per the brief, "the objective is scenario-based estimation, not absolute
prediction of future climate." Its job is to answer "how does LST tend to
respond to a change in these features, at this kind of location" in a way
that's explainable (see ml/explain.py), not to hit a target R^2.

Because of that framing, `evaluate_directional_validity` matters more than
plain R^2 here: it backtests whether the model correctly predicts the *sign
and rough magnitude* of an LST change when fed the *actual* feature changes
that occurred historically between two study years in held-out zones. That's
the right validation question for a counterfactual/scenario tool - it
doesn't validate absolute prediction, it validates whether the model's
learned sensitivity to feature change matches what actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ModelType = RandomForestRegressor | GradientBoostingRegressor


def train_lst_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target",
    model_type: str = "random_forest",
    n_estimators: int = 300,
    max_depth: int | None = 12,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> ModelType:
    """Fit an interpretable tree-ensemble regressor for LST."""
    missing = [c for c in feature_columns if c not in train_df.columns]
    if missing:
        raise ValueError(f"train_df is missing feature columns: {missing}")

    X = train_df[feature_columns].to_numpy()
    y = train_df[target_column].to_numpy()

    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth or 3,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'; use 'random_forest' or 'gradient_boosting'")

    model.fit(X, y)
    model.feature_names_in_list_ = feature_columns  # convenience attribute for downstream code
    return model


@dataclass
class EvaluationResult:
    r_squared: float
    mae: float
    n_test_samples: int
    directional_accuracy: float | None = None
    n_backtest_pairs: int | None = None


def evaluate_holdout(model: ModelType, test_df: pd.DataFrame, feature_columns: list[str], target_column: str = "target") -> EvaluationResult:
    """Standard held-out R^2/MAE - reported for transparency, secondary to directional validity."""
    X_test = test_df[feature_columns].to_numpy()
    y_test = test_df[target_column].to_numpy()
    y_pred = model.predict(X_test)
    return EvaluationResult(
        r_squared=float(r2_score(y_test, y_pred)),
        mae=float(mean_absolute_error(y_test, y_pred)),
        n_test_samples=len(y_test),
    )


def evaluate_directional_validity(
    model: ModelType,
    features_before: pd.DataFrame,
    features_after: pd.DataFrame,
    observed_lst_before: np.ndarray,
    observed_lst_after: np.ndarray,
    feature_columns: list[str],
) -> EvaluationResult:
    """
    Backtest: for a set of held-out zones with known features/LST at two real
    time points, check whether the model - given the *actual* feature values
    at each time point - predicts an LST change with the same sign (and
    similar order of magnitude) as what was actually observed.

    This is the validation metric that matters for a scenario/what-if tool:
    it tests whether the model's learned sensitivity to feature change is
    trustworthy, without pretending the model can forecast absolute LST.
    """
    if not (len(features_before) == len(features_after) == len(observed_lst_before) == len(observed_lst_after)):
        raise ValueError("features_before/after and observed LST arrays must have matching, equal lengths")

    pred_before = model.predict(features_before[feature_columns].to_numpy())
    pred_after = model.predict(features_after[feature_columns].to_numpy())

    predicted_delta = pred_after - pred_before
    observed_delta = np.asarray(observed_lst_after) - np.asarray(observed_lst_before)

    same_sign = np.sign(predicted_delta) == np.sign(observed_delta)
    # Treat near-zero observed changes (< 0.1 deg C) as "no meaningful change" and
    # exclude them from the directional check - sign comparisons on noise are meaningless.
    meaningful = np.abs(observed_delta) >= 0.1
    if meaningful.sum() == 0:
        directional_accuracy = float("nan")
    else:
        directional_accuracy = float(same_sign[meaningful].mean())

    magnitude_mae = float(mean_absolute_error(observed_delta, predicted_delta))

    return EvaluationResult(
        r_squared=float("nan"),
        mae=magnitude_mae,
        n_test_samples=len(observed_delta),
        directional_accuracy=directional_accuracy,
        n_backtest_pairs=int(meaningful.sum()),
    )
