"""
Land use / land cover (LULC) classification.

Two interchangeable strategies, both pure numpy/sklearn (no GIS-stack
dependency), because the emphasis in this project is explainability:

1. `rule_based_lulc` - a transparent, threshold-based classifier using
   NDVI/NDBI/NDWI. Every decision is inspectable and defensible without any
   training data, which matters for study years where labeled samples don't
   exist (e.g. WorldCover only covers 2020/2021). Thresholds live in
   config.yaml, not hard-coded, so they can be tuned/validated per study year.
2. `train_rf_classifier` / `classify_rf` - a Random Forest trained on a small
   labeled sample set, for years/areas where you *do* have reference points
   (field survey, high-res imagery digitising, or a reference product like
   Dynamic World/WorldCover used as training labels).

Both return the same class labels so downstream code (hotspot/relationship
analysis) doesn't need to know which strategy produced them.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

CLASS_LABELS = ["water", "vegetation", "built_up", "bare_open"]


def rule_based_lulc(
    ndvi: np.ndarray,
    ndbi: np.ndarray,
    ndwi: np.ndarray,
    ndvi_vegetation_min: float = 0.30,
    ndwi_water_min: float = 0.30,
    ndbi_built_up_min: float = 0.0,
) -> np.ndarray:
    """
    Transparent threshold-based LULC classifier.

    Precedence (checked in order, first match wins):
      1. water:      ndwi >= ndwi_water_min
      2. vegetation: ndvi >= ndvi_vegetation_min
      3. built_up:   ndbi >= ndbi_built_up_min AND ndvi < ndvi_vegetation_min
      4. bare_open:  everything else

    Returns an integer array with the same shape as the inputs, where values
    index into CLASS_LABELS (0=water, 1=vegetation, 2=built_up, 3=bare_open).
    NaN input pixels are labeled -1 (nodata).
    """
    for name, arr in [("ndvi", ndvi), ("ndbi", ndbi), ("ndwi", ndwi)]:
        if arr.shape != ndvi.shape:
            raise ValueError(f"Shape mismatch for '{name}': {arr.shape} vs {ndvi.shape}")

    labels = np.full(ndvi.shape, 3, dtype="int16")  # default: bare_open
    valid = ~(np.isnan(ndvi) | np.isnan(ndbi) | np.isnan(ndwi))

    is_water = valid & (ndwi >= ndwi_water_min)
    is_vegetation = valid & ~is_water & (ndvi >= ndvi_vegetation_min)
    is_built_up = valid & ~is_water & ~is_vegetation & (ndbi >= ndbi_built_up_min)

    labels[is_water] = 0
    labels[is_vegetation] = 1
    labels[is_built_up] = 2
    labels[~valid] = -1
    return labels


def train_rf_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = 15,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest LULC classifier on labeled samples.

    X_train: (n_samples, n_features) array, e.g. columns [ndvi, ndbi, ndwi, ...].
    y_train: (n_samples,) integer class labels matching CLASS_LABELS indices.
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train has {X_train.shape[0]} samples but y_train has {y_train.shape[0]}"
        )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def classify_rf(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Apply a trained RF classifier to a feature array, returning class labels."""
    return model.predict(X)
