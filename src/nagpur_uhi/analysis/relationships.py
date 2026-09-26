"""
Relationship analysis between land cover indicators (NDVI/NDBI/...) and LST.

Two levels, matching the brief's "relationship analysis between land cover
and surface temperature":

1. Global: Pearson correlation + a multivariate linear fit, for a single
   headline statistic per study year.
2. Local (approximate GWR): the LST-vs-predictor relationship is well known
   to vary spatially across a city rather than following one global slope
   (a park in a dense core cools differently than a park at the city edge).
   `local_regression_map` fits an independent linear regression inside a
   moving window at every pixel, giving a slope/R^2 raster - a lightweight,
   dependency-free approximation of Geographically Weighted Regression. Swap
   in the `mgwr` package for a more rigorous kernel-weighted GWR if desired;
   the moving-window version is intentionally simple so it needs only numpy.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def global_correlation(lst: np.ndarray, predictor: np.ndarray) -> dict[str, float]:
    """Pearson correlation between LST and a predictor (e.g. NDVI or NDBI), over valid pixels."""
    if lst.shape != predictor.shape:
        raise ValueError(f"Shape mismatch: lst {lst.shape} vs predictor {predictor.shape}")
    valid = ~(np.isnan(lst) | np.isnan(predictor))
    if valid.sum() < 3:
        raise ValueError("Need at least 3 valid, co-located pixels to compute a correlation")
    r, p_value = stats.pearsonr(lst[valid], predictor[valid])
    return {"pearson_r": float(r), "p_value": float(p_value), "n_valid_pixels": int(valid.sum())}


def multivariate_linear_fit(lst: np.ndarray, predictors: dict[str, np.ndarray]) -> dict[str, object]:
    """
    Ordinary least squares LST ~ predictors, using numpy's lstsq (avoids a
    statsmodels dependency). Returns coefficients, intercept, and R^2.
    Use this for a quick multi-predictor summary; use the ML module
    (ml/model.py) when you need a non-linear, explainable model instead.
    """
    names = list(predictors.keys())
    stacked = np.stack([predictors[name] for name in names] + [lst], axis=0)
    valid = ~np.any(np.isnan(stacked), axis=0)
    if valid.sum() < len(names) + 2:
        raise ValueError("Not enough valid co-located pixels for the number of predictors given")

    X = np.stack([predictors[name][valid] for name in names], axis=1)
    y = lst[valid]
    X_design = np.hstack([X, np.ones((X.shape[0], 1))])

    coeffs, residuals, rank, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "coefficients": dict(zip(names, coeffs[:-1].tolist())),
        "intercept": float(coeffs[-1]),
        "r_squared": r_squared,
        "n_valid_pixels": int(valid.sum()),
    }


def local_regression_map(
    lst: np.ndarray, predictor: np.ndarray, window_size: int = 15
) -> dict[str, np.ndarray]:
    """
    Approximate GWR: fit an independent simple linear regression (LST ~ predictor)
    inside a `window_size` x `window_size` window centred on every pixel.

    Returns per-pixel {"slope", "r_squared"} arrays showing how the
    LST-predictor relationship's strength and direction vary across the city.
    Windows with fewer than 5 valid pixels are set to NaN.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    if lst.shape != predictor.shape:
        raise ValueError(f"Shape mismatch: lst {lst.shape} vs predictor {predictor.shape}")

    half = window_size // 2
    rows, cols = lst.shape
    slope_map = np.full(lst.shape, np.nan, dtype="float64")
    r2_map = np.full(lst.shape, np.nan, dtype="float64")

    for r in range(rows):
        r0, r1 = max(0, r - half), min(rows, r + half + 1)
        for c in range(cols):
            c0, c1 = max(0, c - half), min(cols, c + half + 1)
            y_win = lst[r0:r1, c0:c1].ravel()
            x_win = predictor[r0:r1, c0:c1].ravel()
            valid = ~(np.isnan(y_win) | np.isnan(x_win))
            if valid.sum() < 5 or np.std(x_win[valid]) == 0:
                continue
            slope, _intercept, r_value, _p, _se = stats.linregress(x_win[valid], y_win[valid])
            slope_map[r, c] = slope
            r2_map[r, c] = r_value**2

    return {"slope": slope_map, "r_squared": r2_map}
