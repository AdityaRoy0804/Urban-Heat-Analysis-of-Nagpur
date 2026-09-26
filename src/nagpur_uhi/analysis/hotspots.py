"""
Spatial hotspot analysis: Getis-Ord Gi* statistic and multi-year persistence.

Implemented from first principles with numpy/scipy (no `esda`/`libpysal`
dependency) because raster grid data has a natural fixed-distance-band /
rook-contiguity spatial-weight structure that a square moving window computes
directly and efficiently via `scipy.ndimage.uniform_filter`. This makes the
statistic testable with plain numpy in any environment, and is mathematically
equivalent to Gi* with a binary fixed-distance-band weight matrix (Ord &
Getis, 1995) restricted to a square window rather than a circular one.

Why Gi* instead of a naive "top 10% of LST pixels" rule: an arbitrary
percentile cutoff doesn't account for spatial clustering or local variance -
it can flag isolated warm pixels as "hotspots" even when they aren't part of
a statistically significant cluster. Gi* explicitly tests whether a location
and its neighbours are hotter/colder than would be expected under spatial
randomness, which is the correct statistical framing for "identification of
hotter and cooler zones" in the problem statement.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

# Category codes returned by classify_hotspots
NODATA = -128
NOT_SIGNIFICANT = 0
HOT_90, HOT_95, HOT_99 = 1, 2, 3
COLD_90, COLD_95, COLD_99 = -1, -2, -3


def getis_ord_gi_star(
    x: np.ndarray, window_size: int = 3, nodata_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the Gi* z-score for every pixel in `x` using a square window of
    side `window_size` (must be odd, >= 3) as the spatial-weight neighbourhood
    (weights = 1 within the window including the centre pixel, 0 outside -
    this is the "Gi*", not "Gi", variant, since the centre pixel is included).

    Global mean/variance are computed once over all valid pixels in `x`;
    local sums are computed per-window. Returns a float array of z-scores,
    the same shape as `x`, with NaN at nodata pixels or pixels whose window
    has fewer than 2 valid neighbours.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    x = np.asarray(x, dtype="float64")
    if nodata_mask is None:
        nodata_mask = np.isnan(x)
    valid = ~nodata_mask

    n = int(valid.sum())
    if n < 2:
        raise ValueError("Need at least 2 valid pixels to compute Gi*")

    valid_values = x[valid]
    x_bar = valid_values.mean()
    s = np.sqrt((valid_values**2).mean() - x_bar**2)
    if s == 0:
        raise ValueError("Global standard deviation is zero (constant raster) - Gi* is undefined")

    x_filled = np.where(valid, x, 0.0)
    valid_f = valid.astype("float64")
    window_area = window_size**2

    # uniform_filter gives the window *mean* over a zero-padded boundary, so
    # multiplying back by the window area recovers the window *sum* while
    # correctly excluding out-of-image and nodata pixels from the sum.
    w_count = uniform_filter(valid_f, size=window_size, mode="constant", cval=0.0) * window_area
    local_sum = uniform_filter(x_filled * valid_f, size=window_size, mode="constant", cval=0.0) * window_area

    with np.errstate(invalid="ignore", divide="ignore"):
        denom_inner = w_count * (n - w_count) / (n - 1)
        denom = s * np.sqrt(np.where(denom_inner > 0, denom_inner, np.nan))
        z = (local_sum - x_bar * w_count) / denom

    z = np.where(valid & (w_count >= 2) & (denom_inner > 0), z, np.nan)
    return z


def classify_hotspots(z: np.ndarray, confidence_levels: tuple[float, float, float] = (1.65, 1.96, 2.58)) -> np.ndarray:
    """
    Classify Gi* z-scores into confidence-graded hot/cold categories.

    Returns an int8 array with values:
        3 / -3  hot / cold at 99% confidence  (|z| >= confidence_levels[2])
        2 / -2  hot / cold at 95% confidence
        1 / -1  hot / cold at 90% confidence
        0       not statistically significant
       -128     nodata (input was NaN)
    """
    z90, z95, z99 = confidence_levels
    cat = np.zeros(z.shape, dtype="int8")
    cat = np.where(z >= z99, HOT_99, cat)
    cat = np.where((z >= z95) & (z < z99), HOT_95, cat)
    cat = np.where((z >= z90) & (z < z95), HOT_90, cat)
    cat = np.where((z <= -z90) & (z > -z95), COLD_90, cat)
    cat = np.where((z <= -z95) & (z > -z99), COLD_95, cat)
    cat = np.where(z <= -z99, COLD_99, cat)
    cat = np.where(np.isnan(z), NODATA, cat).astype("int8")
    return cat


def persistent_hotspots(
    hotspot_categories_by_year: list[np.ndarray],
    min_confidence_level: int = HOT_95,
    persistent_threshold: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Overlay multiple years of `classify_hotspots` output to find *persistent*
    heat hotspots - pixels flagged hot (at >= `min_confidence_level`) in at
    least `persistent_threshold` of the years that have valid data there.

    Returns (frequency, persistent_mask):
        frequency: float array, fraction of valid years each pixel was hot
        persistent_mask: bool array, True where frequency >= persistent_threshold
    """
    if len(hotspot_categories_by_year) < 2:
        raise ValueError("Need at least 2 years of hotspot classifications for a persistence analysis")

    stack = np.stack(hotspot_categories_by_year, axis=0)
    valid = stack != NODATA
    is_hot = (stack >= min_confidence_level) & valid

    n_valid_years = valid.sum(axis=0)
    frequency = np.full(n_valid_years.shape, np.nan, dtype="float64")
    has_data = n_valid_years > 0
    frequency[has_data] = is_hot.sum(axis=0)[has_data] / n_valid_years[has_data]

    persistent_mask = np.where(has_data, frequency >= persistent_threshold, False)
    return frequency, persistent_mask
