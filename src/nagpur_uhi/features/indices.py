"""
Spectral index and Land Surface Temperature (LST) computations.

Pure numpy on already-aligned raster arrays - no GIS I/O here by design, so
this is the most heavily unit-tested module in the repo (see tests/test_indices.py).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _safe_normalized_difference(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """(a-b)/(a+b) with division-by-zero and NaN-nodata handling."""
    band_a = band_a.astype("float64")
    band_b = band_b.astype("float64")
    denom = band_a + band_b
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(np.abs(denom) > _EPS, (band_a - band_b) / denom, np.nan)
    return result


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index. Range [-1, 1]; higher = more vegetation."""
    _validate_same_shape(nir, red)
    return _safe_normalized_difference(nir, red)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Built-up Index. Higher = more built-up/impervious surface."""
    _validate_same_shape(swir, nir)
    return _safe_normalized_difference(swir, nir)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters). Higher = more open water."""
    _validate_same_shape(green, nir)
    return _safe_normalized_difference(green, nir)


def sar_vv_vh_ratio_db(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """
    Sentinel-1 VV/VH ratio in dB-difference form (VV_db - VH_db), used as a
    texture/roughness proxy that discriminates bare ground, construction
    activity, and finished built-up surfaces independent of cloud cover.
    """
    _validate_same_shape(vv_db, vh_db)
    return vv_db.astype("float64") - vh_db.astype("float64")


def lst_celsius_from_landsat_c2(st_b10_dn: np.ndarray) -> np.ndarray:
    """
    Convert a raw Landsat Collection-2 Level-2 ST_B10 digital number to
    surface temperature in Celsius, using USGS's official scale/offset:
        ST(Kelvin) = DN * 0.00341802 + 149.0

    This assumes the *raw* uint16 DN, not the already-scaled band (if you
    acquired data through gee_client.get_landsat_composite, the scaling is
    already applied and this function is not needed - it exists for the case
    of ingesting Collection-2 GeoTIFFs directly from USGS EarthExplorer).
    """
    kelvin = st_b10_dn.astype("float64") * 0.00341802 + 149.0
    return kelvin - 273.15


def _validate_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Band shape mismatch: {a.shape} vs {b.shape}. Inputs must be pre-aligned.")
