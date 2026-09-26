#!/usr/bin/env python3
"""
Stage 0 (optional) - Generate SYNTHETIC demo data for Nagpur.

Produces the exact same on-disk artifacts that scripts/02_extract_features.py
would have produced from real satellite data (data/processed/features_{year}.npz
and data/processed/terrain_context.npz) - so scripts/03 and 04, the API, and the
frontend all run immediately, end to end, with zero external data, credentials,
or network access.

THIS IS NOT REAL SATELLITE DATA. It's a deterministic, physically-plausible
synthetic scene: an urban core and a faster-growing peri-urban cluster (modelled
loosely on a "Hingna Gramin"-style edge-of-city location) that expand over the
study years, two fixed water bodies, and a green belt - with LST, NDVI, NDBI,
NDWI and SAR-ratio generated from a consistent underlying land-cover field, so
the relationships the ML model learns from it are directionally sensible
(more built-up -> hotter, more vegetation -> cooler, water -> cooler).

Use this to:
  - develop/demo the analysis, ML, API and frontend layers immediately
  - sanity-check the pipeline's plumbing (shapes, dtypes, file formats) before
    spending GEE/CDSE/download quota on real data

Do NOT use this to draw any real conclusion about Nagpur - replace it with
scripts/01+02 (GEE or STAC) or manual downloads (see README) before that.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from nagpur_uhi.config import load_config
from nagpur_uhi.features.lulc import rule_based_lulc
from nagpur_uhi.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

GRID_SIZE = 200          # synthetic scene is 200x200 "pixels" (resolution is arbitrary for a demo)
BASE_SEED = 20260926     # fixed for reproducibility


def _gaussian_blob(size: int, center: tuple[float, float], sigma: float, amplitude: float = 1.0) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    cy, cx = center
    return amplitude * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)))


def _smoothed_noise(size: int, rng: np.random.Generator, sigma: float = 6.0, amplitude: float = 0.05) -> np.ndarray:
    noise = rng.normal(0, 1, size=(size, size))
    return amplitude * gaussian_filter(noise, sigma=sigma)


def generate_year(year: int, all_years: list[int], size: int = GRID_SIZE, base_seed: int = BASE_SEED) -> dict[str, np.ndarray]:
    """Generate one synthetic study year's feature stack."""
    rng = np.random.default_rng(base_seed + year)
    span = max(1, max(all_years) - min(all_years))
    progress = (year - min(all_years)) / span  # 0.0 (earliest year) -> 1.0 (latest year)

    # Two built-up clusters: an established urban core, and a faster-growing
    # peri-urban cluster (stands in for an edge-of-city area like Hingna Gramin)
    # that starts small and expands more aggressively over the study years.
    core = _gaussian_blob(size, center=(95, 90), sigma=18 + 4 * progress, amplitude=0.55 + 0.35 * progress)
    peri_urban = _gaussian_blob(size, center=(150, 55), sigma=8 + 8 * progress, amplitude=0.08 + 0.55 * progress)
    built_up_intensity = np.clip(core + peri_urban, 0, 1)

    # Two fixed water bodies (stand-ins for Nagpur's urban lakes).
    lake_1 = _gaussian_blob(size, center=(70, 70), sigma=6, amplitude=1.0)
    lake_2 = _gaussian_blob(size, center=(150, 140), sigma=5, amplitude=1.0)
    water_mask = (lake_1 > 0.5) | (lake_2 > 0.5)
    n_water = int(water_mask.sum())

    # Green belt near two corners, mildly eroded by urban growth over time.
    green_belt = _gaussian_blob(size, center=(20, 180), sigma=40, amplitude=0.8) + _gaussian_blob(
        size, center=(185, 15), sigma=35, amplitude=0.7
    )
    base_vegetation = np.clip(0.30 + green_belt - 0.4 * progress * green_belt, 0, 1)

    ndvi = np.clip(
        base_vegetation - 0.55 * built_up_intensity + _smoothed_noise(size, rng, sigma=5, amplitude=0.05), -0.1, 0.9
    )
    ndbi = np.clip(
        -0.30 + 0.90 * built_up_intensity + _smoothed_noise(size, rng, sigma=4, amplitude=0.05), -0.6, 0.6
    )
    ndwi = np.clip(
        -0.30 + 0.30 * (1 - built_up_intensity) + _smoothed_noise(size, rng, sigma=5, amplitude=0.04), -0.6, 0.6
    )
    sar_vv_vh_ratio = (
        4.0 + 6.0 * built_up_intensity - 3.0 * np.clip(ndvi, 0, 1) + _smoothed_noise(size, rng, sigma=3, amplitude=0.3)
    )

    if n_water:
        ndvi[water_mask] = rng.uniform(-0.2, -0.05, size=n_water)
        ndbi[water_mask] = rng.uniform(-0.5, -0.3, size=n_water)
        ndwi[water_mask] = rng.uniform(0.4, 0.7, size=n_water)
        sar_vv_vh_ratio[water_mask] = rng.uniform(-6.0, -3.0, size=n_water)  # specular reflection -> low ratio

    base_temp_celsius = 34.0  # plausible pre-monsoon Nagpur baseline
    lst = (
        base_temp_celsius
        + 6.0 * built_up_intensity
        - 5.0 * np.clip(ndvi, 0, 1)
        + _smoothed_noise(size, rng, sigma=6, amplitude=0.6)
        + rng.normal(0, 0.15, size=(size, size))  # pixel-level sensor-like noise
    )
    if n_water:
        lst[water_mask] -= 3.0

    lulc_class = rule_based_lulc(ndvi, ndbi, ndwi)

    return {
        "lst": lst,
        "ndvi": ndvi,
        "ndbi": ndbi,
        "ndwi": ndwi,
        "sar_vv_vh_ratio": sar_vv_vh_ratio,
        "lulc_class": lulc_class,
    }


def generate_terrain_context(size: int = GRID_SIZE, base_seed: int = BASE_SEED, pixel_size_m: float = 30.0) -> dict[str, np.ndarray]:
    """Gentle, mostly-flat synthetic terrain, broadly consistent with Nagpur's real topography (~305m, low relief)."""
    rng = np.random.default_rng(base_seed)
    y, x = np.mgrid[0:size, 0:size]
    elevation = (
        305.0
        + 8.0 * np.sin(x / 40.0)
        + 6.0 * np.cos(y / 55.0)
        + _smoothed_noise(size, rng, sigma=10.0, amplitude=2.0)
    )
    dz_dy, dz_dx = np.gradient(elevation, pixel_size_m, pixel_size_m)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect_deg = (np.degrees(np.arctan2(-dz_dx, dz_dy)) + 360.0) % 360.0
    return {"elevation_m": elevation, "slope_deg": slope_deg, "aspect_deg": aspect_deg}


def main() -> None:
    configure_logging()
    cfg = load_config()
    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    years = [sy.year for sy in cfg.study_years]
    logger.warning(
        "Generating SYNTHETIC demo data for %d study years (%s) - this is NOT real satellite "
        "observation data. Use it to develop/demo the pipeline; replace with scripts/01+02 "
        "(GEE or STAC) or manual downloads before drawing any real conclusion about Nagpur.",
        len(years), years,
    )

    for year in years:
        feature_stack = generate_year(year, years)
        out_path = processed_dir / f"features_{year}.npz"
        np.savez_compressed(out_path, **feature_stack)
        logger.info(
            "Wrote %s (mean LST=%.1fC, mean NDVI=%.2f, built-up pixel fraction=%.1f%%)",
            out_path, np.nanmean(feature_stack["lst"]), np.nanmean(feature_stack["ndvi"]),
            100.0 * np.mean(feature_stack["lulc_class"] == 2),
        )

    terrain = generate_terrain_context()
    np.savez_compressed(processed_dir / "terrain_context.npz", **terrain)
    logger.info("Wrote terrain_context.npz")
    logger.info("Synthetic demo data ready. Run scripts/03_run_analysis.py next.")


if __name__ == "__main__":
    main()
