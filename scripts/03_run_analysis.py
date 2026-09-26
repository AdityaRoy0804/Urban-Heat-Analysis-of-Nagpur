#!/usr/bin/env python3
"""
Stage 3 - Multi-temporal spatial analysis.

Loads each year's features_{year}.npz (from scripts/02), computes the Gi*
hotspot classification per year, overlays them into a persistent-hotspot
frequency map, and computes the global + local (approximate GWR) relationship
between LST and NDVI/NDBI for each year.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from nagpur_uhi.analysis import hotspots, relationships
from nagpur_uhi.config import load_config
from nagpur_uhi.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    cfg = load_config()
    processed_dir = Path(cfg.paths.processed_dir)

    hotspot_categories_by_year = []
    years = [sy.year for sy in cfg.study_years]

    for year in years:
        data = np.load(processed_dir / f"features_{year}.npz")
        lst = data["lst"]

        z = hotspots.getis_ord_gi_star(lst, window_size=cfg.hotspot.window_size)
        categories = hotspots.classify_hotspots(z, confidence_levels=tuple(cfg.hotspot.confidence_levels))
        hotspot_categories_by_year.append(categories)
        np.savez_compressed(processed_dir / f"hotspots_{year}.npz", z_score=z, category=categories)
        logger.info(
            "%s: %.1f%% of valid pixels are hot at >=95%% confidence",
            year,
            100.0 * np.mean((categories >= hotspots.HOT_95) & (categories != hotspots.NODATA)),
        )

        corr_ndvi = relationships.global_correlation(lst, data["ndvi"])
        corr_ndbi = relationships.global_correlation(lst, data["ndbi"])
        logger.info(
            "%s: LST-NDVI r=%.3f (p=%.4g), LST-NDBI r=%.3f (p=%.4g)",
            year, corr_ndvi["pearson_r"], corr_ndvi["p_value"], corr_ndbi["pearson_r"], corr_ndbi["p_value"],
        )

    frequency, persistent_mask = hotspots.persistent_hotspots(
        hotspot_categories_by_year,
        min_confidence_level=hotspots.HOT_95,
        persistent_threshold=cfg.hotspot.persistent_threshold,
    )
    np.savez_compressed(
        processed_dir / "persistent_hotspots.npz", frequency=frequency, persistent_mask=persistent_mask
    )
    logger.info(
        "Persistent hotspot coverage: %.1f%% of valid pixels (threshold=%.0f%% of years)",
        100.0 * np.nanmean(persistent_mask.astype(float)),
        100.0 * cfg.hotspot.persistent_threshold,
    )
    logger.info("Stage 3 complete.")


if __name__ == "__main__":
    main()
