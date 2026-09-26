#!/usr/bin/env python3
"""
Stage 4 - Dataset assembly + explainable ML model training.

Aggregates pixel-level features into regular grid zones, builds one training
row per zone per study year, trains the LST surrogate model with a spatial
block train/test split, evaluates it, and persists:

- fitted model
- background sample for SHAP/occlusion explanations
- zones_dataset.csv for API/history/scenario use

NOTE
----
The current zoning scheme uses a regular grid of
ZONE_GRID_SIZE_PX x ZONE_GRID_SIZE_PX pixels as a working stand-in for
real administrative/locality boundaries.

For production, this can later be replaced with proper ward/locality
polygons using rasterstats/geopandas.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from nagpur_uhi.config import load_config
from nagpur_uhi.features.lulc import CLASS_LABELS
from nagpur_uhi.ml.dataset import spatial_block_split
from nagpur_uhi.ml.model import (
    evaluate_directional_validity,
    evaluate_holdout,
    train_lst_model,
)
from nagpur_uhi.utils.logging_config import configure_logging


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# 10 x 10 pixels at 30 m resolution ~= 300 m x 300 m zones.
ZONE_GRID_SIZE_PX = 10


# ---------------------------------------------------------------------
# Array aggregation helpers
# ---------------------------------------------------------------------

def _block_reduce_mean(
    array: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    Mean-pool a 2D array into block_size x block_size cells.

    NaN values are ignored.

    Any incomplete pixels at the bottom/right edge are trimmed so that
    the array can be reshaped safely.
    """
    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            f"_block_reduce_mean expects a 2D array, got shape={array.shape}"
        )

    rows, cols = array.shape

    usable_rows = rows - (rows % block_size)
    usable_cols = cols - (cols % block_size)

    if usable_rows == 0 or usable_cols == 0:
        raise ValueError(
            f"Array shape {array.shape} is too small for "
            f"block size {block_size}"
        )

    trimmed = array[:usable_rows, :usable_cols]

    reshaped = trimmed.reshape(
        usable_rows // block_size,
        block_size,
        usable_cols // block_size,
        block_size,
    )

    with np.errstate(invalid="ignore"):
        result = np.nanmean(reshaped, axis=(1, 3))

    return result


def _block_reduce_mode(
    array: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    Mode-pool an integer LULC raster into block_size x block_size cells.

    Negative values are treated as invalid/no-data.

    Returns:
        2D integer array containing the majority class for each zone.
        -1 means no valid LULC value existed in that zone.
    """
    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            f"_block_reduce_mode expects a 2D array, got shape={array.shape}"
        )

    rows, cols = array.shape

    usable_rows = rows - (rows % block_size)
    usable_cols = cols - (cols % block_size)

    if usable_rows == 0 or usable_cols == 0:
        raise ValueError(
            f"Array shape {array.shape} is too small for "
            f"block size {block_size}"
        )

    trimmed = array[:usable_rows, :usable_cols]

    reshaped = trimmed.reshape(
        usable_rows // block_size,
        block_size,
        usable_cols // block_size,
        block_size,
    )

    n_zone_rows = usable_rows // block_size
    n_zone_cols = usable_cols // block_size

    result = np.full(
        (n_zone_rows, n_zone_cols),
        -1,
        dtype=np.int16,
    )

    for r in range(n_zone_rows):
        for c in range(n_zone_cols):
            values = reshaped[r, :, c, :].ravel()

            # Ignore negative/no-data classes.
            values = values[values >= 0]

            if values.size > 0:
                result[r, c] = np.bincount(values.astype(np.int64)).argmax()

    return result


# ---------------------------------------------------------------------
# Shape alignment
# ---------------------------------------------------------------------

def _trim_to_common_shape(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """
    Trim all 2D arrays to the smallest common shape.

    This is intentionally conservative: it does not resample or distort
    any data. It only removes incomplete edge zones caused by slightly
    different raster dimensions.

    Returns:
        aligned_arrays, (common_rows, common_cols)
    """

    if not arrays:
        raise ValueError("No arrays supplied for alignment.")

    shapes = {}

    for name, array in arrays.items():
        array = np.asarray(array)

        if array.ndim != 2:
            raise ValueError(
                f"Array '{name}' must be 2D, got shape={array.shape}"
            )

        shapes[name] = array.shape

    common_rows = min(shape[0] for shape in shapes.values())
    common_cols = min(shape[1] for shape in shapes.values())

    if common_rows == 0 or common_cols == 0:
        raise ValueError(
            f"Invalid common shape: {(common_rows, common_cols)}"
        )

    if len(set(shapes.values())) > 1:
        logger.warning(
            "Zone-grid dimensions differ. Trimming all layers to "
            "common shape %dx%d. Original shapes: %s",
            common_rows,
            common_cols,
            shapes,
        )

    aligned = {
        name: array[:common_rows, :common_cols]
        for name, array in arrays.items()
    }

    return aligned, (common_rows, common_cols)


# ---------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------

def build_zone_table(
    cfg,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Build one training row per spatial zone per study year.
    """

    # -------------------------------------------------------------
    # Terrain context
    # -------------------------------------------------------------

    terrain_path = processed_dir / "terrain_context.npz"

    if not terrain_path.exists():
        raise FileNotFoundError(
            f"Missing terrain context file: {terrain_path}"
        )

    terrain = np.load(terrain_path)

    required_terrain = [
        "elevation_m",
        "slope_deg",
    ]

    for key in required_terrain:
        if key not in terrain.files:
            raise KeyError(
                f"'{key}' not found in {terrain_path}. "
                f"Available: {terrain.files}"
            )

    elevation_zone = _block_reduce_mean(
        terrain["elevation_m"],
        ZONE_GRID_SIZE_PX,
    )

    slope_zone = _block_reduce_mean(
        terrain["slope_deg"],
        ZONE_GRID_SIZE_PX,
    )

    # -------------------------------------------------------------
    # Persistent hotspot context
    # -------------------------------------------------------------

    persistent_path = processed_dir / "persistent_hotspots.npz"

    if not persistent_path.exists():
        raise FileNotFoundError(
            f"Missing persistent hotspot file: {persistent_path}"
        )

    persistent = np.load(persistent_path)

    if "frequency" not in persistent.files:
        raise KeyError(
            f"'frequency' not found in {persistent_path}. "
            f"Available: {persistent.files}"
        )

    hotspot_freq_zone = _block_reduce_mean(
        persistent["frequency"],
        ZONE_GRID_SIZE_PX,
    )

    rows: list[dict] = []

    # -------------------------------------------------------------
    # Process every study year
    # -------------------------------------------------------------

    for study_year in cfg.study_years:

        year = study_year.year

        logger.info("Building zones for %s", year)

        features_path = (
            processed_dir / f"features_{year}.npz"
        )

        if not features_path.exists():
            raise FileNotFoundError(
                f"Missing feature file: {features_path}"
            )

        feats = np.load(features_path)

        required_features = [
            "lst",
            "ndvi",
            "ndbi",
            "ndwi",
            "sar_vv_vh_ratio",
            "lulc_class",
        ]

        missing = [
            key
            for key in required_features
            if key not in feats.files
        ]

        if missing:
            raise KeyError(
                f"{features_path} is missing required arrays: {missing}. "
                f"Available: {feats.files}"
            )

        # ---------------------------------------------------------
        # Aggregate continuous features
        # ---------------------------------------------------------

        lst_zone = _block_reduce_mean(
            feats["lst"],
            ZONE_GRID_SIZE_PX,
        )

        ndvi_zone = _block_reduce_mean(
            feats["ndvi"],
            ZONE_GRID_SIZE_PX,
        )

        ndbi_zone = _block_reduce_mean(
            feats["ndbi"],
            ZONE_GRID_SIZE_PX,
        )

        ndwi_zone = _block_reduce_mean(
            feats["ndwi"],
            ZONE_GRID_SIZE_PX,
        )

        sar_zone = _block_reduce_mean(
            feats["sar_vv_vh_ratio"],
            ZONE_GRID_SIZE_PX,
        )

        # ---------------------------------------------------------
        # Aggregate categorical LULC using majority class
        # ---------------------------------------------------------

        lulc_mode = _block_reduce_mode(
            feats["lulc_class"],
            ZONE_GRID_SIZE_PX,
        )

        # ---------------------------------------------------------
        # Align all zone-level arrays
        #
        # Different source rasters can have slightly different
        # dimensions because of export extent/resolution/alignment.
        #
        # We therefore trim all derived zone arrays to their
        # common dimensions before indexing them together.
        # ---------------------------------------------------------

        aligned, common_shape = _trim_to_common_shape(
            {
                "lst": lst_zone,
                "ndvi": ndvi_zone,
                "ndbi": ndbi_zone,
                "ndwi": ndwi_zone,
                "sar_vv_vh_ratio": sar_zone,
                "elevation_m": elevation_zone,
                "slope_deg": slope_zone,
                "hotspot_frequency": hotspot_freq_zone,
                "lulc_mode": lulc_mode,
            }
        )

        lst_zone = aligned["lst"]
        ndvi_zone = aligned["ndvi"]
        ndbi_zone = aligned["ndbi"]
        ndwi_zone = aligned["ndwi"]
        sar_zone = aligned["sar_vv_vh_ratio"]
        elevation_zone_year = aligned["elevation_m"]
        slope_zone_year = aligned["slope_deg"]
        hotspot_freq_zone_year = aligned["hotspot_frequency"]
        lulc_mode = aligned["lulc_mode"]

        n_rows, n_cols = common_shape

        logger.info(
            "%s: common zone grid = %d x %d",
            year,
            n_rows,
            n_cols,
        )

        # ---------------------------------------------------------
        # Create zone IDs
        # ---------------------------------------------------------

        zone_ids = np.array(
            [
                [f"z_{r}_{c}" for c in range(n_cols)]
                for r in range(n_rows)
            ]
        )

        # ---------------------------------------------------------
        # Convert zone arrays to records
        # ---------------------------------------------------------

        for r in range(n_rows):

            for c in range(n_cols):

                target = lst_zone[r, c]

                # LST is the prediction target.
                # A zone without valid LST cannot be used.
                if not np.isfinite(target):
                    continue

                lulc_code = int(lulc_mode[r, c])

                if (
                    lulc_code >= 0
                    and lulc_code < len(CLASS_LABELS)
                ):
                    lulc_label = CLASS_LABELS[lulc_code]
                else:
                    lulc_label = "unknown"

                hotspot_frequency = (
                    hotspot_freq_zone_year[r, c]
                )

                # -------------------------------------------------
                # Record
                # -------------------------------------------------

                rows.append(
                    {
                        "zone_id": zone_ids[r, c],
                        "row": r,
                        "col": c,
                        "year": year,

                        # Target
                        "target": float(target),

                        # Remote-sensing predictors
                        "ndvi": ndvi_zone[r, c],
                        "ndbi": ndbi_zone[r, c],
                        "ndwi": ndwi_zone[r, c],
                        "sar_vv_vh_ratio": sar_zone[r, c],

                        # Terrain
                        "elevation_m": elevation_zone_year[r, c],
                        "slope_deg": slope_zone_year[r, c],

                        # Hotspot context
                        "hotspot_frequency": hotspot_frequency,

                        "is_persistent_hotspot": bool(
                            np.isfinite(hotspot_frequency)
                            and hotspot_frequency
                            >= cfg.hotspot.persistent_threshold
                        ),

                        # LULC
                        "lulc_class": lulc_label,
                    }
                )

    # -------------------------------------------------------------
    # Construct DataFrame
    # -------------------------------------------------------------

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "Zone dataset is empty. No valid LST zones were generated."
        )

    logger.info(
        "Raw zone table: %d rows",
        len(df),
    )

    # -------------------------------------------------------------
    # Remove rows with missing model variables
    # -------------------------------------------------------------

    numeric_columns = [
        "target",
        "ndvi",
        "ndbi",
        "ndwi",
        "sar_vv_vh_ratio",
        "elevation_m",
        "slope_deg",
        "hotspot_frequency",
    ]

    before_dropna = len(df)

    df = (
        df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=numeric_columns)
        .reset_index(drop=True)
    )

    dropped = before_dropna - len(df)

    logger.info(
        "Removed %d rows with missing/invalid numeric features",
        dropped,
    )

    if df.empty:
        raise RuntimeError(
            "All zone rows were removed because of missing/invalid features."
        )

    return df


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:

    configure_logging()

    cfg = load_config()

    processed_dir = Path(
        cfg.paths.processed_dir
    )

    models_dir = Path(
        cfg.paths.models_dir
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Build zone-level dataset
    # -------------------------------------------------------------

    logger.info(
        "Building zone-level training table"
    )

    zones_df = build_zone_table(
        cfg,
        processed_dir,
    )

    zones_path = (
        processed_dir / "zones_dataset.csv"
    )

    zones_df.to_csv(
        zones_path,
        index=False,
    )

    logger.info(
        "Zone table: %d rows across %d zones and %d years",
        len(zones_df),
        zones_df["zone_id"].nunique(),
        zones_df["year"].nunique(),
    )

    # -------------------------------------------------------------
    # Validate required years
    # -------------------------------------------------------------

    years_available = sorted(
        zones_df["year"].unique()
    )

    if len(years_available) < 2:
        raise RuntimeError(
            "At least two study years are required for "
            "directional backtesting."
        )

    # -------------------------------------------------------------
    # Model features
    # -------------------------------------------------------------

    feature_columns = list(
        cfg.scenario.feature_columns
    )

    missing_model_features = [
        col
        for col in feature_columns
        if col not in zones_df.columns
    ]

    if missing_model_features:
        raise KeyError(
            "Configured model features are missing from "
            f"zones_dataset: {missing_model_features}"
        )

    # -------------------------------------------------------------
    # Spatial block split
    #
    # Each zone is approximately:
    #
    # 10 pixels × 30 m = 300 m
    #
    # Therefore a 500 m spatial block corresponds to approximately
    # 500 / 300 = 1.67 zones.
    #
    # Minimum 2 zones gives a meaningful spatial block.
    # -------------------------------------------------------------

    native_pixel_size_m = 30.0

    zone_size_m = (
        ZONE_GRID_SIZE_PX
        * native_pixel_size_m
    )

    block_size_in_zones = max(
        2,
        round(
            cfg.model.spatial_block_size_m
            / zone_size_m
        ),
    )

    logger.info(
        "Spatial block split: zone size=%.1fm, "
        "block size=%d zones (~%.1fm)",
        zone_size_m,
        block_size_in_zones,
        block_size_in_zones * zone_size_m,
    )

    train_df, test_df = spatial_block_split(
        zones_df,
        block_size_px=block_size_in_zones,
        test_size=cfg.model.test_size,
    )

    logger.info(
        "Train rows=%d, test rows=%d",
        len(train_df),
        len(test_df),
    )

    if train_df.empty:
        raise RuntimeError(
            "Spatial split produced an empty training set."
        )

    if test_df.empty:
        raise RuntimeError(
            "Spatial split produced an empty test set."
        )

    # -------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------

    logger.info(
        "Training %s LST model",
        cfg.model.type,
    )

    model = train_lst_model(
        train_df,
        feature_columns=feature_columns,
        model_type=cfg.model.type,
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        min_samples_leaf=cfg.model.min_samples_leaf,
        random_state=cfg.model.random_state,
    )

    # -------------------------------------------------------------
    # Holdout evaluation
    # -------------------------------------------------------------

    holdout_eval = evaluate_holdout(
        model,
        test_df,
        feature_columns,
    )

    logger.info(
        "Held-out R^2=%.3f, MAE=%.2f C (n=%d) - "
        "secondary metric, see directional validity below",
        holdout_eval.r_squared,
        holdout_eval.mae,
        holdout_eval.n_test_samples,
    )

    # -------------------------------------------------------------
    # Directional-validity backtest
    #
    # Compare first study year against last study year for zones
    # that contain complete data in both periods.
    # -------------------------------------------------------------

    pivoted = zones_df.pivot_table(
        index="zone_id",
        columns="year",
        values=feature_columns + ["target"],
    )

    years_sorted = sorted(
        zones_df["year"].unique()
    )

    first_year = years_sorted[0]
    last_year = years_sorted[-1]

    required_columns = (
        [(c, first_year) for c in feature_columns + ["target"]]
        +
        [(c, last_year) for c in feature_columns + ["target"]]
    )

    common_zones = pivoted.dropna(
        subset=required_columns
    ).index

    logger.info(
        "Complete zones available for directional "
        "backtest (%s -> %s): %d",
        first_year,
        last_year,
        len(common_zones),
    )

    if len(common_zones) >= 5:

        features_before = pd.DataFrame(
            {
                c: pivoted.loc[
                    common_zones,
                    (c, first_year),
                ]
                for c in feature_columns
            }
        )

        features_after = pd.DataFrame(
            {
                c: pivoted.loc[
                    common_zones,
                    (c, last_year),
                ]
                for c in feature_columns
            }
        )

        lst_before = pivoted.loc[
            common_zones,
            ("target", first_year),
        ].to_numpy()

        lst_after = pivoted.loc[
            common_zones,
            ("target", last_year),
        ].to_numpy()

        directional_eval = evaluate_directional_validity(
            model,
            features_before,
            features_after,
            lst_before,
            lst_after,
            feature_columns,
        )

        logger.info(
            "Directional validity (%s->%s): %.0f%% correct sign "
            "over %d zones with meaningful change",
            first_year,
            last_year,
            100 * (
                directional_eval.directional_accuracy
                or 0
            ),
            directional_eval.n_backtest_pairs,
        )

    else:

        logger.warning(
            "Fewer than 5 zones with complete data in both "
            "%s and %s - skipping directional backtest",
            first_year,
            last_year,
        )

    # -------------------------------------------------------------
    # Background sample for explanations
    # -------------------------------------------------------------

    background_size = min(
        200,
        len(train_df),
    )

    background_sample = (
        train_df[
            feature_columns
        ]
        .sample(
            background_size,
            random_state=cfg.model.random_state,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Save model artifact
    # -------------------------------------------------------------

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "background_data": background_sample,
    }

    model_path = (
        models_dir / "lst_model.pkl"
    )

    with open(
        model_path,
        "wb",
    ) as f:
        pickle.dump(
            artifact,
            f,
        )

    logger.info(
        "Saved model artifact: %s",
        model_path,
    )

    logger.info(
        "Stage 4 complete. Model artifact and "
        "zones_dataset.csv written to %s / %s",
        models_dir,
        processed_dir,
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()