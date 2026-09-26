"""
Turn aligned raster feature stacks into tabular ML datasets, and split them
in a spatially-aware way.

A plain random train/test split on pixel data from a spatially autocorrelated
raster leaks information (neighbouring train/test pixels are nearly
identical), inflating apparent accuracy. `spatial_block_split` instead splits
whole contiguous blocks of pixels between train and test, which is the
standard fix for this in spatial ML and matters here because we explicitly
want an honest "directional validity" assessment (see ml/model.py), not an
inflated accuracy number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_training_table(
    feature_stack: dict[str, np.ndarray],
    target: np.ndarray,
    row_col_index: bool = True,
) -> pd.DataFrame:
    """
    Flatten a dict of same-shape 2D raster arrays plus a target raster into a
    tidy DataFrame, one row per pixel, dropping pixels where any feature or
    the target is NaN.
    """
    shapes = {name: arr.shape for name, arr in feature_stack.items()}
    if len(set(shapes.values()) | {target.shape}) != 1:
        raise ValueError(f"All feature/target rasters must share one shape; got {shapes} vs target {target.shape}")

    rows, cols = target.shape
    data = {name: arr.ravel() for name, arr in feature_stack.items()}
    data["target"] = target.ravel()

    if row_col_index:
        row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        data["row"] = row_idx.ravel()
        data["col"] = col_idx.ravel()

    df = pd.DataFrame(data)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        df.attrs["n_dropped_nodata_rows"] = dropped
    return df


def spatial_block_split(
    df: pd.DataFrame,
    block_size_px: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split `df` (must contain 'row' and 'col' columns from build_training_table)
    into train/test by randomly assigning whole `block_size_px` x `block_size_px`
    spatial blocks to the test set, rather than splitting individual pixels.
    """
    if "row" not in df.columns or "col" not in df.columns:
        raise ValueError("df must contain 'row' and 'col' columns (build with row_col_index=True)")

    block_row = df["row"] // block_size_px
    block_col = df["col"] // block_size_px
    block_id = block_row.astype(str) + "_" + block_col.astype(str)

    unique_blocks = np.array(block_id.unique(), dtype=object)
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_blocks)
    n_test_blocks = max(1, int(len(unique_blocks) * test_size))
    test_blocks = set(unique_blocks[:n_test_blocks])

    is_test = block_id.isin(test_blocks)
    train_df = df[~is_test].reset_index(drop=True)
    test_df = df[is_test].reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Spatial block split produced an empty train or test set - "
            "reduce block_size_px or increase the AOI/sample size."
        )
    return train_df, test_df
