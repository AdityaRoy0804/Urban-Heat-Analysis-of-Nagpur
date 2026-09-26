import unittest

import numpy as np

from nagpur_uhi.ml.dataset import build_training_table, spatial_block_split


class TestBuildTrainingTable(unittest.TestCase):
    def test_flattens_and_drops_nodata_rows(self):
        ndvi = np.array([[0.1, 0.2], [np.nan, 0.4]])
        lst = np.array([[30.0, 31.0], [32.0, np.nan]])
        df = build_training_table({"ndvi": ndvi}, lst)
        # Row (0,0) and (0,1) are valid; (1,0) has nan target->dropped via ndvi? actually
        # (1,0) ndvi=nan -> dropped; (1,1) target nan -> dropped. Only 2 valid rows remain.
        self.assertEqual(len(df), 2)
        self.assertIn("row", df.columns)
        self.assertIn("col", df.columns)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_training_table({"ndvi": np.zeros((2, 2))}, np.zeros((3, 3)))


class TestSpatialBlockSplit(unittest.TestCase):
    def test_split_produces_nonoverlapping_blocks(self):
        rng = np.random.default_rng(0)
        size = 40
        ndvi = rng.uniform(0, 1, size=(size, size))
        lst = 30 - 5 * ndvi
        df = build_training_table({"ndvi": ndvi}, lst)

        train_df, test_df = spatial_block_split(df, block_size_px=5, test_size=0.25, random_state=1)
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)

        train_blocks = set(zip(train_df["row"] // 5, train_df["col"] // 5))
        test_blocks = set(zip(test_df["row"] // 5, test_df["col"] // 5))
        self.assertEqual(len(train_blocks & test_blocks), 0, "train/test blocks must not overlap")

    def test_missing_row_col_columns_raises(self):
        import pandas as pd

        df = pd.DataFrame({"ndvi": [0.1, 0.2]})
        with self.assertRaises(ValueError):
            spatial_block_split(df)


if __name__ == "__main__":
    unittest.main()
