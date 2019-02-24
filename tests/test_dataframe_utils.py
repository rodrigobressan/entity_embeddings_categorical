import os
import unittest

import pandas as pd

from entity_embeddings.util.DataframeUtils import create_random_dataframe, create_random_csv, remove_random_csv, \
    get_all_columns_except
from entity_embeddings.util.ValidationUtils import check_not_empty_dataframe


class TestDataframeUtils(unittest.TestCase):
    def test_create_random_dataframe(self):
        rows = 5
        cols = 5
        columns = 'ABCDE'

        df = create_random_dataframe(rows, cols, columns)

        self.__check_dataframe_data(df, rows, cols, columns)

    def test_create_and_remove_random_csv(self):
        rows = 5
        cols = 5
        columns = 'ABCDE'

        csv_path = create_random_csv(rows, cols, columns)

        exists_csv = os.path.exists(csv_path)
        self.assertEqual(exists_csv, True)

        df = pd.read_csv(csv_path)
        self.__check_dataframe_data(df, rows, cols, columns)

        remove_random_csv()

        exists_csv = os.path.exists(csv_path)
        self.assertEqual(exists_csv, False)

    def test_load_guarantee_not_empty(self):
        df = pd.DataFrame()
        self.assertRaises(ValueError, check_not_empty_dataframe, df)

    def test_get_all_columns_except(self):
        columns = 'ABCD'
        to_remove = 'D'
        df = create_random_dataframe(rows=4, cols=4, columns=columns)

        df = get_all_columns_except(df, to_remove)
        self.assertEqual(''.join(list(df)), columns.replace(to_remove, ''))

    def __check_dataframe_data(self, df: pd.DataFrame, rows: int, cols: int, columns: str) -> None:
        self.assertEqual(df.shape[0], rows)
        self.assertEqual(df.shape[1], cols)
        self.assertEqual(''.join(list(df)), columns)
