import unittest

import pandas as pd

from entity_embeddings.util import DataframeUtils
from entity_embeddings.util.ValidationUtils import check_csv_data, check_not_empty_dataframe, check_target_name, \
    check_target_existent_in_df, check_train_ratio, check_epochs, check_batch_size, check_weights_output


class TestValidationUtils(unittest.TestCase):
    def test_check_csv_data_not_existent(self) -> None:
        csv_path = 'not_existent_path.csv'
        self.assertRaises(ValueError, check_csv_data, csv_path)

    def test_check_csv_data_empty(self) -> None:
        csv_path = ''
        self.assertRaises(ValueError, check_csv_data, csv_path)

    def test_check_not_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        self.assertRaises(ValueError, check_not_empty_dataframe, df)

    def test_check_target_name(self) -> None:
        target_name = ''
        self.assertRaises(ValueError, check_target_name, target_name)

    def test_check_target_existent_in_df(self) -> None:
        df = DataframeUtils.create_random_dataframe(columns='ABCD')
        self.assertRaises(ValueError, check_target_existent_in_df, 'E', df)

    def test_check_train_ratio_greater_than_one(self) -> None:
        ratio = 1.1
        self.assertRaises(ValueError, check_train_ratio, ratio)

    def test_check_train_ratio_equal_one(self) -> None:
        ratio = 1
        self.assertRaises(ValueError, check_train_ratio, ratio)

    def test_check_train_ratio_smaller_than_zero(self) -> None:
        ratio = -0.1
        self.assertRaises(ValueError, check_train_ratio, ratio)

    def test_check_train_ratio_equals_zero(self) -> None:
        ratio = 0.0
        self.assertRaises(ValueError, check_train_ratio, ratio)

    def test_check_epochs(self) -> None:
        epochs = 0
        self.assertRaises(ValueError, check_epochs, epochs)

    def test_check_batch_size(self) -> None:
        batch_size = 0
        self.assertRaises(ValueError, check_batch_size, batch_size)

    def test_check_weights_output(self) -> None:
        weights_output = ''
        self.assertRaises(ValueError, check_weights_output, weights_output)
