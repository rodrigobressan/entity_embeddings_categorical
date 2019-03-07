import unittest

import numpy as np

from entity_embeddings.util import dataframe_utils
from entity_embeddings.util.preprocessing_utils import transpose_to_list, series_to_list, get_X_y, label_encode


class TestPreprocessingUtils(unittest.TestCase):
    def test_transpose_to_list(self):
        X_array = np.random.randint(0, 10, size=(10, 4))

        feature_list = transpose_to_list(X_array)

        self.assertEqual(len(feature_list), X_array.shape[1])
        self.__check_items(X_array, feature_list)

    def test_series_to_list(self):
        df = dataframe_utils.create_random_dataframe(rows=5, cols=5, columns='ABCDE')

        for row in df.iterrows():
            output_list = series_to_list(row)
            self.assertListEqual(row[1].values.tolist(), output_list[1].values.tolist())

    def test_label_encode(self):
        data = [[1], [0], [1], [0], [1]]
        data_encoded, labels = label_encode(data)

        for index in range(data_encoded.shape[0]):
            self.assertEqual(data_encoded[index][0], data[index][0])

        self.assertEqual(labels[0].classes_[0], 0)
        self.assertEqual(labels[0].classes_[1], 1)

    def test_get_X_y(self):
        df = dataframe_utils.create_random_dataframe(rows=5, cols=5, columns='ABCDE')
        target = 'E'

        X_list, y_list = get_X_y(df, target)

        # check y values
        for index, y_final in enumerate(y_list):
            self.assertEqual(df.loc[:, df.columns == target].values.tolist()[index][0], y_final)

        for index_list, X_list in enumerate(X_list):
            for index_item, item_x in enumerate(X_list):
                current_df = df.loc[:, df.columns != target].values.tolist()[index_list][index_item]

                self.assertEqual(current_df, item_x)

    def __check_items(self, X_array, feature_list):
        for index in range(X_array.shape[1]):
            for item_index, item_value in enumerate(X_array[index:index + 1, :-1].tolist()[0]):
                self.assertEqual(item_value, feature_list[item_index][index])
