import unittest

import numpy as np

from entity_embeddings.PreprocessingUtils import transpose_to_list


class TestPreprocessingUtils(unittest.TestCase):
    def test_transpose_to_list_without_last_col(self):
        X_array = np.random.randint(0, 10, size=(10, 4))

        feature_list = transpose_to_list(X_array)

        self.assertEqual(len(feature_list), X_array.shape[1] - 1)

        for index in range(X_array.shape[1]):
            for item_index, item_value in enumerate(X_array[index:index + 1, :-1].tolist()[0]):
                self.assertEqual(item_value, feature_list[item_index][index])

    def test_transpose_to_list_with_last_col(self):
        X_array = np.random.randint(0, 10, size=(10, 4))

        feature_list = transpose_to_list(X_array, keep_last_column=True)

        self.assertEqual(len(feature_list), X_array.shape[1])

        for index in range(X_array.shape[1]):
            for item_index, item_value in enumerate(X_array[index:index + 1, :-1].tolist()[0]):
                self.assertEqual(item_value, feature_list[item_index][index])
