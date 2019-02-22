import unittest

import numpy as np
import pandas as pd

from EmbeddingConfig import EmbeddingConfig, get_embedding_size


class TestEmbeddingConfig(unittest.TestCase):
    def test_init_with_empty_dataframe(self):
        df_empty = pd.DataFrame()
        target = 'cat_a'

        self.assertRaises(ValueError, EmbeddingConfig, df_empty, target)

    def test_init_with_empty_column(self):
        df = self.make_default_dataframe()
        column_empty = ''

        self.assertRaises(ValueError, EmbeddingConfig, df, column_empty)

    def test_init_with_not_existent_category(self):
        df = self.make_default_dataframe()
        not_existent_column = 'D'

        self.assertRaises(ValueError, EmbeddingConfig, df, not_existent_column)

    def test_valid_init(self):
        df = self.make_default_dataframe()
        target = 'C'

        config = EmbeddingConfig(df, target)
        self.assertEqual(len(config.categories), 2)

    def test_embedding_size(self):
        embedding_1 = get_embedding_size(10)
        self.assertEqual(embedding_1, 5)

        embedding_2 = get_embedding_size(15)
        self.assertEqual(embedding_2, 8)

        embedding_3 = get_embedding_size(20)
        self.assertEqual(embedding_3, 10)

    @staticmethod
    def make_default_dataframe():
        return pd.DataFrame(np.random.randint(0, 10, size=(10, 3)), columns=list('ABC'))
