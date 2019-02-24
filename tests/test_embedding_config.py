import unittest

import numpy as np
import pandas as pd

from entity_embeddings.EmbeddingConfig import EmbeddingConfig, get_embedding_size


class TestEmbeddingConfig(unittest.TestCase):
    # def test_valid_init(self):
    #     df = self.make_default_dataframe()
    #     target = 'C'
    #
    #     config = EmbeddingNetworkConfig(df, target)
    #     self.assertEqual(len(config.categories), 2)

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
