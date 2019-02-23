import unittest
from typing import List

import numpy as np
import pandas as pd
from keras.layers import Embedding

from entity_embeddings.EmbeddingConfig import EmbeddingConfig, get_embedding_size, Category
from entity_embeddings.EmbeddingNetwork import EmbeddingNetwork
from entity_embeddings.EmbeddingNetwork import transpose_to_list


class EmbeddingNetworkTest(unittest.TestCase):
    def test_model_embedding_size(self):
        df = pd.DataFrame(np.random.randint(0, 10, size=(10, 4)), columns=list('ABCD'))
        target = 'D'

        config = EmbeddingConfig(df, target)

        network = EmbeddingNetwork(config, epochs=1)

        for layer in network.model.layers:
            if type(layer) is Embedding:
                embedding_size = int(layer.embeddings.initial_value.shape[1])
                self.assertEqual(get_embedding_size(df[layer.name].nunique()), embedding_size)


    def test_model_regression(self):
        pass

    def test_model_binary_classification(self):
        pass

    def test_model_others(self):
        pass
