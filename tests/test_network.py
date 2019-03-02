import unittest

from keras.layers import Embedding

from entity_embeddings.config import Config, get_embedding_size
from entity_embeddings.network.network import EmbeddingNetwork
from entity_embeddings.processor.target_type import TargetType
from entity_embeddings.util.dataframe_utils import create_random_csv, remove_random_csv


class TestNetwork(unittest.TestCase):
    def test_model_embedding_size(self):

        random_csv = create_random_csv()
        target = 'D'

        config = Config.make_default_config(csv_path=random_csv,
                                            target_name=target,
                                            target_type=TargetType.BINARY_CLASSIFICATION,
                                            train_ratio=0.9)

        network = EmbeddingNetwork(config)

        for layer in network.model.layers:
            if isinstance(layer, Embedding):
                embedding_size = int(layer.embeddings.initial_value.shape[1])
                self.assertEqual(get_embedding_size(config.df[layer.name].nunique()), embedding_size)

        remove_random_csv()

    def test_output_for_regression(self):
        pass

    def test_output_for_binary_classification(self):
        pass
