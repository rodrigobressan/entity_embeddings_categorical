import unittest
from unittest import mock
from unittest.mock import Mock

import numpy as np

from entity_embeddings.config import Config
from entity_embeddings.embedder import Embedder
from entity_embeddings.processor.target_type import TargetType
from entity_embeddings.util.dataframe_utils import create_random_csv, remove_random_csv
from entity_embeddings.util.model_utils import get_weights_from_layer


class TestEmbedder(unittest.TestCase):
    def setUp(self):
        config = self.make_default_config()
        self.embedder = Embedder(config)

        self.embedder.prepare_data = Mock()

        self.embedder.prepare_data.return_value = [np.random.randint(0, 1, size=(10, 3)),
                                                   np.random.randint(0, 1, size=(10, 3)),
                                                   np.random.randint(0, 1, size=(10, 1)),
                                                   np.random.randint(0, 1, size=(10, 1))]

    def tearDown(self):
        remove_random_csv()

    def test_call_prepare_data_on_perform_embedding(self):
        self.embedder.perform_embedding()
        self.assertTrue(self.embedder.prepare_data.called)

    def make_default_config(self) -> Config:
        random_csv = create_random_csv()
        target = 'D'

        config = Config.make_default_config(csv_path=random_csv,
                                            target_name=target,
                                            target_type=TargetType.BINARY_CLASSIFICATION,
                                            train_ratio=0.9)

        return config
