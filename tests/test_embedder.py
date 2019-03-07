import shutil
import unittest
from unittest.mock import Mock

import numpy as np
import os

from entity_embeddings.config import Config
from entity_embeddings.embedder import Embedder
from entity_embeddings.processor.target_type import TargetType
from entity_embeddings.util.dataframe_utils import create_random_csv, remove_random_csv


class TestEmbedder(unittest.TestCase):
    def setUp(self):
        self.config = self.make_default_config()
        self.embedder = Embedder(self.config)

        self.embedder.prepare_data = Mock()

        self.embedder.prepare_data.return_value = [np.random.randint(0, 1, size=(10, 3)),
                                                   np.random.randint(0, 1, size=(10, 3)),
                                                   np.random.randint(0, 1, size=(10, 1)),
                                                   np.random.randint(0, 1, size=(10, 1)),
                                                   []]

    def tearDown(self):
        remove_random_csv()


    def test_make_artifact_dir_when_not_existent(self):
        self.config.artifacts_path = 'not_existent_path'
        self.embedder.perform_embedding()

        dir_created = os.path.exists(self.config.artifacts_path)
        self.assertTrue(dir_created, True)

        # clean up
        shutil.rmtree(self.config.artifacts_path)

    def make_default_config(self) -> Config:
        random_csv = create_random_csv()
        target = 'D'

        config = Config.make_default_config(csv_path=random_csv,
                                            target_name=target,
                                            target_type=TargetType.BINARY_CLASSIFICATION,
                                            train_ratio=0.9)

        return config
