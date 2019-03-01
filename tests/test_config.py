import unittest
from typing import List

import numpy as np
import pandas as pd
from keras.engine import Layer
from keras.layers import Activation, Dense, Concatenate
from sklearn.preprocessing import LabelEncoder

from entity_embeddings.config import Config, get_embedding_size
from entity_embeddings.network import ModelAssembler
from entity_embeddings.processor.processor import TargetProcessor
from entity_embeddings.processor.target_type import TargetType
from entity_embeddings.util.dataframe_utils import create_random_csv, remove_random_csv


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        random_csv = create_random_csv()
        target = 'D'

        config = Config.make_default_config(csv_path=random_csv,
                                            target_name=target,
                                            target_type=TargetType.BINARY_CLASSIFICATION,
                                            train_ratio=0.9)
        self.assertIsNotNone(config)
        remove_random_csv()

    def test_custom_config(self):
        random_csv = create_random_csv()
        target = 'D'

        processor = CustomProcessor()
        assembler = CustomAssembler()

        config = Config.make_custom_config(csv_path=random_csv,
                                           target_name=target,
                                           train_ratio=0.9,
                                           target_processor=processor,
                                           model_assembler=assembler)

        self.assertIsNotNone(config)
        remove_random_csv()

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


class CustomProcessor(TargetProcessor):
    def process_target(self, y: List) -> np.ndarray:
        # just for example purposes, let's use a LabelEncoder
        return LabelEncoder().fit_transform(y)


class CustomAssembler(ModelAssembler):
    def make_final_layer(self, previous_layer: Layer):
        output_model = Dense(1)(previous_layer)
        output_model = Activation('sigmoid')(output_model)
        return output_model

    def compile_model(self, model):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    """
    You can aso customize the hidden layers of the network
    """

    def make_hidden_layers(self, outputs: List[Layer]):
        output_model = Concatenate()(outputs)
        output_model = Dense(5000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        return output_model
