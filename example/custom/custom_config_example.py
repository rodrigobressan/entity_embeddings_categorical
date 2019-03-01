from typing import List

import numpy as np
from keras.engine import Layer
from keras.layers import Dense, Activation, Concatenate
from sklearn.preprocessing import LabelEncoder

from entity_embeddings.config import Config
from entity_embeddings.embedder import Embedder
from entity_embeddings.network import ModelAssembler
from entity_embeddings.processor.processor import TargetProcessor


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


def main():
    custom_processor = CustomProcessor()
    custom_assembler = CustomAssembler()

    data_path = "../mock_categorical.csv"
    config = Config.make_custom_config(csv_path=data_path,
                                       target_name='vivo',
                                       train_ratio=0.9,
                                       target_processor=custom_processor,
                                       model_assembler=custom_assembler,
                                       epochs=10,
                                       verbose=True,
                                       weights_output='weights.pickle')

    embedder = Embedder(config)
    embedder.perform_embedding()


if __name__ == '__main__':
    main()
