from typing import List

import numpy as np
import pandas as pd
from keras.engine import Layer
from keras.layers import Dense, Activation, Concatenate
from sklearn.preprocessing import MinMaxScaler

from entity_embeddings import Config, Embedder
from entity_embeddings.network import ModelAssembler
from entity_embeddings.processor import TargetProcessor
from entity_embeddings.util import visualization_utils


class CustomProcessor(TargetProcessor):
    def process_target(self, y: List) -> np.ndarray:
        # just for example purposes, let's use a MinMaxScaler
        return MinMaxScaler().fit_transform(pd.DataFrame(y))


class CustomAssembler(ModelAssembler):
    def make_final_layer(self, previous_layer: Layer):
        output_model = Dense(1)(previous_layer)
        output_model = Activation('sigmoid')(output_model)
        return output_model

    def compile_model(self, model):
        model.compile(loss='mean_absolute_error', optimizer='adam')
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

    data_path = "../ross_short.csv"
    config = Config.make_custom_config(csv_path=data_path,
                                       target_name='Sales',
                                       train_ratio=0.9,
                                       target_processor=custom_processor,
                                       model_assembler=custom_assembler,
                                       epochs=1,
                                       verbose=True,
                                       artifacts_path='artifacts')

    embedder = Embedder(config)
    embedder.perform_embedding()

    visualization_utils.make_visualizations_from_config(config)


if __name__ == '__main__':
    main()
