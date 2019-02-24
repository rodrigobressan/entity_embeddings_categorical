"""
This file contains the implementation of the class EmbeddingNetwork, which will be responsible for the initialization
and setup of our entity embedding network
"""
from typing import List, Tuple

import numpy as np
from keras.engine import Layer
from keras.layers import Concatenate
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Model as KerasModel

from entity_embeddings.EmbeddingConfig import EmbeddingConfig
from entity_embeddings.util.PreprocessingUtils import transpose_to_list

np.random.seed(42)


class EmbeddingNetwork:
    """
    This class is used to provide a Entity Embedding Network from a given EmbeddingConfig object
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        self.model = self.__make_model()

    def __make_model(self) -> KerasModel:
        """
        This method is used to generate our KerasModel containing the Embedding layers alongside with the output layers
        :return: a compiled KerasModel object
        """
        inputs, outputs = self.__make_embedding_layers()

        output_model = Concatenate()(outputs)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(2)(output_model)
        output_model = Activation('sigmoid')(output_model)

        model = KerasModel(inputs=inputs, outputs=output_model)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def __make_embedding_layers(self) -> Tuple[List[Layer], List[Layer]]:
        """
        This method is used to generate the list of inputs and output layers where our Embedding layers will be placed
        :return: a tuple containing two lists: the first, the input layers; the second, the output layers
        """
        embedding_inputs = []
        embedding_outputs = []

        for category in self.config.categories:
            input_category = Input(shape=(1,))
            output_category = Embedding(input_dim=category.unique_values,
                                        output_dim=category.embedding_size,
                                        name=category.alias)(input_category)
            output_category = Reshape(target_shape=(category.embedding_size,))(output_category)

            embedding_inputs.append(input_category)
            embedding_outputs.append(output_category)

        return embedding_inputs, embedding_outputs

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        This method is used to fit a given training and validation data into our entity embeddings model
        :param X_train: training features
        :param y_train: training targets
        :param X_val: validation features
        :param y_val: validation targets
        """
        self.model.fit(transpose_to_list(X_train), y_train,
                       validation_data=(transpose_to_list(X_val), y_val),
                       epochs=self.config.epochs, batch_size=self.config.batch_size,
                       # callbacks=[self.checkpointer],
                       )

    def get_weights_from_layer(self, layer_name):
        return self.model.get_layer(layer_name).get_weights()[0]
