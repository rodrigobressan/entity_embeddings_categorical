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
from entity_embeddings.processor.TargetType import TargetType
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

        output_model = self.__make_hidden_layers(outputs)
        output_model = self.__make_final_layer(output_model)

        model = KerasModel(inputs=inputs, outputs=output_model)
        model = self.__compile_model(model)
        return model

    def __compile_model(self, model):
        if self.config.target_type == TargetType.BINARY_CLASSIFICATION:
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif self.config.target_type == TargetType.MULTICLASS_CLASSIFICATION:
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        elif self.config.target_type == TargetType.REGRESSION:
            model.compile(loss='mse', optimizer='adam')

        return model

    def __make_hidden_layers(self, outputs) -> Layer:
        output_model = Concatenate()(outputs)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        return output_model

    def __make_final_layer(self, previous_layer):
        if self.config.target_type == TargetType.BINARY_CLASSIFICATION:
            return self.__make_binary_layer(previous_layer)
        elif self.config.target_type == TargetType.REGRESSION:
            return self.__make_regression_layer(previous_layer)
        elif self.config.target_type == TargetType.MULTICLASS_CLASSIFICATION:
            return self.__make_multiclass_layer(previous_layer)

    def __make_binary_layer(self, previous_layer) -> Layer:
        output_model = Dense(1)(previous_layer)
        output_model = Activation('sigmoid')(output_model)
        return output_model

    def __make_regression_layer(self, previous_layer) -> Layer:
        output_model = Dense(1)(previous_layer)
        output_model = Activation('linear')(output_model)
        return output_model

    def __make_multiclass_layer(self, previous_layer) -> Layer:
        output_model = Dense(self.config.unique_classes)(previous_layer)
        output_model = Activation('softmax')(output_model)
        return output_model

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

    def make_binary_classification_layer(previous_layer: Layer) -> Layer:
        output_model = Dense(1)(previous_layer)
        output_model = Activation('sigmoid')(output_model)

        return output_model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        This method is used to fit a given training and validation data into our entity embeddings model
        :param X_train: training features
        :param y_train: training targets
        :param X_val: validation features
        :param y_val: validation targets
        """
        self.model.fit(x=transpose_to_list(X_train),
                       y=y_train,
                       validation_data=(transpose_to_list(X_val), y_val),
                       epochs=self.config.epochs,
                       batch_size=self.config.batch_size)
