"""
This file contains the implementation of the class EmbeddingNetwork, which will be responsible for the initialization
and setup of our entity embedding network
"""
from typing import Tuple, List

import numpy as np
from keras import Input
from keras.callbacks import History
from keras.engine import Layer
from keras.layers import Embedding, Reshape
from keras.models import Model as KerasModel

from entity_embeddings.config import Config
from entity_embeddings.util.preprocessing_utils import transpose_to_list

np.random.seed(42)


class EmbeddingNetwork:
    """
    This class is used to provide a Entity Embedding Network from a given EmbeddingConfig object
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = self.__make_model()

    def __make_model(self) -> KerasModel:
        """
        This method is used to generate our KerasModel containing the Embedding layers alongside with the output layers
        :return: a compiled KerasModel object
        """

        inputs, outputs = self._make_embedding_layers()
        output_model = self.config.model_assembler.make_hidden_layers(outputs)
        output_model = self.config.model_assembler.make_final_layer(output_model)

        model = KerasModel(inputs=inputs, outputs=output_model)
        model = self.config.model_assembler.compile_model(model)
        return model

    def _make_embedding_layers(self) -> Tuple[List[Layer], List[Layer]]:
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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> History:
        """
        This method is used to fit a given training and validation data into our entity embeddings model
        :param X_train: training features
        :param y_train: training targets
        :param X_val: validation features
        :param y_val: validation targets
        :return a History object
        """

        self.max_log_y = max(np.max(np.log(y_train)), np.max(np.log(y_val)))

        history = self.model.fit(x=transpose_to_list(X_train),
                                 y=self._val_for_fit(y_train),
                                 validation_data=(transpose_to_list(X_val), self._val_for_fit(y_val)),
                                 epochs=self.config.epochs,
                                 batch_size=self.config.batch_size, )
        return history

    def _val_for_fit(self, val):
        val = np.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return np.exp(val * self.max_log_y)