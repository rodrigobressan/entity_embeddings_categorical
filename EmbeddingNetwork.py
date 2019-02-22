from typing import List

import numpy as np
from keras import Model
from keras.layers import Concatenate
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Model as KerasModel

from EmbeddingConfig import EmbeddingConfig, Category

np.random.seed(42)


class EmbeddingNetwork(Model):
    def __init__(self, config: EmbeddingConfig, epochs=10):
        super().__init__()
        self.config = config
        self.epochs = epochs
        self.__make_model()

    @staticmethod
    def split_features_from_categories(X: np.ndarray, categories: List[Category]) -> List:
        X_list = []
        for index, value in enumerate(categories):
            X_list.append(X[..., [index]])

        return X_list

    def __make_model(self):
        categories = self.config.categories

        embedding_inputs = []
        embedding_outputs = []
        for category in categories:
            input_category = Input(shape=(1,))
            output_category = Embedding(category.unique_values, category.embedding_size, name=category.alias)(
                input_category)
            output_category = Reshape(target_shape=(category.embedding_size,))(output_category)

            embedding_inputs.append(input_category)
            embedding_outputs.append(output_category)

        output_model = Concatenate()(embedding_outputs)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(2)(output_model)
        output_model = Activation('sigmoid')(output_model)

        self.model = KerasModel(inputs=embedding_inputs, outputs=output_model)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(self.split_features_from_categories(X_train, self.config.categories), y_train,
                       validation_data=(self.split_features_from_categories(X_val, self.config.categories), y_val),
                       epochs=self.epochs, batch_size=128,
                       # callbacks=[self.checkpointer],
                       )
