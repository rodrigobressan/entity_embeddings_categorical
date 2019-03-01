import pickle

from keras import Model
from keras.engine import Layer
from keras.layers import Dense, Activation

from entity_embeddings import config
from entity_embeddings.processor.target_type import TargetType
from entity_embeddings.util import dataframe_utils


def save_weights(model: Model, config: config) -> None:
    weights_embeddings = []
    for column in dataframe_utils.get_all_columns_except(config.df, config.target_name):
        weights = get_weights_from_layer(model, column)
        weights_embeddings.append(weights)

    with open(config.weights_output, 'wb') as f:
        pickle.dump(weights_embeddings, f, -1)


def get_weights_from_layer(model, layer_name):
    return model.get_layer(layer_name).get_weights()[0]
