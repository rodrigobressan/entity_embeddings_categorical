import pickle
from typing import List

from keras import Model

from entity_embeddings import Config
from entity_embeddings.util import dataframe_utils


def get_weights(model: Model, config: Config) -> List:
    weights_embeddings = []
    for column in dataframe_utils.get_all_columns_except(config.df, config.target_name):
        weights = get_weights_from_layer(model, column)
        weights_embeddings.append(weights)

    return weights_embeddings


def save_weights(weights: List, config: Config) -> None:
    with open(config.get_weights_path(), 'wb') as f:
        pickle.dump(weights, f, -1)


def save_labels(labels: List, config: Config) -> None:
    with open(config.get_labels_path(), 'wb') as f:
        pickle.dump(labels, f, -1)


def get_weights_from_layer(model, layer_name):
    return model.get_layer(layer_name).get_weights()[0]
