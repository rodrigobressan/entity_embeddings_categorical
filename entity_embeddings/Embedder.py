import pickle
from typing import Tuple

import numpy as np

from entity_embeddings.EmbeddingConfig import EmbeddingConfig
from entity_embeddings.EmbeddingNetwork import EmbeddingNetwork
from entity_embeddings.processor.TargetType import TargetType
from entity_embeddings.util import DataframeUtils
from entity_embeddings.util.PreprocessingUtils import sample, get_X_y, label_encode


def main():
    data_path = "../mock_categorical.csv"
    config = EmbeddingConfig(csv_path=data_path,
                             target_name='vivo',
                             target_type=TargetType.BINARY_CLASSIFICATION,
                             train_ratio=0.9,
                             epochs=10,
                             verbose=True,
                             weights_output='weights.pickle')

    embedder = Embedder(config)
    embedder.perform_embedding()


class Embedder():
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.X_train, self.X_val, self.y_train, self.y_val = prepare_data(self.config)

    def perform_embedding(self) -> None:
        network = EmbeddingNetwork(self.config)
        network.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        save_weights(network, self.config)


def prepare_data(config: EmbeddingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = get_X_y(config.df, config.target_name)

    # pre processing of X and Y
    X = label_encode(X)
    y = np.array(y)

    num_records = len(X)
    train_size = int(config.train_ratio * num_records)

    X_train = X[:train_size]
    X_val = X[train_size:]
    y_train = y[:train_size]
    y_val = y[train_size:]

    X_train, y_train = sample(X_train, y_train, 1000)  # Simulate data sparsity

    y_train = config.processor.process_target(y_train.tolist())
    y_val = config.processor.process_target(y_val.tolist())

    return X_train, X_val, y_train, y_val


def save_weights(network: EmbeddingNetwork, config: EmbeddingConfig) -> None:
    weights_embeddings = []
    for column in DataframeUtils.get_all_columns_except(config.df, config.target_name):
        weights = network.get_weights_from_layer(column)
        weights_embeddings.append(weights)

    with open(config.weights_output, 'wb') as f:
        pickle.dump(weights_embeddings, f, -1)


if __name__ == '__main__':
    main()
