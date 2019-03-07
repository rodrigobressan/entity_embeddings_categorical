import os
from typing import Tuple, List

import numpy as np

from entity_embeddings.config import Config
from entity_embeddings.network.network import EmbeddingNetwork
from entity_embeddings.util import model_utils, preprocessing_utils


class Embedder():
    def __init__(self, config: Config):
        self.config = config
        self.network = EmbeddingNetwork(config)

    def prepare_data(self, config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
        X, y = preprocessing_utils.get_X_y(config.df, config.target_name)

        # pre processing of X and Y
        X, labels = preprocessing_utils.label_encode(X)
        y = np.array(y)

        num_records = len(X)
        train_size = int(config.train_ratio * num_records)

        X_train = X[:train_size]
        X_val = X[train_size:]
        y_train = y[:train_size]
        y_val = y[train_size:]

        X_train, y_train = preprocessing_utils.sample(X_train, y_train, 1000)  # Simulate data sparsity

        y_train = config.target_processor.process_target(y_train.tolist())
        y_val = config.target_processor.process_target(y_val.tolist())

        return X_train, X_val, y_train, y_val, labels

    def perform_embedding(self) -> None:
        self.X_train, self.X_val, self.y_train, self.y_val, self.labels = self.prepare_data(self.config)
        self.network.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        if not os.path.exists(self.config.artifacts_path):
            os.makedirs(self.config.artifacts_path, exist_ok=True)

        weights = model_utils.get_weights(self.network.model, self.config)
        model_utils.save_weights(weights, self.config)
        model_utils.save_labels(self.labels, self.config)
