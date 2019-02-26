from typing import Tuple

import numpy as np

from entity_embeddings.EmbeddingConfig import EmbeddingConfig
from entity_embeddings.EmbeddingNetwork import EmbeddingNetwork
from entity_embeddings.util import ModelUtils, PreprocessingUtils


class Embedder():
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.X_train, self.X_val, self.y_train, self.y_val = self.prepare_data(self.config)

    def prepare_data(self, config: EmbeddingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, y = PreprocessingUtils.get_X_y(config.df, config.target_name)

        # pre processing of X and Y
        X = PreprocessingUtils.label_encode(X)
        y = np.array(y)

        num_records = len(X)
        train_size = int(config.train_ratio * num_records)

        X_train = X[:train_size]
        X_val = X[train_size:]
        y_train = y[:train_size]
        y_val = y[train_size:]

        X_train, y_train = PreprocessingUtils.sample(X_train, y_train, 1000)  # Simulate data sparsity

        y_train = config.target_processor.process_target(y_train.tolist())
        y_val = config.target_processor.process_target(y_val.tolist())

        return X_train, X_val, y_train, y_val

    def perform_embedding(self) -> None:
        network = EmbeddingNetwork(self.config)
        network.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        ModelUtils.save_weights(network.model, self.config)
