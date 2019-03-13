import os
from typing import Tuple, List

import numpy as np

from entity_embeddings.config import Config
from entity_embeddings.network.network import EmbeddingNetwork
from entity_embeddings.util import model_utils, preprocessing_utils, visualization_utils


class Embedder:
    """
    This class should be used to perform the entity embedding on our Neural Network. For initializing it, you should
    provide a valid Config object.
    """

    def __init__(self, config: Config):
        self.config = config
        self.network = EmbeddingNetwork(config)
        self.X_train, self.X_val, self.y_train, self.y_val, self.labels = self.prepare_data()

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
        """
        This method is used to perform all the required pre-processing steps on the provided set of features and the
        targets, such as label encoding and sampling.
        :return: a tuple containing 5 different elements in the following order: X_train, X_val, y_train, y_val and the
        encoded labels
        """
        X, y = preprocessing_utils.get_X_y(self.config.df, self.config.target_name)

        # pre processing of X and Y
        X, labels = preprocessing_utils.label_encode(X)
        y = np.array(y)

        num_records = len(X)
        train_size = int(self.config.train_ratio * num_records)

        X_train = X[:train_size]
        X_val = X[train_size:]
        y_train = y[:train_size]
        y_val = y[train_size:]

        X_train, y_train = preprocessing_utils.sample(X_train, y_train, 1000)  # Simulate data sparsity

        y_train = self.config.target_processor.process_target(y_train.tolist())
        y_val = self.config.target_processor.process_target(y_val.tolist())

        return X_train, X_val, y_train, y_val, labels

    def perform_embedding(self) -> None:
        """
        This method is the main method in our Embedded class, being responsible to prepare our data and then feed our
        Entity Embedding Network, as well as to save the weights into the disk.
        """

        history = self.network.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        if not os.path.exists(self.config.artifacts_path):
            os.makedirs(self.config.artifacts_path, exist_ok=True)

        weights = model_utils.get_weights(self.network.model, self.config)

        # save artifacts
        model_utils.save_weights(weights, self.config)
        model_utils.save_labels(self.labels, self.config)

        visualization_utils.make_plot_from_history(history, self.config.artifacts_path)
