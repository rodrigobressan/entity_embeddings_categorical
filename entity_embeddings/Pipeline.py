import pickle

import numpy as np
import pandas as pd

from entity_embeddings.EmbeddingConfig import EmbeddingConfig
from entity_embeddings.EmbeddingNetwork import EmbeddingNetwork
from entity_embeddings.PreprocessingUtils import sample, get_X_y, label_encode
from entity_embeddings.processor.TargetType import get_processor, TargetType


def train(csv_path: str, name_target: str, type: TargetType, train_ratio: float = 0.9,
          weights_output: str = 'weights_embeddings.pickle') -> None:
    # TODO precheck target, df..
    df = pd.read_csv(csv_path)
    X, y = get_X_y(df, name_target)

    print('Samples for training: %d' % (len(y)))

    # pre processing of X and Y
    X = label_encode(X)
    y = np.array(y)

    num_records = len(X)
    train_size = int(train_ratio * num_records)

    X_train = X[:train_size]
    X_val = X[train_size:]
    y_train = y[:train_size]
    y_val = y[train_size:]

    X_train, y_train = sample(X_train, y_train, 1000)  # Simulate data sparsity

    processor = get_processor(type)
    y_train = processor.process_target(y_train.tolist())
    y_val = processor.process_target(y_val.tolist())

    print("Number of samples used for training: " + str(y_train.shape[0]))

    config = EmbeddingConfig(df, name_target)
    embedding_network = EmbeddingNetwork(config, epochs=1)
    embedding_network.fit(X_train, y_train, X_val, y_val)

    weights_embeddings = []
    for column in df.loc[:, df.columns != name_target]:
        weights = embedding_network.model.get_layer(column).get_weights()[0]
        print('Column: %s' % column)
        print('Weights: %s' % weights)
        weights_embeddings.append(weights)

    with open(weights_output, 'wb') as f:
        pickle.dump(weights_embeddings, f, -1)


if __name__ == '__main__':
    data_path = "./mock_categorical.csv"
    train(csv_path=data_path,
          name_target='vivo',
          type=TargetType.BINARY_CLASSIFICATION)
