from entity_embeddings.embedder import Embedder
from entity_embeddings.config import Config
from entity_embeddings.processor.target_type import TargetType


def main():
    data_path = "./example/mock_categorical.csv"
    config = Config(csv_path=data_path,
                    target_name='vivo',
                    target_type=TargetType.BINARY_CLASSIFICATION,
                    train_ratio=0.9,
                    epochs=10,
                    verbose=True,
                    weights_output='weights.pickle')

    embedder = Embedder(config)
    embedder.perform_embedding()


if __name__ == '__main__':
    main()
