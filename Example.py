from entity_embeddings.Embedder import Embedder
from entity_embeddings.EmbeddingConfig import EmbeddingConfig
from entity_embeddings.processor.TargetType import TargetType


def main():
    data_path = "./mock_categorical.csv"
    config = EmbeddingConfig(csv_path=data_path,
                             target_name='vivo',
                             target_type=TargetType.MULTICLASS_CLASSIFICATION,
                             train_ratio=0.9,
                             epochs=10,
                             verbose=True,
                             weights_output='weights.pickle')

    embedder = Embedder(config)
    embedder.perform_embedding()


if __name__ == '__main__':
    main()
