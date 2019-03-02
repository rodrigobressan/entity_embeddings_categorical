from entity_embeddings import Config, Embedder, TargetType


def main():
    data_path = "../rossmann.csv"
    config = Config.make_default_config(csv_path=data_path,
                                        target_name='Sales',
                                        target_type=TargetType.REGRESSION,
                                        train_ratio=0.9,
                                        epochs=10,
                                        verbose=True,
                                        weights_output='weights.pickle')

    print(config)
    # embedder = Embedder(config)
    # embedder.perform_embedding()


if __name__ == '__main__':
    main()
