import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.figure import Figure
from sklearn import manifold
from sklearn.preprocessing import LabelEncoder

from entity_embeddings import Config

TITLE_FORMAT = 'Weights for %s'
FILENAME_FORMAT = '%s_embedding.%s'


def make_visualizations(labels: List[LabelEncoder],
                        embeddings: List[np.array],
                        df: pandas.DataFrame,
                        output_path: str = None,
                        format: str = 'pdf'):
    figures = []
    for index in range(df.shape[1] - 1):
        column = df.columns[index]

        if is_not_single_embedding(labels[index]):
            labels_column = labels[index]
            embeddings_column = embeddings[index]

            tsne = manifold.TSNE(init='pca', random_state=0, method='exact')
            Y = tsne.fit_transform(embeddings_column)

            fig = plt.figure(figsize=(10, 10))
            figures.append(fig)
            plt.scatter(-Y[:, 0], -Y[:, 1])
            plt.title(TITLE_FORMAT % column)
            for i, text in enumerate(labels_column.classes_):
                plt.annotate(text, (-Y[i, 0], -Y[i, 1]), xytext=(-20, 10), textcoords='offset points')

            if output_path:
                os.makedirs(output_path, exist_ok=True)
                plt.savefig(os.path.join(output_path, FILENAME_FORMAT % (column, format)))

    return figures


def make_visualizations_from_config(config: Config,
                                    format: str = 'pdf') -> List[Figure]:
    with open(config.get_labels_path(), 'rb') as f:
        labels = pickle.load(f)

    with open(config.get_weights_path(), 'rb') as f:
        embeddings = pickle.load(f)

    return make_visualizations(labels, embeddings, config.df, config.get_visualizations_dir(), format)


def is_not_single_embedding(label: LabelEncoder):
    return label.classes_.shape[0] > 1
