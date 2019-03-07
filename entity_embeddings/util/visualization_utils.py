import os
import pickle
from typing import List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn import manifold
from sklearn.preprocessing import LabelEncoder

from entity_embeddings import Config

TITLE_FORMAT = 'Weights for %s'
FILENAME_FORMAT = '%s_embedding.pdf'


def make_visualizations(config: Config,
                        persist: bool = True) -> List[Figure]:
    with open(config.get_labels_path(), 'rb') as f:
        labels = pickle.load(f)

    with open(config.get_weights_path(), 'rb') as f:
        embeddings = pickle.load(f)

    figures = []
    for index in range(config.df.shape[1] - 1):
        column = config.df.columns[index]

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

            if persist:
                os.makedirs(config.get_visualizations_dir(), exist_ok=True)
                plt.savefig(os.path.join(config.get_visualizations_dir(), FILENAME_FORMAT % column))

    return figures


def is_not_single_embedding(label: LabelEncoder):
    return label.classes_.shape[0] > 1
