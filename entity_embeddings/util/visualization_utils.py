"""
Contain methods that are useful to generate visualizations from the weights of the Embedding layers. Also provides
methods to plot the model history, such as loss over epochs.
"""
import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas
from keras.callbacks import History
from matplotlib.figure import Figure
from sklearn import manifold
from sklearn.preprocessing import LabelEncoder

from entity_embeddings import Config
from sklearn.decomposition import PCA


TITLE_FORMAT = 'Weights for %s'
SCATTER_EMBEDDINGS_FORMAT = '%s_embedding.%s'
PLOT_LOSS_FORMAT = 'loss_epochs.%s'


def make_visualizations(labels: List[LabelEncoder],
                        embeddings: List[np.array],
                        df: pandas.DataFrame,
                        output_path: str = None,
                        extension: str = 'pdf') -> List[Figure]:
    """
    Used to generate the embedding visualizations for each categorical variable

    :param labels: a list of the LabelEncoders of each categorical variable
    :param embeddings: a Numpy array containing the weights from the categorical variables
    :param df: the dataframe from where the weights were extracted
    :param output_path: (optional) where the visualizations will be saved
    :param extension: (optional) the extension to be used when saving the artifacts
    :return: the list of figures for each categorical variable
    """
    figures = []
    for index in range(df.shape[1] - 1):
        column = df.columns[index]

        if is_not_single_embedding(labels[index]):
            labels_column = labels[index]
            embeddings_column = embeddings[index]

            # tsne = manifold.TSNE(init='pca', random_state=0, method='exact')
            pca = PCA(n_components=2)
            Y = pca.fit_transform(embeddings_column)

            fig = plt.figure(figsize=(10, 10))
            figures.append(fig)
            plt.scatter(-Y[:, 0], -Y[:, 1])
            plt.title(TITLE_FORMAT % column)
            for i, text in enumerate(labels_column.classes_):
                plt.annotate(text, (-Y[i, 0], -Y[i, 1]), xytext=(-20, 10), textcoords='offset points')

            if output_path:
                os.makedirs(output_path, exist_ok=True)
                plt.savefig(os.path.join(output_path, SCATTER_EMBEDDINGS_FORMAT % (column, extension)))

    return figures


def make_visualizations_from_config(config: Config,
                                    extension: str = 'pdf') -> List[Figure]:
    """
    Used to generate the embedding visualizations from a given Config object
    :param config: the Config used
    :param extension: the extension to be saved the artifacts
    :return: the list of figures for each categorical variable
    """
    with open(config.get_labels_path(), 'rb') as f:
        labels = pickle.load(f)

    with open(config.get_weights_path(), 'rb') as f:
        embeddings = pickle.load(f)

    return make_visualizations(labels, embeddings, config.df, config.get_visualizations_dir(), extension)


def is_not_single_embedding(label: LabelEncoder) -> bool:
    """
    Used to check if there is more than one class in a given LabelEncoder
    :param label: label encoder to be checked
    :return: a boolean if the embedding contains more than one class
    """
    return label.classes_.shape[0] > 1


def make_plot_from_history(history: History,
                           output_path: str = None,
                           extension: str = 'pdf') -> Figure:
    """
    Used to make a Figure object containing the loss curve between the epochs.
    :param history: the history outputted from the model.fit method
    :param output_path: (optional) where the image will be saved
    :param extension: (optional) the extension of the file
    :return: a Figure object containing the plot
    """
    loss = history.history['loss']

    fig = plt.figure(figsize=(10, 10))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(loss)

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, PLOT_LOSS_FORMAT % extension))

    return fig
