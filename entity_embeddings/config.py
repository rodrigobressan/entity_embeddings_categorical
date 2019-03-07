"""
This file contains both the Category and EmbeddingConfig classes, which are responsible to store data related to the
categories. This data will be later on used on our EmbeddingNetwork class.
"""
from typing import List

import numpy as np

from entity_embeddings.network.assembler import get_model_assembler
from entity_embeddings.processor.target_type import TargetType
from entity_embeddings.util.dataframe_utils import load_guarantee_not_empty
from entity_embeddings.util.processor_utils import get_target_processor
from entity_embeddings.util.validation_utils import *


def get_embedding_size(unique_values: int) -> int:
    """
    Return the embedding size to be used on the Embedding layer
    :param unique_values: the number of unique values in the given category
    :return: the size to be used on the embedding layer
    """
    size = int(min(np.ceil(unique_values / 2), 50))
    if size < 2:
        return 2
    else:
        return size


def generate_categories_from_df(df: pd.DataFrame, target_name: str) -> List:
    """
    Returns a list of the categories from a given pandas DataFrame, with the exception of the provided target name
    :param df: the DataFrame
    :param target_name: the name of the target column to not be included
    :return: a List of Category with the df columns except the provided one
    """
    category_list = []

    for category in df:
        if not category == target_name:
            category_list.append(Category(category, df[category].nunique()))

    return category_list


class Category:
    """
    Used to store fields related to a given category, such as its name, count of unique values and the size of each
    embedding layer
    """

    def __init__(self, alias: str, unique_values: int):
        self.alias = alias
        self.unique_values = unique_values
        self.embedding_size = get_embedding_size(unique_values)


class Config:
    """
    Used to store all the configuration (dataframes, target type, epochs, artifacts..) which will be
    used on our Embeddings Network
    """

    def __init__(self,
                 csv_path: str,
                 target_name: str,
                 train_ratio: float,
                 target_processor: TargetProcessor,
                 model_assembler: ModelAssembler,
                 epochs: int = 10,
                 batch_size: int = 128,
                 verbose: bool = False,
                 artifacts_path: str = 'artifacts'):
        # input validations
        check_csv_data(csv_path)
        check_target_name(target_name)
        check_train_ratio(train_ratio)
        check_epochs(epochs)
        check_batch_size(batch_size)

        check_target_processor(target_processor)
        check_model_assembler(model_assembler)

        self.csv_path = csv_path
        self.target_name = target_name
        self.train_ratio = train_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.artifacts_path = artifacts_path

        self.target_processor = target_processor
        self.model_assembler = model_assembler

        self.df = load_guarantee_not_empty(self.csv_path)
        check_target_existent_in_df(self.target_name, self.df)

        self.unique_classes = self.df[self.target_name].nunique()

        self.categories: List[Category] = generate_categories_from_df(self.df, self.target_name)

        # artifacts related fields
        self.DEFAULT_WEIGHTS_FILENAME = 'weights.pickle'
        self.DEFAULT_LABELS_FILENAME = 'labels.pickle'
        self.DEFAULT_PATH_VISUALIZATIONS = 'visualizations'

    @classmethod
    def make_default_config(cls,
                            csv_path: str,
                            target_name: str,
                            target_type: TargetType,
                            train_ratio: float,
                            epochs: int = 10,
                            batch_size: int = 128,
                            verbose: bool = False,
                            artifacts_path: str = 'artifacts'):
        """
        Used to create a default Config object.

        :param csv_path: where the csv containing both the features and target is located
        :param target_name: the name of the target/output variable
        :param target_type: the TargetType to be used (BINARY, REGRESSION, MULTICLASS)
        :param train_ratio: the proportion to be used for the training subset
        :param epochs: how many epochs should the model be trained
        :param batch_size: the size of the batch size
        :param verbose: if logs should be outputted or not
        :param artifacts_path: where the artifacts (weights, labels, visualizations) should be stored
        :return: a Config object
        """
        df = load_guarantee_not_empty(csv_path)
        check_target_existent_in_df(target_name, df)
        n_unique_classes = df[target_name].nunique()

        target_processor = get_target_processor(target_type)
        model_assembler = get_model_assembler(target_type, n_unique_classes)

        return cls(csv_path,
                   target_name,
                   train_ratio,
                   target_processor,
                   model_assembler,
                   epochs,
                   batch_size,
                   verbose,
                   artifacts_path)

    @classmethod
    def make_custom_config(cls,
                           csv_path: str,
                           target_name: str,
                           train_ratio: float,
                           target_processor: TargetProcessor,
                           model_assembler: ModelAssembler,
                           epochs: int = 10,
                           batch_size: int = 128,
                           verbose: bool = False,
                           artifacts_path: str = 'artifacts'):
        """
        Used to create a custom Config object. Mostly should be used when you want to have a custom TargetProcessor
        and/or a custom ModelAssembler.

        :param csv_path: where the csv containing both the features and target is located
        :param target_name: the name of the target/output variable
        :param train_ratio: the proportion to be used for the training subset
        :param target_processor: the TargetProcessor to be used
        :param model_assembler: the ModelAssembler to be used
        :param epochs: how many epochs should the model be trained
        :param batch_size: the size of the batch size
        :param verbose: if logs should be outputted or not
        :param artifacts_path: where the artifacts (weights, labels, visualizations) should be stored
        :return: a Config object
        """
        return cls(csv_path,
                   target_name,
                   train_ratio,
                   target_processor,
                   model_assembler,
                   epochs,
                   batch_size,
                   verbose,
                   artifacts_path)

    def get_weights_path(self):
        """
        Used to return the path of the stored weights
        :return: the pah of the stored weights on disk
        """
        return os.path.join(self.artifacts_path, self.DEFAULT_WEIGHTS_FILENAME)

    def get_labels_path(self):
        """
        Used to return the path of the stored labels
        :return: the pah of the stored labels on disk
        """
        return os.path.join(self.artifacts_path, self.DEFAULT_LABELS_FILENAME)

    def get_visualizations_dir(self):
        """
        Used to return the path of the stored visualizations
        :return: the pah of the stored visualizations on disk
        """
        return os.path.join(self.artifacts_path, self.DEFAULT_PATH_VISUALIZATIONS)
