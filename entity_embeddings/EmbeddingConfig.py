"""
This file contains both the Category and EmbeddingConfig classes, which are responsible to store data related to the
categories. This data will be later on used on our EmbeddingNetwork class.
"""
from typing import List

import numpy as np

from entity_embeddings.processor.TargetType import TargetType
from entity_embeddings.util.DataframeUtils import load_guarantee_not_empty
from entity_embeddings.util.ProcessorUtils import get_target_processor
from entity_embeddings.util.ValidationUtils import *


def get_embedding_size(unique_values: int) -> int:
    """
    This method is used to generate the embedding size to be used on our Embedding layer
    :param unique_values: the number of unique values in the given category
    :return:
    """
    return int(min(np.ceil(unique_values / 2), 50))


def generate_categories_from_df(df: pd.DataFrame, target_name: str):
    category_list = []

    for category in df:
        if not category == target_name:
            category_list.append(Category(category, df[category].nunique()))

    return category_list


class Category:
    """
    This class is used to store data related to the used category, such as the embedding size to be used
    """

    def __init__(self, alias: str, unique_values: int):
        self.alias = alias
        self.unique_values = unique_values
        self.embedding_size = get_embedding_size(unique_values)


class EmbeddingConfig:
    """
    This class is used to store all the configuration (dataframes, target type, epochs, artifacts..) which will be
    used on our Embeddings Network
    """

    def __init__(self,
                 csv_path: str,
                 target_name: str,
                 target_type: TargetType,
                 train_ratio: float,
                 epochs: int = 10,
                 batch_size: int = 128,
                 verbose: bool = False,
                 weights_output: str = 'weights_embeddings.pickle'):
        # input validations
        check_csv_data(csv_path)
        check_target_name(target_name)
        check_train_ratio(train_ratio)
        check_epochs(epochs)
        check_batch_size(batch_size)
        check_weights_output(weights_output)

        self.csv_path = csv_path
        self.target_name = target_name
        self.target_type = target_type
        self.train_ratio = train_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.weights_output = weights_output
        self.target_processor = get_target_processor(target_type)

        self.df = load_guarantee_not_empty(self.csv_path)
        check_target_existent_in_df(self.target_name, self.df)

        self.unique_classes = self.df[self.target_name].nunique()
        self.categories: List[Category] = generate_categories_from_df(self.df, self.target_name)
