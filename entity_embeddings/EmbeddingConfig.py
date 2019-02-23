"""
This file contains both the Category and EmbeddingConfig classes, which are responsible to store data related to the
categories. This data will be later on used on our EmbeddingNetwork class.
"""
from typing import List

import numpy as np
import pandas as pd


def get_embedding_size(unique_values: int) -> int:
    """
    This method is used to generate the embedding size to be used on our Embedding layer
    :param unique_values: the number of unique values in the given category
    :return:
    """
    return int(min(np.ceil(unique_values / 2), 50))


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
    This class is used to store all the categories which will be used on our Embeddings Network
    """

    def __init__(self, df: pd.DataFrame, target_var: str):
        self.validate_inputs(df, target_var)

        self.categories: List[Category] = []
        self.target_var = target_var

        for category in df:
            if not category == target_var:
                self.add_category(Category(category, df[category].nunique()))

    def add_category(self, category: Category) -> None:
        """
        Method to add a new category
        :param category: the category to be added
        """
        self.categories.append(category)

    def validate_inputs(self, df: pd.DataFrame, target_var: str) -> None:
        """
        Method used to validate the received initialization inputs
        :param df: the dataframe to be used
        :param target_var: the name of the target variable
        """
        self.check_not_empty_dataframe(df)
        self.check_not_empty_target_var(target_var)
        self.check_target_in_df(df, target_var)

    @staticmethod
    def check_not_empty_dataframe(df: pd.DataFrame) -> None:
        """
        Used to check if the received dataframe is not empty
        :param df:
        """
        if df.empty:
            raise ValueError("You should provide a non-empty pandas dataframe")

    @staticmethod
    def check_not_empty_target_var(target_var: str) -> None:
        """
        Used to check if the target var is not empty
        :param target_var:
        """
        if not target_var:
            raise ValueError("You should provide a not null target var")

    @staticmethod
    def check_target_in_df(df: pd.DataFrame, target_var: str) -> None:
        """
        Used to check if the passed target var exists in the dataframe
        :param df: the dataframe to be used
        :param target_var: the target/y var name
        """
        if target_var not in df.columns:
            raise ValueError("You should provide a target variable that is existent on the dataframe")
