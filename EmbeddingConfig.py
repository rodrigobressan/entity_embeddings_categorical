import numpy as np
import pandas as pd


def get_embedding_size(unique_values: int):
    return int(min(np.ceil(unique_values / 2), 50))


class Category:
    def __init__(self, alias: str, unique_values: int):
        self.alias = alias
        self.unique_values = unique_values
        self.embedding_size = get_embedding_size(unique_values)


class EmbeddingConfig:
    def __init__(self, df: pd.DataFrame, target_var: str):
        self.validate_inputs(df, target_var)

        self.categories = []
        self.target_var = target_var

        for category in df:
            if not category == target_var:
                self.add_category(Category(category, df[category].nunique()))

    def add_category(self, category):
        self.categories.append(category)

    def validate_inputs(self, df: pd.DataFrame, target_var: str):
        self.check_not_empty_dataframe(df)
        self.check_not_empty_target_var(target_var)
        self.check_target_in_df(df, target_var)

    @staticmethod
    def check_not_empty_dataframe(df: pd.DataFrame):
        if df.empty:
            raise ValueError("You should provide a non-empty pandas dataframe")

    @staticmethod
    def check_not_empty_target_var(target_var: str):
        if not target_var:
            raise ValueError("You should provide a not null target var")

    @staticmethod
    def check_target_in_df(df: pd.DataFrame, target_var: str):
        if target_var not in df.columns:
            raise ValueError("You should provide a target variable that is existent on the dataframe")
