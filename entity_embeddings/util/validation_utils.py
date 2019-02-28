import os

import pandas as pd


def check_csv_data(csv_path: str) -> None:
    if not csv_path:
        raise ValueError("You should provide a csv path where the data is located")

    if not os.path.exists(csv_path):
        raise ValueError("You should provide an existent csv path")


def check_not_empty_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("You should provide a non-empty pandas dataframe")


def check_target_name(target_name: str) -> None:
    if not target_name:
        raise ValueError("You should provide a non-empty target name")


def check_target_existent_in_df(target_name: str, df: pd.DataFrame) -> None:
    if target_name not in df.columns:
        raise ValueError("You should provide a target variable that is existent on the dataframe")


def check_train_ratio(train_ratio: float) -> None:
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("You should provide a train ratio greater than 0 and smaller than 1")


def check_epochs(epochs: int) -> None:
    if epochs <= 0:
        raise ValueError("You should provide a epoch greater than zero")


def check_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("You should provide a batch size greater than zero")


def check_weights_output(weights_output: str) -> None:
    if not weights_output:
        raise ValueError("You should provide a output file for the embeddings weights")
