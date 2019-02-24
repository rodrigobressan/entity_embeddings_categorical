import os
import shutil

import numpy as np
import pandas as pd

from entity_embeddings.util.ValidationUtils import check_not_empty_dataframe


def load_guarantee_not_empty(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    check_not_empty_dataframe(df)

    return df


def get_all_columns_except(df: pd.DataFrame, column_to_skip: str) -> pd.DataFrame:
    return df.loc[:, df.columns != column_to_skip]


# Below are some methods that are used mostly for testing purposes

BASE_OUTPUT_DIR = 'temp_artifacts'
BASE_OUTPUT_FILENAME = 'temp_dataframe.csv'


def create_random_dataframe(rows: int = 4,
                            cols: int = 4,
                            columns: str = 'ABCD') -> pd.DataFrame:
    return pd.DataFrame(np.random.randint(0, 10, size=(rows, cols)), columns=list(columns))


def create_random_csv(rows: int = 4,
                      cols: int = 4,
                      columns: str = 'ABCD') -> str:
    df = create_random_dataframe(rows, cols, columns)

    full_path = os.path.join(BASE_OUTPUT_DIR, BASE_OUTPUT_FILENAME)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    df.to_csv(full_path, index=False)

    return full_path


def remove_random_csv() -> None:
    shutil.rmtree(BASE_OUTPUT_DIR)
