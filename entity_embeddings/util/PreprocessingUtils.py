from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing


def series_to_list(series: pd.Series) -> List:
    """
    This method is used to convert a given pd.Series object into a list
    :param series: the list to be converted
    :return: the list containing all the elements from the Series object
    """
    list_cols = []
    for item in series:
        list_cols.append(item)

    return list_cols


def sample(X: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    num_row = X.shape[0]
    indices = np.random.randint(num_row, size=n)
    return X[indices, :], y[indices]


def get_X_y(df: pd.DataFrame, name_target: str) -> Tuple[List, List]:
    """
    This method is used to gather the X (features) and y (targets) from a given dataframe based on a given
    target name
    :param df: the dataframe to be used as source
    :param name_target: the name of the target variable
    :return: the list of features and targets
    """
    X_list = []
    y_list = []

    for index, record in df.iterrows():
        fl = series_to_list(record.drop(name_target))
        X_list.append(fl)
        y_list.append(int(record[name_target]))

    return X_list, y_list


def label_encode(data: List) -> np.ndarray:
    """
    This method is used to perform Label Encoding on a given list
    :param data: the list containing the items to be encoded
    :return: the encoded np.ndarray
    """
    data_encoded = np.array(data)
    for i in range(data_encoded.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(data_encoded[:, i])
        data_encoded[:, i] = le.transform(data_encoded[:, i])
    data_encoded = data_encoded.astype(int)
    return data_encoded


def transpose_to_list(X: np.ndarray) -> List[np.ndarray]:
    """
    :param X: the ndarray to be used as source
    :return: a list of nd.array containing the elements from the numpy array
    """

    features_list = []
    for index in range(X.shape[1]):
        features_list.append(X[..., [index]])

    return features_list
