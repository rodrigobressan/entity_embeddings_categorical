from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler


class TargetProcessor(ABC):
    @abstractmethod
    def process_target(self, y: List) -> np.ndarray:
        pass


class BinaryClassificationProcessor(TargetProcessor):
    def process_target(self, y: List) -> np.ndarray:
        return LabelEncoder().fit_transform(y)


class MulticlassClassificationProcessor(TargetProcessor):
    def process_target(self, y: List) -> np.ndarray:
        return OneHotEncoder().fit_transform(pd.DataFrame(y))


class RegressionProcessor(TargetProcessor):
    def process_target(self, y: List) -> np.ndarray:
        return np.array(y)
        # return MinMaxScaler().fit_transform(pd.DataFrame(y))
