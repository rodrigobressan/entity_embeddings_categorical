from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from entity_embeddings.processor.BaseProcessor import TargetProcessor


class MulticlassClassificationProcessor(TargetProcessor):
    def process_target(self, y: List) -> np.ndarray:
        return OneHotEncoder().fit_transform(pd.DataFrame(y))
