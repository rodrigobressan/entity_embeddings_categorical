from typing import List

import numpy as np
from sklearn.preprocessing import LabelEncoder

from entity_embeddings.processor.BaseProcessor import TargetProcessor


class BinaryClassificationProcessor(TargetProcessor):
    def process_target(self, y: List) -> np.ndarray:
        return LabelEncoder().fit_transform(y)
