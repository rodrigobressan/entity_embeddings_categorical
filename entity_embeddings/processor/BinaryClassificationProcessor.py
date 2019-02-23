from typing import List

import numpy as np
from keras.utils import to_categorical

from entity_embeddings.processor.BaseTargetProcessor import BaseTargetProcessor


class BinaryClassificationProcessor(BaseTargetProcessor):
    def process_target(self, y: List) -> np.ndarray:
        """
        Converts a given set of targets to categorical (binary class)
        :param y: the
        :return:
        """
        return to_categorical(y)
