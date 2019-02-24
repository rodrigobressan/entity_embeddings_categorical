from typing import List

import numpy as np

from entity_embeddings.processor.TargetProcessor import TargetProcessor


class RegressionProcessor(TargetProcessor):
    def process_target(self, y: List) -> np.ndarray:
        pass
