from typing import List

import numpy as np

from entity_embeddings.processor.BaseTargetProcessor import BaseTargetProcessor


class RegressionProcessor(BaseTargetProcessor):
    def process_target(self, y: List) -> np.ndarray:
        pass
