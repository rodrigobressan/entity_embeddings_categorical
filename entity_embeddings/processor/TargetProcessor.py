from abc import ABC, abstractmethod
from typing import List

import numpy as np


class TargetProcessor(ABC):
    @abstractmethod
    def process_target(self, y: List) -> np.ndarray:
        pass
