from entity_embeddings.processor.BinaryClassificationProcessor import BinaryClassificationProcessor
from entity_embeddings.processor.RegressionProcessor import RegressionProcessor
from entity_embeddings.processor.TargetType import TargetType


def get_processor(type: int) -> int:
    """
    This method is used to get a target processor from a given target type
    :param type:
    :return:
    """
    if type == TargetType.REGRESSION:
        return RegressionProcessor()
    elif type == TargetType.BINARY_CLASSIFICATION:
        return BinaryClassificationProcessor()

    raise ValueError("You should provide a valid target type")
