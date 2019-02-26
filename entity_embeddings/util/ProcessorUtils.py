from entity_embeddings.processor import BaseProcessor
from entity_embeddings.processor.BinaryClassificationProcessor import BinaryClassificationProcessor
from entity_embeddings.processor.MulticlassClassificationProcessor import MulticlassClassificationProcessor
from entity_embeddings.processor.RegressionProcessor import RegressionProcessor
from entity_embeddings.processor.TargetType import TargetType


def get_target_processor(type: int) -> BaseProcessor:
    """
    This method is used to get a target processor from a given target type
    :param type:
    :return:
    """
    if type == TargetType.REGRESSION:
        return RegressionProcessor()
    elif type == TargetType.BINARY_CLASSIFICATION:
        return BinaryClassificationProcessor()
    elif type == TargetType.MULTICLASS_CLASSIFICATION:
        return MulticlassClassificationProcessor()

    raise ValueError("You should provide a valid target type")
