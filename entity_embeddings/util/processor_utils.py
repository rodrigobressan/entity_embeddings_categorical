from entity_embeddings.processor import processor, RegressionProcessor, MulticlassClassificationProcessor, \
    BinaryClassificationProcessor
from entity_embeddings.processor.target_type import TargetType


def get_target_processor(type: int) -> processor:
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
