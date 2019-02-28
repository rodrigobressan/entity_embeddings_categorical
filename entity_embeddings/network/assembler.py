from abc import ABC, abstractmethod

from keras.engine import Layer
from keras.layers import Dense, Activation

from entity_embeddings.config import Config
from entity_embeddings.processor.target_type import TargetType


def get_assembler(config: Config):
    if config.target_type == TargetType.BINARY_CLASSIFICATION:
        return BinaryClassificationAssembler(config)
    elif config.target_type == TargetType.MULTICLASS_CLASSIFICATION:
        return MulticlassClassificationAssembler(config)
    elif config.target_type == TargetType.REGRESSION:
        return RegressionClassificationAssembler(config)


class ModelAssembler(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def make_final_layer(self):
        pass

    @abstractmethod
    def compile_model(self):
        pass


class BinaryClassificationAssembler(ModelAssembler):
    def make_final_layer(self, previous_layer: Layer) -> Layer:
        output_model = Dense(1)(previous_layer)
        output_model = Activation('sigmoid')(output_model)
        return output_model

    def compile_model(self, model):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class MulticlassClassificationAssembler(ModelAssembler):
    def make_final_layer(self, previous_layer: Layer) -> Layer:
        output_model = Dense(self.config.unique_classes)(previous_layer)
        output_model = Activation('softmax')(output_model)
        return output_model

    def compile_model(self, model):
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model


class RegressionClassificationAssembler(ModelAssembler):
    def make_final_layer(self, previous_layer: Layer) -> Layer:
        output_model = Dense(1)(previous_layer)
        output_model = Activation('linear')(output_model)
        return output_model

    def compile_model(self, model):
        model.compile(loss='mse', optimizer='adam')
        return model
