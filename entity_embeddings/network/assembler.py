from abc import ABC, abstractmethod
from typing import List

from keras.engine import Layer
from keras.layers import Dense, Activation, Concatenate
from keras.models import Model as KerasModel

from entity_embeddings.processor.target_type import TargetType


def get_model_assembler(target_type: TargetType, n_unique_classes: int):
    if target_type == TargetType.BINARY_CLASSIFICATION:
        return BinaryClassificationAssembler()
    elif target_type == TargetType.MULTICLASS_CLASSIFICATION:
        return MulticlassClassificationAssembler(n_unique_classes)
    elif target_type == TargetType.REGRESSION:
        return RegressionClassificationAssembler()


class ModelAssembler(ABC):
    @abstractmethod
    def make_final_layer(self, previous_layer: Layer) -> Layer:
        raise NotImplementedError("Your model assembler should override the method make_final_layer")

    @abstractmethod
    def compile_model(self, model: KerasModel) -> KerasModel:
        raise NotImplementedError("Your model assembler should override the method compile_model")

    def make_hidden_layers(self, outputs: List[Layer]) -> Layer:
        output_model = Concatenate()(outputs)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        return output_model


class BinaryClassificationAssembler(ModelAssembler):
    def make_final_layer(self, previous_layer: Layer) -> Layer:
        output_model = Dense(1)(previous_layer)
        output_model = Activation('sigmoid')(output_model)
        return output_model

    def compile_model(self, model):
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class MulticlassClassificationAssembler(ModelAssembler):
    def __init__(self, n_unique_classes: int):
        self.n_unique_classes = n_unique_classes

    def make_final_layer(self, previous_layer: Layer) -> Layer:
        output_model = Dense(self.n_unique_classes)(previous_layer)
        output_model = Activation('softmax')(output_model)
        return output_model

    def compile_model(self, model):
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model


class RegressionClassificationAssembler(ModelAssembler):
    def make_final_layer(self, previous_layer: Layer) -> Layer:
        output_model = Dense(1)(previous_layer)
        output_model = Activation('sigmoid')(output_model)
        return output_model

    def compile_model(self, model):
        model.compile(loss='mean_absolute_error', optimizer='adam')
        return model
