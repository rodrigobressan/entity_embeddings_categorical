import unittest

import keras
from keras import Input, Model
from keras.engine import Layer
from keras.optimizers import Optimizer

from entity_embeddings.config import Config
from entity_embeddings.network import ModelAssembler
from entity_embeddings.processor.target_type import TargetType
from entity_embeddings.util.dataframe_utils import create_random_csv, remove_random_csv

class TestAssembler(unittest.TestCase):
    def make_config_for_type(self, type: TargetType):
        csv_path = create_random_csv(5, 5, 'ABCDE')
        config = Config.make_default_config(csv_path=csv_path,
                                            target_name='E',
                                            target_type=type,
                                            train_ratio=0.9)

        return config

    def make_sample_layer(self):
        input_layer = Input(shape=(1,))
        return input_layer

    def test_activation_for_binary_is_sigmoid(self):
        config = self.make_config_for_type(TargetType.BINARY_CLASSIFICATION)
        previous_layer = self.make_sample_layer()

        layer = config.model_assembler.make_final_layer(previous_layer)

        self.check_layer_properties(layer=layer,
                                    activation="Sigmoid",
                                    outputs=1,
                                    operation='dense')

        remove_random_csv()

    def test_activation_for_multiclass_is_softmax(self):
        config = self.make_config_for_type(TargetType.MULTICLASS_CLASSIFICATION)
        previous_layer = self.make_sample_layer()

        layer = config.model_assembler.make_final_layer(previous_layer)
        self.check_layer_properties(layer=layer,
                                    activation="Softmax",
                                    outputs=config.unique_classes,
                                    operation='dense')
        remove_random_csv()

    def test_activation_for_regression_is_sigmoid(self):
        config = self.make_config_for_type(TargetType.REGRESSION)
        previous_layer = self.make_sample_layer()

        layer = config.model_assembler.make_final_layer(previous_layer)
        self.check_layer_properties(layer=layer,
                                    activation="Sigmoid",
                                    outputs=1,
                                    operation='dense')

        remove_random_csv()

    def test_model_params_for_binary_classification(self):
        config = self.make_config_for_type(TargetType.BINARY_CLASSIFICATION)
        previous_layer = self.make_sample_layer()

        layer = config.model_assembler.make_final_layer(previous_layer)
        model = Model(inputs=previous_layer, outputs=layer)

        model = config.model_assembler.compile_model(model)
        self.check_model_parameters(model=model,
                                    optimizer=keras.optimizers.Adam,
                                    loss="binary_crossentropy",
                                    metrics=["accuracy"])

        remove_random_csv()

    def test_model_params_for_multiclass_classification(self):
        config = self.make_config_for_type(TargetType.MULTICLASS_CLASSIFICATION)
        previous_layer = self.make_sample_layer()

        layer = config.model_assembler.make_final_layer(previous_layer)
        model = Model(inputs=previous_layer, outputs=layer)

        model = config.model_assembler.compile_model(model)
        self.check_model_parameters(model=model,
                                    optimizer=keras.optimizers.Adam,
                                    loss="categorical_crossentropy",
                                    metrics=[])

        remove_random_csv()

    def test_model_params_for_regression_classification(self):
        config = self.make_config_for_type(TargetType.REGRESSION)
        previous_layer = self.make_sample_layer()

        layer = config.model_assembler.make_final_layer(previous_layer)
        model = Model(inputs=previous_layer, outputs=layer)

        model = config.model_assembler.compile_model(model)
        self.check_model_parameters(model=model,
                                    optimizer=keras.optimizers.Adam,
                                    loss="mean_absolute_error",
                                    metrics=[])

        remove_random_csv()

    def check_model_parameters(self, model: Model, optimizer: Optimizer, loss: str, metrics) -> None:
        self.assertIsInstance(model.optimizer, optimizer)
        self.assertEqual(model.loss, loss)
        self.assertEqual(model.metrics, metrics)

    def check_layer_properties(self, layer: Layer, activation: str, outputs: int, operation: str):
        self.assertEqual(layer.op.type, activation)

        # TODO move those things to layer_utils class
        n_output = layer.shape[1].value
        self.assertEqual(n_output, outputs)

        op = layer.op._inputs[0].op.name.split("_")[0]
        self.assertEqual(op, operation)
