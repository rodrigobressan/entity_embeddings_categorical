import unittest

from keras import Input
from keras.engine import Layer

from entity_embeddings.config import Config
from entity_embeddings.network.assembler import get_assembler
from entity_embeddings.processor.target_type import TargetType
from entity_embeddings.util.dataframe_utils import create_random_csv


class TestAssembler(unittest.TestCase):
    def make_config_for_type(self, type: TargetType):
        csv_path = create_random_csv(5, 5, 'ABCDE')
        config = Config(csv_path=csv_path,
                        target_name='E',
                        target_type=type,
                        train_ratio=0.9)

        return config

    def make_sample_layer(self):
        input_category = Input(shape=(1,))
        return input_category

    def test_activation_for_binary_is_sigmoid(self):
        config = self.make_config_for_type(TargetType.BINARY_CLASSIFICATION)

        assembler = get_assembler(config)
        previous_layer = self.make_sample_layer()

        layer = assembler.make_final_layer(previous_layer)

        self.check_layer_properties(layer=layer,
                                    activation="Sigmoid",
                                    outputs=1,
                                    operation='dense')

    def test_activation_for_multiclass_is_softmax(self):
        config = self.make_config_for_type(TargetType.MULTICLASS_CLASSIFICATION)

        assembler = get_assembler(config)
        previous_layer = self.make_sample_layer()

        layer = assembler.make_final_layer(previous_layer)
        self.check_layer_properties(layer=layer,
                                    activation="Softmax",
                                    outputs=config.unique_classes,
                                    operation='dense')

    def test_activation_for_regression_is_identity(self):
        config = self.make_config_for_type(TargetType.REGRESSION)

        assembler = get_assembler(config)
        previous_layer = self.make_sample_layer()

        layer = assembler.make_final_layer(previous_layer)
        self.check_layer_properties(layer=layer,
                                    activation="Identity",
                                    outputs=1,
                                    operation='dense')


    def check_layer_properties(self, layer: Layer, activation: str, outputs: int, operation: str):
        self.assertEqual(layer.op.type, activation)

        n_output = layer.shape[1].value
        self.assertEqual(n_output, outputs)

        op = layer.op._inputs[0].op.name.split("_")[0]
        self.assertEqual(op, operation)