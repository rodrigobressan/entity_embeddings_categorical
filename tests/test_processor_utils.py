import unittest

from entity_embeddings.util import processor_utils


class TestProcessorUtils(unittest.TestCase):
    def test_get_invalid_target_processor(self):
        self.assertRaises(ValueError, processor_utils.get_target_processor, 1000)
