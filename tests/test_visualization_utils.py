import os
import shutil
import unittest

from pandas.tests.extension.numpy_.test_numpy_nested import np

from entity_embeddings.util import dataframe_utils
from entity_embeddings.util.preprocessing_utils import label_encode
from entity_embeddings.util.visualization_utils import make_visualizations


class TestVisualizationUtils(unittest.TestCase):
    def test_make_visualizations_generate_artifacts(self):
        labels = [[0], [1]]

        data, labels = label_encode(labels)

        embeddings = np.array([[1, 0], [0, 1]])

        embeddings_list = []
        embeddings_list.append(embeddings)

        df = dataframe_utils.create_random_dataframe(2, 2, 'AB')

        path = 'test_visualization'
        make_visualizations(list(labels), embeddings_list, df, output_path=path)

        path_created = os.path.exists(path)

        self.assertTrue(path_created)

        shutil.rmtree(path)
