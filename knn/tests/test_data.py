import unittest
import pandas

import knn.data as data


class DataTest(unittest.TestCase):
    def test_load_learning_data(self):
        result = data.load_learning_data()

        self.assertIsInstance(result, pandas.DataFrame)
