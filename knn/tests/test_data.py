import unittest
import pandas

import knn.data as data


class DataTest(unittest.TestCase):
    def test_load_learning_data(self):
        result = data.load_learning_data()

        self.assertIsInstance(result, pandas.DataFrame)

    def test_load_training_data(self):
        result = data.load_testing_data()

        self.assertIsInstance(result, pandas.DataFrame)

    def test_load_data(self):
        result = data.load_data()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], pandas.DataFrame)
        self.assertIsInstance(result[1], pandas.DataFrame)
