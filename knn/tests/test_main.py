import unittest
import pandas

import knn.main as main


class MainTest(unittest.TestCase):
    def test_load_learning_data(self):
        result = main.load_learning_data()

        self.assertIsInstance(result, pandas.DataFrame)
