import unittest

import knn.knn as knn


class kNNTest(unittest.TestCase):
    def test_constructor_accepts_arguments(self):
        knn.kNN(123, [1, 2, 3])

    def test_predict(self):
        solution = knn.kNN(123, [1, 2, 3])

        result = solution.predict([])

        self.assertIsInstance(result, list)
