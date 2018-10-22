import unittest

import knn.knn as knn


class kNNTest(unittest.TestCase):
    def test_constructor_accepts_arguments(self):
        knn.kNN(123, [1, 2, 3])
