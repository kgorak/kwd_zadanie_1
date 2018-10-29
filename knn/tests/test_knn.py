import unittest
import pandas

import knn.knn as knn


class kNNTest(unittest.TestCase):
    def test_constructor_accepts_arguments(self):
        knn.kNN(123, [1, 2, 3])

    def test_predict(self):
        row = [1, 1, 1, 1, 'label']
        data = pandas.DataFrame([row])
        solution = knn.kNN(123, data)

        result = solution.predict(data)

        self.assertIsInstance(result, list)
        self.assertEqual(result, ['label'])

    def test_score_good_label(self):
        row = [1, 1, 1, 1, 'label']
        data = pandas.DataFrame([row])
        solution = knn.kNN(123, data)

        result = solution.score(data, ['label'])

        self.assertIsInstance(result, float)
        self.assertEqual(result, 1.0)

    def test_score_bad_label(self):
        row = [1, 1, 1, 1, 'label']
        data = pandas.DataFrame([row])
        solution = knn.kNN(123, data)

        result = solution.score(data, ['bad_label'])

        self.assertIsInstance(result, float)
        self.assertEqual(result, 0)
