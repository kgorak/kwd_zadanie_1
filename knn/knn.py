from scipy.spatial.distance import euclidean
import pandas


def split_data(data):
    points = data[:4]
    label = data[4]

    return points, label


class CountOccurences():
    def __init__(self):
        self._dict = {}

    def put(self, key):
        try:
            self._dict[key] += 1
        except KeyError:
            self._dict[key] = 1

    def get_max(self):
        retVal = None
        max_val = -1
        for key in self._dict:
            if self._dict[key] > max_val:
                max_val = self._dict[key]
                retVal = key

        return retVal


class kNN:
    def __init__(self, k: float, learning_data: pandas.DataFrame):
        self.k = k
        self.learning_data = learning_data

    def distance(self, array_a, array_b):
        return euclidean(array_a, array_b)

    def predict(self, objects_to_classify: list):
        retVal = []

        for c_ind, c_obj in objects_to_classify.iterrows():
            c_point = c_obj[:4]
            distance = []

            for r_ind, r_obj in self.learning_data.iterrows():
                r_point, r_label = r_obj[:4], r_obj[4]

                distance.append((self.distance(r_point, c_point), r_label))

            sorted_distance = sorted(distance, key=lambda d: d[0])

            occurences = CountOccurences()
            for dist in sorted_distance[:self.k]:
                occurences.put(dist[1])

            predicted_label = occurences.get_max()

            retVal.append(predicted_label)

        return retVal

    def score(self, objects_to_classify, correct_labels):
        assert(len(objects_to_classify) == len(correct_labels))

        number_of_labels = len(objects_to_classify)

        predicted_labels = self.predict(objects_to_classify)

        correct = 0

        for predicted_label, correct_label in zip(predicted_labels, correct_labels):
            if predicted_label == correct_label:
                correct += 1

        return '{}/{} = {}'.format(
            correct, number_of_labels, correct/number_of_labels)
