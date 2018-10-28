import os
import pandas


def load_learning_data():
    root = os.path.dirname(__file__)
    filepath = os.path.join(root, '../iris.data.learning')
    learning_data = pandas.read_csv(filepath, header=None)

    return learning_data


def load_testing_data():
    root = os.path.dirname(__file__)
    filepath = os.path.join(root, '../iris.data.test')
    test_data = pandas.read_csv(filepath, header=None)

    return test_data


def load_data():
    learning_data = load_learning_data()
    testing_data = load_testing_data()

    return learning_data, testing_data
