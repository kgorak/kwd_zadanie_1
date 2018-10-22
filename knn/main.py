import os
import pandas


def load_learning_data():
    root = os.path.dirname(__file__)
    filepath = os.path.join(root, '../iris.data.learning')
    learning_data = pandas.read_csv(filepath)

    return learning_data


def main():
    pass


if __name__ == '__main__':
    main()
