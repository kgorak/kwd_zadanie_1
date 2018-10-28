from knn import *


def main():
    l, t = load_data()

    k = kNN(3, l)

    score = k.score(t, t[4])
    print(score)


if __name__ == '__main__':
    main()
