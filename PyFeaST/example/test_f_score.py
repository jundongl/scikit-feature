import numpy as np
import csv
from FS_package.function.statistics_based import f_score


def main():
    # get number of features
    with open('../data/test_lung_s3.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            num_columns = len(row)
            break

    # load data
    mat = np.loadtxt('../data/test_lung_s3.csv', delimiter=',', skiprows=1, usecols=range(0, num_columns))
    X = mat[:, 1:num_columns]  # data
    X = X.astype(float)
    y = mat[:, 0]  # label

    # feature selection
    num_fea = 5
    F = f_score.f_score(X, y)
    idx = f_score.feature_ranking(F)
    print idx[0:5]


if __name__ == '__main__':
    main()