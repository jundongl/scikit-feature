import numpy as np
import csv
from FS_package.function.statistics_based import t_score


def main():
    # get number of features
    with open('../data/test_colon_s3.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            num_columns = len(row)
            break

    # load data
    mat = np.loadtxt('../data/test_colon_s3.csv', delimiter=',', skiprows=1, usecols=range(0, num_columns))
    X = mat[:, 1:num_columns]  # data
    X = X.astype(float)
    y = mat[:, 0]  # label

    # feature selection
    num_fea = 5
    F = t_score.t_score(X, y)
    idx = t_score.feature_ranking(F)
    print X[:, idx[0:num_fea]]


if __name__ == '__main__':
    main()