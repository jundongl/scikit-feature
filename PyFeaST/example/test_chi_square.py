import scipy.io
from FS_package.function.statistics_based import chi_square


def main():
    # get number of features
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    y = y[:, 0]

    # feature selection
    num_fea = 2
    F = chi_square.chi_square(X, y)
    idx = chi_square.feature_ranking(F)
    print idx[0:num_fea]


if __name__ == '__main__':
    main()