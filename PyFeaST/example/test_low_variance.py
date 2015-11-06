from FS_package.function.statistics_based import low_variance


def main():
    # load data
    X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]

    # feature selection
    # p is parameter of function low_variance_feature_selection
    p = .8
    features = low_variance.low_variance_feature_selection(X, p)
    print features


if __name__ == '__main__':
    main()