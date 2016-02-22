import numpy as np
from scipy.sparse import rand
from skfeature.function.structure import tree_fs


def main():
    n_samples = 50    # specify the number of samples in the simulated data
    n_features = 100    # specify the number of features in the simulated data

    # simulate the dataset
    X = np.random.rand(n_samples, n_features)

    # simulate the feature weight
    w_orin = rand(n_features, 1, 1).toarray()
    w_orin[0:50] = 0

    # obtain the ground truth of the simulated dataset
    noise = np.random.rand(n_samples, 1)
    y = np.dot(X, w_orin) + 0.01 * noise
    y = y[:, 0]


    z = 0.01  # specify the regularization parameter of regularization parameter of L2 norm for the non-overlapping group

    # specify the tree structure among features
    idx = np.array([[-1, -1, 1], [1, 20, np.sqrt(20)], [21, 40, np.sqrt(20)], [41, 50, np.sqrt(10)],
                    [51, 70, np.sqrt(20)], [71, 100, np.sqrt(30)], [1, 50, np.sqrt(50)], [51, 100, np.sqrt(50)]]).T
    idx = idx.astype(int)

    # perform feature selection and obtain the feature weight of all the features
    w, obj, value_gamma = tree_fs.tree_fs(X, y, z, idx, verbose=True)


if __name__ == '__main__':
    main()
