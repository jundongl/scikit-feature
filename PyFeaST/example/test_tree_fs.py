import numpy as np
from scipy.sparse import rand
from FS_package.function.structure import tree_fs


def main():
    n_samples = 50
    n_features = 100
    X = np.random.rand(n_samples, n_features)
    w_orin = rand(n_features, 1, 1).toarray()
    w_orin[0:50] = 0
    noise = np.random.rand(n_samples, 1)
    y = np.dot(X, w_orin) + 0.01 * noise
    y = y[:, 0]
    z = 0.01
    idx = np.array([[-1, -1, 1], [1, 20, np.sqrt(20)], [21, 40, np.sqrt(20)], [41, 50, np.sqrt(10)],
                    [51, 70, np.sqrt(20)], [71, 100, np.sqrt(30)], [1, 50, np.sqrt(50)], [51, 100, np.sqrt(50)]]).T
    idx = idx.astype(int)
    w, obj, value_gamma = tree_fs.tree_fs(X, y, z, idx, verbose=True)


if __name__ == '__main__':
    main()
