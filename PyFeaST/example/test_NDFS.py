import scipy.io
from FS_package.function.sparse_learning_based import NDFS
from FS_package.utility import construct_W
from FS_package.utility.sparse_learning import feature_ranking
from FS_package.utility.unsupervised_evaluation import evaluation


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]

    kwargs = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs)

    # NDFS feature selection
    W = NDFS.ndfs(X, W=W, n_clusters=20, verbose=False)
    idx = feature_ranking(W)

    # evaluation
    n_selected_features = 100
    X_selected = X[:, idx[0:n_selected_features]]
    ari, nmi, acc = evaluation(X_selected=X_selected, n_clusters=20, y=y)

    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc


if __name__ == '__main__':
    main()
