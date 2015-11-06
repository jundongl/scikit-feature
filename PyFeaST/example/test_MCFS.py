import scipy.io
from FS_package.function.sparse_learning_based import MCFS
from FS_package.utility import construct_W
from FS_package.utility import unsupervised_evaluation


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    y = y[:, 0]

    # construct W
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs)

    # mcfs feature selection
    n_selected_features = 100
    S = MCFS.mcfs(X, n_selected_features, W=W, n_clusters=20)
    idx = MCFS.feature_ranking(S)

    # evaluation
    X_selected = X[:, idx[0:n_selected_features]]
    ari, nmi, acc = unsupervised_evaluation.evaluation(X_selected=X_selected, n_clusters=20, y=y)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc

if __name__ == '__main__':
    main()