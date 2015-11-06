import scipy.io
from FS_package.function.sparse_learning_based import UDFS
from FS_package.utility.unsupervised_evaluation import evaluation
from FS_package.utility.sparse_learning import feature_ranking


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']
    X = X.astype(float)
    label = mat['Y']
    label = label[:, 0]

    # UDFS feature selection
    n_selected_features = 100
    S = UDFS.udfs(X, n_clusters=20, k=5, verbose=True)
    idx = feature_ranking(S)
    X_selected = X[:, idx[0:n_selected_features]]

    # evaluation
    ari, nmi, acc = evaluation(X_selected=X_selected, n_clusters=20, y=label)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc

if __name__ == '__main__':
    main()