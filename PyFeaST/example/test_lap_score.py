import scipy.io
from FS_package.function.similarity_based import lap_score
from FS_package.utility import construct_W
from FS_package.utility import unsupervised_evaluation


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']    # data
    y = mat['Y']    # label
    y = y[:, 0]
    X = X.astype(float)
    n_samples, n_features = X.shape

    # construct affinity matrix
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs_W)

    # feature selection
    score = lap_score.lap_score(X, W = W)

    idx = lap_score.feature_ranking(score)

    # evaluation
    num_fea = 100
    selected_features = X[:, idx[0:num_fea]]
    ari, nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=20, y=y)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc


if __name__ == '__main__':
    main()