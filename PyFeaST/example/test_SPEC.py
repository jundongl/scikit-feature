import scipy.io
from FS_package.function.similarity_based import SPEC
from FS_package.utility import unsupervised_evaluation


def main():
    # load data
    mat = scipy.io.loadmat('../data/gisette.mat')
    X = mat['X']
    y = mat['Y']
    y = y[:, 0]
    X = X.astype(float)


    # feature selection
    kwargs = {'style': 0}
    score = SPEC.spec(X, **kwargs)
    idx = SPEC.feature_ranking(score, **kwargs)

    # evaluation
    num_fea = 100
    selected_features = X[:, idx[0:num_fea]]
    ari, nmi, acc = unsupervised_evaluation.evaluation(selected_features=selected_features, n_clusters=2, y=y)
    print 'ARI:', ari
    print 'NMI:', nmi
    print 'ACC:', acc


if __name__ == '__main__':
    main()