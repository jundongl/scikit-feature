import scipy
import numpy as np
from sklearn import linear_model
from skfeature.utility.construct_W import construct_W


def mcfs(X, n_selected_features, **kwargs):
    """
    This function implements unsupervised feature selection for multi-cluster data.

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    n_selected_features: {int}
        number of features to select
    kwargs: {dictionary}
        W: {sparse matrix}, shape (n_samples, n_samples)
            affinity matrix
        n_clusters: {int}
            number of clusters (default is 5)

    Output
    ------
    W: {numpy array}, shape(n_features, n_clusters)
        feature weight matrix

    Reference
    ---------
    Cai, Deng et al. "Unsupervised Feature Selection for Multi-Cluster Data." KDD 2010.
    """

    # use the default affinity matrix
    if 'W' not in kwargs:
        W = construct_W(X)
    else:
        W = kwargs['W']
    # default number of clusters is 5
    if 'n_clusters' not in kwargs:
        n_clusters = 5
    else:
        n_clusters = kwargs['n_clusters']

    # solve the generalized eigen-decomposition problem and get the top K
    # eigen-vectors with respect to the smallest eigenvalues
    W = W.toarray()
    W = (W + W.T) / 2
    W_norm = np.diag(np.sqrt(1 / W.sum(1)))
    W = np.dot(W_norm, np.dot(W, W_norm))
    WT = W.T
    W[W < WT] = WT[W < WT]
    eigen_value, ul = scipy.linalg.eigh(a=W)
    Y = np.dot(W_norm, ul[:, -1*n_clusters-1:-1])

    # solve K L1-regularized regression problem using LARs algorithm with cardinality constraint being d
    n_sample, n_feature = X.shape
    W = np.zeros((n_feature, n_clusters))
    for i in range(n_clusters):
        clf = linear_model.Lars(n_nonzero_coefs=n_selected_features)
        clf.fit(X, Y[:, i])
        W[:, i] = clf.coef_
    return W


def feature_ranking(W):
    """
    This function computes MCFS score and ranking features according to feature weights matrix W
    """
    mcfs_score = W.max(1)
    idx = np.argsort(mcfs_score, 0)
    idx = idx[::-1]
    return idx