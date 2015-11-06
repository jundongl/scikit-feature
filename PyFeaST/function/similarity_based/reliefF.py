import numpy as np
from FS_package.utility.construct_W import construct_W


def reliefF(X, y):
    """
    This function implements the reliefF feature selection, steps are as follows:
    1. Construct the affinity matrix W in reliefF way
    2. For the r-th feature, we define fr = X(:,r), reliefF score for the r-th feature is -1+fr'*W*fr

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels

    Output
    ------
    score: {numpy array}, shape (n_features,)
        reliefF score for each feature

    Reference
    ---------
    Zhao, Zheng et al. "On Similarity Preserving Feature Selection." TKDE 2013.
    """

    # construct the affinity matrix W
    kwargs = {"neighbor_mode": "supervised", "reliefF": True, 'y': y}
    W = construct_W(X, **kwargs)
    n_samples, n_features = X.shape
    score = np.zeros(n_features)
    for i in range(n_features):
        score[i] = -1 + np.dot(np.transpose(X[:, i]), W.dot(X[:, i]))
    return score


def feature_ranking(score):
    """
    Rank features in descending order according to reliefF score, the higher the reliefF score, the more important the
    feature is
    """
    idx = np.argsort(score, 0)
    return idx[::-1]

