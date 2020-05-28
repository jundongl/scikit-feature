import numpy.matlib
import numpy as np
from scipy.sparse import *
from sklearn.metrics.pairwise import rbf_kernel
from numpy import linalg as LA


def spec(X, **kwargs):
    """
    This function implements the SPEC feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        style: {int}
            style == -1, the first feature ranking function, use all eigenvalues
            style == 0, the second feature ranking function, use all except the 1st eigenvalue
            style >= 2, the third feature ranking function, use the first k except 1st eigenvalue
        W: {sparse matrix}, shape (n_samples, n_samples}
            input affinity matrix

    Output
    ------
    w_fea: {numpy array}, shape (n_features,)
        SPEC feature score for each feature

    Reference
    ---------
    Zhao, Zheng and Liu, Huan. "Spectral Feature Selection for Supervised and Unsupervised Learning." ICML 2007.
    """

    if 'style' not in kwargs:
        kwargs['style'] = 0
    if 'W' not in kwargs:
        kwargs['W'] = rbf_kernel(X, gamma=1)

    style = kwargs['style']
    W = kwargs['W']
    if type(W) is numpy.ndarray:
        W = csc_matrix(W)

    n_samples, n_features = X.shape

    # build the degree matrix
    X_sum = np.array(W.sum(axis=1))
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        D[i, i] = X_sum[i]

    # build the laplacian matrix
    L = D - W
    d1 = np.power(np.array(W.sum(axis=1)), -0.5)
    d1[np.isinf(d1)] = 0
    d2 = np.power(np.array(W.sum(axis=1)), 0.5)
    v = np.dot(np.diag(d2[:, 0]), np.ones(n_samples))
    v = v/LA.norm(v)

    # build the normalized laplacian matrix
    L_hat = (np.matlib.repmat(d1, 1, n_samples)) * np.array(L) * np.matlib.repmat(np.transpose(d1), n_samples, 1)

    # calculate and construct spectral information
    s, U = np.linalg.eigh(L_hat)
    s = np.flipud(s)
    U = np.fliplr(U)

    # begin to select features
    w_fea = np.ones(n_features)*1000

    for i in range(n_features):
        f = X[:, i]
        F_hat = np.dot(np.diag(d2[:, 0]), f)
        l = LA.norm(F_hat)
        if l < 100*np.spacing(1):
            w_fea[i] = 1000
            continue
        else:
            F_hat = F_hat/l
        a = np.array(np.dot(np.transpose(F_hat), U))
        a = np.multiply(a, a)
        a = np.transpose(a)

        # use f'Lf formulation
        if style == -1:
            w_fea[i] = np.sum(a * s)
        # using all eigenvalues except the 1st
        elif style == 0:
            a1 = a[0:n_samples-1]
            w_fea[i] = np.sum(a1 * s[0:n_samples-1])/(1-np.power(np.dot(np.transpose(F_hat), v), 2))
        # use first k except the 1st
        else:
            a1 = a[n_samples-style:n_samples-1]
            w_fea[i] = np.sum(a1 * (2-s[n_samples-style: n_samples-1]))

    if style != -1 and style != 0:
        w_fea[w_fea == 1000] = -1000

    return w_fea


def feature_ranking(score, **kwargs):
    if 'style' not in kwargs:
        kwargs['style'] = 0
    style = kwargs['style']

    # if style = -1 or 0, ranking features in descending order, the higher the score, the more important the feature is
    if style == -1 or style == 0:
        idx = np.argsort(score, 0)
        return idx[::-1]
    # if style != -1 and 0, ranking features in ascending order, the lower the score, the more important the feature is
    elif style != -1 and style != 0:
        idx = np.argsort(score, 0)
        return idx