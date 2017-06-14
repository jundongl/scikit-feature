import numpy as np
import sys
import math
import sklearn.cluster
from skfeature.utility.construct_W import construct_W


def ndfs(X, **kwargs):
    """
    This function implement unsupervised feature selection using nonnegative spectral analysis, i.e.,
    min_{F,W} Tr(F^T L F) + alpha*(||XW-F||_F^2 + beta*||W||_{2,1}) + gamma/2 * ||F^T F - I||_F^2
    s.t. F >= 0
    
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        W: {sparse matrix}, shape {n_samples, n_samples}
            affinity matrix
        alpha: {float}
            Parameter alpha in objective function
        beta: {float}
            Parameter beta in objective function
        gamma: {float}
            a very large number used to force F^T F = I
        F0: {numpy array}, shape (n_samples, n_clusters)
            initialization of the pseudo label matirx F, if not provided
        n_clusters: {int}
            number of clusters
        verbose: {boolean}
            True if user want to print out the objective function value in each iteration, false if not

    Output
    ------
    W: {numpy array}, shape(n_features, n_clusters)
        feature weight matrix
        
    Reference: 
        Li, Zechao, et al. "Unsupervised Feature Selection Using Nonnegative Spectral Analysis." AAAI. 2012.
    """

    # default gamma is 10e8
    if 'gamma' not in kwargs:
        gamma = 10e8
    else:
        gamma = kwargs['gamma']
    # use the default affinity matrix
    if 'W' not in kwargs:
        W = construct_W(X)
    else:
        W = kwargs['W']
    if 'alpha' not in kwargs:
        alpha = 1
    else:
        alpha = kwargs['alpha']
    if 'beta' not in kwargs:
        beta = 1
    else:
        beta = kwargs['beta']
    if 'F0' not in kwargs:
        if 'n_clusters' not in kwargs:
            print >>sys.stderr, "either F0 or n_clusters should be provided"
        else:
            # initialize F
            n_clusters = kwargs['n_clusters']
            F = kmeans_initialization(X, n_clusters)
    else:
        F = kwargs['F0']
    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']
    
    n_samples, n_features = X.shape

    # initialize D as identity matrix
    D = np.identity(n_features)
    I = np.identity(n_samples)

    # build laplacian matrix
    L = np.array(W.sum(1))[:, 0] - W

    max_iter = 1000
    obj = np.zeros(max_iter)
    for iter_step in range(max_iter):
        # update W
        T = np.linalg.inv(np.dot(X.transpose(), X) + beta * D + 1e-6*np.eye(n_features))
        W = np.dot(np.dot(T, X.transpose()), F)
        # update D
        temp = np.sqrt((W*W).sum(1))
        temp[temp < 1e-16] = 1e-16
        temp = 0.5 / temp
        D = np.diag(temp)
        # update M
        M = L + alpha * (I - np.dot(np.dot(X, T), X.transpose()))
        M = (M + M.transpose())/2
        # update F
        denominator = np.dot(M, F) + gamma*np.dot(np.dot(F, F.transpose()), F)
        temp = np.divide(gamma*F, denominator)
        F = F*np.array(temp)
        temp = np.diag(np.sqrt(np.diag(1 / (np.dot(F.transpose(), F) + 1e-16))))
        F = np.dot(F, temp)

        # calculate objective function
        obj[iter_step] = np.trace(np.dot(np.dot(F.transpose(), M), F)) + gamma/4*np.linalg.norm(np.dot(F.transpose(), F)-np.identity(n_clusters), 'fro')
        if verbose:
            print('obj at iter {0}: {1}'.format(iter_step+1, obj[iter_step]))

        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step-1]) < 1e-3:
            break
    return W


def kmeans_initialization(X, n_clusters):
    """
    This function uses kmeans to initialize the pseudo label

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    n_clusters: {int}
        number of clusters

    Output
    ------
    Y: {numpy array}, shape (n_samples, n_clusters)
        pseudo label matrix
    """

    n_samples, n_features = X.shape
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                                    tol=0.0001, precompute_distances=True, verbose=0,
                                    random_state=None, copy_x=True, n_jobs=1)
    kmeans.fit(X)
    labels = kmeans.labels_
    Y = np.zeros((n_samples, n_clusters))
    for row in range(0, n_samples):
        Y[row, labels[row]] = 1
    T = np.dot(Y.transpose(), Y)
    F = np.dot(Y, np.sqrt(np.linalg.inv(T)))
    F = F + 0.02*np.ones((n_samples, n_clusters))
    return F


def calculate_obj(X, W, F, L, alpha, beta):
    """
    This function calculates the objective function of NDFS
    """
    # Tr(F^T L F)
    T1 = np.trace(np.dot(np.dot(F.transpose(), L), F))
    T2 = np.linalg.norm(np.dot(X, W) - F, 'fro')
    T3 = (np.sqrt((W*W).sum(1))).sum()
    obj = T1 + alpha*(T2 + beta*T3)
    return obj