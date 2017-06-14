import numpy as np
import scipy
import math
from skfeature.utility.sparse_learning import generate_diagonal_matrix, calculate_l21_norm
from sklearn.metrics.pairwise import pairwise_distances


def udfs(X, **kwargs):
    """
    This function implements l2,1-norm regularized discriminative feature
    selection for unsupervised learning, i.e., min_W Tr(W^T M W) + gamma ||W||_{2,1}, s.t. W^T W = I

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        gamma: {float}
            parameter in the objective function of UDFS (default is 1)
        n_clusters: {int}
            Number of clusters
        k: {int}
            number of nearest neighbor
        verbose: {boolean}
            True if want to display the objective function value, false if not

    Output
    ------
    W: {numpy array}, shape(n_features, n_clusters)
        feature weight matrix

    Reference
    Yang, Yi et al. "l2,1-Norm Regularized Discriminative Feature Selection for Unsupervised Learning." AAAI 2012.
    """

    # default gamma is 0.1
    if 'gamma' not in kwargs:
        gamma = 0.1
    else:
        gamma = kwargs['gamma']
    # default k is set to be 5
    if 'k' not in kwargs:
        k = 5
    else:
        k = kwargs['k']
    if 'n_clusters' not in kwargs:
        n_clusters = 5
    else:
        n_clusters = kwargs['n_clusters']
    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    # construct M
    n_sample, n_feature = X.shape
    M = construct_M(X, k, gamma)

    D = np.eye(n_feature)
    max_iter = 1000
    obj = np.zeros(max_iter)
    for iter_step in range(max_iter):
        # update W as the eigenvectors of P corresponding to the first n_clusters
        # smallest eigenvalues
        P = M + gamma*D
        eigen_value, eigen_vector = scipy.linalg.eigh(a=P)
        W = eigen_vector[:, 0:n_clusters]
        # update D as D_ii = 1 / 2 / ||W(i,:)||
        D = generate_diagonal_matrix(W)

        obj[iter_step] = calculate_obj(X, W, M, gamma)
        if verbose:
            print('obj at iter {0}: {1}'.format(iter_step+1, obj[iter_step]))

        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step-1]) < 1e-3:
            break
    return W


def construct_M(X, k, gamma):
    """
    This function constructs the M matrix described in the paper
    """
    n_sample, n_feature = X.shape
    Xt = X.T
    D = pairwise_distances(X)
    # sort the distance matrix D in ascending order
    idx = np.argsort(D, axis=1)
    # choose the k-nearest neighbors for each instance
    idx_new = idx[:, 0:k+1]
    H = np.eye(k+1) - 1/(k+1) * np.ones((k+1, k+1))
    I = np.eye(k+1)
    Mi = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        Xi = Xt[:, idx_new[i, :]]
        Xi_tilde =np.dot(Xi, H)
        Bi = np.linalg.inv(np.dot(Xi_tilde.T, Xi_tilde) + gamma*I)
        Si = np.zeros((n_sample, k+1))
        for q in range(k+1):
            Si[idx_new[q], q] = 1
        Mi = Mi + np.dot(np.dot(Si, np.dot(np.dot(H, Bi), H)), Si.T)
    M = np.dot(np.dot(X.T, Mi), X)
    return M


def calculate_obj(X, W, M, gamma):
    """
    This function calculates the objective function of ls_l21 described in the paper
    """
    return np.trace(np.dot(np.dot(W.T, M), W)) + gamma*calculate_l21_norm(W)