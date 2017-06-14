import math
import numpy as np
from numpy import linalg as LA
from skfeature.utility.sparse_learning import generate_diagonal_matrix
from skfeature.utility.sparse_learning import calculate_l21_norm


def rfs(X, Y, **kwargs):
    """
    This function implementS efficient and robust feature selection via joint l21-norms minimization
    min_W||X^T W - Y||_2,1 + gamma||W||_2,1

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    Y: {numpy array}, shape (n_samples, n_classes)
        input class label matrix, each row is a one-hot-coding class label
    kwargs: {dictionary}
        gamma: {float}
            parameter in RFS
        verbose: boolean
            True if want to display the objective function value, false if not

    Output
    ------
    W: {numpy array}, shape(n_samples, n_features)
        feature weight matrix

    Reference
    ---------
    Nie, Feiping et al. "Efficient and Robust Feature Selection via Joint l2,1-Norms Minimization" NIPS 2010.
    """

    # default gamma is 1
    if 'gamma' not in kwargs:
        gamma = 1
    else:
        gamma = kwargs['gamma']
    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    n_samples, n_features = X.shape
    A = np.zeros((n_samples, n_samples + n_features))
    A[:, 0:n_features] = X
    A[:, n_features:n_features+n_samples] = gamma*np.eye(n_samples)
    D = np.eye(n_features+n_samples)

    max_iter = 1000
    obj = np.zeros(max_iter)
    for iter_step in range(max_iter):
        # update U as U = D^{-1} A^T (A D^-1 A^T)^-1 Y
        D_inv = LA.inv(D)
        temp = LA.inv(np.dot(np.dot(A, D_inv), A.T) + 1e-6*np.eye(n_samples))  # (A D^-1 A^T)^-1
        U = np.dot(np.dot(np.dot(D_inv, A.T), temp), Y)
        # update D as D_ii = 1 / 2 / ||U(i,:)||
        D = generate_diagonal_matrix(U)

        obj[iter_step] = calculate_obj(X, Y, U[0:n_features, :], gamma)

        if verbose:
            print('obj at iter {0}: {1}'.format(iter_step+1, obj[iter_step]))
        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step-1]) < 1e-3:
            break

    # the first d rows of U are the feature weights
    W = U[0:n_features, :]
    return W


def calculate_obj(X, Y, W, gamma):
    """
    This function calculates the objective function of rfs
    """
    temp = np.dot(X, W) - Y
    return calculate_l21_norm(temp) + gamma*calculate_l21_norm(W)