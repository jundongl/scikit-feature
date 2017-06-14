import numpy as np


def soft_threshold(A,b):
    """
    This function implement the soft-threshold operator
    Input:
        A: {numpy scalar, vector, or matrix}
        b: scalar}
    """
    res = np.zeros(A.shape)
    res[A > b] = A[A > b] - b
    res[A < -b] = A[A < -b] + b
    return res


def calculate_obj(X, y, w, lambda1, lambda2, T):
    return 1/2 * (np.linalg.norm(y- np.dot(X, w), 'fro'))**2 + lambda1*np.abs(w).sum() + lambda2*np.abs(np.dot(T, w)).sum()


def graph_fs(X, y, **kwargs):
    """
    This function implement the graph structural feature selection algorithm GOSCAR

    Objective Function
        min_{w} 1/2 ||X*w - y||_F^2 + lambda1 ||w||_1 + lambda2 \sum_{(i,j) \in E} max{|w_i|, |w|_j}

    Input:
        X: {numpy array}, shape (n_samples, n_features)
            Input data, guaranteed to be a numpy array
        y: {numpy array}, shape (n_samples, 1)
            Input data, the label matrix
        edge_list: {numpy array}, shape (n_edges, 2)
            Input data, each row is a pair of linked features, note feature index should start from 0
        lambda1: {float}
            Parameter lambda1 in objective function
        lambda2: {float}
            Parameter labmda2 in objective function
        rho: {flot}
            parameter used for optimization
        max_iter: {int}
            maximal iteration
        verbose: {boolean} True or False
            True if we want to print out the objective function value in each iteration, False if not

    Output:
        w: the weights of the features
        obj: the value of the objective function in each iteration
    """

    if 'lambda1' not in kwargs:
        lambda1 = 0.8
    else:
        lambda1 = kwargs['lambda1']
    if 'lambda2' not in kwargs:
        lambda2 = 0.8
    else:
        lambda2 = kwargs['lambda2']
    if 'edge_list' not in kwargs:
        print('Error using function, the network structure E is required')
        raise()
    else :
        edge_list = kwargs['edge_list']
    if 'max_iter' not in kwargs:
        max_iter = 300
    else:
        max_iter = kwargs['max_iter']
    if 'verbose' not in kwargs:
        verbose = 0
    else:
        verbose = kwargs['verbose']
    if 'rho' not in kwargs:
        rho = 5
    else:
        rho = kwargs['rho']

    n_samples, n_features = X.shape

    # construct T from E
    ind1 = edge_list[:, 0]
    ind2 = edge_list[:, 1]
    num_edge = ind1.shape[0]
    T = np.zeros((num_edge*2, n_features))
    for i in range(num_edge):
        T[i, ind1[i]] = 0.5
        T[i, ind2[i]] = 0.5
        T[i+num_edge, ind1[i]] = 0.5
        T[i+num_edge, ind2[i]] = -0.5

    # calculate F = X^T X + rho(I + T^T * T)
    F = np.dot(X.T, X) + rho*(np.identity(n_features) + np.dot(T.T, T))

    # Cholesky factorization of F = R^T R
    R = np.linalg.cholesky(F)  # NOTE, this return F = R R^T
    R = R.T
    Rinv = np.linalg.inv(R)
    Rtinv = Rinv.T

    # initialize p, q, mu , v to be zero vectors
    p = np.zeros((2*num_edge, 1))
    q = np.zeros((n_features, 1))
    mu = np.zeros((n_features, 1))
    v = np.zeros((2*num_edge, 1))

    # start the main loop
    iter = 0
    obj = np.zeros((max_iter,1))
    while iter < max_iter:
        print(iter)
        # update w
        b = np.dot(X.T, y) - mu - np.dot(T.T, v) + rho*np.dot(T.T,p) + rho*q
        w_hat = np.dot(Rtinv, b)
        w = np.dot(Rinv, w_hat)

        # update q
        q = soft_threshold(w + 1/rho*mu, lambda1/rho)
        # update p

        p = soft_threshold(np.dot(T, w)+1/rho*v, lambda2/rho)
        # update mu, v
        mu += rho*(w - q)
        v += rho*(np.dot(T, w) - p)

        # calculate objective function
        obj[iter] = calculate_obj(X, y, w, lambda1, lambda2, T)
        if verbose:
            print('obj at iter {0}: {1}'.format(iter, obj[iter]))
        iter += 1
    return w, obj, q

def feature_ranking(w):
    T = w.abs()
    idx = np.argsort(T, 0)
    return idx[::-1]
