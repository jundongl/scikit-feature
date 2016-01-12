import numpy as np
from numpy import linalg as LA


def feature_ranking(W):
    """
    This function ranks features according to the feature weights matrix W

    Input:
    -----
    W: {numpy array}, shape (n_features, n_classes)
        feature weights matrix

    Output:
    ------
    idx: {numpy array}, shape {n_features,}
        feature index ranked in descending order by feature importance
    """
    T = (W*W).sum(1)
    idx = np.argsort(T, 0)
    return idx[::-1]


def generate_diagonal_matrix(U):
    """
    This function generates a diagonal matrix D from an input matrix U as D_ii = 0.5 / ||U[i,:]||

    Input:
    -----
    U: {numpy array}, shape (n_samples, n_features)

    Output:
    ------
    D: {numpy array}, shape (n_samples, n_samples)
    """
    temp = np.sqrt(np.multiply(U, U).sum(1))
    temp[temp < 1e-16] = 1e-16
    temp = 0.5 / temp
    D = np.diag(temp)
    return D


def calculate_l21_norm(X):
    """
    This function calculates the l21 norm of a matrix X, i.e., \sum ||X[i,:]||_2

    Input:
    -----
    X: {numpy array}, shape (n_samples, n_features)

    Output:
    ------
    l21_norm: {float}
    """
    return (np.sqrt(np.multiply(X, X).sum(1))).sum()


def construct_label_matrix(label):
    """
    This function converts a 1d numpy array to a 2d array, for each instance, the class label is 1 or 0

    Input:
    -----
    label: {numpy array}, shape(n_samples,)

    Output:
    ------
    label_matrix: {numpy array}, shape(n_samples, n_classes)
    """

    n_samples = label.shape[0]
    unique_label = np.unique(label)
    n_classes = unique_label.shape[0]
    label_matrix = np.zeros((n_samples, n_classes))
    for i in range(n_classes):
        label_matrix[label == unique_label[i], i] = 1

    return label_matrix.astype(int)


def construct_label_matrix_pan(label):
    """
    This function converts a 1d numpy array to a 2d array, for each instance, the class label is 1 or -1

    Input:
    -----
    label: {numpy array}, shape(n_samples,)

    Output:
    ------
    label_matrix: {numpy array}, shape(n_samples, n_classes)
    """
    n_samples = label.shape[0]
    unique_label = np.unique(label)
    n_classes = unique_label.shape[0]
    label_matrix = np.zeros((n_samples, n_classes))
    for i in range(n_classes):
        label_matrix[label == unique_label[i], i] = 1
    label_matrix[label_matrix == 0] = -1

    return label_matrix.astype(int)


def euclidean_projection(V, n_features, n_classes, z, gamma):
    """
    L2 Norm regularized euclidean projection min_W  1/2 ||W- V||_2^2 + z * ||W||_2
    """
    W_projection = np.zeros((n_features, n_classes))
    for i in range(n_features):
        if LA.norm(V[i, :]) > z/gamma:
            W_projection[i, :] = (1-z/(gamma*LA.norm(V[i, :])))*V[i, :]
        else:
            W_projection[i, :] = np.zeros(n_classes)
    return W_projection


def tree_lasso_projection(v, n_features, idx, n_nodes):
    """
    This functions solves the following optimization problem min_w 1/2 ||w-v||_2^2 + \sum z_i||w_{G_{i}}||
    where w and v are of dimensions of n_features; z_i >=0, and G_{i} follows the tree structure
    """
    # test whether the first node is special
    if idx[0, 0] == -1 and idx[1, 0] == -1:
        w_projection = np.zeros(n_features)
        z = idx[2, 0]
        for j in range(n_features):
            if v[j] > z:
                w_projection[j] = v[j] - z
            else:
                if v[j] < -z:
                    w_projection[j] = v[j] + z
                else:
                    w_projection[j] = 0
        i = 1

    else:
        w = v.copy()
        i = 0

    # sequentially process each node
    while i < n_nodes:
        # compute the L2 norm of this group
        two_norm = 0
        start_idx = int(idx[0, i] - 1)
        end_idx = int(idx[1, i])
        for j in range(start_idx, end_idx):
            two_norm += w_projection[j] * w_projection[j]
        two_norm = np.sqrt(two_norm)
        z = idx[2, i]
        if two_norm > z:
            ratio = (two_norm - z) / two_norm
            # shrinkage this group by ratio
            for j in range(start_idx, end_idx):
                w_projection[j] *= ratio
        else:
            for j in range(start_idx, end_idx):
                w_projection[j] = 0
        i += 1
    return w_projection


def tree_norm(w, n_features, idx, n_nodes):
    """
    This function computes \sum z_i||w_{G_{i}}||
    """
    obj = 0
    # test whether the first node is special
    if idx[0, 0] == -1 and idx[1, 0] == -1:
        z = idx[2, 0]
        for j in range(n_features):
            obj += np.abs(w[j])
        obj *= z
        i = 1
    else:
        i = 0

    # sequentially process each node
    while i < n_nodes:
        two_norm = 0
        start_idx = int(idx[0, i] - 1)
        end_idx = int(idx[1, i])
        for j in range(start_idx, end_idx):
            two_norm += w[j] * w[j]
        two_norm = np.sqrt(two_norm)
        z = idx[2, i]
        obj += z*two_norm
        i += 1
    return obj

