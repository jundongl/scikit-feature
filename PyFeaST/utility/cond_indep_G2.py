import numpy as np
import csv
from scipy import special


def chi_squared_prob(x2, v):
    """
    This function computes the chi-squared probability.
    It returns P(X2|v), the probability of observing a chi-squared value <= X2 with v degrees of freedom
    """
    return special.gammainc(v/2, float(x2)/2)


def cond_indep_G2(X, a, b, s):
    """
    This function test if a is independent with b given s using likelihood ratio test G2

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a discrete data matrix
    a: {int}
        index of variable a in the input X
    b: {int}
        index of variable y in Data matrix
    s: {numpy array}, shape (n_features,)
        indexes of variables in set s

    Output
    ------
        ci: {int}
            test result (1 = conditional independent, 0 = no)
        G2: {float}
            G2 value(-1 if not enough data to perform the test, i.e., CI = 0)

    Reference
    ---------
    Leray, Philippe and Francois, Olivier. "BNT Structure Learning Package: Documentation and Experiments" 2004
    """

    alpha = 0.05
    n_samples, n_features = X.shape
    col_max = X.max(axis=0)
    s_max = np.array([col_max[i] for i in s])

    if len(s) == 0:
        n_ij = np.zeros((col_max[a], col_max[b]))
        t_ij = np.zeros((col_max[a], col_max[b]))
        df = np.prod(np.array([col_max[i] for i in np.array([a, b])])-1) * np.prod(s_max)
    else:
        tmp = np.zeros(len(s_max))
        tmp[0] = 1
        tmp[1:] = np.cumprod(s_max[0:len(s_max)-1])
        qs = 1 + np.dot(s_max-1, tmp)
        n_ijk = np.zeros((col_max[a], col_max[a], qs))
        t_ijk = np.zeros((col_max[b], col_max[b], qs))
        df = np.prod(np.array([col_max[i] for i in np.array([a, b])])-1)*qs

    if n_samples < 10*df:
        # not enough data to perform the test
        G2 = -1
        ci = 0
    elif len(s) == 0:
        for i in range(col_max[a]):
            for j in range(col_max[b]):
                col = X[:, 0]
                n_ij[i, j] = len(col[(X[:, a] == i+1) & (X[:, b] == j+1)])
        col_sum_nij = np.sum(n_ij, axis=0)
        if len(col_sum_nij[col_sum_nij == 0]) != 0:
            temp_nij = np.array([])
            for i in range(len(col_sum_nij)):
                if col_sum_nij[i] != 0:
                    temp_nij = np.append(temp_nij, n_ij[:, i])
            temp_nij = temp_nij.reshape((len(temp_nij)/len(n_ij[:, 0], len(n_ij[:, 0]))))
            n_ij = np.transpose(temp_nij)
        row = np.sum(n_ij, axis=1)
        col = np.sum(n_ij, axis=0)
        for i in range(len(row)):
            for j in range(len(col)):
                t_ij[i, j] = float(row[i] * col[j])/n_samples
        tmp = np.zeros((col_max[a], col_max[b]))
        for i in range(col_max[a]):
            for j in range(col_max[b]):
                if t_ij[i, j] == 0:
                    tmp[i, j] = float('Inf')
                else:
                    tmp[i, j] = n_ij[i, j] / t_ij[i, j]
        tmp[(tmp == float('Inf')) | (tmp == 0)] = 1
        tmp = 2 * n_ij * np.log(tmp)

        G2 = np.sum(tmp)
        alpha2 = 1 - chi_squared_prob(G2, df)
        ci = (alpha2 >= alpha)
    else:
        for example in range(n_samples):
            i = X[example, a]
            j = X[example, b]
            si = X[example, [element for element in s]]-1
            k = int(1+np.dot(si, tmp))
            n_ijk[i-1, j-1, k-1] += 1
        n_ik = np.sum(n_ijk, axis=1)
        n_jk = np.sum(n_ijk, axis=0)
        n2 = np.sum(n_jk, axis=0)
        tmp = np.zeros((col_max[a], col_max[b], qs))
        for k in range(int(qs)):
            if n2[k] == 0:
                t_ijk[:, :, k] = 0
            else:
                for i in range(col_max[a]):
                    for j in range(col_max[b]):
                        if n2[k] == 0:
                            t_ijk[i, j, k] = float('Inf')
                        else:
                            t_ijk[i, j, k] = n_ik[i, k]*n_jk[j, k]/n2[k]
                        if t_ijk[i, j, k] == 0:
                            tmp[i, j, k] = float('Inf')
                        else:
                            tmp[i, j, k] = n_ijk[i, j, k] / t_ijk[i, j, k]
        tmp[(tmp == float('Inf')) | (tmp == 0)] = 1
        tmp = 2 * n_ijk * np.log(tmp)
        G2 = np.sum(tmp)
        alpha2 = 1 - chi_squared_prob(G2, df)
        ci = (alpha2 >= alpha)

    return int(ci), G2