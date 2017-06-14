import numpy as np
from skfeature.utility.construct_W import construct_W


def trace_ratio(X, y, n_selected_features, **kwargs):
    """
    This function implements the trace ratio criterion for feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    n_selected_features: {int}
        number of features to select
    kwargs: {dictionary}
        style: {string}
            style == 'fisher', build between-class matrix and within-class affinity matrix in a fisher score way
            style == 'laplacian', build between-class matrix and within-class affinity matrix in a laplacian score way
        verbose: {boolean}
            True if user want to print out the objective function value in each iteration, False if not

    Output
    ------
    feature_idx: {numpy array}, shape (n_features,)
        the ranked (descending order) feature index based on subset-level score
    feature_score: {numpy array}, shape (n_features,)
        the feature-level score
    subset_score: {float}
        the subset-level score

    Reference
    ---------
    Feiping Nie et al. "Trace Ratio Criterion for Feature Selection." AAAI 2008.
    """

    # if 'style' is not specified, use the fisher score way to built two affinity matrix
    if 'style' not in kwargs.keys():
        kwargs['style'] = 'fisher'
    # get the way to build affinity matrix, 'fisher' or 'laplacian'
    style = kwargs['style']
    n_samples, n_features = X.shape

    # if 'verbose' is not specified, do not output the value of objective function
    if 'verbose' not in kwargs:
        kwargs['verbose'] = False
    verbose = kwargs['verbose']

    if style is 'fisher':
        kwargs_within = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
        # build within class and between class laplacian matrix L_w and L_b
        W_within = construct_W(X, **kwargs_within)
        L_within = np.eye(n_samples) - W_within
        L_tmp = np.eye(n_samples) - np.ones([n_samples, n_samples])/n_samples
        L_between = L_within - L_tmp

    if style is 'laplacian':
        kwargs_within = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        # build within class and between class laplacian matrix L_w and L_b
        W_within = construct_W(X, **kwargs_within)
        D_within = np.diag(np.array(W_within.sum(1))[:, 0])
        L_within = D_within - W_within
        W_between = np.dot(np.dot(D_within, np.ones([n_samples, n_samples])), D_within)/np.sum(D_within)
        D_between = np.diag(np.array(W_between.sum(1)))
        L_between = D_between - W_between

    # build X'*L_within*X and X'*L_between*X
    L_within = (np.transpose(L_within) + L_within)/2
    L_between = (np.transpose(L_between) + L_between)/2
    S_within = np.array(np.dot(np.dot(np.transpose(X), L_within), X))
    S_between = np.array(np.dot(np.dot(np.transpose(X), L_between), X))

    # reflect the within-class or local affinity relationship encoded on graph, Sw = X*Lw*X'
    S_within = (np.transpose(S_within) + S_within)/2
    # reflect the between-class or global affinity relationship encoded on graph, Sb = X*Lb*X'
    S_between = (np.transpose(S_between) + S_between)/2

    # take the absolute values of diagonal
    s_within = np.absolute(S_within.diagonal())
    s_between = np.absolute(S_between.diagonal())
    s_between[s_between == 0] = 1e-14  # this number if from authors' code

    # preprocessing
    fs_idx = np.argsort(np.divide(s_between, s_within), 0)[::-1]
    k = np.sum(s_between[0:n_selected_features])/np.sum(s_within[0:n_selected_features])
    s_within = s_within[fs_idx[0:n_selected_features]]
    s_between = s_between[fs_idx[0:n_selected_features]]

    # iterate util converge
    count = 0
    while True:
        score = np.sort(s_between-k*s_within)[::-1]
        I = np.argsort(s_between-k*s_within)[::-1]
        idx = I[0:n_selected_features]
        old_k = k
        k = np.sum(s_between[idx])/np.sum(s_within[idx])
        if verbose:
            print('obj at iter {0}: {1}'.format(count+1, k))
        count += 1
        if abs(k - old_k) < 1e-3:
            break

    # get feature index, feature-level score and subset-level score
    feature_idx = fs_idx[I]
    feature_score = score
    subset_score = k

    return feature_idx, feature_score, subset_score



