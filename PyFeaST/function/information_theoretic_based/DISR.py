from FS_package.utility.entropy_estimators import *
from FS_package.utility.mutual_information import conditional_entropy


def disr(X, y, **kwargs):
    """
    This function implement the DISR feature selection.
    The scoring criteria is calculated based on the formula j_disr=sum_j(I(f,fj;y)/H(f,fj,y))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a discrete data matrix
    y: {numpy array}, shape (n_samples,)
        input class labels

    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features, F[1] is the most important feature

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """

    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False

    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True

    # sum stores sum_j(I(f,fj;y)/H(f,fj,y)) for each feature f
    sum = np.zeros(n_features)

    # make sure that j_cmi is positive at the very beginning
    j_disr = 1

    while True:
        if len(F) == 0:
            # t1 stores I(f;y) for each feature f
            t1 = np.zeros(n_features)
            for i in range(n_features):
                f = X[:, i]
                t1[i] = midd(f, y)
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            f_select = X[:, idx]

        if is_n_selected_features_specified is True:
            if len(F) == n_selected_features:
                break
        if is_n_selected_features_specified is not True:
            if j_disr <= 0:
                break

        # we assign an extreme small value to j_disr to ensure that it is smaller than all possible value of j_disr
        j_disr = -1000000000000
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t1 = midd(f_select, y) + cmidd(f, y, f_select)
                t2 = entropyd(f) + conditional_entropy(f_select, f) + (conditional_entropy(y, f_select) - ee.cmidd(y, f, f_select))
                sum[i] += np.true_divide(t1, t2)
                # record the largest j_disr and the corresponding feature index
                if sum[i] > j_disr:
                    j_disr = sum[i]
                    idx = i
        F.append(idx)
        f_select = X[:, idx]

    return np.array(F)

