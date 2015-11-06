import numpy as np
from FS_package.utility import cond_indep_G2


def subset_list(s_size):
    """
    This function returns all subsets of a input set except empty set

    Input
    -----
    s: {numpy array}, shape (n_features,)

    Output
    ------
    subset: {numpy array}, shape (2^s_size - 1, s_size)
        all subsets of a input set except empty set (in each subset, the element is either true or false)
        for example, if s_size = 5, and the subset is (1,3,4), then the corresponding returned subset value is
        (True, False, True, True, False)
    """
    if s_size == 0:
        s_subset = []
    else:
        num_subset = 2 ** s_size - 1
        s_subset = np.zeros((num_subset, s_size))
        s_subset = s_subset.astype(bool)
        for num in range(num_subset):
            if num != 0:
                position = 0
                while num:
                    num, bi = num/2, num % 2
                    if bi == 1:
                        s_subset[num, position] = True
                    position += 1
    return s_subset


def OSFS(X, y):
    """
    This function implements the streamwise online streaming feature selection algorithm (OSFS)

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, assume feature arrives one at each time step
    y: {numpy array},shape (n_samples,)
        input class labels

    Output
    ------
    BCF: {numpy array}, shape (n_features,)
        indexes of the selected features in a streamwise way
    """
    n_samples, n_features = X.shape
    # indexes of the best candidate features so far
    BCF = []
    count = 1
    empty = []
    data = y
    while count < n_features:
        # online relevance analysis
        # If added=1, feature fi is relevance to class label.
        # Otherwise if added=0, feature fi is not relevant to the class label.
        added = 0
        fi = X[:, count-1]
        data = np.append(data, fi)
        data = np.reshape(data, (n_samples, len(data)/n_samples))
        ci = cond_indep_G2.cond_indep_G2(count, 0, empty, data)
        if ci == 0:
            BCF.append(count)
            added = 1
        # online redundancy analysis
        if added:
            subset_index = subset_list(len(BCF)-1)
            n_subsets = (2 ** (len(BCF)-1)-1)
            # indexes of feature in BCF which need to be deleted
            delete_list = []
            num_delete_list_pre = 0
            for f in BCF:
                if f not in delete_list:
                    if len(delete_list) != 0:
                        temp_delete_list = delete_list
                        temp_delete_list.append(f)
                        temp_BCF = BCF
                        np.delete(temp_BCF, temp_delete_list)
                        if num_delete_list_pre != len(delete_list):
                            num_delete_list_pre = len(delete_list)
                            subset_index = subset_list(len(temp_BCF))
                            n_subsets = 2 ** (len(temp_BCF)) - 1
                    for i in range(n_subsets):
                        subset = BCF[subset_index[i, :]]
                        ci = cond_indep_G2.cond_indep_G2(BCF[f], 0, subset, data)
                        if ci == 1:
                            delete_list.append(f)
                            break
            np.delete(BCF, delete_list)
            return np.array(BCF)


def main():
    print subset_list(5)


if __name__ == '__main__':
    main()