import numpy as np
from PyFeaST.utility import cond_indep_G2


def subset_list(s_size):
    """
    This function returns all subsets of an input set except empty set

    Input
    -----
    s_size: {int}, number of elements in input set.

    Output
    ------
    s_subset: {numpy array}, shape (2^s_size - 1, s_size)
        All subsets of an input set except empty set (in each subset, the element is index of )
        For example, if s_size = 2, there will be 3 subsets of the input set.Thus, s_subset[0] = [0],s_subset[1] = [1],
        s_subset[2]=[0,1](here, 0 and 1 represent the index of elements in input set)
    """
    # if input set is an empty set, its subset is also empty set.
    if s_size == 0:
        s_subset = []
    else:
        # except empty set, number of subsets for an input set with size of s_size is 2 ** s_size-1
        num_subset = 2 ** s_size-1
        s_subset = [[] for i in range(num_subset)]
        for num in range(1,num_subset+1):
            number = num
            position = 0
            while num:
                num, bi = num/2, num % 2
                if bi == 1:
                    s_subset[number-1].append(s_size-1-position)
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
        indexes of the selected features in a streaming way
    """
    n_samples, n_features = X.shape
    # a list contains indexes of the best candidate features so far
    BCF = []
    count = 1
    empty = []
    temp_data = y
    while count < n_features:
        # online relevance analysis
        # If added=1, feature fi is relevance to class label.
        # Otherwise if added=0, feature fi is not relevant to the class label.
        added = 0
        # add features to temp_data one by one (simulate streaming data)
        fi = X[:, count-1]
        temp_data = np.append(temp_data,fi)
        # reshape temp_data into a data matrix with size of (n_samples,n_features)
        data = np.reshape(temp_data, (len(temp_data)/n_samples ,n_samples))
        data = np.transpose(data)
        # check if new coming feature is relevance to class label (1 = conditional independent, 0 = dependent)
        ci = cond_indep_G2.cond_indep_G2(data, count, 0, empty)
        # add new coming feature into BCF if it is relevance to class label
        if ci == 0:
            BCF.append(count)
            added = 1
        # online redundancy analysis
        if added:
            delete_list = []
            # check if exist a subset S of BCF-ele, s.t.Ind(ele,label|S), then remove ele from BCF
            for ele in BCF:
                if ele not in delete_list:
                    temp_BCF = [i for i in BCF]
                    temp_BCF.remove(ele)
                    for delete_ele in delete_list:
                        temp_BCF.remove(delete_ele)
                    subset_index = subset_list(len(temp_BCF))
                    is_delete = False
                    for i in range(len(subset_index)):
                        subset = [temp_BCF[idx] for idx in subset_index[i]]
                        ci = cond_indep_G2.cond_indep_G2(data, ele, 0, subset)
                        if ci == 1:
                            is_delete = True
                            break
                    if is_delete is True:
                        delete_list.append(ele)
            for ele in delete_list:
                BCF.remove(ele)
        count += 1
    #  return indexes of the selected features in a streamwise way
    return np.array(BCF)