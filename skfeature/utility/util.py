import numpy as np
from scipy.stats import rankdata
from random import shuffle

def reverse_argsort(X, size=None):
    """
    This function takes the indexes of features (0 being most important -1 being least)
    and converts them to a rank based system aligned with `sklearn.SelectKBest`
    
    Input
    -----
    X: {numpy array} shape(n_features, ), the indices of the feature with the first 
        element being most important and the last element being the least important
    
    Output
    ------
    F: {numpy array} ranking of the feature indices that are sklearn friendly
    
    """
    if size is None:
        X = np.array(X)
        return np.array(rankdata(-X)-1, dtype=np.int)
    
    else:
        # else we have to pad it with the values...
        X_all = list(range(size))
        X_raw = list(X)
        X_unseen = [x for x in X_all if x not in X_raw]
        shuffle(X_unseen)
        X_obj = X_raw + X_unseen
        return reverse_argsort(X_obj[:])
    
    
    