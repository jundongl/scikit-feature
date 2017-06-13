import numpy as np
from scipy.stats import rankdata
import random 

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
    def dedup(seq):
        """
        Based on uniqifiers benchmarks
        https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order
        """
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
    
    X = dedup(list(X))
    if size is None:
        X = np.array(X)
        return np.array(rankdata(-X)-1, dtype=np.int)
    
    else:
        # else we have to pad it with the values...
        X_all = list(range(size))
        X_raw = X[:]
        X_unseen = [x for x in X_all if x not in X_raw]
        X_unseen = list(set(X_unseen))
        X_unseen_shuffle = random.sample(X_unseen[:], len(X_unseen))
        X_obj = X_raw + X_unseen_shuffle
        return reverse_argsort(X_obj[:])
    
    
    