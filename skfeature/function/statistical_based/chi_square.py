import numpy as np
from sklearn.feature_selection import chi2
from scipy.stats import rankdata

def chi_square(X, y, mode="rank"):
    """
    This function implements the chi-square feature selection (existing method for classification in scikit-learn)

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array},shape (n_samples,)
        input class labels
    mode: whether this function should return the "raw" scores, "rank" scores, 
        or ordered "index".

        "raw" will be the raw score if statistical. 
        "rank" will be index ordering for compatibility with sklearn. 
        "index" will be the original method shown in this repository.

    Output
    ------
    F: {numpy array}, shape (n_features,)
        chi-square score for each feature
    """
    if mode not in ['rank', 'raw', 'index']:
        print('mode is not one of "rank", "raw", "index"')
        raise()
    
    F, pval = chi2(X, y)
    
    if mode == "raw":
        return F
    elif mode == 'rank':
        return rankdata(F)
    else:
        idx = np.argsort(F)
        return idx[::-1]
        
