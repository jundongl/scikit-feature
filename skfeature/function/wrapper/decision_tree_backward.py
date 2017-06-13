import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from skfeature.utility.util import reverse_argsort


def decision_tree_backward(X, y, mode="rank", n_selected_features=None):
    """
    This function implements the backward feature selection algorithm based on decision tree

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    n_selected_features : {int}
        number of selected features

    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features
    """

    n_samples, n_features = X.shape
    if n_selected_features is None:
        n_selected_features = n_features
    # using 10 fold cross validation
    kfold = KFold(n_splits=10, shuffle=True)
    # choose decision tree as the classifier
    clf = DecisionTreeClassifier()

    # selected feature set, initialized to contain all features
    F = range(n_features)
    count = n_features

    while count > n_selected_features:
        max_acc = 0
        for i in range(n_features):
            if i in F:
                F.remove(i)
                X_tmp = X[:, F]                
                results = cross_val_score(clf, X_tmp, y, cv=kfold)
                acc = results.mean()    
                F.append(i)
                # record the feature which results in the largest accuracy
                if acc > max_acc:
                    max_acc = acc
                    idx = i
        # delete the feature which results in the largest accuracy
        F.remove(idx)
        count -= 1
    if mode == "index":
        return np.array(F)
    else:
        return reverse_argsort(F)


