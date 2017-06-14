import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def svm_backward(X, y, n_selected_features):
    """
    This function implements the backward feature selection algorithm based on SVM

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    n_selected_features: {int}
        number of selected features

    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features
    """

    n_samples, n_features = X.shape
    # using 10 fold cross validation
    cv = KFold(n_samples, n_folds=10, shuffle=True)
    # choose SVM as the classifier
    clf = SVC()

    # selected feature set, initialized to contain all features
    F = range(n_features)
    count = n_features

    while count > n_selected_features:
        max_acc = 0
        for i in range(n_features):
            if i in F:
                F.remove(i)
                X_tmp = X[:, F]
                acc = 0
                for train, test in cv:
                    clf.fit(X_tmp[train], y[train])
                    y_predict = clf.predict(X_tmp[test])
                    acc_tmp = accuracy_score(y[test], y_predict)
                    acc += acc_tmp
                acc = float(acc)/10
                F.append(i)
                # record the feature which results in the largest accuracy
                if acc > max_acc:
                    max_acc = acc
                    idx = i
        # delete the feature which results in the largest accuracy
        F.remove(idx)
        count -= 1
    return np.array(F)


