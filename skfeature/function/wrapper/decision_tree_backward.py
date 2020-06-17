import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score


def decision_tree_backward(X, y,metric):
    """
    This function implements the backward feature selection algorithm based on decision tree

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels

    metric: metric to be while performing backward selection

    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features
    """

    n_samples, n_features = X.shape
    # using 5 fold stratified cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True,random_state=1)
    # choose decision tree as the classifier
    clf = DecisionTreeClassifier()

    # selected feature set, initialized to contain all features
    F = list(range(n_features))
    count = n_features
    acc = 0
    #Finding the f1-score/error of the initial set of features
    for train, test in cv.split(X,y):
        clf.fit(X[train], y[train])
        y_predict = clf.predict(X[test])
        if metric == "log-loss":
            acc_tmp = log_loss(y[test], y_predict)
        else:
            acc_tmp = f1_score(y[test],y_predict,average="micro")
        acc += acc_tmp
    max_acc = float(acc)/5
    #This loop will keep on iterating till we find a set of features beyond which the f1-score/error does not improve
    while True:

        idx = -1
        for i in range(n_features):
            if i in F:
                F.remove(i)
                X_tmp = X[:, F]
                acc = 0
                #Finding the f1-score/error after removing a particular feature
                for train, test in cv.split(X_tmp,y):
                    clf.fit(X_tmp[train], y[train])
                    y_predict = clf.predict(X_tmp[test])
                    if metric == "log-loss":
                        acc_tmp = log_loss(y[test], y_predict)
                    else:
                        acc_tmp = f1_score(y[test],y_predict,average="micro")
                    acc += acc_tmp
                acc = float(acc)/5
                F.append(i)
                # record the feature, removing which results in the largest f1-score or the smallest error
                if metric == "log-loss" and acc < max_acc:
                    max_acc = acc
                    idx = i
                elif metric == "f1-score" and acc>max_acc:
                    max_acc = acc
                    idx = i
        # delete the feature, removing which results in the largest f1-score or the smallest error
        if idx!=-1:
            F.remove(idx)
            count -= 1
        else:
            break
    return (np.array(F),max_acc)
