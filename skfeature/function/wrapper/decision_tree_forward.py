import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


def decision_tree_forward(X, y, metric):
    """
    This function implements the forward feature selection algorithm based on decision tree

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples, )
        input class labels
    metric: metric to be used while performing forward selection
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features
    """

    n_samples, n_features = X.shape
    # using 5 fold stratified cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    # choose decision tree as the classifier
    clf = DecisionTreeClassifier()

    # selected feature set, initialized to be empty
    F = []
    count = 0
    max_acc = 0
    max_error = 100
    #This loop will keep on iterating till we find a set of features beyond which the f1-score/error does not improve
    while True:
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                X_tmp = X[:, F]
                acc = 0
                error = 0
                #Finding the f1-score/error after adding a particular feature
                for train, test in cv.split(X_tmp,y):
                    clf.fit(X_tmp[train], y[train])
                    y_predict = clf.predict(X_tmp[test])
                    if metric == "log-loss":
                        acc_tmp = log_loss(y[test], y_predict)
                        error += acc_tmp
                    else:
                        acc_tmp = f1_score(y[test], y_predict,average="micro")
                        acc += acc_tmp
                acc = float(acc)/5
                error = float(error)/5
                F.pop()
                # record the feature adding which results in the largest f1-score or the lowest error
                if metric == "log-loss" and error<max_error:
                    max_error = error
                    idx = i
                elif metric == "f1-score" and acc > max_acc:
                    max_acc = acc
                    idx = i

        # add the feature adding which results in the largest f1-score or the lowest score
        if idx!=-1:
            F.append(idx)
            count += 1
        else:
            break
    if metric =="log-loss":
        return (np.array(F),max_error)
    else:
        return (np.array(F),max_acc)
