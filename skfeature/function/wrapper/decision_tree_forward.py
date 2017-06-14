import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def decision_tree_forward(X, y, n_selected_features):
    """
    This function implements the forward feature selection algorithm based on decision tree

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples, )
        input class labels
    n_selected_features: {int}
        number of selected features

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features
    """

    n_samples, n_features = X.shape
    # using 10 fold cross validation
    cv = KFold(n_samples, n_folds=10, shuffle=True)
    # choose decision tree as the classifier
    clf = DecisionTreeClassifier()

    # selected feature set, initialized to be empty
    F = []
    count = 0
    while count < n_selected_features:
        max_acc = 0
        for i in range(n_features):
            if i not in F:
                F.append(i)
                X_tmp = X[:, F]
                acc = 0
                for train, test in cv:
                    clf.fit(X_tmp[train], y[train])
                    y_predict = clf.predict(X_tmp[test])
                    acc_tmp = accuracy_score(y[test], y_predict)
                    acc += acc_tmp
                acc = float(acc)/10
                F.pop()
                # record the feature which results in the largest accuracy
                if acc > max_acc:
                    max_acc = acc
                    idx = i
        # add the feature which results in the largest accuracy
        F.append(idx)
        count += 1
    return np.array(F)

