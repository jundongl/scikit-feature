import csv
import numpy as np
from sklearn.cross_validation import KFold
from FS_package.function.wrapper import decision_tree_forward
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main():
    # obtain the number of features in the dataset
    with open('../data/test_lung_s3.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            num_columns = len(row)
            break

    # load data
    mat = np.loadtxt('../data/test_lung_s3.csv', delimiter=',', skiprows=1, usecols=range(0, num_columns))
    X = mat[:, 1:num_columns]  # data
    y = mat[:, 0]  # label
    X = X.astype(float)
    y = y.astype(float)
    n_samples, n_features = X.shape

    # using 10 fold cross validation
    cv = KFold(n_samples, n_folds=10, shuffle=True)

    # evaluation
    n_features = 10
    neigh = KNeighborsClassifier(n_neighbors=1)
    acc = 0

    for train, test in cv:
        idx = decision_tree_forward.decision_tree_forward(X[train], y[train], n_features)
        print idx
        X_selected = X[:, idx]
        neigh.fit(X_selected[train], y[train])
        y_predict = neigh.predict(X_selected[test])
        acc_tmp = accuracy_score(y[test], y_predict)
        print acc_tmp
        acc += acc_tmp
    print 'ACC', float(acc)/10


if __name__ == '__main__':
    main()
