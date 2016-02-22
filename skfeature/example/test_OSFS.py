import scipy.io
import numpy as np
from sklearn.metrics import accuracy_score
from skfeature.function.streaming import OSFS
from sklearn import cross_validation
from sklearn import svm


def main():
    # load data
    mat = scipy.io.loadmat('../data/colon.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    X = X.astype(int)
    y = y.astype(int)
    n_samples, n_features = X.shape    # number of samples and number of features

    # process the data to make its feature value larger than 0
    for i in range(n_features):
        unique_num = np.unique(X[:, i])
        max_num = np.max(unique_num)
        count = 1
        if max_num >= 1:
            for ele in unique_num:
                if ele < 1:
                    X[:, i][X[:, i] == ele] = max_num + count
                    count += 1
        else:
            for ele in unique_num:
                X[:, i][X[:, i] == ele] = count
                count += 1
    unique_y = np.unique(y)
    max_y = np.max(unique_y)
    count = 1
    if max_y >= 1:
        for ele in unique_y:
            if ele < 1:
                y[y == ele] = max_y + count
                count += 1
    else:
        for ele in unique_y:
            y[y == ele] = count
            count += 1

    # split data into 10 folds
    ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)

    # perform evaluation on classification task
    clf = svm.LinearSVC()    # linear SVM

    correct = 0
    for train, test in ss:
        # obtain the index of selected features
        idx = OSFS.OSFS(X[train], y[train])

        # obtain the dataset on the selected features
        selected_features = X[:, idx]

        # train a classification model with the selected features on the training dataset
        clf.fit(selected_features[train], y[train])

        # predict the class labels of test data
        y_predict = clf.predict(selected_features[test])

        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc

    # output the average classification accuracy over all 10 folds
    print 'Accuracy:', float(correct)/10

if __name__ == '__main__':
    main()