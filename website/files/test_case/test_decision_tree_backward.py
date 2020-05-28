import scipy.io
from sklearn.cross_validation import KFold
from skfeature.function.wrapper import decision_tree_backward
from sklearn import svm
from sklearn.metrics import accuracy_score


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape    # number of samples and number of features

    # split data into 10 folds
    ss = KFold(n_samples, n_folds=10, shuffle=True)

    # perform evaluation on classification task
    clf = svm.LinearSVC()    # linear SVM

    correct = 0
    for train, test in ss:
        # obtain the idx of selected features from the training set
        idx = decision_tree_backward.decision_tree_backward(X[train], y[train], n_features)

        # obtain the dataset on the selected features
        X_selected = X[:, idx]

        # train a classification model with the selected features on the training dataset
        clf.fit(X_selected[train], y[train])

        # predict the class labels of test data
        y_predict = clf.predict(X_selected[test])

        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc

    # output the average classification accuracy over all 10 folds
    print 'Accuracy:', float(correct)/10

if __name__ == '__main__':
    main()