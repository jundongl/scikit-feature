import scipy.io
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from FS_package.function.sparse_learning_based import RFS
from FS_package.utility.sparse_learning import construct_label_matrix, feature_ranking


def main():
    # load MATLAB data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']    # data
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape
    X = X.astype(float)
    Y = construct_label_matrix(y)

    # split data
    cv = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)

    # evaluation
    n_selected_features = 100
    clf = svm.LinearSVC()
    correct = 0

    for train, test in cv:
        W = RFS.erfs(X[train, :], Y[train, :], gamma=0.1, verbose=True)
        idx = feature_ranking(W)
        X_selected = X[:, idx[0:n_selected_features]]
        clf.fit(X_selected[train, :], y[train])
        y_predict = clf.predict(X_selected[test, :])
        acc = accuracy_score(y[test], y_predict)
        print acc
        correct = correct + acc
    print 'ACC', float(correct)/10


if __name__ == '__main__':
    main()