import scipy.io
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score
from FS_package.function.statistics_based import gini_index
from sklearn import cross_validation


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    y = mat['gnd']
    y = y[:, 0]
    X = mat['fea']
    n_samples, n_features = X.shape
    X = X.astype(float)

    # split data
    ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)

    # cross validation
    num_fea = 100
    clf = svm.LinearSVC()
    correct = 0

    for train, test in ss:
        score = gini_index.gini_index(X[train], y[train])
        idx = gini_index.feature_ranking(score)
        selected_features = X[:, idx[0:num_fea]]
        clf.fit(selected_features[train], y[train])
        y_predict = clf.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        print acc
        correct = correct + acc
    print 'ACC', float(correct)/10

if __name__ == '__main__':
    main()

