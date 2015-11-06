import scipy.io
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score
from FS_package.function.similarity_based import trace_ratio


def main():
    # load matlab data
    mat = scipy.io.loadmat('../data/ORL.mat')
    X = mat['X']  # data
    y = mat['Y']  # label
    y = y[:, 0]
    X = X.astype(float)
    n_samples, n_features = X.shape

    # split data
    ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)

    # evaluation
    num_fea = 100
    clf = svm.SVC(kernel='linear')
    correct = 0

    for train, test in ss:
        idx, feature_score, subset_score = trace_ratio.trace_ratio(X[train], y[train], num_fea, style='fisher')
        selected_features = X[:, idx[0:num_fea]]
        clf.fit(selected_features[train], y[train])
        y_predict = clf.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        print acc
        correct = correct + acc
    print 'ACC', float(correct)/10


if __name__ == '__main__':
    main()
