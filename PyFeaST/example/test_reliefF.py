import scipy.io
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from FS_package.function.similarity_based import reliefF


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']      # data
    y = mat['Y']  # label
    y = y[:, 0]
    X = X.astype(float)

    # normalize data, necessary!
    X = preprocessing.normalize(X, norm='l2', axis=0)
    n_samples, n_features = X.shape

    # split data
    ss = cross_validation.KFold(n_samples, n_folds=10)

    # evaluation
    num_fea = 100
    neigh = KNeighborsClassifier(n_neighbors=1)
    correct = 0

    for train, test in ss:
        score = reliefF.reliefF(X[train], y[train])
        idx = reliefF.feature_ranking(score)
        selected_features = X[:, idx[0:num_fea]]
        print selected_features
        neigh.fit(selected_features[train], y[train])
        y_predict = neigh.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        print acc
        correct = correct + acc
    print 'ACC', float(correct)/10


if __name__ == '__main__':
    main()