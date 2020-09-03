import scipy.io
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn import svm
from skfeature.function.information_theoretical_based import CIFE


def main():
    # load data
    mat = scipy.io.loadmat('../data/colon.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape    # number of samples and number of features

    # split data into 10 folds
    ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)

    # perform evaluation on classification task
    num_fea = 10    # number of selected features
    clf = svm.LinearSVC()    # linear SVM

    correct = 0
    for train, test in ss:
        # obtain the index of each feature on the training set
        idx = CIFE.cife(X[train], y[train], n_selected_features=num_fea)

        # obtain the dataset on the selected features
        features = X[:, idx[0:num_fea]]

        # train a classification model with the selected features on the training dataset
        clf.fit(features[train], y[train])

        # predict the class labels of test data
        y_predict = clf.predict(features[test])

        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test], y_predict)
        correct = correct + acc

    # output the average classification accuracy over all 10 folds
    print 'Accuracy:', float(correct)/10

if __name__ == '__main__':
    main()
