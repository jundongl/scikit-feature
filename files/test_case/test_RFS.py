import scipy.io
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from skfeature.function.sparse_learning_based import RFS
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    Y = construct_label_matrix(y)
    n_samples, n_features = X.shape

    # split data into 10 folds
    ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)

    # perform evaluation on classification task
    num_fea = 100    # number of selected features
    clf = svm.LinearSVC()    # linear SVM

    correct = 0
    for train, test in ss:
        # obtain the feature weight matrix
        Weight = RFS.rfs(X[train, :], Y[train, :], gamma=0.1)

        # sort the feature scores in an ascending order according to the feature scores
        idx = feature_ranking(Weight)

        # obtain the dataset on the selected features
        selected_features = X[:, idx[0:num_fea]]

        # train a classification model with the selected features on the training dataset
        clf.fit(selected_features[train], y[train])

        # predict the class labels of test data
        y_predict = clf.predict(selected_features[test])

        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test], y_predict)
        print acc
        correct = correct + acc

    # output the average classification accuracy over all 10 folds
    print 'Accuracy:', float(correct)/10

if __name__ == '__main__':
    main()