from nose.tools import *
import scipy.io
from sklearn import svm
from sklearn.metrics import accuracy_score
from skfeature.function.statistical_based import CFS
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from skfeature.function.wrapper import decision_tree_backward, decision_tree_forward

def test_decision_tree_backward():
    # load data
    mat = scipy.io.loadmat('./data/COIL20.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    n_samples, n_features = X.shape    # number of samples and number of features

    # reduce cols to speed up test - rather than wait a minute
    X = X[:, :100]    
    num_fea = 10
    
    # split data into 10 folds
    kfold = KFold(n_splits=10, shuffle=True)
    
    # build pipeline
    pipeline = []
    pipeline.append(('select top k', SelectKBest(score_func=decision_tree_backward.decision_tree_backward, k=num_fea)))
    pipeline.append(('linear svm', svm.LinearSVC()))
    model = Pipeline(pipeline)
    
    results = cross_val_score(model, X, y, cv=kfold)
    print("Accuracy: {}".format(results.mean()))
    assert_true(results.mean() > 0.1)

def test_decision_tree_forward():
    # load data
    mat = scipy.io.loadmat('./data/COIL20.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:100, 0]
    n_samples, n_features = X.shape    # number of samples and number of features

    # reduce cols to speed up test - rather than wait a minute
    X = X[:100, :30]    
    num_fea = 10
    
    # split data into 10 folds
    kfold = KFold(n_splits=10, shuffle=True)
    
    # build pipeline
    pipeline = []
    pipeline.append(('select top k', SelectKBest(score_func=decision_tree_forward.decision_tree_forward, k=num_fea)))
    pipeline.append(('linear svm', svm.LinearSVC()))
    model = Pipeline(pipeline)
    
    print(X.shape)
    print(y.shape)
    
    results = cross_val_score(model, X, y, cv=kfold)
    print("Accuracy: {}".format(results.mean()))
    assert_true(results.mean() > 0.1)
