from nose.tools import *
import scipy.io
from skfeature.function.streaming import alpha_investing
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

def test_alphainvesting():
    # load data
    mat = scipy.io.loadmat('./data/COIL20.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    y = y.astype(float)
    n_samples, n_features = X.shape    # number of samples and number of features

    # split data into 10 folds
    kfold = KFold(n_splits=10, shuffle=True)
    
    # build pipeline
    pipeline = []
    pipeline.append(('alphainvesting', alpha_investing.AlphaInvesting(w0=0.05, dw=0.05)))
    pipeline.append(('linear svm', svm.LinearSVC()))
    model = Pipeline(pipeline)
    
    results = cross_val_score(model, X, y, cv=kfold)
    assert_true(results.mean() > 0.95)

