import scipy.io
from skfeature.function.sparse_learning_based import MCFS
from skfeature.utility import construct_W
from skfeature.utility import unsupervised_evaluation


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]

    # construct affinity matrix
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs)

    num_fea = 100    # specify the number of selected features
    num_cluster = 20    # specify the number of clusters, it is usually set as the number of classes in the ground truth

    # obtain the feature weight matrix
    Weight = MCFS.mcfs(X, n_selected_features=num_fea, W=W, n_clusters=20)

    # sort the feature scores in an ascending order according to the feature scores
    idx = MCFS.feature_ranking(Weight)

    # obtain the dataset on the selected features
    selected_features = X[:, idx[0:num_fea]]

    # perform kmeans clustering based on the selected features and repeats 20 times
    nmi_total = 0
    acc_total = 0
    for i in range(0, 20):
        nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=num_cluster, y=y)
        nmi_total += nmi
        acc_total += acc

    # output the average NMI and average ACC
    print 'NMI:', float(nmi_total)/20
    print 'ACC:', float(acc_total)/20

if __name__ == '__main__':
    main()