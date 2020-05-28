import scipy.io
from skfeature.function.sparse_learning_based import UDFS
from skfeature.utility import unsupervised_evaluation
from skfeature.utility.sparse_learning import feature_ranking


def main():
    # load data
    mat = scipy.io.loadmat('../data/COIL20.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]

    # perform evaluation on clustering task
    num_fea = 100    # number of selected features
    num_cluster = 20    # number of clusters, it is usually set as the number of classes in the ground truth

    # obtain the feature weight matrix
    Weight = UDFS.udfs(X, gamma=0.1, n_clusters=num_cluster)

    # sort the feature scores in an ascending order according to the feature scores
    idx = feature_ranking(Weight)

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