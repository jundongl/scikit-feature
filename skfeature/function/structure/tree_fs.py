import math
import numpy as np
from skfeature.utility.sparse_learning import tree_lasso_projection, tree_norm


def tree_fs(X, y, z, idx, **kwargs):
    """
    This function implements tree structured group lasso regularization with least square loss, i.e.,
    min_{w} ||Xw-Y||_2^2 + z\sum_{i}\sum_{j} h_{j}^{i}|||w_{G_{j}^{i}}|| where h_{j}^{i} is the weight for the j-th group
    from the i-th level (the root node is in level 0)

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels or regression target
    z: {float}
        regularization parameter of L2 norm for the non-overlapping group
    idx: {numpy array}, shape (3, n_nodes)
        3*nodes matrix, where nodes denotes the number of nodes of the tree
        idx(1,:) contains the starting index
        idx(2,:) contains the ending index
        idx(3,:) contains the corresponding weight (w_{j})
    kwargs: {dictionary}
        verbose: {boolean}
            True if user want to print out the objective function value in each iteration, false if not

    Output
    ------
        w: {numpy array}, shape (n_features,)
            weight vector
        obj: {numpy array}, shape (n_iterations,)
            objective function value during iterations
        value_gamma: {numpy array}, shape (n_iterations,)
            suitable step size during iterations

    Note for input parameter idx:
    (1) For idx, if each entry in w is a leaf node of the tree and the weight for this leaf node are the same, then
    idx[0,0] = -1 and idx[1,0] = -1, idx[2,0] denotes the common weight
    (2) In idx, the features of the left tree is smaller than the right tree (idx[0,i] is always smaller than idx[1,i])

    Reference:
        Liu, Jun, et al. "Moreau-Yosida Regularization for Grouped Tree Structure Learning." NIPS. 2010.
        Liu, Jun, et al. "SLEP: Sparse Learning with Efficient Projections." http://www.public.asu.edu/~jye02/Software/SLEP, 2009.
    """

    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    # starting point initialization
    n_samples, n_features = X.shape

    # compute X'y
    Xty = np.dot(np.transpose(X), y)

    # initialize a starting point
    w = np.zeros(n_features)

    # compute Xw = X*w
    Xw = np.dot(X, w)

    # starting the main program, the Armijo Goldstein line search scheme + accelerated gradient descent
    # initialize step size gamma = 1
    gamma = 1

    # assign wp with w, and Xwp with Xw
    Xwp = Xw
    wwp = np.zeros(n_features)
    alphap = 0
    alpha = 1

    # indicates whether the gradient step only changes a little
    flag = False

    max_iter = 1000
    value_gamma = np.zeros(max_iter)
    obj = np.zeros(max_iter)
    for iter_step in range(max_iter):
        # step1: compute search point s based on wp and w (with beta)
        beta = (alphap-1)/alpha
        s = w + beta*wwp

        # step2: line search for gamma and compute the new approximation solution w
        Xs = Xw + beta*(Xw - Xwp)
        # compute X'* Xs
        XtXs = np.dot(np.transpose(X), Xs)

        # obtain the gradient g
        G = XtXs - Xty

        # copy w and Xw to wp and Xwp
        wp = w
        Xwp = Xw

        while True:
            # let s walk in a step in the antigradient of s to get v and then do the L1/L2-norm regularized projection
            v = s - G/gamma
            # tree overlapping group lasso projection
            n_nodes = int(idx.shape[1])
            idx_tmp = idx.copy()
            idx_tmp[2, :] = idx[2, :] * z / gamma
            w = tree_lasso_projection(v, n_features, idx_tmp, n_nodes)
            # the difference between the new approximate solution w and the search point s
            v = w - s
            # compute Xw = X*w
            Xw = np.dot(X, w)
            Xv = Xw - Xs
            r_sum = np.inner(v, v)
            l_sum = np.inner(Xv, Xv)
            # determine weather the gradient step makes little improvement
            if r_sum <= 1e-20:
                flag = True
                break

            # the condition is ||Xv||_2^2 <= gamma * ||v||_2^2
            if l_sum <= r_sum*gamma:
                break
            else:
                gamma = max(2*gamma, l_sum/r_sum)
        value_gamma[iter_step] = gamma

        # step3: update alpha and alphap, and check weather converge
        alphap = alpha
        alpha = (1+math.sqrt(4*alpha*alpha+1))/2

        wwp = w - wp
        Xwy = Xw -y
        # calculate the regularization part
        tree_norm_val = tree_norm(w, n_features, idx, n_nodes)

        # function value = loss + regularization
        obj[iter_step] = np.inner(Xwy, Xwy)/2 + z*tree_norm_val

        if verbose:
            print('obj at iter {0}: {1}'.format(iter_step+1, obj[iter_step]))

        if flag is True:
            break

        # determine whether converge
        if iter_step >= 2 and math.fabs(obj[iter_step] - obj[iter_step-1]) < 1e-3:
            break

    return w, obj, value_gamma




