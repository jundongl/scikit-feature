import math
import numpy as np
from skfeature.utility.sparse_learning import tree_lasso_projection, tree_norm


def group_fs(X, y, z1, z2, idx, **kwargs):
    """
    This function implements supervised sparse group feature selection with least square loss, i.e.,
    min_{w} ||Xw-y||_2^2 + z_1||w||_1 + z_2*sum_{i} h_{i}||w_{G_{i}}|| where h_i is the weight for the i-th group

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels or regression target
    z1: {float}
        regularization parameter of L1 norm for each element
    z2: {float}
        regularization parameter of L2 norm for the non-overlapping group
    idx: {numpy array}, shape (3, n_nodes)
        3*nodes matrix, where nodes denotes the number of groups
        idx[1,:] contains the starting index of a group
        idx[2,: contains the ending index of a group
        idx[3,:] contains the corresponding weight (w_{j})
    kwargs: {dictionary}
        verbose: {boolean}
            True if user want to print out the objective function value in each iteration, false if not

    Output
    ------
    w: {numpy array}, shape (n_features, )
        weight matrix
    obj: {numpy array}, shape (n_iterations, )
        objective function value during iterations
    value_gamma: {numpy array}, shape (n_iterations, )
        suitable step size during iterations

    Reference
    ---------
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
            idx_tmp = np.zeros((3, n_nodes+1))
            idx_tmp[0:2, :] = np.concatenate((np.array([[-1], [-1]]), idx[0:2, :]), axis=1)
            idx_tmp[2, :] = np.concatenate((np.array([z1/gamma]), z2/gamma*idx[2, :]), axis=1)
            w = tree_lasso_projection(v, n_features, idx_tmp, n_nodes+1)
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
        idx_tmp = np.zeros((3, n_nodes+1))
        idx_tmp[0:2, :] = np.concatenate((np.array([[-1], [-1]]), idx[0:2, :]), axis=1)
        idx_tmp[2, :] = np.concatenate((np.array([z1]), z2*idx[2, :]), axis=1)
        tree_norm_val = tree_norm(w, n_features, idx_tmp, n_nodes+1)

        # function value = loss + regularization
        obj[iter_step] = np.inner(Xwy, Xwy)/2 + tree_norm_val

        if verbose:
            print('obj at iter {0}: {1}'.format(iter_step+1, obj[iter_step]))

        if flag is True:
            break

        # determine weather converge
        if iter_step >= 2 and math.fabs(obj[iter_step] - obj[iter_step-1]) < 1e-3:
            break

    return w, obj, value_gamma


