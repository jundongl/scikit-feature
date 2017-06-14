import math
import numpy as np
from numpy import linalg as LA
from skfeature.utility.sparse_learning import euclidean_projection, calculate_l21_norm


def proximal_gradient_descent(X, Y, z, **kwargs):
    """
    This function implements supervised sparse feature selection via l2,1 norm, i.e.,
    min_{W} ||XW-Y||_F^2 + z*||W||_{2,1}

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a numpy array
    Y: {numpy array}, shape (n_samples, n_classes)
        input class labels, each row is a one-hot-coding class label
    z: {float}
        regularization parameter
    kwargs: {dictionary}
        verbose: {boolean}
            True if user want to print out the objective function value in each iteration, false if not

    Output
    ------
        W: {numpy array}, shape (n_features, n_classes)
            weight matrix
        obj: {numpy array}, shape (n_iterations,)
            objective function value during iterations
        value_gamma: {numpy array}, shape (n_iterations,)
            suitable step size during iterations

    Reference
    ---------
        Liu, Jun, et al. "Multi-Task Feature Learning Via Efficient l2,1-Norm Minimization." UAI. 2009.
    """

    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    # starting point initialization
    n_samples, n_features = X.shape
    n_samples, n_classes = Y.shape

    # compute X'Y
    XtY = np.dot(np.transpose(X), Y)

    # initialize a starting point
    W = XtY

    # compute XW = X*W
    XW = np.dot(X, W)

    # compute l2,1 norm of W
    W_norm = calculate_l21_norm(W)

    if W_norm >= 1e-6:
        ratio = init_factor(W_norm, XW, Y, z)
        W = ratio*W
        XW = ratio*XW

    # starting the main program, the Armijo Goldstein line search scheme + accelerated gradient descent
    # initialize step size gamma = 1
    gamma = 1

    # assign Wp with W, and XWp with XW
    XWp = XW
    WWp =np.zeros((n_features, n_classes))
    alphap = 0
    alpha = 1

    # indicate whether the gradient step only changes a little
    flag = False

    max_iter = 1000
    value_gamma = np.zeros(max_iter)
    obj = np.zeros(max_iter)
    for iter_step in range(max_iter):
        # step1: compute search point S based on Wp and W (with beta)
        beta = (alphap-1)/alpha
        S = W + beta*WWp

        # step2: line search for gamma and compute the new approximation solution W
        XS = XW + beta*(XW - XWp)
        # compute X'* XS
        XtXS = np.dot(np.transpose(X), XS)
        # obtain the gradient g
        G = XtXS - XtY
        # copy W and XW to Wp and XWp
        Wp = W
        XWp = XW

        while True:
            # let S walk in a step in the antigradient of S to get V and then do the L1/L2-norm regularized projection
            V = S - G/gamma
            W = euclidean_projection(V, n_features, n_classes, z, gamma)
            # the difference between the new approximate solution W and the search point S
            V = W - S
            # compute XW = X*W
            XW = np.dot(X, W)
            XV = XW - XS
            r_sum = LA.norm(V, 'fro')**2
            l_sum = LA.norm(XV, 'fro')**2

            # determine weather the gradient step makes little improvement
            if r_sum <= 1e-20:
                flag = True
                break

            # the condition is ||XV||_2^2 <= gamma * ||V||_2^2
            if l_sum < r_sum*gamma:
                break
            else:
                gamma = max(2*gamma, l_sum/r_sum)
        value_gamma[iter_step] = gamma

        # step3: update alpha and alphap, and check weather converge
        alphap = alpha
        alpha = (1+math.sqrt(4*alpha*alpha+1))/2

        WWp = W - Wp
        XWY = XW -Y

        # calculate obj
        obj[iter_step] = LA.norm(XWY, 'fro')**2/2
        obj[iter_step] += z*calculate_l21_norm(W)

        if verbose:
            print('obj at iter {0}: {1}'.format(iter_step+1, obj[iter_step]))

        if flag is True:
            break

        # determine weather converge
        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step-1]) < 1e-3:
            break
    return W, obj, value_gamma


def init_factor(W_norm, XW, Y, z):
    """
    Initialize the starting point of W, according to the author's code
    """
    n_samples, n_classes = XW.shape
    a = np.inner(np.reshape(XW, n_samples*n_classes), np.reshape(Y, n_samples*n_classes)) - z*W_norm
    b = LA.norm(XW, 'fro')**2
    ratio = a / b
    return ratio