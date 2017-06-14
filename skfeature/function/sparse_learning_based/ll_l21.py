import math
import numpy as np
from numpy import linalg as LA
from skfeature.utility.sparse_learning import euclidean_projection, calculate_l21_norm


def proximal_gradient_descent(X, Y, z, **kwargs):
    """
    This function implements supervised sparse feature selection via l2,1 norm, i.e.,
    min_{W} sum_{i}log(1+exp(-yi*(W'*x+C))) + z*||W||_{2,1}

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    Y: {numpy array}, shape (n_samples, n_classes)
        input class labels, each row is a one-hot-coding class label, guaranteed to be a numpy array
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
    value_gamma: {numpy array}, shape (n_iterations,s)
        suitable step size during iterations


    Reference:
        Liu, Jun, et al. "Multi-Task Feature Learning Via Efficient l2,1-Norm Minimization." UAI. 2009.
    """

    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    # Starting point initialization #
    n_samples, n_features = X.shape
    n_samples, n_classes = Y.shape

    # the indices of positive samples
    p_flag = (Y == 1)
    # the total number of positive samples
    n_positive_samples = np.sum(p_flag, 0)
    # the total number of negative samples
    n_negative_samples = n_samples - n_positive_samples
    n_positive_samples = n_positive_samples.astype(float)
    n_negative_samples = n_negative_samples.astype(float)

    # initialize a starting point
    W = np.zeros((n_features, n_classes))
    C = np.log(np.divide(n_positive_samples, n_negative_samples))

    # compute XW = X*W
    XW = np.dot(X, W)

    # starting the main program, the Armijo Goldstein line search scheme + accelerated gradient descent
    # the intial guess of the Lipschitz continuous gradient
    gamma = 1.0/(n_samples*n_classes)

    # assign Wp with W, and XWp with XW
    XWp = XW
    WWp =np.zeros((n_features, n_classes))
    CCp = np.zeros((1, n_classes))

    alphap = 0
    alpha = 1

    # indicates whether the gradient step only changes a little
    flag = False

    max_iter = 1000
    value_gamma = np.zeros(max_iter)
    obj = np.zeros(max_iter)
    for iter_step in range(max_iter):
        # step1: compute search point S based on Wp and W (with beta)
        beta = (alphap-1)/alpha
        S = W + beta*WWp
        SC = C + beta*CCp

        # step2: line search for gamma and compute the new approximation solution W
        XS = XW + beta*(XW - XWp)
        aa = -np.multiply(Y, XS+np.tile(SC, (n_samples, 1)))
        # fun_S is the logistic loss at the search point
        bb = np.maximum(aa, 0)
        fun_S = np.sum(np.log(np.exp(-bb)+np.exp(aa-bb))+bb)/(n_samples*n_classes)
        # compute prob = [p_1;p_2;...;p_m]
        prob = 1.0/(1+np.exp(aa))

        b = np.multiply(-Y, (1-prob))/(n_samples*n_classes)
        # compute the gradient of C
        GC = np.sum(b, 0)
        # compute the gradient of W as X'*b
        G = np.dot(np.transpose(X), b)

        # copy W and XW to Wp and XWp
        Wp = W
        XWp = XW
        Cp = C

        while True:
            # let S walk in a step in the antigradient of S to get V and then do the L1/L2-norm regularized projection
            V = S - G/gamma
            C = SC - GC/gamma
            W = euclidean_projection(V, n_features, n_classes, z, gamma)

            # the difference between the new approximate solution W and the search point S
            V = W - S
            # compute XW = X*W
            XW = np.dot(X, W)
            aa = -np.multiply(Y, XW+np.tile(C, (n_samples, 1)))
            # fun_W is the logistic loss at the new approximate solution
            bb = np.maximum(aa, 0)
            fun_W = np.sum(np.log(np.exp(-bb)+np.exp(aa-bb))+bb)/(n_samples*n_classes)

            r_sum = (LA.norm(V, 'fro')**2 + LA.norm(C-SC, 2)**2) / 2
            l_sum = fun_W - fun_S - np.sum(np.multiply(V, G)) - np.inner((C-SC), GC)

            # determine weather the gradient step makes little improvement
            if r_sum <= 1e-20:
                flag = True
                break

            # the condition is fun_W <= fun_S + <V, G> + <C ,GC> + gamma/2 * (<V,V> + <C-SC,C-SC> )
            if l_sum < r_sum*gamma:
                break
            else:
                gamma = max(2*gamma, l_sum/r_sum)
        value_gamma[iter_step] = gamma

        # step3: update alpha and alphap, and check weather converge
        alphap = alpha
        alpha = (1+math.sqrt(4*alpha*alpha+1))/2

        WWp = W - Wp
        CCp = C - Cp

        # calculate obj
        obj[iter_step] = fun_W
        obj[iter_step] += z*calculate_l21_norm(W)

        if verbose:
            print('obj at iter {0}: {1}'.format(iter_step+1, obj[iter_step]))

        if flag is True:
            break

        # determine weather converge
        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step-1]) < 1e-3:
            break
    return W, obj, value_gamma
