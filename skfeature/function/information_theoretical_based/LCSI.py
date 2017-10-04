from skfeature.utility.entropy_estimators import *
import concurrent.futures


def parallel_loop(i, f, t1, t2, t3, f_select, y, beta, gamma):
    t2[i] += midd(f_select, f)
    t3[i] += cmidd(f_select, f, y)
    # calculate j_cmi for feature i (not in F)
    t = t1[i] - beta * t2[i] + gamma * t3[i]
    return [int(i), t, t2, t3]


def lcsi(X, y, **kwargs):
    """
    This function implements the basic scoring criteria for linear combination of shannon information term.
    The scoring criteria is calculated based on the formula j_cmi=I(f;y)-beta*sum_j(I(fj;f))+gamma*sum(I(fj;f|y))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a discrete data matrix
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        Parameters for different feature selection algorithms.
        beta: {float}
            beta is the parameter in j_cmi=I(f;y)-beta*sum(I(fj;f))+gamma*sum(I(fj;f|y))
        gamma: {float}
            gamma is the parameter in j_cmi=I(f;y)-beta*sum(I(fj;f))+gamma*sum(I(fj;f|y))
        function_name: {string}
            name of the feature selection function
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape: (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """

    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMI = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False
    # initialize the parameters
    if 'beta' in kwargs.keys():
        beta = kwargs['beta']
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True
    n_jobs = None
    if "n_jobs" in kwargs.keys():
        n_jobs = kwargs["n_jobs"]

    # select the feature whose j_cmi is the largest
    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)
    # t2 stores sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    t3 = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

    # make sure that j_cmi is positive at the very beginning
    j_cmi = 1

    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_CMI.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        ## Exit conditions from the inifinite loop
        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmi < 0:
                break

        # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
        j_cmi = -1E30
        if 'function_name' in kwargs.keys():
            if kwargs['function_name'] == 'MRMR':
                beta = 1.0 / len(F)
            elif kwargs['function_name'] == 'JMI':
                beta = 1.0 / len(F)
                gamma = 1.0 / len(F)

        ## Assign job queue, with function and args packed, the results are stored in future_queue
        job_queue = []
        future_queue = []
        for i in range(n_features):
            if i not in F:
                job_queue.append( [parallel_loop, 
                                   i, 
                                   X[:, i], 
                                   t1, 
                                   t2, 
                                   t3, 
                                   f_select, 
                                   y, 
                                   beta, 
                                   gamma] )
    
        ## Execute the job in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for job in job_queue:
                future_queue.append( executor.submit( job[0], ## parallel_loop() function
                                                      job[1], ## i
                                                      job[2], ## y = X[:, i]
                                                      job[3], ## t1
                                                      job[4], ## t2
                                                      job[5], ## t3
                                                      job[6], ## f_select
                                                      job[7], ## y
                                                      job[8], ## beta
                                                      job[9]  ## gamma
                                                    ) ) 
    
        ## Unpack the results, adding back single threaded logic
        for future in future_queue:
            
            i = future.result()[0]
            t = future.result()[1]
            t2 = future.result()[2]
            t3 = future.result()[3]
    
            ## record the largest j_cmi and the corresponding feature index
            if t > j_cmi:
                j_cmi = t
                idx = i
    
        F.append(idx)
        J_CMI.append(j_cmi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]
        
    return np.array(F), np.array(J_CMI), np.array(MIfy)


