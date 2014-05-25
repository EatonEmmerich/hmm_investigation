'''
Module implementing Hidden Markov model parameter estimation.

To avoid repeated warnings of the form "Warning: divide by zero encountered in log", 
it is recommended that you use the command "np.seterr(divide="ignore")" before 
invoking methods in this module.  This warning arises from the code using the 
fact that python sets - log 0 to "inf", to keep the code simple.

Created on Mar 28, 2012

@author: kroon
'''

from __future__ import division
from warnings import warn

import numpy as np
import numpy.testing as npt

from gaussian import Gaussian




def lrtrans(K):
    '''
    Generate a transition matrix enforcing a left-to-right topology, with 'K' 
    emitting states
    
    Generate a transition matrix with 'K' emitting states, and an initial and 
    final non-emitting state, enforcing a left-to-right topology.  This means 
    broadly: no transitions from higher-numbered to lower-numbered states are 
    permitted, while all other transitions are permitted. 
    The following exceptions apply:
    -The initial state may not transition to the final state
    -The final state may not transition (all transition probabilities from 
     this state should be 0
    In the above description, the initial state is numbered 0, the emitting 
    states 1 to K, and the final state K+1.
    All legal transitions from a given state should be equally likely
    
    Parameter
    ---------
    
    K : int
        Number of emitting states for the transition matrix
    
    Return
    ------

    trans : (K+2,K+2) ndarray
        The generated transition matrix

    Example
    -------
    >>>print lrtrans(4)
    [[ 0.          0.25        0.25        0.25        0.25        0.        ]
     [ 0.          0.2         0.2         0.2         0.2         0.2       ]
     [ 0.          0.          0.25        0.25        0.25        0.25      ]
     [ 0.          0.          0.          0.33333333  0.33333333  0.33333333]
     [ 0.          0.          0.          0.          0.5         0.5       ]
     [ 0.          0.          0.          0.          0.          0.        ]]  
    '''
    trans = np.zeros((K + 2, K + 2))
    for col in range(1, K + 1):
        trans[0, col] = 1. / K
    for row in range(1, K + 1):
        for col in range(row, K + 2):
            prob = 1./(K + 2 - row)
            trans[row, col] = prob
    return trans

def lrinit(N, signals, trans):
    '''
    Initial allocation of the observations to states in a left-to-right manner
    
    Each signal consists of a number of observations. Each observation is 
    allocated to one of the 'K' emitting states in a left-to-right manner
    by splitting the observations of each signal into approximately equally-sized 
    chunks of increasing state number, with the number of chunks determined by the 
    number of emitting states specified in 'trans'.
    If the number of such states is 'maxstates', the allocation for a signal is 
    specified by:
    np.floor(np.linspace(0, maxstates, n, endpoint=False))
    where 'n' is the number of time-steps in that signal.
    
    Parameters
    ----------
    N : int
        Total number of samples in all the signals combined.
    signals : ((d, N_i),) list
        List of signals to be provided with initial allocation
        with sum of N_i = N
    trans : (K+2,K+2) ndarray
        Transition matrix for HMM

    Return
    ------
    
    states : (N, K) ndarray
        Initial allocation of signal time-steps to states.
        'states[:,j]' specifies the allocation of all the observations to state j.
         
    
    Example
    -------
    
    >>>states = lrinit(sum(range(7)), [np.zeros((3, i+1)) for i in range(6)], np.zeros((5, 5)))
    >>>print states
    [[ 1.  0.  0.]
     [ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]
     [ 1.  0.  0.]
     [ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]
     [ 1.  0.  0.]
     [ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]
     [ 1.  0.  0.]
     [ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]
     [ 0.  0.  1.]] 
    '''
    maxstates = np.shape(trans)[0] - 2 # Exclude initial and final states
    states = np.zeros((N, maxstates))
    i = 0
    for s in signals:
        vals = np.floor(np.linspace(0, maxstates, num=np.shape(s)[1], endpoint=False))
        for v in vals:
            states[i][v] = 1
            i += 1
    return states

def negloglik(signals, prop=1.0, trans=None, dists=None):
    '''
    Return the negative log-likelihood of the training set, 'signals', 
    adjusted by a prior probability

    Given a hidden Markov model with 'K' emitting states, 
    transition matrix 'trans',  and state densities 'dists', 
    calculate the negative log-likelihood of the observations in 'signals', 
    using the most likely underlying state sequence for each signal, as given
    by the Viterbi algorithm.  An adjustment is allowed for a prior probability 
    of 'prop'.
    
    The keyword arguments, 'trans' and 'dists' need to be populated.
    
    Parameters
    ----------
    signals : ((d,N_i),) list
        Signals for which the negative log-likelihood is being calculated
    prop : float
        Prior probability of the model
    trans : (K+2,K+2) ndarray
        Transition matrix of the HMM, including non-emitting states
    dists : (K,) list
        List of current DensityFunc densities
        
    Return
    ------
    
    nll : float
        Negative log-likelihood
    
    Example
    -------
    
    >>> prop = 0.3
    >>> signal = np.array([[ 1. ,  1.1,  0.8,  0.2,  1.6,  1.7,  3.4,  1.4,  1.1]])
    >>> signals = [signal] * 2
    >>> trans = np.array([[ 0.  ,  1./3 ,  1./3 ,  1./3, 0.  ],
    ... [ 0.  ,  0.45,  0.45,  0.,  0.1 ],
    ... [ 0.  ,  0.45,  0.45,  0.,  0.1 ],
    ... [ 0.  ,  0.  ,  0.  ,  1.,  0.  ],
    ... [ 0.  ,  0.  ,  0.  ,  0.,  0.  ]])
    >>> dists = [Gaussian(mean=np.array([1]),cov=np.array([[1]])), 
    ... Gaussian(mean=np.array([2]),cov=np.array([[1]])), 
    ... Gaussian(mean=np.array([1.5]),cov=np.array([[1]]))]
    >>> nll = negloglik(signals, prop, trans, dists)
    >>> print nll    
    40.3933843048    
    ''' 
    nll = calcstates(signals, trans, dists)[2]
    return nll - np.log(prop)

def viterbi(signal, trans, dists):
    '''
    Apply the Viterbi algorithm to the observations provided in 'signal'.
    
    Apply the Viterbi algorithm to 'signal' for an HMM with 
    transition matrix 'trans', and densities 'dists'.   
    Return the maximum likelihood state sequence as well as the negative 
    log-likelihood of that sequence.  
    
    Parameters
    ----------
    
    signal : (d,n) ndarray
        Signal for which the optimal state sequence is to be calculated
    trans : (K+2,K+2) ndarray
        Transition matrix to use for calculating the state sequence, including 
        non-emitting states
	defined as [[0 to 0, 1 to 0, 2 to 0]
		    [0 to 1, 1 to 1, 2 to 1]
		    [0 to 2, 1 to 2, 2 to 2]]
    dists : (K,) list
        list of DensityFunc objects. 'K' is the number of emitting states. 
        
    Return
    ------
    
    vals : (n,) ndarray
        The state sequence for the signal (excluding non-emitting states)
    nll : float
        The negative log-likelihood associated with the sequence
    
    Example
    -------
    
    >>> signal = np.array([[ 1. ,  1.1,  0.8,  0.2,  1.6,  1.7,  3.4,  1.4,  1.1]])
    >>> trans = np.array([[ 0.  ,  1./3 ,  1./3 ,  1./3, 0.  ],
    ... [ 0.  ,  0.45,  0.45,  0.,  0.1 ],
    ... [ 0.  ,  0.45,  0.45,  0.,  0.1 ],
    ... [ 0.  ,  0.  ,  0.  ,  1.,  0.  ],
    ... [ 0.  ,  0.  ,  0.  ,  0.,  0.  ]])
    >>> dists = [Gaussian(mean=np.array([1]),cov=np.array([[1]])), 
    ... Gaussian(mean=np.array([2]),cov=np.array([[1]])), 
    ... Gaussian(mean=np.array([1.5]),cov=np.array([[1]]))]
    >>> vals, nll = viterbi(signal, trans, dists)
    >>> print 'State sequence: ', vals
    State sequence:  [1 1 1 1 2 2 2 1 1]    
    >>> print 'Negative log-likelihood:', nll
    Negative log-likelihood: 19.5947057502    
    '''
    n = np.shape(signal)[1] # Number of observations
    S = np.shape(trans)[0] # number of states, including starting and ending states
    nlltable = np.zeros((S, n + 2)) # table containing NLLs of ML state sequence
    backtable = np.zeros((S, n + 2), dtype="int") # Back pointers for ML state sequence (0 is start)
    nlltable[0, 0] = 0
    for state in xrange(1, S):
        nlltable[state, 0] = float("inf")
    emissionNLLs = np.array([[s.negloglik(x) for s in dists] for x in np.hsplit(signal, n)]) # Shape = (K+2,n)
    #what is emmisionNLLs?
    #maybe:  (signal index)
#   (states)  (negloglik)
    emissionNLLs = emissionNLLs.T
    ##^^##
    # Insert Viterbi code here
    probMatrix = np.zeros((S,n+2))
    nlltrans = -1*np.log(trans)
#    print "K+2: %d,imax" %S
#    print "n: %d,xmax" %n
#    print "EMMISIONNLLS:"
#    print np.shape(emissionNLLs)
#    print emissionNLLs
#    print emissionNLLs+++===>>>>
#    print "ProbMatrix:"
#    print np.shape(probMatrix)
#    print "nlltrans"
#    print np.shape(nlltrans)
#    print nlltrans
    probMatrix[0:,0:n] = nlltable[:,0:n]
    for x in range (n+1):
	if (x == 0):
	    for i in range (S-1):
		if (i < np.shape(emissionNLLs)[0]):
		    probMatrix[i+1,x+1] = emissionNLLs[i,x]+nlltrans[0,i+1]
		else:
		    probMatrix[i+1,x+1] = 0+nlltrans[0,i+1]
		backtable[i+1,x+1] = 0
	    #also do for i=0 and i = S
	elif (x > 0 and x < n):
	    for i in range (S-1):
		probVector = []
		if (i < np.shape(emissionNLLs)[0]):
		    for j in range (S-1):
			probVector.append(probMatrix[j+1,x]+nlltrans[j+1,i+1])
		    probMatrix[i+1,x+1] = emissionNLLs[i,x]+np.min(probVector)
		    backtable[i+1,x+1] = (np.argmin(probVector)+1)
		else:
		    for j in range (S-1):
                        probVector.append(probMatrix[j+1,x]+nlltrans[j+1,i+1])
                    probMatrix[i+1,x+1] = 0+np.min(probVector)
                    backtable[i+1,x+1] = (np.argmin(probVector)+1)
		print np.min(probVector)

	    #also do for i=0 and i = S
	    #I don't know whats going on anymore
	    #I thaught I knew a few lines of code ago
	else:
	    for i in range (S-1):
		probVector = []
	        for j in range (S-1):
		    probVector.append(probMatrix[j+1,x] + nlltrans[j+1,i+1])
		probMatrix[i+1,x+1] = 0 + np.min(probVector)
		backtable[i+1,x+1] = np.argmin(probVector)+1
		    

    """My code not working presently:
    for x in range (n+2):
	if (x <1):
	    for i in range (S):
		if (i >= 1 and i < S-1):
		    probMatrix[x,i] = emissionNLLs[x,i-1]+nlltrans[0,i]
		    backtable[i,x] = i
		else:
		    probMatrix[x,i] = 1+nlltrans[0,i]
		    backtable[i,x] = i
	if (x >=1 and x<n+1):
	    for i in range (S):
		probVector = []
		if (i >= 1 and i< S-1):
		    for j in range (0,S):
#			print "i: %d" %i
#			print "x: %d" %x
			probVector.append(probMatrix[x-1,j]+nlltrans[j,i])
    		    probMatrix[x,i] = emissionNLLs[x,i-1]+np.min(probVector)
		    print np.argmin(probVector)
		    print probVector
		    backtable[i,x] = np.argmin(probVector)
		else:
		    for j in range (0,S):
			probVector.append(probMatrix[x-1,j]+nlltrans[j,i])
		    probMatrix[x,i] = 1+np.min(probVector)
		    backtable[i,x] = np.argmin(probVector)
	else:
	    print"""
    
    print "probmatrix"
    print probMatrix
    nlltable[:,:] = probMatrix
    #
    ##VV##
    vals = np.zeros(n, dtype="int")
    vals[n-1] = backtable[S-1, n+1]
    for i in xrange(n-2, -1, -1):
        vals[i] = backtable[vals[i+1], i+2]
    print "nlltable"
    print nlltable
    print backtable
    return vals, nlltable[S-1, n+1]

def calcstates(signals, trans, dists):
    '''
    Calculate state sequences on the 'signals' maximizing the likelihood for 
    the given HMM parameters
    
    Calculate the state sequences for each of the given 'signals', maximizing the 
    likelihood of the given parameters of a HMM model. This allocates each of the
    observations, in all the equences, to one of the states. 
    
    Use the state allocation to calculate an updated transition matrix.   
    
    As part of this updated transition matrix calculation, emitting states which 
    are not used in the new state allocation are removed. 
    
    In what follows, 'K' is the number of emitting states described in 'trans', 
    while 'K' ' is the new number of emitting states. 
    
    Parameters
    ----------
    
    signals : ((d,N_i),) list
        Signals from which parameters are being estimated
    trans : (K+2,K+2) ndarray
        Transition matrix to use for calculating the state sequences, including 
        non-emitting states
    dists : (K,) list 
    list of DensityFunc objects
    
    Return
    ------
    
    states : (N,K') ndarray
        The updated state allocations of each observation is all signals
    trans : (K'+ 2,K'+ 2) ndarray
        Updated transition matrix 
    nll : float
        Negative log-likelihood of all the data
    
    Example
    -------

    >>> signal = np.array([[ 1. ,  1.1,  0.8,  0.2,  1.6,  1.7,  3.4,  1.4,  1.1]])
    >>> signals = [signal] * 2
    >>> trans = np.array([[ 0.  ,  1./3 ,  1./3 ,  1./3, 0.  ],
    ... [ 0.  ,  0.45,  0.45,  0.,  0.1 ],
    ... [ 0.  ,  0.45,  0.45,  0.,  0.1 ],
    ... [ 0.  ,  0.  ,  0.  ,  1.,  0.  ],
    ... [ 0.  ,  0.  ,  0.  ,  0.,  0.  ]])
    >>> dists = [Gaussian(mean=np.array([1]),cov=np.array([[1]])), 
    ... Gaussian(mean=np.array([2]),cov=np.array([[1]])), 
    ... Gaussian(mean=np.array([1.5]),cov=np.array([[1]]))]
    >>> states, newtrans, nll = calcstates(signals, trans, dists)
    >>> print states
    [[ 1.  0.]
     [ 1.  0.]
     [ 1.  0.]
     [ 1.  0.]
     [ 0.  1.]
     [ 0.  1.]
     [ 0.  1.]
     [ 1.  0.]
     [ 1.  0.]
     [ 1.  0.]
     [ 1.  0.]
     [ 1.  0.]
     [ 1.  0.]
     [ 0.  1.]
     [ 0.  1.]
     [ 0.  1.]
     [ 1.  0.]
     [ 1.  0.]]
    >>> print newtrans
    [[ 0.          1.          0.          0.        ]
     [ 0.          0.66666667  0.16666667  0.16666667]
     [ 0.          0.33333333  0.66666667  0.        ]
     [ 0.          0.          0.          0.        ]]    
    >>> print nll
    39.1894115005
    '''
    maxstates = np.shape(trans)[0] - 2 # Exclude initial and final states
    newtrans = np.zeros_like(trans)
    used = [True] + [False] * maxstates + [True]
    
    N = sum([dat.shape[1] for dat in signals])
    states = np.zeros((N, maxstates))
    # Passed as param now: dists = [Gaussian(mean=means[:,i], cov=covs[i]) for i in range(maxstates)]
    i = 0
    totalnll = 0
    for s in signals:
        vals, nll = viterbi(s, trans, dists)
        totalnll += nll
        oldv = 0
        for v in vals:
            states[i][v-1] = 1
            newtrans[oldv, v] += 1
            oldv = v
            used[v] = True
            i += 1
        newtrans[vals[-1], maxstates+1] += 1
    newstates = np.where(used)[0]
    newtrans = newtrans[newstates, :]
    newtrans = newtrans[:, newstates]
    # Normalize all except transitions from end state:
    rowsums = np.sum(newtrans, axis=1)
    rowsums[-1] = 1
    newtrans = newtrans/rowsums[:, np.newaxis]
    return states[:, [s - 1 for s in newstates[1:-1]]], newtrans, totalnll # Remove unused states by indexing

def updatenums(weights):
    '''
    Update number of observations per state for current HMM labelling of data
    
    Update the number of observations allocated to each of the 'K' emitting 
    HMM states using the labelling of all the observation,  provided in 'weights'.
    
    Parameters
    ----------
    
    weights : (N,K) ndarray
        Current labelling of observations in model
        
    Return
    ------
    
    nums : (K,) ndarray
        The calculated number of observations allocated to each state

    Example
    -------

    >>> w = np.reshape(np.array(3 * [1, 0] + 2 * [0, 1]), (5, 2))
    >>> print updatenums(w)
    [3 2]  
    '''
    # By the time this is called, all values returned in nums in this calculation should be > 0
    nums = np.sum(weights, axis=0)
    return nums

def updatemeans(weights, nums, data):
    '''Update estimates of mean vectors for each HMM state
    
    Update the means of each of the 'K' emitting HMM states for the given 
    allocation of the observations to states.
    
    Parameters
    ----------
    
    weights : (N,K) ndarray
        Current state allocations of all the observations
    nums : (K,) ndarray
        Number of observations per state
    data : (d,N) ndarray
        Data from which parameters are being estimated

    Return
    ------
    
    means : (d,K) ndarray
        The estimated means
    
    Example
    -------
    
    >>> data = np.array(range(20)).reshape(4,5)
    >>> w = np.reshape(np.array(3 * [1, 0] + 2 * [0, 1]), (5, 2))
    >>> means = updatemeans(w, updatenums(w), data)
    >>> print means
    [[  1.    3.5]
     [  6.    8.5]
     [ 11.   13.5]
     [ 16.   18.5]]    
    '''
    means = np.dot(data, weights)
    means = means / nums # Note that /= does not respect __future__ division import
    return means

def updatecovs(weights, means, nums, data, diagcov=False):
    '''
    Update estimates of covariance matrices for each HMM state
    
    Estimate the covariance matrices for each of the 'K' emitting HMM states for 
    the given allocation of the observations to states. 
    If 'diagcov' is true, diagonal covariance matrices are returned.
    
    Parameters
    ----------
    
    weights : (N,K) ndarray
        Current state allocations in model
    means : (d,K) ndarray
        Current means in model
    nums : (K,) ndarray
        Number of points per state
    data : (d,N) ndarray
        Data from which parameters are being estimated
    diagcov : boolean
        diagcov = True, estimates diagonal covariance matrix
        diagcov = False, estimates full covariance matrix
    
    Return
    ------
    
    covs : (K,d,d) ndarray
        The estimated covariance matrices
    
    
    Example
    -------
    
    >>> data = np.array([[ 0.3323902 ,  1.39952168],
    ...        [-3.09393968,  0.85202915],
    ...        [ 0.3932616 ,  4.14018981],
    ...        [ 2.71301182,  1.48606545],
    ...        [ 0.76624929,  1.48450185],
    ...        [-2.68682389, -2.20487651],
    ...        [-1.50746076, -1.3965284 ],
    ...        [-3.35436652, -2.70017904],
    ...        [ 0.62831278, -0.14266899],
    ...        [-3.13713063, -1.35515249]])
    >>> data = np.transpose(data)
    >>> w   = np.reshape(3 * [1, 0] + 4 * [0, 1] + 3 * [1, 0], (10, 2))
    >>> n   = updatenums(w)
    >>> m   = updatemeans(w, n, data)
    >>> cov = updatecovs(w, m, n, data) # Full covariance
    >>> print cov
    [[[ 3.33881418  2.61431608]
      [ 2.61431608  4.69524387]]

     [[ 4.32780425  3.27144308]
      [ 3.27144308  2.78110481]]]
    >>> cov = updatecovs(w, m, n, data, diagcov=True) # Diagonal covariance
    >>> print cov
    [[[ 3.33881418  0.        ]
      [ 0.          4.69524387]]

     [[ 4.32780425  0.        ]
      [ 0.          2.78110481]]]    
    '''
    d = means.shape[0]
    covs = []
    for i, m in enumerate(np.hsplit(means, means.shape[1])):
        cov = np.zeros((d, d))
        for j, p in enumerate(np.hsplit(data, data.shape[1])):
            x = p - m
            cov += weights[j, i] * np.outer(x, x)
        covs.append(cov)
    covs = np.array(covs)
    covs = covs / np.reshape(nums, (-1, 1, 1))
    if diagcov:
        covs = np.array([np.diag(np.diag(c)) for c in covs])
    return covs
    
def hmm(data, lengths, trans, init=lrinit, diagcov=False, maxiters=20, rtol=1e-4):
    '''
    Perform parameter estimation for a hidden Markov model (HMM)
    
    Perform parameter estimation for a HMM using multi-dimensional Gaussian 
    states.  The signals for the training data are packed into 'data' with 'trans' 
    providing an initial transition matrix for constraining the topology of the 
    HMM. 'init' is used for an initial allocation of the data points into HMM 
    states.  Thereafter, parameters are estimated using Viterbi re-estimation.  
    In what follows 'K' is the original number of emittting states, while 'K' '
    is the final number of emitting states.
    
    Parameters
    ----------
    
    data : (d,N) ndarray
        The data used to estimate the parameters. 'data' consists of 'l'  
        signals packed together into a single matrix. 
        'd' is the dimension of each observation.
        'N' is the total number of observations.
    lengths : (l,) list
        Length of each signal in 'data' where 'l' is the number of signals.  
        The sum of the elements of 'lengths' must be N
    trans : (K+2,K+2) ndarray
        Initial transition matrix for the HMM, including the starting and ending 
        non-emitting states. 'K' is the initial number of emitting states.
    init : Function keyword
        Function used for initial state allocations.  This function should take 
        the same parameters as the default function, 'lrinit'
    diagcov : boolean 
        'True' uses a diagonal covariance matrix  
        'False' uses a full covariance matrix 
    maxiters : int
        The maximum number of iterations of the EM algorithm to be performed. 
        A warning is issued if 'maxiters' is exceeded. 
    rtol: float
        Threshold of estimated relative  error in negative log-likelihood (NLL).
        
    Return
    ------
    
    trans : (K+2,K+2) ndarray
        Updated transition matrix
    dists : (K',) list
        DensityFunc object of each component.
        'K' ' is the final number of emitting states, after those states to 
        which no observations were assgined, have been removed.
    newNLL : float
        Negative log-likelihood of parameters at convergence.
        
    Example
    -------

    >>> signal = np.array([[ 1. ,  1.1,  0.8,  0.2,  1.6,  1.7,  3.4,  1.4,  1.1]])
    >>> data = np.hstack([signal] * 2)
    >>> lengths = [9] * 2
    >>> trans = np.array([[ 0.  ,  1./3 ,  1./3 ,  1./3, 0.  ],
    ... [ 0.  ,  0.45,  0.45,  0.,  0.1 ],
    ... [ 0.  ,  0.45,  0.45,  0.,  0.1 ],
    ... [ 0.  ,  0.  ,  0.  ,  1.,  0.  ],
    ... [ 0.  ,  0.  ,  0.  ,  0.,  0.  ]])
    >>> states = np.array([ 1,  1,  1,  2,  2,  2,  2,  2,  1])
    >>> newtrans = np.array([[ 0.  ,  1. ,  0. ,  0.  ],
    ... [ 0.  ,  0.5,  0.25,  0.25 ],
    ... [ 0.  ,  0.2,  0.8 ,  0    ],
    ... [ 0.  ,  0.  ,  0.  ,  0.  ]])
    >>> newmeans = np.array([[ 1.  ,  1.66]])
    >>> newcovs = np.array([[[ 0.015 ]], [[ 1.0464]]])
    >>> nll = 22.290642196869609
    >>> trans, dists, nll = hmm(data, lengths, trans)
    >>> print trans
    [[ 0.    1.    0.    0.  ]
     [ 0.    0.5   0.25  0.25]
     [ 0.    0.2   0.8   0.  ]
     [ 0.    0.    0.    0.  ]]
    >>> print print [a.mean() for a in dists]
    [array([ 1.]), array([ 1.66])]
    >>> print [a.cov() for a in dists] 
    [array([[ 0.015]]), array([[ 1.0464]])]
    >>> print 'Negative log-likelihood: ', nll
    22.2906421969
    
    >>> npt.assert_almost_equal([a.mean() for a in ans[1]], newmeans.transpose(), decimal=4)
    >>> npt.assert_almost_equal([a.cov() for a in ans[1]], newcovs, decimal=4)
    >>> npt.assert_almost_equal(ans[2], nll, decimal=4)
    >>> print 'Optimal state sequence: ', viterbi(signal, trans, dists)[0]   
    Optimal state sequence: [1 1 1 2 2 2 2 2 1]
    '''
    N = data.shape[1] 
    newstarts = np.cumsum(lengths)[:-1]
    signals = np.hsplit(data, newstarts)
    states = init(N, signals, trans)
    nums = updatenums(states)
    means = updatemeans(states, nums, data)
    covs = updatecovs(states, means, nums, data, diagcov)
    
    
    dists = [Gaussian(mean=means[:,i], cov=covs[i]) for i in xrange(covs.shape[0])]
    oldstates, trans, newNLL = calcstates(signals, trans, dists)
    converged = False
    iters = 0
    while not converged and iters <  maxiters:
		
		nums = updatenums(states)
		means = updatemeans(states, nums, data)
		covs = updatecovs(states, means, nums, data, diagcov)
		dists = [Gaussian(mean=means[:,i], cov=covs[i]) for i in xrange(covs.shape[0])]
		oldstates, trans, oldNLL = calcstates(signals, trans, dists)
		if(np.abs(np.abs(oldNLL)-np.abs(newNLL)) < 1e-9):
			converged = True
		newNLL = oldNLL
		iters = iters +1
        # Insert EM code here to calculate the transition matrix: 'trans',
        # the state densities: 'dists',
        # and the negative log-likelihood: 'newNLL'
        
    if iters >= maxiters:
        warn("Maximum number of iterations reached - HMM parameters may not have converged")
    return trans, dists, newNLL
