import numpy as np
from gaussian import Gaussian
import hmm

signal =  np.array([[ 1. ,  1.1,  0.8,  0.2,  1.6,  1.7,  3.4,  1.4,  1.1]])
trans = np.array([[ 0.  ,  1./3 ,  1./3 ,  1./3, 0.  ],
[ 0.  ,  0.45,  0.45,  0.,  0.1 ],
[ 0.  ,  0.45,  0.45,  0.,  0.1 ],
[ 0.  ,  0.  ,  0.  ,  1.,  0.  ],
[ 0.  ,  0.  ,  0.  ,  0.,  0.  ]])
dists = [Gaussian(mean=np.array([1]),cov=np.array([[1]])), 
Gaussian(mean=np.array([2]),cov=np.array([[1]])), 
Gaussian(mean=np.array([1.5]),cov=np.array([[1]]))]
vals, nll = hmm.viterbi(signal, trans, dists)
print 'State sequence: ', vals
#State sequence:  [1 1 1 1 2 2 2 1 1]    
print 'Negative log-likelihood:', nll
#Negative log-likelihood: 19.5947057502    

