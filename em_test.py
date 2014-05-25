import numpy as np
import numpy.testing as npt
import hmm

signal = np.array([[ 1. ,  1.1,  0.8,  0.2,  1.6,  1.7,  3.4,  1.4,  1.1]])
data = np.hstack([signal] * 2)
lengths = [9] * 2
trans = np.array([[ 0.  ,  1./3 ,  1./3 ,  1./3, 0.  ],
[ 0.  ,  0.45,  0.45,  0.,  0.1 ],
[ 0.  ,  0.45,  0.45,  0.,  0.1 ],
[ 0.  ,  0.  ,  0.  ,  1.,  0.  ],
[ 0.  ,  0.  ,  0.  ,  0.,  0.  ]])
states = np.array([ 1,  1,  1,  2,  2,  2,  2,  2,  1])
newtrans = np.array([[ 0.  ,  1. ,  0. ,  0.  ],
[ 0.  ,  0.5,  0.25,  0.25 ],
[ 0.  ,  0.2,  0.8 ,  0    ],
[ 0.  ,  0.  ,  0.  ,  0.  ]])
newmeans = np.array([[ 1.  ,  1.66]])
newcovs = np.array([[[ 0.015 ]], [[ 1.0464]]])
nll = 22.290642196869609
trans, dists, nll = hmm.hmm(data, lengths, trans)
print trans
#     [[ 0.    1.    0.    0.  ]
#      [ 0.    0.5   0.25  0.25]
#      [ 0.    0.2   0.8   0.  ]
#      [ 0.    0.    0.    0.  ]]
print [a.mean() for a in dists]
#     [array([ 1.]), array([ 1.66])]
print [a.cov() for a in dists] 
#     [array([[ 0.015]]), array([[ 1.0464]])]
print 'Negative log-likelihood: ', nll
#   22.2906421969
    
npt.assert_almost_equal([a.mean() for a in ans[1]], newmeans.transpose(), decimal=4)
npt.assert_almost_equal([a.cov() for a in ans[1]], newcovs, decimal=4)
npt.assert_almost_equal(ans[2], nll, decimal=4)
print 'Optimal state sequence: ', viterbi(signal, trans, dists)[0]   
#   Optimal state sequence: [1 1 1 2 2 2 2 2 1]

