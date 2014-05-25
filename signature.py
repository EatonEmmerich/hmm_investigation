import numpy as np
import glob
from dimred import PCA
import matplotlib.pyplot as plt
import hmm

"""
Cycle through all signatures and show figeres of the centered_scaled_rot 
Superimposed for every signature
"""

signatures = []
for signcount in range (1,6):
	tempsig = []
	globfiles = glob.glob('./data/signatures/sign%d/*.txt' % signcount)

	for numcount in globfiles:
		data = np.loadtxt(numcount,comments = '%')

		pca1 = PCA(data[:,0:2].T)
		csrdata = pca1.center_scale_rot()
		#plt.plot(csrdata[0,:],csrdata[1,:])
		tempsig.append(csrdata)
	#plt.show()
signatures.append(tempsig)

diagcov = False
print np.shape(signatures)
print "Full Covariance Matrix"
for k in range (1,7):
	MarkovModel = []
	for a in range (1,6):
		trans = hmm.lrtrans(k)
		llist = []
		for b in range(0,4):
			llist.append([len(signatures[a-1][b,:])])
		MarkovModel.append(hmm.hmm(np.column_stack(signatures[a-1][0:3,:]),llist,trans,diagcov = diagcov))
	print np.shape(testdataStack)
#   print np.shape(testdataStack[0][0])
	TestClassification = []
	OriginalTestClassification = []
	TrainClassification = []
	OriginalTrainClassification = []

