import numpy as np
import glob
from dimred import PCA
import matplotlib.pyplot as plt
import hmm
import utils

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
	#print signcount
	#print "with length:"
	#print len(tempsig)
	#plt.show()
	signatures.append(tempsig)

diagcov = False
#print np.shape(signatures)
#print np.shape(np.column_stack(signatures[1][0:3]).T)
print "Full Covariance Matrix"
for k in range (1,7):
	MarkovModel = []
	for a in range (0,5):
		trans = hmm.lrtrans(k)
		llist = []
		for b in range(0,3):
#			print np.shape(signatures[a][b])
			llist.append([len(signatures[a][b])])
		MarkovModel.append(hmm.hmm(np.column_stack(signatures[a][0:3]),llist,trans,diagcov = diagcov))
	#print np.shape(testdataStack)
#   print np.shape(testdataStack[0][0])
	TestClassification = []
	OriginalTestClassification = []
	TrainClassification = []
	OriginalTrainClassification = []
	for a in range (0,5):
		for b in range (3,len(signatures[a])):
			OriginalTestClassification.append(a)
			templist = []
			for c in range (0,5):
				#print np.shape(np.array([np.column_stack(signatures[a][b]).T]))
				templist.append(hmm.negloglik(np.array([np.column_stack(signatures[a][b]).T]),trans = MarkovModel[c][0],dists = MarkovModel[c][1]))
			TestClassification.append(np.argmin(templist))
	utils.confusion(OriginalTestClassification,TestClassification)
	print "completed for k = %d" %k
diagcov = True
#print np.shape(signatures)
#print np.shape(np.column_stack(signatures[1][0:3]).T)
print "Diagonal Covariance Matrix"
for k in range (1,7):
        MarkovModel = []
        for a in range (0,5):
                trans = hmm.lrtrans(k)
                llist = []
                for b in range(0,3):
#                       print np.shape(signatures[a][b])
                        llist.append([len(signatures[a][b])])
                MarkovModel.append(hmm.hmm(np.column_stack(signatures[a][0:3]),llist,trans,diagcov = diagcov))
        #print np.shape(testdataStack)
#   print np.shape(testdataStack[0][0])
        TestClassification = []
        OriginalTestClassification = []
        TrainClassification = []
        OriginalTrainClassification = []
        for a in range (0,5):
                for b in range (3,len(signatures[a])):
                        OriginalTestClassification.append(a)
                        templist = []
                        for c in range (0,5):
                                #print np.shape(np.array([np.column_stack(signatures[a][b]).T]))
                                templist.append(hmm.negloglik(np.array([np.column_stack(signatures[a][b]).T]),trans = MarkovModel[c][0],dists = MarkovModel[c][1]))
                        TestClassification.append(np.argmin(templist))
        utils.confusion(OriginalTestClassification,TestClassification)
        print "completed for k = %d" %k


	

