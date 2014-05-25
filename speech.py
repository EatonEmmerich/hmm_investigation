import numpy as np
import pickle
import hmm
import utils

def normalize(tstack,trstack):
	"""
	"""
	for i in range (len(tstack)):
		tstackS = np.column_stack(tstack[i])
		mean = np.mean(tstackS,axis = 1)
		for j in range(len(tstack[i])):
			for k in range ((np.shape(tstack[i][j])[1])):
				tstack[i][j][:,k] = tstack[i][j][:,k]-mean
	for i in range (len(trstack)):
                trstackS = np.column_stack(trstack[i])
                mean = np.mean(trstackS,axis = 1)
                for j in range(len(trstack[i])):
                        for k in range ((np.shape(trstack[i][j])[1])):
                                trstack[i][j][:,k] = trstack[i][j][:,k]-mean


	return tstack,trstack

def getlistoflengths(stack):
	"""
	stack: (b,d,n)
	---------------------------------
	returns:
	lengthlist: n corresponding to each b in stack
	"""
	tlength = []
	Bt = np.shape(stack)[0]
	for b in range(Bt):
		tlength.append(np.shape(stack[b])[1])
	return tlength
	

# Read in pickled data:
f = open('./data/speech.dat') 
data = np.load(f)


# put pickled data in useable form
# print data.keys()
testdata = data.get('test')
traindata = data.get('train')
keyList = testdata.keys()
testdataStack = []
traindataStack = []
for a in keyList:
	testdataList = []
	traindataList = []
	for b in range(len(testdata.get(a))):
		testdataList.append((testdata.get(a)[b].T))
	for b in range(len(traindata.get(a))):
		traindataList.append((traindata.get(a)[b].T))
	testdataStack.append((testdataList))
	traindataStack.append((traindataList))
testdataStack,traindataStack = normalize(testdataStack, traindataStack)
#print np.shape(testdataStack[0][0])[1]
#llist = getlistoflengths(traindataStack[0])
#print llist
diagcov = False
print "Full Covariance Matrix"
for k in range (1,7):
	MarkovModel = []
	for a in traindataStack:
		trans = hmm.lrtrans(k)
		llist = getlistoflengths(a)
		MarkovModel.append(hmm.hmm(np.column_stack(a),llist,trans,diagcov = diagcov))
#	print np.shape(testdataStack)
#	print np.shape(testdataStack[0][0])
	TestClassification = []
	OriginalTestClassification = []
	TrainClassification = []
	OriginalTrainClassification = []
	for a in range (np.shape(testdataStack)[0]):
		llist = getlistoflengths(testdataStack[a])
		classifiedtests = []
		origtests = []
		for b in range (len(llist)):
			#origtests.append(keyList[a])
			temp = np.array([testdataStack[a][b]])
#			print np.shape(temp)
			templistlen = []
			for c in range (len(keyList)):
				templistlen.append(hmm.negloglik(temp,trans = MarkovModel[c][0],dists = MarkovModel[c][1]))
			#lassifiedtests.append(keyList[(np.argmin(templistlen))])
			TestClassification.append(keyList[(np.argmin(templistlen))])
			OriginalTestClassification.append(keyList[a])
		llist = getlistoflengths(traindataStack[a])
		for b in range (len(llist)):
			temp = np.array([traindataStack[a][b]])
			templistlen = []
			for c in range (len(keyList)):
				templistlen.append(hmm.negloglik(temp,trans = MarkovModel[c][0],dists = MarkovModel[c][1]))
			TrainClassification.append(keyList[(np.argmin(templistlen))])
			OriginalTrainClassification.append(keyList[a])
	print "Test Confusion Matrix"
	utils.confusion(OriginalTestClassification,TestClassification)
	print "Train Confusion Matrix"
	utils.confusion(OriginalTrainClassification,TrainClassification)
	print k
diagcov = True
print "Diagonal Covariance Matrix"
for k in range (1,7):
	MarkovModel = []
	for a in traindataStack:
		trans = hmm.lrtrans(k)
		llist = getlistoflengths(a)
		MarkovModel.append(hmm.hmm(np.column_stack(a),llist,trans,diagcov = diagcov))
#	print np.shape(testdataStack)
#	print np.shape(testdataStack[0][0])
	TestClassification = []
	OriginalTestClassification = []
	TrainClassification = []
	OriginalTrainClassification = []
	for a in range (np.shape(testdataStack)[0]):
		llist = getlistoflengths(testdataStack[a])
		classifiedtests = []
		origtests = []
		for b in range (len(llist)):
			#origtests.append(keyList[a])
			temp = np.array([testdataStack[a][b]])
#			print np.shape(temp)
			templistlen = []
			for c in range (len(keyList)):
				templistlen.append(hmm.negloglik(temp,trans = MarkovModel[c][0],dists = MarkovModel[c][1]))
			#lassifiedtests.append(keyList[(np.argmin(templistlen))])
			TestClassification.append(keyList[(np.argmin(templistlen))])
			OriginalTestClassification.append(keyList[a])
		llist = getlistoflengths(traindataStack[a])
		for b in range (len(llist)):
			temp = np.array([traindataStack[a][b]])
			templistlen = []
			for c in range (len(keyList)):
				templistlen.append(hmm.negloglik(temp,trans = MarkovModel[c][0],dists = MarkovModel[c][1]))
			TrainClassification.append(keyList[(np.argmin(templistlen))])
			OriginalTrainClassification.append(keyList[a])
	print "Test Confusion Matrix"
	utils.confusion(OriginalTestClassification,TestClassification)
	print "Train Confusion Matrix"
	utils.confusion(OriginalTrainClassification,TrainClassification)
	print k
