import numpy as np
import pickle
import hmm

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
MarkovModel = []
for k in range (2,4):
	for a in traindataStack:
		trans = hmm.lrtrans(k)
		llist = getlistoflengths(a)
		MarkovModel.append(hmm.hmm(np.column_stack(a),llist,trans))
	print np.shape(testdataStack)
	print np.shape(testdataStack[0][0])
	classifiedtests = []
	for a in range (np.shape(testdataStack)[0]):
		llist = getlistoflengths(testdataStack[a])
		for b in range (len(llist)):
			temp = np.array([testdataStack[a][b]])
			print np.shape(temp)
			templistlen = []
			for c in range (len(keyList)):
				templistlen.append(hmm.negloglik(temp,trans = MarkovModel[c][0],dists = MarkovModel[c][1]))
			classifiedtests.append(keyList[(np.argmin(templistlen))])
		print classifiedtests
	print k
#for i in range (2,4):
#	print MarkovModel[i]
