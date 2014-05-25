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

MarkovModel = []
for a in traindataStack:
	MarkovModel.append(hmm.hmm())
