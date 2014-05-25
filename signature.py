import numpy as np
import glob
from dimred import PCA
import matplotlib.pyplot as plt

"""
Cycle through all signatures and show figeres of the centered_scaled_rot 
Superimposed for every signature
"""
for signcount in range (1,5):

	globfiles = glob.glob('./data/sign/sign%d/*.txt' % signcount)
	plt.figure()

	for numcount in globfiles:
		data = np.loadtxt(numcount,comments = '%')

		pca1 = PCA(data[:,0:2].T)
		csrdata = pca1.center_scale_rot()
		#plt.plot(csrdata[0,:],csrdata[1,:])

	#plt.show()


