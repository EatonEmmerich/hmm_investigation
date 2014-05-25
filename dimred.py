"""Module containing classes for dimension reduction

 - PCA - class for principal components analysis given d-dimensianal data
 - LDA - class for linear discriminant analysis

"""

# Python 3 compatibility
from __future__ import division


import numpy as np


class PCA(object):
    """
    Class for performing and querying Principal Component Analysis (PCA).

    Parameters
    ----------
    data : (d, n) ndarray
        Array of n d-dimensional samples, one sample per column.

    Attributes
    ----------
    d : int
        Dimensionality of data-set.
    n : int
        Number of samples.
    mean : (d,) ndarray
        Mean of data.
    u : (d, d) ndarray
        Rotation matrix, aligns data with axes.
    s : (d, d) ndarray
        Diagonal scaling matrix.
    vh : (d, n) ndarray
        Whitened data.
    rank : int
        Rank of covariance matrix.
    ncomp : int
        Number of principal components to be used in reconstruction.  May be
        set by the user.
    explained_proportion
        The proportion of variance explained by the current
        number of principal components.


    Methods
    -------
    get_ncomp
        Returns the current number of principal components.
    set_ncomp
        Change the current number of principal components
    keep_fraction
        Set number of principle components by fraction.
    explained_proportion
        The fraction of the information captured by the current number of
        principal components.
    project
        Project data onto the current principal components
    recon
        Reconstruct the data from the projection coefficients using the 
        current principal components
    get_pca_comp
        Returns the current principal components.        
    whiten
        Whiten the data (also squeeze out any empty dimensions)
    center_scale_rot
        Remove the mean of the data, rotate to align with principal axes,
        scale so that the largest singular value equals one.

    References
    ----------
    .. [1] Patrec 813 lecture notes,
           http://courses.ee.sun.ac.za/Pattern_Recognition_813/lectures/lecture01/node5.html

    Examples
    --------
    ...

    """
    def __init__(self, data):
        # Calculate shape of data
	d = data.shape
	n = d[1]
	d = d[0]

        if not n > d:  # Sanity check
            print ('Warning: The number of data points is less than the dimension')
                             
        # Calculate mean and remove from data 
	mean = np.mean(data,axis = 1)[:,np.newaxis]
	dn = data-mean

        # Calculate the reduced form of the SVD
	u,s,vh = np.linalg.svd(dn,full_matrices = 0)
	

        # Compute rank of covariance matrix 
	cov = np.cov(dn,bias=1)
        rank = np.size(np.where(s > 1e-12))
        
        # Pass values to class
        self.d, self.n, self.s, self.mean, self.rank = d, n, s, mean, rank
	self.u, self.vh = u,vh

        # Scale the principal components so the largest singular value is 1
        self.props = self.s / self.s[0]

        # Default number of components to be used
        self.ncomp = self.rank

    @property
    def get_ncomp(self):
        return self.ncomp

    def set_ncomp(self, n):
        if n > self.rank:
            print("Warning: only %d principal components - using all of them"
                  % self.rank)

        self.ncomp = min(self.rank, n)

    def keep_fraction(self, frac):
        """
        Keep only enough components to represent a certain fraction of the
        variance in the data.

        Parameters
        ----------
        frac : float
            Value between 0 and 1 representing the fraction of components to
            keep.

        Notes
        -----
        If you want to use zero principal components, set `ncomp` explicitly.

        """
        if frac >= 1.:
            self.ncomp = self.rank
        elif frac < 1.:
            ncomp = np.searchsorted(self.props[::-1], frac) + 1
            self.ncomp = min(ncomp,self.rank)
     

    @property
    def explained_proportion(self):
        """
        Get the proportion of variance explained by the current principal
        components.

        Returns
        -------
        p : float
            The proportion of variance in the transformed original data
            explained by the currently used principal components.  The value
            is between 0 and 1 (inclusive).

        """
        if self.ncomp < self.rank:
            return 1. - self.props[self.ncomp]
        else:
            return 1


    def project(self, data):
        """
        Project data onto the current principal components.

        Parameters
        ----------
        data : (d, m) ndarray
            Data set to project.

        Returns
        -------
        out : (ncomp, m) ndarray
            The result after normalizing `b` in the same way as the original
            data and projecting this normalized data onto the axes (there are
            `ncomp` of them) of the current principal components.

        """
        # First normalize data as done with original matrix. Then
        # transformation is by left-multiplication with first `ncomp` rows of
        # Ut. For b of shape (d, m), return transformed data of shape
        # (ncomp, m)
	dn = data-self.mean
	Ut = self.u.T
	Ut = Ut[:self.ncomp,:]
        # Calculate the shape of the data
	b = np.dot(Ut,dn)
	d, m = np.shape(data)
        # Sanity check
#        if not m == self.d:
#            raise ValueError('The dimension of the data %s does not match '
#                             'the dimension of the original dataset %s' % \
#                             (b.shape, (self.d, self.n)))

        # Calculate the projection coefficients
	coef = b


	
        

        return coef

    def recon(self, coef):
        """
        Given projection coefficients, reconstruct data.

        Parameters
        ----------
        coef : (p, n) ndarray
            Each column of `coef` contains one set of projection coefficients.
            The number of projection coefficients must be <= self.rank.

        Returns
        -------
        reconstructed_data : (d, n) ndarray
            Reconstructed data.

        """
        # Find the shape of the coefficients
        p, n = coef.shape

        # Sanity check
        if not p <= self.rank:
            raise ValueError("Dimension of coefficients must be <= than data rank")
        
        # Project back to original dimension, and restore the mean
        data = np.dot(self.u[:,:self.ncomp],coef)
	data = data + self.mean

        return data
    
    def get_pca_comp(self):
        """
        Current principal components in the original basis.

        Returns
        -------
        pc : (d, ncomp) ndarray
            Principal components, one per column, in the original system of
            axes.
        """
        
        return self.u[:, :self.ncomp]
    
    def whiten(self):
        """
        Return the whitened, reduced data.
        No diemsionality reduction is done, apart from discarding the `empty`
        dimensions.

        Returns
        -------
        w : (rank, n)
            Whitened, reduced data.
        """
        # Calculate whitened data
	w = self.vh[:self.ncomp,:]
        
        return w

    def center_scale_rot(self):
        """
        Remove the mean of the original data, rescale, and rotate so that the
        principal directions are aligned with the coordinate axes.  
        The data is scaled so that the largest singular value equals one.
        The `empty' dimensions are discarded but no further dimensionality 
        reduction is done. 

        Returns
        -------
        scdata : (d x n) ndarray
            Centered, scaled data.
        """
        
        # Calculate the centered, scaled, and rotated data
        dat_rot = np.diag(self.s).dot(self.vh)
	scdata = dat_rot/self.s[0]
	scdata = scdata[:self.ncomp,:]
        
        return scdata





class LDA(object):
    """
    Class for performing multi-class LDA.
        
        Parameters
        ----------
        data : (d, n) ndarray
            Array of n d-dimensional samples, one sample per column.
        classes :
        
        labels :
        
        Attributes
        ----------
        d : int
            Dimensionality of data-set.
        n : int
            Number of samples.
        nmbr_cl : scalar
            Number of classes
        max_pr_dim : scalar
            Maximum dimension of projection space
            max_pr_dim = min(nmbr_cl-1,r) where `r` is the rank of the 
            within-class scatter matrix
        pr_dim : scalar
            The current projection dimension (pr_dim <= max_pr_dim)
            
        
        Methods
        -------
        get_transformation_matrix
            Returns the matrix that transforms the data
        set_pr_dim
            Sets the projection dimension
        tr_data
            Projects data onto lower dimensional subspace.
            The dimension is dtermined by the value of pr_dim
        
        References
        ----------
        .. [1] Patrec 813 lecture notes,
            http://courses.ee.sun.ac.za/Pattern_Recognition_813/lectures/lecture01/node5.html
        
        Examples
        --------
    
    
    
    """


    def __init__(self, data, classes, labels):
        
        d,n       = data.shape       # Shape of data
        nmbr_cl   = len(classes)     # The number of classes
        
        # Global mean
	globalmean = np.zeros((d,1))
        for count1 in range(0,d):
		globalmean[count1,0] = np.mean(data[count1,:])
	
	classprob = np.zeros((nmbr_cl,1))
	for count1 in range(0,nmbr_cl):
		classprob[count1,0] = np.size(np.where(labels == classes[count1]))

	classmean = np.zeros((d,nmbr_cl))
	for count1 in range(0,nmbr_cl):
		for count2 in range(0,n): 
			if(labels[count2]==classes[count1]):
				classmean[:,count1] = classmean[:,count1] + data[:,count2]
		classmean[:,count1] = classmean[:,count1]/classprob[count1,0]
	classprob = classprob/n

        # Initiate within-, and between-class scatter
        whin_class_scat = np.zeros((d, d))
        btwn_class_scat = np.zeros((d, d))
		

        
        
        # Calculate within-class scatter, and between-class scatter
	whin_Q = np.zeros((d,d))
	btwn_Q = np.zeros((d,d))

        for count1 in range(0,nmbr_cl):
		classdata = np.squeeze(data[:,np.where(labels == classes[count1])])
		whin_class_scat = whin_class_scat + classprob[count1,0]*np.cov(classdata,bias = 1)
	
	btwn_class_scat = np.cov(data,bias = 1)-whin_class_scat	 
           
        # Calculate the PCA components (Q), and the eigenvalues (eig) of the
        # scatter matrices, using the svd.
        # Note: It is cheaper to calculate the eigenvalues/eigenvectors 
        # (using np.linalg.eigh), in which case one has
        # to sort the eigenvalues in decreasing order.
        whin_Q,whin_sing,void = np.linalg.svd(whin_class_scat,full_matrices=0)
	btwn_Q,btwn_sing,void = np.linalg.svd(btwn_class_scat,full_matrices=0)
        
        # Prune the singular part of within class scatter
        whin_sing = whin_sing[np.where(whin_sing > 1e-12)]
        
        # Now do the simultaneous diagonalization
        
        
        # Find maximum projection dimension
        max_pr_dim = d
       	whin_rank = d 
         
        
        # Pass variables to class
        
        self.whin_class_scat = whin_class_scat
        self.btwn_class_scat = btwn_class_scat
        self.whin_Q          = whin_Q[:,:whin_rank]
        self.whin_sing       = whin_sing[:whin_rank]
        self.btwn_Q          = btwn_Q[:,:max_pr_dim]
        self.btwn_sing       = btwn_sing[:max_pr_dim]
        self.whin_rank       = whin_rank
        self.nmbr_cl         = nmbr_cl
        self.proj_dim        = max_pr_dim
        self.max_pr_dim      = max_pr_dim
            
    @property
    def get_transformation_matrix(self):
        """
        Get transformation matrix, using the current dimension of the projection 
        space.
        The default value is the maximum projection dimension
        Data is projected by: project = W.T * data
        
        Return
        ------
        W : (d,d') ndarray
           d is original data dimension, d' is the reduced dimension
        """
        # Calculate W
        
        
        return W
        
    
    
    
    def set_proj_dim(self, n):
        if (n > self.max_pr_dim):
            print("Warning: specified projection dimension too large - use max")
        
        self.proj_dim = min(self.max_pr_dim, n)

    

    def tr_data(self, data, proj_dim = None):
        """
        Project data  onto the space determined by the original data.
        
        Parameters
        -----------
        data : (d,m) ndarray
        The data that needs to be projected
        
        proj_dim : (scalar)
        The dimension of the subspace onto which the data is projected.
        If proj_dim == None: The current projection dimension as set by 
        set_proj_dim() is used. 
        The default value is the rank of the within class scatter matrix
        Note: The maximum dimension of the projected space is one less than 
        the number of classes, i.e. within_rank <= k (number of classes)
        
        Return
        ------
        proj_data : (proj_dim,m) ndarray
        The projected data
        """
         
        # Sanity Check        
        if not proj_dim == None:
            if proj_dim > self.max_pr_dim:
                print("Warning: The projection dimension exceeds maximum.")
                proj_dim  = max_pr_dim
            else:
                proj_dim  = proj_dim
        if proj_dim == None:
            proj_dim = self.max_pr_dim

        self.set_proj_dim(proj_dim)
        W   = self.get_transformation_matrix
        return W.T.dot(data)



