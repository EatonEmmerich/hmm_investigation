'''Module containing various utility functions

@since: 10 Jan 2012

@author: skroon
'''
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt




def confusion(truth, pred):
    '''
    Generate and print a confusion matrix.  
    
    For the printing, the column widths containing
    the numbers should all be equal, and should be wide enough to accommodate the widest class name as
    well as the widest value in the matrix.
    
    Parameters
    ----------
    truth : (n,) list
        A list of the true class label for each data value.
        There are n data values.
    pred  : (n,) list
        A list of the class labels as returned by the system.
        
    Return
    ------
    result : dict
        A dictionary of the confusion matrix.
        
    Example
    -------

    >>> orig = ["Yellow", "Yellow", "Green", "Green", "Blue", "Yellow"]
    >>> pred = ["Yellow", "Green", "Green", "Blue", "Blue", "Yellow"]
    >>> result = confusion(orig, pred)
             Blue  Green Yellow
      Blue      1      0      0
     Green      1      1      0
    Yellow      0      1      2
    >>> result
    {('Yellow', 'Green'): 1, ('Green', 'Blue'): 1, ('Green', 'Green'): 1, ('Blue', 'Blue'): 1, ('Yellow', 'Yellow'): 2}
    '''
    print_  = True
    classes = set(truth)
    classes.union(set(pred))
    classes = list(classes)
    conf = {}
    for i, c in enumerate(truth):
        if conf.get((c, pred[i]), 0):
            conf[c, pred[i]] += 1
        else:
            conf[c, pred[i]] = 1
    if print_:
        max_ = 0
        for c in classes:
            if len(str(c)) > max_:
                max_ = len(str(c))
        for c in classes:
            for d in classes:
                if len(str(conf.get((c, d), 0))) > max_:
                    max_ = len(str(conf.get((c, d), 0)))
        print "%*s" % (max_, " "),
        for c in classes:
            print "%*s" % (max_, c),
        print
        for c in classes:
            print "%*s" % (max_, c),
            for d in classes:
                print "%*s" % (max_, conf.get((c, d), 0)),
            print
    return conf
###############################################################

def loadimages():
    import matplotlib.pyplot as plt
    import os
    import fnmatch
    """
    Load all the gray scale images in all the subdirectories with suffix `png`.
    The images are flattened and each image is represented as an (d,) array.
    
    Return
    ------
    
    images : (d,n) ndarray
       returns n, d-dimensional images.
    
    """
    matches = []
    for root, dirs, files in os.walk("./data/faces"):
        for filename in fnmatch.filter(files, '*.png'):
            matches.append(os.path.join(root, filename))
    data = []
    for m in matches:
        data.append(plt.imread(m).flatten())
    return np.column_stack(data)

def read_files_in_directory(dir_path):
    """
    Read diferent files from a directory.
    The path to the directory relative to current directory.
    This is a snippet that should be adapted for use in your 
    code
    
    Parameters
    ----------
    
    dir_path : char
       The directory containing the files
       
    Output
    ------
    
    In this snippet all files will be copied to to *.out
    
    Example
    -------
    read_files_in_directory('./data/sign/sign1/*.txt')
    """
    import glob
    list_of_files = glob.glob(dir_path)           # create the list of file
    for file_name in list_of_files:
       FI = open(file_name, 'r')
       FO = open(file_name.replace('txt', 'out'), 'w') 
    for line in FI:
       FO.write(line)

    FI.close()
    FO.close()


