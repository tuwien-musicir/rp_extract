# PLOTTING FUNCTIONS for RP_EXTRACT features

# 2015-04 by Thomas Lidy

import matplotlib.pyplot as plt
# from scipy.io import wavfile
import pylab
#from pylab import pcolor, show, colorbar, xticks, yticks
from numpy import corrcoef, sum, log, arange

# there is a pre-packaged namespace that contains the whole Numpy-Scipy-Matplotlib stack in one piece:
# from pylab import *
# however: `%matplotlib` prevents importing * from pylab and numpy

#sys.path.append('../RP_extract_python')
#import rp_extract_python as rp

#np.set_printoptions(suppress=True)

# enable inline graphics / plots in ipython notebook
get_ipython().magic(u'pylab inline')

def plotmatrix(matrix):
    pylab.figure()
    pylab.imshow(matrix, origin='lower', aspect='auto',interpolation='nearest')
    plt.xlabel('Mod. Frequency Index')
    pylab.ylabel('Frequency [Bark]')
    pylab.show()

# alternate version using pcolor 
def plotmatrix2(matrix):
    pcolor(matrix)
    #colorbar()
    #yticks(arange(0.5,10.5),range(0,10))
    #xticks(arange(0.5,10.5),range(0,10))
    show()
    
def plotrp(matrix):
    plotmatrix(matrix)
    
def plotssd(matrix):
    pylab.figure()
    pylab.imshow(matrix, origin='lower', aspect='auto',interpolation='nearest')
    pylab.xticks(range(0,7), ['mean', 'var', 'skew', 'kurt', 'median', 'min', 'max'])
    pylab.ylabel('Frequency [Bark]')
    pylab.show()

def plotrh(hist):
    plt.bar(range(0,60),hist) # 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel('Mod. Frequency Index')
    #plt.ylabel('Probability')
    plt.title('Rhythm Histogram')
    plt.show()
    

# PLOTTING EXAMPLES

## This is how to RESHAPE in case needed
# rpf = feat["rp"].reshape(24,60,order='F')  # order='F' means Fortran compatible; Alex uses it in rp_extract flatten() to be Matlab compatible
# print rpf.shape
# plotmatrix(rpf)

# ssd = feat["ssd"].reshape(24,7,order='F')  # order='F' means Fortran compatible; Alex uses it in rp_extract flatten() to be Matlab compatible
# print ssd.shape

# plotssd(ssd)

# plotrh(feat["rh"])

