"""
Here we attempt to implement sklearn with the photometric data.
"""
import matplotlib
import matplotlib.pyplot as pl
matplotlib.style.use('ggplot')
import sklearn
from sklearn import datasets, svm, metrics, tree
from astropy.io import fits
import scipy as sp
import numpy as np
from astropy.io.fits import getheader
from itertools import product # For decision boundary plotting
import pandas as pd

pl.close('all')
fncsc = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Data\\atlas3d\\linkingphotometric.fit'
hdulistcsc = fits.open(fncsc)
hdulistcsc.info()
colnames = hdulistcsc[0]
#print colnames
tbdata = hdulistcsc[1]
#print tbdata
cols = hdulistcsc[1].columns
#cols.info()
#cols.names

#We import fast and slow pd dataframes to analyse the distributions individually THESE ARE IN PANDAS FORMAT WHEN READ IN
slow_rots1 = pd.read_csv(r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\slow_rots_fin.csv')
fast_rots1 = pd.read_csv(r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\fast_rots_fin.csv')

slow_rots1.plot.scatter(x='lam_sqrt_eps',y='n_b',color='DarkBlue',label='Slow Rotators')
fast_rots1.plot.scatter(x='lam_sqrt_eps',y='n_b',color='DarkGreen',label='Fast Rotators')
pl.axes()
pl.legend()
pl.show()
