import scipy as sp
from scipy import stats
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits

fncsc = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Data\\atlas3d\\linkingphotometric.fit'
hdulistcsc = fits.open(fncsc)
hdulistcsc.info()
colnames = hdulistcsc[0]
#print colnames
tbdatacsc = hdulistcsc[1]
#print tbdata
cols = tbdatacsc.columns
datacsc = hdulistcsc[1].data
csc = datacsc.field(5)[1:] #Using from 1 excludes the field name which is included in the data CERSIC INDEX FROM SINGLE FIT
print 'This is csc: ', csc

fnrotlam = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Code\\atlas3dcode\\atlas3d.fit'
hdulistlam = fits.open(fnrotlam)
colinfoatlas = hdulistlam[0].data
print colinfoatlas
datalam = hdulistlam[1].data
#vsig = datalam.field(7)
#print 'This is vsig: ', vsig
lam = datalam.field(9)
#print 'This is lam', lam
eps = datalam.field(4) #ellipticity

#We create arrays for to be able to plot the data, rather than as a list
cscarray = sp.array(csc)
lamarray = sp.array(lam)
lameps = lam/eps

cscfloat = cscarray.astype(np.float)

#We perform a linear regression using least squares fit
slope, intercept, r_value, p_value, std_err = stats.linregress(cscfloat,lamarray)
print slope, intercept, r_value, p_value, std_err

slope, intercept, r_value, p_value, std_err = stats.linregress(cscfloat,lameps)
print 'Including eps ',slope, intercept, r_value, p_value, std_err