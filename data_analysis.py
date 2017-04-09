import scipy as sp
from scipy import stats
import numpy as np
np.set_printoptions(threshold='inf')
import matplotlib.pyplot as pl
pl.close('all')
from astropy.io import fits
import pandas as pd

def print_full(x): #Function to see full tables
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns',len(x))
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

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
#print 'This is csc: ', csc

fnrotlam = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Code\\atlas3dcode\\atlas3d.fit'
hdulistlam = fits.open(fnrotlam)
colinfoatlas = hdulistlam[0].data
#print colinfoatlas
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

fs = datalam.field(11)
fcount = 0
scount = 0
#fast_rots = sp.empty([224,18])
#print fast_rots
#slow_rots = sp.empty([36,18])
flist=[]
slist=[]
for i in range(len(fs)):
    if fs[i] == 'F':
        flist.append(i)
    else:
        slist.append(i)

fast_rots = pd.DataFrame.from_dict(datalam[flist])
fast_rots1 = fast_rots.assign(id=flist)#we add column with position in original fits file
fast_rots_cersic = pd.DataFrame.from_dict(datacsc[flist])
fast_rots_cersic.columns = fast_rots_cersic.iloc[0]
#print fast_rots_cersic
frames_fast_rots = [fast_rots1,fast_rots_cersic]
#print frames_fast_rots
fast_rots = pd.concat(frames_fast_rots,axis=1)
fast_rots = fast_rots[1:]
slow_rots = pd.DataFrame.from_dict(datalam[slist])
slow_rots1 = slow_rots.assign(id=slist)
slow_rots_cersic = pd.DataFrame.from_dict(datacsc[slist])
slow_rots_cersic.columns = fast_rots_cersic.columns
frames_slow_rots = [slow_rots1,slow_rots_cersic]
slow_rots = pd.concat(frames_slow_rots,axis=1)
#print 'slow'
#print slow_rots #Slow rotators total panda is GOOD!
#print 'fast'
#print_full(fast_rots)
delete = ['Simbad','NED','LEDA','_RA','_DE','name']
fast_rots.drop(['Simbad','NED','LEDA','_RA','_DE','name     '], axis=1, inplace=True)#Delete repeated and unnecessary columns
#print fast_rots
slow_rots.drop(['Simbad','NED','LEDA','_RA','_DE','name     '], axis=1, inplace=True)#Delete repeated and unnecessary columns
fast_rots_lam_Re = fast_rots.iloc[:,10] # Pick out lamre
fast_rots_eps = fast_rots.iloc[:,5]
fast_lam_sqrt_eps = fast_rots_lam_Re.div(sp.sqrt(fast_rots_eps))
#print fast_lam_sqrt_eps
fast_rots.loc[:,35] = fast_lam_sqrt_eps[:]
fast_rots.rename(columns={35:'lam_sqrt_eps'},inplace=True)
#print_full(slow_rots)
#print fast_rots.iloc[:,10]
slow_rots_lam_Re = slow_rots.iloc[:,10]
slow_rots_eps = slow_rots.iloc[:,5]
slow_lam_sqrt_eps = slow_rots_lam_Re.div(sp.sqrt(slow_rots_eps))
#print fast_lam_sqrt_eps
slow_rots.loc[:,35] = slow_lam_sqrt_eps[:]
slow_rots.rename(columns={35:'lam_sqrt_eps'},inplace=True)
#print_full(fast_rots)
print list(slow_rots.columns.values)

fast_rots.to_csv(r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\fast_rots_fin.csv')
slow_rots.to_csv(r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\slow_rots_fin.csv')

#print flist, slist
#We perform a linear regression using least squares fit
"""
slope, intercept, r_value, p_value, std_err = stats.linregress(cscfloat,lamarray)
print slope, intercept, r_value, p_value, std_err

slope, intercept, r_value, p_value, std_err = stats.linregress(cscfloat,lameps)
print 'Including eps ',slope, intercept, r_value, p_value, std_err
"""
slow_rots.plot.scatter(x='Rmax', y='lam_sqrt_eps',color='DarkBlue',title='Slow Rotators Rmax Dependence on Lambda')
fast_rots.plot.scatter(x='Rmax', y='lam_sqrt_eps',color='Pink',title='Fast Rotators Rmax Dependence on Lambda')
pl.show()
