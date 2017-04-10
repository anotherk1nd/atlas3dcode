import scipy as sp
from scipy import stats
import numpy as np
np.set_printoptions(threshold='inf')
import matplotlib.pyplot as pl
import matplotlib
matplotlib.style.use('classic')
pl.close('all')
from astropy.io import fits
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import radviz

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
#print list(slow_rots.columns.values)

#We remove spaces in the column names
fast_rots.rename(columns={'mu_e ': 'mu_e', 'R_e  ': 'R_e', 'n   ': 'n', 'n_b ': 'n_b', 'q_b ': 'q_b', 'm0_d ': 'm0_d'}, inplace=True)
fast_rots.rename(columns={'R_d ': 'R_d', 'q_d ': 'q_d', 'D/T ': 'D/T'}, inplace=True)
slow_rots.rename(columns={'mu_e ': 'mu_e', 'R_e  ': 'R_e', 'n   ': 'n', 'n_b ': 'n_b', 'q_b ': 'q_b', 'm0_d ': 'm0_d'}, inplace=True)
slow_rots.rename(columns={'R_d ': 'R_d', 'q_d ': 'q_d', 'D/T ': 'D/T'}, inplace=True)

#fast_rots.to_csv(r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\fast_rots_fin.csv')
#slow_rots.to_csv(r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\slow_rots_fin.csv')

#print flist, slist
#We perform a linear regression using least squares fit
"""
slope, intercept, r_value, p_value, std_err = stats.linregress(cscfloat,lamarray)
print slope, intercept, r_value, p_value, std_err

slope, intercept, r_value, p_value, std_err = stats.linregress(cscfloat,lameps)
print 'Including eps ',slope, intercept, r_value, p_value, std_err
"""
"""
pl.scatter(slow_rots.Rmax,slow_rots.lam_sqrt_eps,c='r',label='Slow Rotators')
pl.scatter(fast_rots.Rmax,fast_rots.lam_sqrt_eps,c='g',label='Fast Rotators')
#slow_rots.pl.scatter(x='Rmax', y='lam_sqrt_eps',color='DarkBlue',title='Slow Rotators Rmax Dependence on Lambda')
#fast_rots.plot.scatter(x='Rmax', y='lam_sqrt_eps',color='Pink',title='Fast Rotators Rmax Dependence on Lambda')
pl.title('Rotators Rmax Dependence on Lambda')
pl.xlabel('Rmax')
pl.ylim(ymin=0.0)
pl.ylabel(r"$\lambda_{Re}$")
pl.show()
"""
slow_rots_edit = slow_rots.drop(['_RAJ2000', '_DEJ2000','id'],axis=1)
slow_rots_edit.info()
params =['Rmax','epse','lam_sqrt_eps','D/T','n','n_b','q_b','m0_d','R_d','q_d','mu_e']
slow_rots_edit[params] = slow_rots_edit[params].astype(float)
slow_rots_edit.info()
#scatter_matrix(slow_rots_edit[params],alpha=0.2,figsize=(12, 12), diagonal='kde')
#pl.show()

fast_rots_edit = fast_rots.drop(['_RAJ2000', '_DEJ2000','id'],axis=1)
#fast_rots_edit.info()
params =['Rmax','epse','lam_sqrt_eps','D/T','n','n_b','q_b','m0_d','R_d','q_d','mu_e']
fast_rots_edit[params] = fast_rots_edit[params].astype(float)
#fast_rots_edit.info()
#scatter_matrix(fast_rots_edit[params],alpha=0.2,figsize=(12, 12), diagonal='kde')
#pl.show()
params1 =['D/T','n','n_b','q_b','m0_d','R_d','q_d','mu_e','F_S'] #INCLUDES F/S
all_rots = pd.concat([fast_rots_edit,slow_rots_edit])
print all_rots[params1]
#radviz(all_rots[params1], 'F_S')
#pl.show()