import matplotlib.pyplot as pl
import sklearn
from sklearn import datasets, svm, metrics, tree
from astropy.io import fits
import scipy as sp
from astropy.io.fits import getheader
import pandas as pd
pl.close('all')

fr = r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\fast_rots_edit.csv'
dffr = pd.read_csv(fr)

sr = r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\slow_rots_edit.csv'
dfsr = pd.read_csv(sr)

al = r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\all_rots_edit.csv'
dfall = pd.read_csv(al)
"""
ax = dffr.plot.scatter(x='lam_sqrt_eps', y='D/T', color='DarkBlue', label='Fast Rotators')
dfsr.plot.scatter(x='lam_sqrt_eps', y='D/T', color='DarkGreen', label='Slow Rotators',ax=ax)
pl.show()
"""
pl.plot(dffr['D/T'],dffr['lam_sqrt_eps'],'rx',label='Fast Rotators')
pl.plot(dfsr['D/T'],dfsr['lam_sqrt_eps'],'go',label='Slow Rotators')
pl.xlabel('D/T')
pl.ylabel(r'$\lambda_{Re}$')
pl.ylim(-0.1,1.0)
pl.legend()
pl.show()
