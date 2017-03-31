import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, svm, metrics
from astropy.io import fits
fn = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Code\\atlas3dcode\\atlas3d.fit'
hdulist = fits.open(fn)
hdulist.info()
tbdata = hdulist[1]
print tbdata
cols = hdulist[1].columns
cols.info()
#cols.names

#print cols[:,0]
#plt.plot(cols[:,0],cols[:,1])
#plt.show()
from astropy.io.fits import getheader
getheader(fn)  # get default HDU (=0), i.e. primary HDU's header
getheader(fn, 0)  # get primary HDU's header
getheader(fn, 1)  # the second extension

data = hdulist[0].data # gives column names
print data
#data1 = hdulist[1].data
#print len(data1)
print data1

#for i in range(len(data1)):
#print data1.field(3) # This returns all the values under the field of column 2
#lamre = data1.field(9) # This holds the LambdaRe value
    