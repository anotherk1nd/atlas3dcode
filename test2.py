"""
This method uses the Support Vector Machines classification machine learning algorithm of sklearn. 
We assign a binary measure of 0 for SR and 1 for FR and loop through the data
to convert the S/F classification to the binary. This is supplied to the tree.DecisionTreeClassifier()
algorithm, and trained on half the dataset, before testing on the second half.

"""
import matplotlib.pyplot as pl
import sklearn
from sklearn import datasets, svm, metrics, tree
from astropy.io import fits
import scipy as sp

pl.close('all')
fn = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Code\\atlas3dcode\\atlas3d.fit'
hdulist = fits.open(fn)
#hdulist.info()
tbdata = hdulist[1]
#print tbdata
cols = hdulist[1].columns
#cols.info()
#cols.names

#print cols[:,0]
#pl.plot(cols[:,0],cols[:,1])
#pl.show()
from astropy.io.fits import getheader
getheader(fn)  # get default HDU (=0), i.e. primary HDU's header
getheader(fn, 0)  # get primary HDU's header
getheader(fn, 1)  # the second extension

#data = hdulist[0].data # gives column names, info
#print data
data1 = hdulist[1].data
#print len(data1)
#print data1

#for i in range(len(data1)):
#print data1.field(3) # This returns all the values under the field of column 2
lamre = data1.field(9) # This holds the LambdaRe value
vsig = data1.field(7) # This holds velocity dispersion value, sigma
fs = data1.field(11) # Holds whether galaxy is fast or slow rotator
print fs
"""
#print lamre
pl.plot(lamre,vsig,'x')
pl.title('V/ vs Lambda at 1Re')#Re = Effective Radius
pl.xlabel(r'$\lambda_R$')
pl.ylabel(r'V/ $\sigma $')
"""
#We count the number fast vs slow rotators
fcount = 0
scount = 0
for i in range(len(fs)):
    if fs[i] == 'F':
        fcount +=1
    else:
        scount +=1
print 'Fcount:', fcount
print 'Scount:', scount

#We create an array containing a binary interpretation of the fast/slow
#categorisation, with 1 indicating fast rotator, so we can pass the array to 
#the classification machine learning algorithm
fslist = sp.zeros(len(fs))

for i in range(len(fs)):
    if fs[i] == 'F':
        fslist[i] = 1
    else:
        fslist[i] = 0
print 'Full list = ',fslist

#We split the array into 2 equally sized arrays to form a training and test set
fstrain = fslist[:len(fslist)/2]
fstest = fslist[len(fslist)/2:]

#We split the target variable list in 2 also
lamtrain = lamre[:len(fslist)/2]
lamtest = lamre[len(fslist)/2:]

#Training and test set formed by dividing arbitrarily in 2 by position in dataset
print 'Training set' ,fstrain
print 'Test set:',fstest

#fslist.reshape(-1,1) # We reshape the data since it has a single feature and doesn't like it otherwise

lamre = lamre[:,None] #Found soln http://stackoverflow.com/questions/32198355/error-with-sklearn-random-forest-regressor
lamtrain = lamtrain[:,None]
lamtest = lamtest[:,None]
#This method came from http://scikit-learn.org/stable/modules/svm.html#svm
clf = tree.DecisionTreeClassifier()
clf.fit(lamtrain,fstrain) # We train the tree using the lamre value and F,S classification as test
prediction = clf.predict(lamtest).copy()
print 'Prediction: ', prediction
print 'True Values: ', fstest

if prediction.all() == fstest.all():
    print 'YES'

