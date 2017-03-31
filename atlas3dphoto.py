"""
Here we attempt to implement sklearn with the photometric data.
"""
import matplotlib.pyplot as pl
import sklearn
from sklearn import datasets, svm, metrics, tree
from astropy.io import fits
import scipy as sp
from astropy.io.fits import getheader

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

#print cols[:,0]
#pl.plot(cols[:,0],cols[:,1])
#pl.show()

#getheader(fn)  # get default HDU (=0), i.e. primary HDU's header
#getheader(fn, 0)  # get primary HDU's header
#getheader(fn, 1)  # the second extension

#colinfo = hdulist[0].data # gives column names, info
#print colinfo
datacsc = hdulistcsc[1].data
csc = datacsc.field(5)[1:] #Using from 1 excludes the field name which is included in the data CERSIC INDEX FROM SINGLE FIT
print csc

#We import the atlas3d data in order to access the lambda value and use the classifier code used in 1st attempt to implement classifier sklearn
fnrotlam = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Code\\atlas3dcode\\atlas3d.fit'
hdulistlam = fits.open(fnrotlam)
datalam = hdulistlam[1].data


fs = datalam.field(11)
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
csctrain = csc[:len(fslist)/2]
csctest = csc[len(fslist)/2:]

#Training and test set formed by dividing arbitrarily in 2 by position in dataset
print 'Training set' ,fstrain
print 'Test set:',fstest

#lamre = lamre[:,None] #Found soln http://stackoverflow.com/questions/32198355/error-with-sklearn-random-forest-regressor
csctrain = csctrain[:,None]
csctest = csctest[:,None]

#This method came from http://scikit-learn.org/stable/modules/svm.html#svm
clf = tree.DecisionTreeClassifier()
clf.fit(csctrain,fstrain) # We train the tree using the lamre value and F,S classification as test
prediction = clf.predict(csctest).copy()
print 'Prediction: ', prediction
print 'True Values: ', fstest

#We assess the accuracy of the predictions. for some reason, the prediction.all method doesn't work,
#so had to code it manually.
true = 0
false = 0
for i in range(len(prediction)):
    if prediction[i] == fstest[i]:
        true += 1
    else:
        false += 1

print 'True: ',true
print 'False: ',false
total = true + false
print 'Success rate: ', round(float(true)/total,2)
#We see a success rate of around 71% compared to what would be 50% for random guesses due to binary nature

#Now we try to apply the sklearn.clf with more than 1 variable. We will use n, the Disk to Total light ratio (D/T) (col 20),





