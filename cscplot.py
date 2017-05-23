# -*- coding: utf-8 -*-
"""
Here we attempt to implement sklearn with the photometric data.
"""
import itertools
import matplotlib.pyplot as pl
import sklearn
from sklearn import datasets, svm, metrics, tree
from sklearn.metrics import confusion_matrix
from astropy.io import fits
import scipy as sp
import numpy as np
from astropy.io.fits import getheader
from itertools import product # For decision boundary plotting
import plotly.plotly as py #for barplot
import plotly.graph_objs as go # for barplot
py.sign_in('funkytimes', 'cQkjfMpg44MA2el0F6VB')
import pandas as pd
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=pl.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    pl.imshow(cm, interpolation='nearest', cmap=cmap)
    pl.title(title)
    pl.colorbar()
    tick_marks = np.arange(len(classes))
    pl.xticks(tick_marks, classes, rotation=45)
    pl.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm,2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pl.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pl.tight_layout()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')

fr = r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\fast_rots_edit_corrected.csv'
dffr = pd.read_csv(fr)

sr = r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\slow_rots_edit.csv'#slow rots didnt need correcting
dfsr = pd.read_csv(sr)

al = r'C:\Users\Joshua\Documents\Term 1\Project\Code\atlas3dcode\all_rots_edit_corrected.csv'
dfall = pd.read_csv(al)

pl.close('all')
fncsc = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Data\\atlas3d\\linkingphotometric.fit'
hdulistcsc = fits.open(fncsc)
hdulistcsc.info()
colnames = hdulistcsc[0]
#print colnames
tbdata = hdulistcsc[1]
#print tbdata
#cols = hdulistcsc[1].columns
#cols.info()
#cols.names

#print cols[:,0]
#pl.plot(cols[:,0],cols[:,1])
#pl.show()

#getheader(fn)  # get default HDU (=0), i.e. primary HDU's header
#getheader(fn, 0)  # get primary HDU's header
#getheader(fn, 1)  # the second extension

colinfo = hdulistcsc[0].data # gives column names, info
#print colinfo
datacsc = hdulistcsc[1].data
csc = datacsc.field(5)[1:] #Using from 1 excludes the field name which is included in the data CERSIC INDEX FROM SINGLE FIT


#We import the atlas3d data in order to access the lambda value and use the classifier code used in 1st attempt to implement classifier sklearn
fnrotlam = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Code\\atlas3dcode\\atlas3d.fit'
hdulistlam = fits.open(fnrotlam)
colinfoatlas = hdulistlam[0].data
#print colinfoatlas
datalam = hdulistlam[1].data
eps = datalam.field(4)
lam = datalam.field(7)
lamtest = lam[len(lam)/2:]
epstest = eps[len(eps)/2:]
lamepstest = sp.divide(lamtest,sp.sqrt(epstest))
print 'This is lameps', lam

fs = datalam.field(11)

#We create an array containing a binary interpretation of the fast/slow
#categorisation, with 1 indicating fast rotator, so we can pass the array to 
#the classification machine learning algorithm
fslist = sp.zeros(len(fs))

for i in range(len(fs)):
    if fs[i] == 'F':
        fslist[i] = 1
    else:
        fslist[i] = 0
#print 'Full list = ',fslist



#We split the array into 2 equally sized arrays to form a training and test set
fstrain = fslist[:len(fslist)/2]
fstest = fslist[len(fslist)/2:]

#We split the target variable list in 2 also
csctrain = csc[:len(fslist)/2]
csctest = csc[len(fslist)/2:]

#Training and test set formed by dividing arbitrarily in 2 by position in dataset
#print 'Training set' ,fstrain
#print 'Test set:',fstest

#lamre = lamre[:,None] #Found soln http://stackoverflow.com/questions/32198355/error-with-sklearn-random-forest-regressor
csctrain = csctrain[:,None]
csctest = csctest[:,None]


#This method came from http://scikit-learn.org/stable/modules/svm.html#svm
clf = tree.DecisionTreeClassifier()
clf.fit(csctrain,fstrain) # We train the tree using the lamre value and F,S classification as test


prediction_csc = clf.predict(csctest).copy()
print 'Prediction: ', prediction_csc
print 'True Values: ', fstest
correct = []
incorrect = []

for i in range(len(fstest)):
    if fstest[i] == prediction_csc[i]:
        correct.append(i)
    if fstest[i] != prediction_csc[i]:
        incorrect.append(i)
        
#print correct
correctarray = []
incorrectarray = []

csc = csc.tolist()
cscarray = sp.asarray(csc,dtype=float)
csctest = csc[len(fslist)/2:]


for i in range(len(fstest)):
    if fstest[i] == prediction_csc[i]:
        correctarray.append([float(csctest[i]),lamepstest[i]])
    if fstest[i] != prediction_csc[i]:
        incorrectarray.append([float(csctest[i]),lamepstest[i]])

print correctarray

def column(matrix, i):
    return [row[i] for row in matrix]

pl.plot(column(correctarray,0),column(correctarray,1),'go')
pl.show()

#pl.title('Success of Classification Predictions Based on Se\'rsic Index')
#pl.xlabel('Se\'rsic Index')
#pl.ylabel('Lambda Value')
#pl.xlim(-0.1,1.0)
#pl.legend([correct, incorrect], ['Correct','Incorrect'])
#pl.show()
        

#print clf.score(csctest,fstest) #This computes the success rate by itself
#print clf.get_params()
