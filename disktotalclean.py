# -*- coding: utf-8 -*-
"""
Here we attempt to implement sklearn with the photometric data.
"""
import matplotlib.pyplot as pl
import sklearn
from sklearn import datasets, svm, metrics, tree
from astropy.io import fits
import scipy as sp
from astropy.io.fits import getheader
from itertools import product # For decision boundary plotting
import plotly.plotly as py #for barplot
import plotly.graph_objs as go # for barplot
py.sign_in('funkytimes', 'cQkjfMpg44MA2el0F6VB')
import pandas as pd

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
lam = datalam.field(7)
#print 'This is lam', lam

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
#clas = clf.fit(csctrain,fstrain) # We train the tree using the lamre value and F,S classification as test

"""
prediction = clf.predict(csctest).copy()
#print 'Prediction: ', prediction
#print 'True Values: ', fstest

print clf.score(csctest,fstest) #This computes the success rate by itself
print clf.get_params()
"""

#We create arrays for to be able to plot the data, rather than as a list
cscarray = sp.array(csc)
datalamarray = sp.array(lam)
lamtest = lam[len(fslist)/2:] 

""" THIS PLOTS CLASSIFICATION OF ROTATION AS A FUNCTION OF CERSIC INDEX
for i in range(len(fs)):
   # correct = pl.plot(cscarray[i],datalamarray[i], 'bo')
    if fs[i] == 'F':
        fast_rotators, = pl.plot(cscarray[i],datalamarray[i],'rx')
    else:
        slow_rotators, = pl.plot(cscarray[i],datalamarray[i],'gx')


#pl.plot(csc,datalam,'o')
pl.title('Classification of Rotation as a function of Cersic Index')
pl.xlabel('Cersic Index')
pl.ylabel('Lambda Value')
pl.legend([fast_rotators, slow_rotators], ['Fast Rotators','Slow Rotators'])
pl.show()
"""


""" 
for i in range(len(fstest)):
    if fstest[i] == prediction[i]:
        correct, = pl.plot(csctest[i],lamtest[i],'bd')
    if fstest[i] != prediction[i]:
        incorrect, = pl.plot(csctest[i],lamtest[i],'m+')
pl.title('Success of Classification Predictions')
pl.xlabel('Cersic Index from Single Fit')
pl.ylabel('Lambda Value')
pl.legend([correct, incorrect], ['Correct','Incorrect'])
pl.show()
raw_input("Now?")
pl.close('all')
"""
#I NEED TO PLOT MY PREDICTIONS TO SHOW THEIR SUCCESS HERE

"""
w = clf.coef_[0] #THIS ONLY WORKS FOR SVM
a = -w[0] / w[1]
xx = sp.linspace(0,12)
yy = a * xx - (clf.intercept_[0]) / w[1]
pl.plot(xx,yy)

pl.show()
"""

#We try to predict based on D/T alone:
dt = datacsc.field(19)[1:]
print dt
dt = dt[:,None]
dttrain = dt[:len(fslist)/2]
dttest = dt[len(fslist)/2:]
clf = tree.DecisionTreeClassifier()
clf.fit(dttrain,fstrain) # We train the tree using the lamre value and F,S classification as test
prediction = clf.predict(dttest).copy()
#print 'Prediction: ', prediction
#print 'True Values: ', fstest
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
print 'D/T Success rate: ', round(float(true)/total,2)
#We find the number of zeros in the dataset
zeros = 0
zero_correct = []
zero_incorrect = []
"""
for i in range(len(dttest)):
    #print dttest[i]
    #print type(dttest[i])
    if dttest[i] == '0.00':
        zeros +=1
    if dttest[i] == '0.00' and prediction[i] == fstest[i]:
        zero_correct.append(i)
    elif dttest[i] == '0.00' and prediction[i] != fstest[i]:
        zero_incorrect.append(i)
"""
#We evaluate how the tree has assigned different predictions for the same value of D/T
print 'correct: ', zero_correct
print 'incorrect: ', zero_incorrect
print 'Number of zeros in data', zeros
#print 'Success of zeros:, ', zero_correct/float(zeros)
totalnumber = len(dttest)
#print totalnumber
#print 'Proportion of zeros in data: '
#print zeros/float(totalnumber)
zero_and_fast = 0
zero_and_slow = 0
for i in range(len(dttrain)):
    if dttrain[i] == '0.00' and fstrain[i] == 1:
        zero_and_fast += 1
    elif dttrain[i] == '0.00' and fstrain[i] == 0:
        zero_and_slow += 1
print 'zero and fast: ', zero_and_fast
print 'zero and slow: ', zero_and_slow
print 'proportion of zero and fast to total zeros', zero_and_fast/79.
#We repeat for the predicted values
pred_zero_and_fast = []
pred_zero_and_slow = []
correct = []
incorrect = []
pred_slow = []
true_slow = []
dt_test_zero = []
dttestfloat = sp.array(dttest).astype(float)#WORKS!, needed to convert strings to floats
dttestfinal = dttestfloat.ravel()
zerocorrect = []
zeroincorrect = []
for i in range(len(prediction)):
    if prediction[i] == 0:
        #print 'prediction is slow'
        pred_slow.append([fstest[i],prediction[i],dttestfinal[i],lamtest[i]])
        #print prediction[i],fstest[i],dttest[i]
    if dttestfloat[i] == 0:
        dt_test_zero.append([fstest[i],prediction[i],dttestfinal[i],lamtest[i]])
    if prediction[i] == fstest[i]:
        #pl.plot(fstest[i],dttestfloat[i],'rx',label='Correct Prediction')
        #pl.xlim(-0.1,1.1)
        correct.append([fstest[i],prediction[i],dttestfinal[i],lamtest[i]])
        if dttestfinal[i] == 0:
            zerocorrect.append([fstest[i],prediction[i],dttestfinal[i],lamtest[i]])
        if dttest[i] == '0.00' and prediction[i] == 1:
            #print 'correct, dt is 0 and pred is fast'
            pred_zero_and_fast.append([fstest[i],prediction[i],dttestfinal[i],lamtest[i]]) #we create array of zerodt and fast
        elif dttest[i] == '0.00' and prediction[i] == 0:
            #print 'correct, dt is 0 and pred is slow'
            pred_zero_and_slow.append([fstest[i],prediction[i],dttestfinal[i],lamtest[i]])
    elif prediction[i] != fstest[i]:
        if dttestfinal[i] == 0:
            zeroincorrect.append([fstest[i],prediction[i],dttestfinal[i],lamtest[i]])
        #pl.plot(fstest[i],dttestfloat[i],'bo',label='Incorrect Prediction')
        incorrect.append([fstest[i],prediction[i],dttestfinal[i],lamtest[i]])#we create array of incorrect
    else:
        print 'woops!'
        break

print 'zero and fast: ', pred_zero_and_fast
print 'zero and slow: ', pred_zero_and_slow

print 'predicted slow: ', pred_slow
print 'zero dt: ', dt_test_zero
print 'correct: ', correct
print 'zero and correct ', zerocorrect
print 'zero and incorrect ', zeroincorrect

def column(matrix, i): #We create a function to extract column from 2d list
    return [row[i] for row in matrix]
print column(correct,1)


pl.plot(column(zerocorrect,0),column(zerocorrect,3),'gx',label='Correct Prediction')
pl.plot(column(zeroincorrect,0),column(zeroincorrect,3),'rx',label='Incorrect Prediction')
pl.xlabel('Fast/slow Status')
pl.ylabel('$\lambda_{Re}$')
pl.xlim(-0.1,1.1)
pl.legend(loc=0)
pl.show()
"""
for i in range(len(fstest)):
    if fstest[i] == prediction[i]:
        correct, = pl.plot(dttest[i],lamtest[i],'b')
    if fstest[i] != prediction[i]:
        incorrect, = pl.plot(dttest[i],lamtest[i],'m+')
#pl.title('Success of Classification Predictions Based on D/T')

pl.xlabel('Disk-to-Total Light Ratio')
pl.ylabel('Lambda Value')
pl.xlim(-0.1,1.0)
pl.legend([correct, incorrect], ['Correct','Incorrect'])
pl.show()
"""


"""
#print dttest
#Here we look at success rate of just zeros
histtrue = 0
histfalse = 0
for i in range(len(prediction)):
    if dttest[i] == '0.00':
            #print dttest[i]
            #print prediction[i], fstest[i]
            if prediction[i] == fstest[i]:
                histtrue +=1
            else:
                histfalse+=1
print histtrue + histfalse #Also equal to total number of zeros
print float(histtrue) / (histtrue + histfalse) #Number of those with zero that were correct
x = ['True','False']
y = [histtrue,histfalse]
data = [go.Bar(
            x=['True', 'False'],
            y=[histtrue,histfalse]
    )]
#py.iplot(data, filename='basic-bar')
#width = 0.5
#pl.bar(x,y, width, color="blue")
#pl.show()
#Now we try to apply the sklearn.clf with more than 1 variable. We will use n, the Disk to Total light ratio (D/T) (col 20),
#Using dt[i+1] excludes the field name which is included in the data CERSIC INDEX FROM SINGLE FIT
# We need to combine the data into an array that has both datapoints in a subarray for each member of the new array
"""
"""
dt = datacsc.field(19)
#print dt
features = sp.zeros([260,2])
for i in sp.arange(259):
    features[i,0] = csc[i]
    features[i,1] = dt[i+1]
#print features

#In the abstract to the Linking Photometric paper, it states that "The median disk-to-total light ratio for fast and slow rotators is
#0.41 and 0.0, respectively." FOR SOME REASON THERE ARE MANY ZEROS IN THE D/T COLUMN, DON'T KNOW WHY

#We now apply decisiontreeclassifier to both these inputs, after splitting the features list in half:
featurestrain = features[:len(fslist)/2]
featurestest = features[len(fslist)/2:]

clf = tree.DecisionTreeClassifier()
clf.fit(featurestrain,fstrain) # We train the tree using the lamre value and F,S classification as test
prediction = clf.predict(featurestest).copy()
#print 'Prediction: ', prediction
#print 'True Values: ', fstest
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


for i in range(len(fstest)):
    if fstest[i] == prediction[i]:
        correct, = pl.plot(dttest[i],lamtest[i],'bd')
    if fstest[i] != prediction[i]:
        incorrect, = pl.plot(dttest[i],lamtest[i],'m+')
pl.title('Success of Classification Predictions Based on D/T')
pl.xlabel('D/T')
pl.ylabel('Lambda Value')
pl.legend([correct, incorrect], ['Correct','Incorrect'])
pl.show()

"""


#Including D/T improves success by 3% compared to using Cersic index alone, but,
#surprisingly, using D/T alone has the highest success rate at a whopping 81%!!

#We should look at the effect on disks and bulges ie ellipticity(?) as the 
#photometric paper states that this should only contribute a 59% success rate
#when using this alone.

#We now import all the variables of both the Atlas3D papers in order to compare
#the effects of including each in different combinations

#We now try the regressor method so we get specific values of the lam value

