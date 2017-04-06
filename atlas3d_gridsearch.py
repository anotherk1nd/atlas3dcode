"""
Here we attempt to implement sklearn with the photometric data.
THERE ARE 224 FR'S AND 36 SR'S
"""
import matplotlib.pyplot as pl
import sklearn
from sklearn import datasets, svm, metrics, tree
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from astropy.io import fits
import scipy as sp
from astropy.io.fits import getheader

pl.close('all')
fnphoto = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Data\\atlas3d\\linkingphotometric.fit'
#This is the data from the linking photometric paper
hdulistphoto = fits.open(fnphoto)
#hdulistphoto.info()
colnames = hdulistphoto[0]
#print colnames
tbdata = hdulistphoto[1]
#print tbdata
cols = hdulistphoto[1].columns
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
dataphoto = hdulistphoto[1].data
csc = dataphoto.field(5)[1:] #Using from 1 excludes the field name which is included in the data CERSIC INDEX FROM SINGLE FIT
#print csc

#We import the atlas3d data in order to access the lambda value and use the classifier code used in 1st attempt to implement classifier sklearn
atlas3d = 'C:\\Users\\Joshua\\Documents\\Term 1\\Project\\Code\\atlas3dcode\\atlas3d.fit'
hdulistatlas3d = fits.open(atlas3d)
dataatlas3d = hdulistatlas3d[1].data

fs = dataatlas3d.field(11)

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
"""
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
"""
#We try to use gridsearch, 
#criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
#max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False
param_grid = [{'criterion':['gini','entropy'],'splitter':['best','random'],'max_features':['sqrt','log2',None],'max_depth':range(1,10)}]
tree = GridSearchCV(tree.DecisionTreeClassifier(), param_grid)
tree.fit(csctrain, fstrain)
tree_preds = tree.predict_proba(csctest)[:, 1]
tree_performance = roc_auc_score(fstest, tree_preds)

print 'DecisionTree: Area under the ROC curve = {}'.format(tree_performance) #Need to study ROC curves more
print(tree.best_estimator_)
print 'Best Score: ',tree.best_score_ #IMPROVED SUCCESS RATE TO 93%!!!!!!!!!
for params, mean_score, scores in tree.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))