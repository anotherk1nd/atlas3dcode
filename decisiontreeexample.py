#Implementing example found at http://scikit-learn.org/stable/modules/tree.html#regression
from sklearn import tree
import scipy as sp
X = [[0, 0], [1, 1]] #Training set, each [] is 
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print clf.predict([[0.7, 0.7]])
