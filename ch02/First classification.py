# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:17:24 2017

@author: hungtran
"""

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
#We load the data with load_iris from sklearn
data = load_iris()
features = data['data']
feature_name = data['feature_names']
target = data['target']
target_names = data['target_names']
labels = target_names[target]
for t,marker,c in zip(xrange(3),">ox","rgb"):
    #We plot each class on its own to get different colored markers
    plt.scatter(features[target==t,3], # 0 means first column of matrix
                 features[target==t,2], # 1 means second column of matrix
                 marker=marker,
                 c=c)
plength = features[:,2] #petal length is stored in third column of matrix
#use numpy operations to get setosa features
is_setosa = (labels == 'setosa')
#This is important step:
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print ('Maximum petal length of setosa: {0}.'.format(max_setosa))
print ('Minimum petal length of setosa: {0}.'.format(min_non_setosa))
# First insight, if the petal length is smaller than two, this is a Setosa
#,otherwise it is either Virginica or Versicolor
'''
if features[:,2]<2:
    print "SETOSA"
else:
    print "Not SETOSA"
'''
# Since we can seperate setosa, we now eliminate setosa examples from our data
features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')
best_acc = -1.0
for fi in xrange(features.shape[1]): #shape[1] is number of columns
    # We are going to generate all possible thresholds for this feature
    thresh = features[:,fi].copy() #copy all values of column fi
    thresh.sort() # sorting 
    # Now test all thresholds
    for t in thresh: #try all value t as a threshhold, pick up the best (maximum accuracy)
        pred = (features[:,fi] > t)
        acc = (pred==virginica).mean()
        rev_acc = (pred==~virginica).mean()
        if rev_acc>acc:
            reverse = True
            acc = rev_acc
        else:
            reverse = False
        if acc>best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
            best_reverse = reverse
print(best_fi, best_t, best_reverse, best_acc)
          
x = np.empty(9)
x.fill(1.6)
y = np.arange(9)
plt.plot(x,y)
plt.show()

    
