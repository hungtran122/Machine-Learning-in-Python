# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:45:37 2017

@author: hungtran
"""

##### SVM #####
# using svm #
'''
input:  x[i] = vectorized[i].toarray()
        y[i] = train_data.target[i]
'''

# training a linear SVM classifier
from sklearn.svm import SVC
import UsingNLTK
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from time import time
# preparation of training and test data
groups = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware']
#train_data = fetch_20newsgroups (subset='train',categories=groups)
#test_data = fetch_20newsgroups (subset='test',categories=groups)
train_data = fetch_20newsgroups (subset='train')
test_data = fetch_20newsgroups (subset='test')
#pprint(test_data.target_names)
#print len(test_data.filenames)
#groups = train_data.target_names
vectorizer = UsingNLTK.StemmedTfidfVectorizer(min_df=30, max_df=.5,stop_words='english')
vectorized = vectorizer.fit_transform(train_data.data)
X = vectorized.toarray()
Y = np.array([train_data.target]).transpose()
num_samples, num_features = X.shape
print ('num of samples: %i, num of features: %i' %(num_samples,num_features))
XY = np.concatenate((X,Y),axis=1)
np.random.shuffle(XY)
X_shuffled = XY[:,:-1]
Y_shuffled = XY[:,-1:].flatten()
test_index = int(num_samples * 0.8) #80% of data will be used as training set, the rest is test set
X_train = X_shuffled[:test_index,:]
Y_train = Y_shuffled[:test_index]
X_test = X_shuffled[test_index:,:]
Y_test = Y_shuffled[test_index:]

t0 = time()
clf = SVC(kernel = 'linear', C = 1, degree = 3, decision_function_shape = 'ovr')
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
accuracy = clf.score(X_test, Y_test)
print("SVM accuracy = %f"%accuracy)
print("SVM done in %0.3fs." % (time() - t0))


