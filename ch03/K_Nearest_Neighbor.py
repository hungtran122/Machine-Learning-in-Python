# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:28:45 2017

@author: hungtran
K nearest neighbor
"""
import UsingNLTK
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
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
#vectorizer = UsingNLTK.StemmedTfidfVectorizer(min_df=30, max_df=.5,stop_words='english')
vectorizer = UsingNLTK.StemmedCountVectorizer(min_df=30, max_df=.5,stop_words='english')
vectorized = vectorizer.fit_transform(train_data.data)
X = vectorized.toarray()
Y = np.array([train_data.target]).transpose()
num_samples, num_features = X.shape
print ('num of samples: %i, num of features: %i' %(num_samples,num_features))
# dividing X, y into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0)
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()
clf = KNeighborsClassifier(n_neighbors=7)
t0 = time()
clf.fit(X_train,Y_train)
preds = clf.predict(X_test)
accuracy = clf.score(X_test,Y_test)
print ('KNN accuracy = %f' %accuracy)
cfm = confusion_matrix(Y_test,preds)
print("KNN done in %0.3fs." % (time() - t0))