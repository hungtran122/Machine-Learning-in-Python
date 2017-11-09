# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:49:44 2017

@author: hungtran
"""
import os
import scipy as sp
import UsingNLTK
from sklearn.datasets import fetch_20newsgroups
from time import time
import numpy as np

# preparation of training and test data
groups = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware']
train_data = fetch_20newsgroups (subset='train',categories=groups)
test_data = fetch_20newsgroups (subset='test',categories=groups)
#train_data = fetch_20newsgroups (subset='train')
#test_data = fetch_20newsgroups (subset='test')
#pprint(test_data.target_names)
#print len(test_data.filenames)
#groups = train_data.target_names
#vectorizer = UsingNLTK.StemmedTfidfVectorizer(min_df=30, max_df=.5,stop_words='english')
#vectorizer = UsingNLTK.StemmedCountVectorizer(min_df=30, max_df=.5,stop_words='english')
vectorizer = UsingNLTK.StemmedTfidfVectorizer(min_df=1,stop_words='english') # using NLTK with Tf_idf
vectorized = vectorizer.fit_transform(train_data.data)
X = vectorized.toarray().transpose()
Y = np.array([train_data.target]).transpose()

num_samples, num_features = X.shape
all_words = vectorizer.get_feature_names()
#print all_words
#print X_train.toarray()

new_post = test_data.data[1]
new_post_vec = vectorizer.transform([new_post]).toarray().transpose()

def dist_raw(v1,v2):
    return sp.linalg.norm((v1-v2).toarray())

##### Normalizing the word count vectors #####
def dist_norm(v1,v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())
def dist(v1,v2):
    v1_normalized = v1/sp.linalg.norm(v1)
    v2_normalized = v2/sp.linalg.norm(v2)
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta)
#####   Removing less important words #####
#####   Using stop_word = 'english'   #####
print sorted(vectorizer.get_stop_words())[0:20]

##### Main #####
import sys
best_doc = None
best_dist = sys.maxint
best_i = None
for i in range(0,num_samples):
#    print post_vec
    d = dist(X[i],new_post_vec)
#    print ("Post %i with dist = %.2f: %s" %(i,d,post))
    if d < best_dist:
        best_dist = d
        best_i = i
print ("Best post %i with dist = %.2f" %(best_i,best_dist))
