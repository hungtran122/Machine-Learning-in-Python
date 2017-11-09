# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:36:06 2017

@author: hungtran
"""

import UsingNLTK
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from time import time

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
vectorizer = UsingNLTK.StemmedCountVectorizer(min_df=30, max_df=.5,stop_words='english')
vectorized = vectorizer.fit_transform(train_data.data)
X = vectorized.toarray()
Y = np.array([train_data.target]).transpose()
num_samples, num_features = X.shape
print ('num of samples: %i, num of features: %i' %(num_samples,num_features))
# dividing X, y into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0,shuffle=True)
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

lda = LatentDirichletAllocation(max_iter=-1,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(vectorized) 
print("Train Latent Dirichlet Allocation done in %0.3fs." % (time() - t0))

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
feature_names = vectorizer.get_feature_names()
n_top_words = 20
print_top_words(lda, feature_names, n_top_words)