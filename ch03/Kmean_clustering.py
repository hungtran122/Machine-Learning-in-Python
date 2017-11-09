# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:50:35 2017

@author: hungtran
"""
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import UsingNLTK
import scipy as sp
from sklearn import svm
from scipy.stats.stats import pearsonr
import numpy as np

groups = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware']
train_data = fetch_20newsgroups (subset='train',categories=groups)
test_data = fetch_20newsgroups (subset='test',categories=groups)
#train_data = fetch_20newsgroups (subset='train')
#test_data = fetch_20newsgroups (subset='test')
#pprint(test_data.target_names)
#print len(test_data.filenames)
#groups = train_data.target_names
vectorizer = UsingNLTK.StemmedCountVectorizer(min_df=30, max_df=.5,stop_words='english')
vectorized = vectorizer.fit_transform(train_data.data)
num_samples, num_features = vectorized.shape
print (num_samples,num_features)
##### Clustering #####
num_clusters = 100
from sklearn.cluster import KMeans

km = KMeans(n_clusters=num_clusters,init='random',n_init=1,verbose=1)
km.fit(vectorized)

##### Prediction #####
new_post = 'Disk drive problems. \
            Hi, I have a problem with my hard disk. \
            After 1 year it is working only sporadically now. \
            I tried to format it, but now it doesn''t boot any more. \
            Any ideas? Thanks.'
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
# Fetch the indices of all vectors in the same cluster in the original dataset
# (km.labels_ == new_post_label) is boolean array
# nonzero converts this array into a smaller array containing indices of the True elements
similar_indices = (km.labels_ == new_post_label).nonzero()[0]
print ('Number of all vectors in the same cluster: %i' %len(similar_indices))
# Build a list of post in this cluster sorted by similarity
similar_posts = list()
for i in similar_indices:
    #dist = sp.linalg.norm((new_post_vec-vectorized[i]).toarray())
    dist = sp.spatial.distance.cosine(new_post_vec.toarray(),vectorized[i].toarray())
    #dist = pearsonr(new_post_vec.toarray(),vectorized[i].toarray())
    similar_posts.append((dist,train_data.data[i]))
similar_posts = sorted(similar_posts)
print(len(similar_posts))
# Give an intuition of the results of clustering
most_similar = similar_posts[0]
in_middle_similar = similar_posts[len(similar_posts)/2]
least_similar = similar_posts[-1]

next_similar_indices = (km.labels_ == new_post_label + 1).nonzero()[0]
print train_data.data[next_similar_indices[0]]

post_group = zip(train_data.data,train_data.target)
z = [(len(post[0]),post[0],train_data.target_names[post[1]]) for post in post_group]
print(sorted(z)[5:7])

for term in ['cs' ,'faq', 'thank', 'know', 'understand', 'document','function','news']:
    print ('IDF (%s) = %.2f'%(term,vectorizer._tfidf.idf_[vectorizer.vocabulary_[term]]))
    


