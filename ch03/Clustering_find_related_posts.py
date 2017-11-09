# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:49:44 2017

@author: hungtran
"""
import os
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
import UsingNLTK
###############################################################
########################## Example 1 ##########################
###############################################################

#vectorizer = CountVectorizer(min_df=1) #min_df = minimum document frequency
vectorizer = UsingNLTK.StemmedTfidfVectorizer(min_df=1,stop_words='english',encoding='utf-8') # using NLTK with Tf_idf
#vectorizer = CountVectorizer(min_df=1,max_df = 2, stop_words='english')
print(vectorizer)
content = ['How to format my hard hard disk, how', 'Hard disk format big issues, we have to fix']
x = vectorizer.fit_transform(content)
all_words = vectorizer.get_feature_names()
print(x.toarray().transpose())



###############################################################
########################## Example 2 ##########################
###############################################################
posts = [open(os.path.join("./Data/",f)).read() for f in os.listdir("./Data/")]
#vectorizer = CountVectorizer(min_df=1) # no use of stop_word option
#vectorizer = CountVectorizer(min_df=1, stop_words='english') # use stop_words option
#vectorizer = UsingNLTK.StemmedCountVectorizer(min_df=1,stop_words='english') # using NLTK
vectorizer = UsingNLTK.StemmedTfidfVectorizer(min_df=1,stop_words='english',encoding='utf-8') # using NLTK with Tf_idf
X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
all_words = vectorizer.get_feature_names()
print all_words
#print X_train.toarray()

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

def dist_raw(v1,v2):
    return sp.linalg.norm((v1-v2).toarray())

##### Normalizing the word count vectors #####
def dist_norm(v1,v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())
#####   Removing less important words #####
#####   Using stop_word = 'english'   #####
print sorted(vectorizer.get_stop_words())[0:20]

##### Main #####
import sys
best_doc = None
best_dist = sys.maxint
best_i = None
for i in range(0,num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
#    print post_vec
    d = dist_norm(post_vec,new_post_vec)
    print ("Post %i with dist = %.2f: %s" %(i,d,post))
    if d < best_dist:
        best_dist = d
        best_i = i
print ("Best post %i with dist = %.2f" %(best_i,best_dist))

print (posts[best_i])
print (new_post)
