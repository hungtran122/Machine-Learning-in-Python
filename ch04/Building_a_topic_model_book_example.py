# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:38:02 2017

@author: hungtran
"""


from __future__ import print_function
from gensim import corpora, models, matutils

#from wordcloud import create_cloud
import wordcloud

import os
from os import path
import matplotlib.pyplot as plt
import numpy as np

if not path.exists('./data/ap.dat'):
    print ('Error: data is not present')
#corpus = corpora.BleiCorpus(os.getcwd() + '\\data\\ap.dat',os.getcwd() + '\\data\\vocab.txt')
corpus = corpora.BleiCorpus('./data/ap.dat',os.getcwd() + './data/vocab.txt')
model = models.LdaModel(corpus,num_topics=100,id2word=corpus.id2word,alpha=None)
#topics = [model[c] for c in corpus]



# Iterate over all the topics in the model
for ti in range(model.num_topics):
    words = model.show_topic(ti, 64)
    tf = sum(f for _, f in words)
    with open('topics.txt', 'w') as output:
        output.write('\n'.join('{}:{}'.format(w, int(1000. * f / tf)) for w, f in words))
        output.write("\n\n\n")

# We first identify the most discussed topic, i.e., the one with the
# highest total weight

topics = matutils.corpus2dense(model[corpus], num_terms=model.num_topics)
weight = topics.sum(1)
max_topic = weight.argmax()


# Get the top 64 words for this topic
# Without the argument, show_topic would return only 10 words
words = model.show_topic(max_topic, 64)

# This function will actually check for the presence of pytagcloud and is otherwise a no-op
wordcloud.WordCloud('cloud_blei_lda.png', words)

num_topics_used = [len(model[doc]) for doc in corpus]
fig,ax = plt.subplots()
ax.hist(num_topics_used, np.arange(42))
ax.set_ylabel('Nr of documents')
ax.set_xlabel('Nr of topics')
fig.tight_layout()
fig.savefig('Figure_04_01.png')


# Now, repeat the same exercise using alpha=1.0
# You can edit the constant below to play around with this parameter
ALPHA = 0.5

model1 = models.ldamodel.LdaModel(
    corpus, num_topics= 100, id2word=corpus.id2word, alpha=ALPHA)
num_topics_used1 = [len(model1[doc]) for doc in corpus]

fig,ax = plt.subplots()
ax.hist([num_topics_used, num_topics_used1], np.arange(42))
ax.set_ylabel('Nr of documents')
ax.set_xlabel('Nr of topics')

# The coordinates below were fit by trial and error to look good
ax.text(9, 223, r'default alpha')
ax.text(26, 156, 'alpha=0.5')
fig.tight_layout()
fig.savefig('Figure_04_02.png')
