# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:37:26 2017
term frequency – inverse document frequency (TF-IDF)
TF(t) = (Number of times term t appears in a document) 
        / (Total number of terms in all the documents)
IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

@author: hungtran
"""
import math
def tfidf(term,doc,docset):
    tf = float(doc.count(term))/(sum([doc_.count(term) for doc_ in docset]))
    idf = math.log(float(len(docset))/(len([doc_ for doc_ in docset if term in doc_])))
    return tf*idf

a, abb, abc = ['a'], ['a', 'b', 'b'], ['a', 'b', 'c']
docset = [a,abb,abc]
print(tfidf('a', a, docset))
print(tfidf('b', abb, docset))
print(tfidf('a', abc, docset))
print(tfidf('b', abc, docset))
print(tfidf('c', abc, docset))