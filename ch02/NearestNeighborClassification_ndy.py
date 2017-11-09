# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:45:33 2017

@author: hungtran
"""
import numpy as np

data_t = np.genfromtxt("seeds.tsv",delimiter='\t',dtype=np.string0)
data_f = np.genfromtxt("seeds.tsv",delimiter='\t')
features = data_f[:,0:7]
labels = data_t[:,7].copy()


def distance(x1,x2):
    return np.sum((x1-x2)**2)
def nn_classification(train_set,labels,new_example):
    dist = np.array([distance(t,new_example) for t in train_set])
    nearest = dist.argmin()
    return labels[nearest]

new = np.array([15.26,14.84,0.871,5.764,3.312,2.221,5.223])
lbl = nn_classification(features,labels,new)

#substract the mean for each feature
features -= features.mean(axis=0)
#divide each feature by its standard deviation
features /= features.std(axis=0)




def load_dataset(dataset_name):
    '''
    data,labels = load_dataset(dataset_name)
    Load a given dataset
    Returns
    -------
    data : numpy ndarray
    labels : list of str
    '''
    data = []
    labels = []
    with open('./{0}.tsv'.format(dataset_name)) as ifile:
        for line in ifile:
            tokens = line.strip().split('\t')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficien',
    'length of kernel groove',
]
features, labels = load_dataset('seeds')

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
from sklearn.cross_validation import KFold