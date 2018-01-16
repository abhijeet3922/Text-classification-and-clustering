#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:36:07 2018
@author: abhijeet
"""

# Importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

# Cleaning the text sentences so that punctuation marks, stop words & digits are removed  
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y


print "There are 10 sentences of following three classes on which K-NN classification and K-means clustering"\
         " is performed : \n1. Cricket \n2. Artificial Intelligence \n3. Chemistry"
path = "Sentences.txt"

train_clean_sentences = []
fp = open(path,'r')
for line in fp:
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)
       
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_clean_sentences)

# Creating true labels for 30 training sentences 
y_train = np.zeros(30)
y_train[10:20] = 1
y_train[20:30] = 2

# Clustering the document with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=5)
modelknn.fit(X,y_train)

# Clustering the training 30 sentences with K-means technique
modelkmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=100)
modelkmeans.fit(X)


test_sentences = ["Chemical compunds are used for preparing bombs based on some reactions",\
                  "Cricket is a boring game where the batsman only enjoys the game",\
                  "Machine learning is an area of Artificial intelligence"]

test_clean_sentence = []
for test in test_sentences:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+","",cleaned)
    test_clean_sentence.append(cleaned)
    
Test = vectorizer.transform(test_clean_sentence) 

true_test_labels = ['Cricket','AI','Chemistry']
predicted_labels_knn = modelknn.predict(Test)
predicted_labels_kmeans = modelkmeans.predict(Test)

print "\nBelow 3 sentences will be predicted against the learned nieghbourhood and learned clusters:\n1. ",\
        test_sentences[0],"\n2. ",test_sentences[1],"\n3. ",test_sentences[2]
print "\n-------------------------------PREDICTIONS BY KNN------------------------------------------"
print "\n",test_sentences[0],":",true_test_labels[np.int(predicted_labels_knn[0])],\
        "\n",test_sentences[1],":",true_test_labels[np.int(predicted_labels_knn[1])],\
        "\n",test_sentences[2],":",true_test_labels[np.int(predicted_labels_knn[2])]

print "\n-------------------------------PREDICTIONS BY K-Means--------------------------------------"
print "\nIndex of Cricket cluster : ",Counter(modelkmeans.labels_[0:10]).most_common(1)[0][0]
print "Index of Artificial Intelligence cluster : ",Counter(modelkmeans.labels_[10:20]).most_common(1)[0][0] 
print "Index of Chemistry cluster : ",Counter(modelkmeans.labels_[20:30]).most_common(1)[0][0]

print "\n",test_sentences[0],":",predicted_labels_kmeans[0],\
        "\n",test_sentences[1],":",predicted_labels_kmeans[1],\
        "\n",test_sentences[2],":",predicted_labels_kmeans[2]