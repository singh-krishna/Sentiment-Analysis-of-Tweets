# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 01:33:06 2019

@author: HP
"""
###############################################################################
#Importing packages
from statistics import median
from collections import defaultdict
from gensim.models import Word2Vec 
import numpy as np
from sklearn import preprocessing,cross_validation,svm
#from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
import csv
from nltk.corpus import stopwords
import re
from matplotlib import pyplot as plt
###############################################################################

###############################################################################
#function for reading file
def openfile(filename):
    columns = defaultdict(list)
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k,v) in row.items():
                columns[k].append(v)
    data=columns['SentimentText']
    return data,columns
###############################################################################
    
###############################################################################
#function for converting every words to lower case
def to_lower(data,columns):
    l1=[]
    for i in range(0,100):#working with first 100 tweets of dataset we can do it for every tweet by just writing len(data) in place of 100
        data[i]=data[i].lower()#set all string to lower case
        l1.append(int((columns['Sentiment'][i])))#append all the value of polarity into a list
    return l1
###############################################################################
    
###############################################################################
#function for finding feature vector of each word
def word2vec(data):
    stop=stopwords.words("english")
    res=[]
    for i in range(0,100):
        sum1=0
        data[i]=re.sub(r'[^\w\s]',r'',data[i])
        sentences=word_tokenize(data[i])
        sentences=[w for w in sentences if not w in stop]
    #    print(sentences)
        x=len(sentences)
        for i in range(x):
            model=Word2Vec(sentences[i], min_count=1)
            words = list(model.wv.vocab)
            target=[]
            for j in range(len(words)):
    #            print(words[j])
                target.append(median(model[words[j]]))
    #        print("result:-",sum(target))
            sum1=sum1+sum(target)
            del target
        res.append(sum1)
    return res
###############################################################################
    
###############################################################################
#function for training and evaluating the accuracy of the classifier model
def evaluate(res,l1):
    print("total number of tweet with polarity",(len(res),len(l1)))
    X= np.array(res).reshape(-1, 1)
    Y=np.array(l1).reshape(-1, 1)
    X=preprocessing.scale(X)
    X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.2)
    clf = LogisticRegression()
    #clf = svm.SVC()
    clf.fit(X_train,Y_train)
    accur=clf.score(X_test,Y_test)*100
    print("Accuracy",accur)
    return X_train,X_test,Y_train,Y_test
###############################################################################
    
###############################################################################
#function for plotting graphs
def plot(X_train,X_test,Y_train,Y_test):    
    plt.scatter(X_train,Y_train,color='red',marker='^')
    plt.show()
    plt.scatter(X_test,Y_test,color='green',marker='*')
    plt.show()
###############################################################################
    
###############################################################################
#calling various functions
data,columns=openfile('c:/Users/HP/Desktop/tweets.csv')
l1=to_lower(data,columns)
res=word2vec(data)
X_train,X_test,Y_train,Y_test=evaluate(res,l1)
plot(X_train,X_test,Y_train,Y_test)
###############################################################################

