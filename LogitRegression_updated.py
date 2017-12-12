# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:05:28 2017

@author: 1542283
"""

#logistic Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cross_validation import KFold

# loading data
iris=load_iris()
data=iris.data

#normalization of columns
def normalization(x):
    return (x-min(x))/(max(x)-min(x))

data=pd.DataFrame(data).apply(normalization)

data1=data.iloc[(iris.target==1)|(iris.target==2),[2,3]]

m=data1.shape[0]
n=data1.shape[1]

X=np.ones((m,n+1))
X[:,1]=data1.iloc[:,0]
X[:,2]=data1.iloc[:,1]
#output vector
idx=np.where((iris.target==1)|(iris.target==2))
y=iris.target[idx]
y=np.array(pd.Series(y).map({1:0,2:1}))
y=y.reshape(m,1)
# 1 : versicolor:0
# 2: virginica: 1
# initializing theta
theta=np.zeros((n+1,1))

#Logistic Regression
#defining sigmoid /logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, target, theta):
    scores = np.dot(X, theta)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll


def logistic_regression(theta,X, target, iterations, learning_rate):
    for step in xrange(iterations):
        h = np.dot(X, theta)
        predictions = sigmoid(h)
        output_error_signal = target - predictions
        gradient = np.dot(X.T, output_error_signal)
        theta += learning_rate * gradient
    return theta
        
    
weights=logistic_regression(theta,X,y,1500,0.01)
weights

#Predictions
opt_theta=weights
Z_predict=np.matmul(X,opt_theta)
y_predict_probability = 1.0 / (1.0 + np.exp(-Z_predict))
y_predict_binary = np.zeros(len(y_predict_probability))
for i in range(len(y_predict_probability)):
    if y_predict_probability[i] >= 0.5:
        y_predict_binary[i] = 1.0
    else:
        y_predict_binary[i] = 0.0
    
    
     
pd.Series(y_predict_binary).value_counts()
pd.Series(y.flatten()).value_counts()
######################################
#SKLEARN IMPLEMENTATION

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(fit_intercept=True, C=1e15)
data1=data.iloc[(iris.target==1)|(iris.target==2),[2,3]]
X1=X[:,[1,2]]
y1=iris.target[idx]
y1=np.array(pd.Series(y1).map({1:0,2:1}))
clf.fit(X,y)
clf.coef_


from sklearn.model_selection import LeaveOneOut
loocv=LeaveOneOut()
loocv.get_n_splits(X)

for train_index,test_index in loocv.split(X):
    X_train,X_test=X[train_index],X[test_index]
    Y_train,y_test=y[train_index],y[test_index]
    








