# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 15:43:03 2016

@author: User
"""
#clear all the data.
import sys
sys.modules[__name__].__dict__.clear()

#data preparing.
import pandas as pd
import numpy as np
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
header=None)
#Prepare the data to numpy array.
#create feature and target data.
from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
print(X.shape)
print(type(X))

y = df.loc[:, 1].values
print(y.shape)
print(type(y))
#transform the label data to int data
#for the target value.
le = LabelEncoder()
y = le.fit_transform(y)
le.transform(['M', 'B'])

##create training dan testing data set.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=5)
    
    
#building support vector machine.
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
               {'clf__C': param_range,
                'clf__gamma': param_range,
                'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

#building the best model.
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
model = Pipeline([('scl', StandardScaler()),
                  ('clf', SVC(kernel = 'rbf', random_state = 0, gamma = 0.001, C = 100))])
model.fit(X_train, y_train)



#get the measure matrices.
#set the things needed.

predicted = model.predict(X_test)

def get_accuracy(predicted, actual):
    return sum(predicted == actual)/len(actual)
accuracy = get_accuracy(predicted, y_test)
print('accuracy: {0}'.format(accuracy))

#definde the function for judge the model.

def get_false_positive(predicted, actual):
    predicted = pd.Series(predicted)
    actual = pd.Series(actual)
    predictPos = predicted[predicted == 1]
    actual = actual[predicted == 1]
    falsePos = sum(predictPos != actual)
    return falsePos
false_positive = get_false_positive(predicted, y_test)


def get_true_positive(predicted, actual):
    predicted = pd.Series(predicted)
    actual = pd.Series(actual)
    predictPos = predicted[predicted == 1]
    actual = actual[predicted == 1]
    truePos = sum(predictPos == actual)
    return truePos
true_positive = get_true_positive(predicted, y_test)

def get_true_negative(predicted, actual):
    predicted = pd.Series(predicted)
    actual = pd.Series(actual)
    predictNeg = predicted[predicted == 0]
    actual = actual[predicted == 0]
    trueNeg = sum(predictNeg == actual)
    return trueNeg
true_negative = get_true_negative(predicted, y_test)    

def get_false_negative(predicted, actual):
    predicted = pd.Series(predicted)
    actual = pd.Series(actual)
    predictNeg = predicted[predicted == 0]
    actual = actual[predicted == 0]
    falseNeg = sum(predictNeg != actual)
    return falseNeg    
false_negative = get_false_negative(predicted, y_test)
#test

def get_false_positive_rate(false_positive, true_negative):
    return false_positive/(false_positive + true_negative)
false_positive_rate = get_false_positive_rate(false_positive, true_negative)
print('fale positve: {0}'.format(false_positive_rate))

def get_true_positive_rate(true_positive, false_negative):
    return true_positive/(false_negative+true_positive)
true_positive_rate = get_true_positive_rate(true_positive, false_negative)
print('true positive rate: {0}'.format(true_positive_rate))

def get_precision(true_positive, false_positive):
    return true_positive/(true_positive + false_positive)
precision = get_precision(true_positive, false_positive)
print('precision: {0}'.format(precision))



#building the model2.
#building the set of feature.
list1 = pd.Series([1, 3, 4, 7, 8, 14, 21, 23, 24, 27, 28])
list1 = list1 -1
X = X[:, list1]
y = y
print(X.shape)
print(y.shape)


#prepare the data.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=5)

#find the best parameter for the model.
#building support vector machine.
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
               {'clf__C': param_range,
                'clf__gamma': param_range,
                'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


#build the best model.
#building the best model.
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
model = Pipeline([('scl', StandardScaler()),
                  ('clf', SVC(kernel = 'linear', random_state = 0, C = 100))])
model.fit(X_train, y_train)

#get the meature matrix.
predicted = model.predict(X_test)
false_positive = get_false_positive(predicted, y_test)
true_positive = get_true_positive(predicted, y_test)
true_negative = get_true_negative(predicted, y_test)  
false_negative = get_false_negative(predicted, y_test)
false_positive_rate = get_false_positive_rate(false_positive, true_negative)
accuracy = get_accuracy(predicted, y_test)
print('accuracy: {0}'.format(accuracy))
print('fale positve: {0}'.format(false_positive_rate))
true_positive_rate = get_true_positive_rate(true_positive, false_negative)
print('true positive rate: {0}'.format(true_positive_rate))
precision = get_precision(true_positive, false_positive)
print('precision: {0}'.format(precision))




