# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:14:29 2016

@author: User
"""
#clear all the object in the environment.
import sys
sys.modules[__name__].__dict__.clear()
#read the data frame with pandas.
import pandas as pd
import numpy as np
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
header=None)
print(df.head())
print(df.shape)
#Prepare the data to numpy array.
#create feature and target data.
from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
print(X.shape)
print(type(X))

y = df.loc[:, 1].values
print(y.shape)
print(type(y))

le = LabelEncoder()
y = le.fit_transform(y)
le.transform(['M', 'B'])

#create training dan testing data set.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=5)

#Model building.
#create the logistic Classifier.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
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


#Perform the cross validation.
import numpy as np
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator= model,
                         X= X_train,
                         y = y_train,
                         cv = 5,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))

from sklearn.cross_validation import cross_val_predict
predicted = cross_val_predict(estimator = model,
                              X = X_train,
                              y = y_train,
                              cv=5,
                              n_jobs=1)
print(predicted.shape)

#building the model2.
#building the set of feature.
list1 = pd.Series([1, 3, 4, 6, 7, 8, 11, 13, 14, 21, 23, 24, 26, 27, 28])
list1 = list1 -1
#print(list1)
##building the df in pandas.
#X = pd.DataFrame(X)
#print(X.head())
X = X[:, list1]
#print(type(X1))
#print(X1[:, 0:5])
print(X.shape)

y = y
#print(y1[0:5])
print(type(y))
print(y.shape)
#create training dan testing data set.
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=5)
    
    
#build the model.
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train, y_train)
predicted = model1.predict(X_test)
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



    
#cross validation.
#==============================================================================
# import numpy as np
# from sklearn.cross_validation import cross_val_score
# scores = cross_val_score(estimator= model1,
#                          X= X1_train,
#                          y = y1_train,
#                          cv = 5,
#                          n_jobs=-1)
# print('CV accuracy scores: %s' % scores)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                       np.std(scores)))
#==============================================================================



#building Logistic Regression model2
list2 = pd.Series([1, 3, 4, 7, 8, 14, 21, 23, 24, 27, 28])
list2 = list2 -1
#prepare the data for model2.
X = X[:, list2]
print(X.shape)
#print(X2[:, 0:5])
y = y
print(y.shape)
#print(y2[0:5])
#create training dan testing data set.
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=5)
#build the model.
from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression()
model2.fit(X_train, y_train)
predicted = model2.predict(X_test)
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
#cross validation.
#==============================================================================
# import numpy as np
# from sklearn.cross_validation import cross_val_score
# scores = cross_val_score(estimator= model2,
#                          X= X2_train,
#                          y = y2_train,
#                          cv = 5,
#                          n_jobs=-1)
# print('CV accuracy scores: %s' % scores)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
#                                       np.std(scores)))    
# 
#==============================================================================

