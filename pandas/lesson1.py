# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:28:12 2017

@author: User
"""

# Import all libraries needed for the tutorial

# General syntax to import specific functions in a library: 
##from (library) import (specific library function)
from pandas import DataFrame, read_csv

# General syntax to import a library but no functions: 
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number
import os
# Enable inline plotting
#%matplotlib inline

#change the working directory.
os.chdir('D:\Toan\1. NCU Master Degree\Semester 4\machineLearning\pandas')
print(os.getcwd())



#get the version of the used package.
print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)


# The inital set of baby names and bith rates

'''
problem 1: create a dataFrame from list and then write it to csv file.
'''
#input data.
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]


#for n, b in zip(names, births): #zip(list1, list2) in looping techniaue.
#    print(n, b)

#create a list of tuple.
#data[(), ()]    
tempData = list(zip(names, births))
print(tempData)

#create dataframe with DataFrame().
#input is a list of tuple

birthData = pd.DataFrame(tempData, columns = ['Name', 'Birth'])
print(birthData)

#birthData is a DataFrame object so it have to_csv method.
birthData.to_csv('birthData.csv', index = False, header = False)


#now you want to read the data from csv file.
location = os.path.join(os.getcwd(), 'birthData.csv')
df = pd.read_csv(location)
print(df)

#now the df dont have header you want to assign name to it.
df = pd.read_csv(location, names = ['name', 'birthRate'])


#read heading data.
df.head()

#get the column data type with dtypes or dtype.
df.dtypes
df.name.dtype
df.birthRate.dtype


#sort the df with column value.
dfSorted = df.sort_values(['birthRate'], ascending=False)
dfSorted.head(1)
#return the max value of birthRate.
df['birthRate'].max()
#type(df['birthRate']) #pandas.core.series.Series
#type(df) #pandas.core.frame.DataFrame

df['birthRate'].plot()

#note this is the one series data plot, very easy.
plt.plot(df['birthRate'])
plt.ylabel('number of birth')
plt.show()

#largest birthRate.
maxBirth = df['birthRate'].max()

#take the data valuespoint base on testing.
maxName = df['name'][df['birthRate'] == df['birthRate'].max()].values
print(maxName)

#tanke the largest data point.
print("The most popular name")
df[df['birthRate'] == df['birthRate'].max()]





