# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:55:58 2017

@author: Bui Xuan Toan
"""

# Import all libraries needed for the tutorial
import pandas as pd
import os
from numpy import random
import matplotlib.pyplot as plt
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number

# Enable inline plotting
#%matplotlib inline
#test the version.
print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)

# The inital set of baby names
names = ['Bob','Jessica','Mary','John','Mel']

#you want to generate list of names randomly from a set of name.
random.seed(300)
randomNames = [names[random.randint(0, len(names))] for i in range(1000)]
print(len(randomNames))
print(randomNames[:10])

#you want to craete a list of birthRate randomly.
random.seed(300)
randomBirthRate = [random.randint(0,1000) for i in range(1000)]
print(len(randomBirthRate))
print(randomBirthRate[:10])

#you want to create a tempData is a lisf of tuple.
#tempData[('name', birthRate)]
tempData = list(zip(randomNames, randomBirthRate))
print(len(tempData))
print(tempData[:10])

#you want to create Data Frame from tempData.
df = pd.DataFrame(tempData, columns = ['name', 'birthRate'])
print(df.head())
print(df.info())

#now you want to write the data into a csv file.
df.to_csv('birthData.csv', index=False, header=False)

#now you want to read the data again.
location = os.path.join(os.getcwd(), 'birthData.csv')
df = pd.read_csv(location)
print(df.head(6))

#you you want the df and assign names as the same time.
#now the df dont have header you want to assign name to it.
df = pd.read_csv(location, names = ['name', 'birthRate'])
print(df.head(6))

#you want to get the information about the df.
df.info()
#you want to see the tail data.
df.tail(10)

#you want to fine the set of name or unique().
df['name'].unique()
#you want to print unique name in the data set.
for n in df['name'].unique():
    print(n)
    
#now you want more information about name series with decribe()
df['name'].describe()

#you want to group the data point and sum birthRate by name.
name = df.groupby('name') #name is like a df with group index by name column.
groupData = name.sum() #after you index how to group the data now you apply function.
print(groupData)

groupData2 = df.groupby('name').sum()
print(groupData2)


#you want to find the largest data point by sort_values().
sortedData = groupData.sort_values(['birthRate'], ascending=False)
print(sortedData.head(1)) #get the first data point.

#get the largest value of birthRate (evidence).
print(groupData['birthRate'].max())

#plot the sortedData.
sortedData['birthRate'].plot.bar()


