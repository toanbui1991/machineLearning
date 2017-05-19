# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:56:20 2017

@author: User
"""

# Import libraries
import pandas as pd
import sys

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)

# Our small data set
d = {'one':[1,1,1,1,1],
     'two':[2,2,2,2,2],
     'letter':['a','a','b','b','c']}

# Create dataframe
df = pd.DataFrame(d) #create data frame from a dictionary.
df

# Create group object
one = df.groupby('letter')

# Apply sum function
one.sum()

#groupby with more than one column
letterone = df.groupby(['letter','one']).sum()
letterone

letterone.index #now you have multipleIndex

#groupby but not change columns into index columns.
letterone = df.groupby(['letter','one'], as_index=False).sum()
letterone


letterone.index #index as default.



