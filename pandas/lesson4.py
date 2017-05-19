# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:58:37 2017

@author: User
"""

# Import libraries
import pandas as pd
import sys

print('Python version ' + sys.version)
print('Pandas version: ' + pd.__version__)

# Our small data set
d = [0,1,2,3,4,5,6,7,8,9]

# Create dataframe
df = pd.DataFrame(d) #each series is a cloumn
df

#Assign name to cloumn.
df.columns = ['Rev']
df

#add new column.
df['NewCol'] = 5
df

#moedify exist column.
df['NewCol'] = df['NewCol'] + 1
df

#delete exist column.
del df['NewCol']
df

#add two new column.
df['test'] = 3
df['col'] = df['Rev']
df
#change the index by assign new vaue to default index.
i = ['a','b','c','d','e','f','g','h','i','j']
df.index = i
df

#using loc
df.loc['a'] #loc use to access row data throuhg row index.
# df.loc[inclusive:inclusive]
df.loc['a':'d'] #use loc to asscess a lock of row data.

# df.iloc[inclusive:exclusive]
# Note: .iloc is strictly integer position based.
#It is available from [version 0.11.0] (http://pandas.pydata.org/pandas-docs/stable/whatsnew.html#v0-11-0-april-22-2013) 
df.iloc[0:3] #iloc vs loc. iloc acces row data base on the order of row (index by number not by key)

#get the data in one column.
df['Rev']

#get access to a set of column
df[['Rev', 'test']]


#get access to data by row and column by using ix.
df.ix[0:3, 'Rev'] #row by index and include, colun by name.
df.ix[5:,'col']
df.ix[:3,['col', 'test']] # a set of column name.

# Select top N number of records (default = 5)
df.head() # default = 5.

# Select bottom N number of records (default = 5)
df.tail()


