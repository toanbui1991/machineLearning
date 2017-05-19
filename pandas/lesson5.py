# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:22:42 2017

@author: User
"""

# Import libraries
import pandas as pd
import sys

print('Python version ' + sys.version)
print('Pandas version: ' + pd.__version__)

# Our small data set
d = {'one':[1,1],'two':[2,2]}
i = ['a','b']

# Create dataframe
df = pd.DataFrame(data = d, index = i) #this time we create the data from a dict (last time is list of tuple)
df

#get the row index by index attribute.
df.index

# Bring the columns and place them in the index
#stack is bring data in column in put it together by row index.
stack = df.stack()
stack


# The index now includes the column names
#index level on is ['a', 'b'], level tow is ['one', 'two']
stack.index

#unstack with unstack method()
unstack = df.unstack()
unstack


#unstack data have two index level
# unstakc totally differe from original data (index is a, b)
#untack index level 0 is ['one', 'two'], level 1: ['a', 'b'].
unstack.index

#tranpose with T attribute.
#tranpose is change index into column and column ito index.
transpose = df.T
transpose

#index of tranpose.
#index is ['one', 'two']
transpose.index