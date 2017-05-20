# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:52:02 2017

@author: User
"""

import pandas as pd
import sys
import os

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
os.chdir('D:\\Toan\\1. NCU Master Degree\\Semester 4\\machineLearning\\pandas')
print(os.getcwd())
# Create DataFrame
d = [1,2,3,4,5,6,7,8,9]
df = pd.DataFrame(d, columns = ['Number'])
df

# Export to Excel
df.to_excel('Lesson10.xlsx', sheet_name = 'testing', index = False)
print('Done')

df.dtypes

df.tail()

df.to_json('Lesson10.json')
print('Done')

# Your path will be different, please modify the path below.
jsonloc = os.path.join(os.getcwd(),'Lesson10.json')

# read json file
df2 = pd.read_json(jsonloc)
df2
df2.dtypes
