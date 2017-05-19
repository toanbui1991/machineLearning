# -*- coding: utf-8 -*-
"""
Created on Thu May 18 00:32:57 2017

@author: Bui Xuan Toan.
"""


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt 
import numpy.random as np
import sys
import matplotlib
import os

print(os.getcwd())
os.chdir('D:\\Toan\\1. NCU Master Degree\\Semester 4\\machineLearning\\pandas')
print(os.getcwd())

#%matplotlib inline


print('Python version ' + sys.version)
print('Pandas version: ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)


# set seed
np.seed(111)

# Function to generate test data
def CreateDataSet(Number=1):
    
    Output = []
    
    for i in range(Number):
        
        # Create a weekly (mondays) date range
        rng = pd.date_range(start='1/1/2009', end='12/31/2012', freq='W-MON')
        
        # Create random data
        data = np.randint(low=25,high=1000,size=len(rng))
        
        # Status pool
        status = [1,2,3]
        
        # Make a random list of statuses
        random_status = [status[np.randint(low=0,high=len(status))] for i in range(len(rng))]
        
        # State pool
        states = ['GA','FL','fl','NY','NJ','TX']
        
        # Make a random list of states 
        random_states = [states[np.randint(low=0,high=len(states))] for i in range(len(rng))]
    
        Output.extend(zip(random_states, random_status, data, rng))
        
    return Output

#randonly create the data set    
dataset = CreateDataSet(4) #4 is the number of iteration fo create data.
df = pd.DataFrame(data=dataset, columns=['State','Status','CustomerCount','StatusDate'])
df.info()

df.head()


#write the data to excel file.
df.to_excel('lesson3.xlsx', index=False)
print('done')


#Read the excel file with pandas.
#pd.read_excel?
location = os.path.join(os.getcwd(), 'lesson3.xlsx')
df2 = pd.read_excel(location,0, index_col = 'StatusDate' ) #0 is the header row.
df2.dtypes

#get the index column.
df2.index

#you want to find a set of states.
df2['State'].unique()

# Clean State Column, convert to upper case
df2['State'] = df2['State'].apply(lambda x: x.upper())
df2['State'].unique()

#choose the data point base on testing.
#Status == 1
df2[df2['Status'] == 1].head(6)

#index the data point base on testing and do action base on that index.
#change from NJ to NY.
mask = df2['State'] == 'NJ'
df2['State'][mask] = 'NY'
df2['State'].unique()

#ploting data line.
df2['CustomerCount'].plot(figsize=(15,5))


#you want to get all the data point in State == 'NY'.
#you also want to sort_index (sort by the index_col)
df2[df2['State'] == 'NY'].sort_index()

#you want to group_by ['State', 'StatusDate']
daily = df2.reset_index().groupby(['State', 'StatusDate']).sum()
daily

#you want to delete a column.
del daily['Status']
daily


#now you want to see the index create by groupby()
#it is multiindex ['State', 'StatusDate']
daily.index

# Select the State index
daily.index.levels[0]

# Select the StatusDate index
daily.index.levels[1]

#now you have index by groupby.
#you want to use it to seperate the data by group.
daily.loc['FL'].plot()
daily.loc['GA'].plot()
daily.loc['NY'].plot()

#you can also seperate the data by two groups.
daily.loc['NY']['2012':].plot()

# Calculate Outliers
StateYearMonth = daily.groupby([daily.index.get_level_values(0), daily.index.get_level_values(1).year, daily.index.get_level_values(1).month])
daily['Lower'] = StateYearMonth['CustomerCount'].transform( lambda x: x.quantile(q=.25) - (1.5*x.quantile(q=.75)-x.quantile(q=.25)) )
daily['Upper'] = StateYearMonth['CustomerCount'].transform( lambda x: x.quantile(q=.75) + (1.5*x.quantile(q=.75)-x.quantile(q=.25)) )
daily['Outlier'] = (daily['CustomerCount'] < daily['Lower']) | (daily['CustomerCount'] > daily['Upper']) 

# Remove Outliers
daily = daily[daily['Outlier'] == False] #note index technique is very important.


# Combine all markets

# group the data by StatusDate.
ALL = pd.DataFrame(daily['CustomerCount'].groupby(daily.index.get_level_values(1)).sum())
#print(ALL)
ALL.columns = ['CustomerCount'] # rename column
#print(ALL)

# Group by Year and Month
YearMonth = ALL.groupby([lambda x: x.year, lambda x: x.month])

# What is the max customer count per Year and Month
ALL['Max'] = YearMonth['CustomerCount'].transform(lambda x: x.max())
ALL.head()

# Create the BHAG dataframe
data = [1000,2000,3000]
idx = pd.date_range(start='12/31/2011', end='12/31/2013', freq='A')
BHAG = pd.DataFrame(data, index=idx, columns=['BHAG'])
BHAG


# Combine the BHAG and the ALL data set 
combined = pd.concat([ALL,BHAG], axis=0)
combined = combined.sort_index(axis=0)
combined.tail()

#plotting.
fig, axes = plt.subplots(figsize=(12, 7))

combined['BHAG'].fillna(method='pad').plot(color='green', label='BHAG')
combined['Max'].plot(color='blue', label='All Markets')
plt.legend(loc='best');


# Group by Year and then get the max value per year
Year = combined.groupby(lambda x: x.year).max()
Year

# Add a column representing the percent change per year
Year['YR_PCT_Change'] = Year['Max'].pct_change(periods=1)
Year
#compute the next year CustomerCount given the same increase rate.
(1 + Year.ix[2012,'YR_PCT_Change']) * Year.ix[2012,'Max']

# First Graph
ALL['Max'].plot(figsize=(10, 5));plt.title('ALL Markets')

# Last four Graphs
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
fig.subplots_adjust(hspace=1.0) ## Create space between plots

daily.loc['FL']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[0,0])
daily.loc['GA']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[0,1]) 
daily.loc['TX']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[1,0]) 
daily.loc['NY']['CustomerCount']['2012':].fillna(method='pad').plot(ax=axes[1,1]) 

# Add titles
axes[0,0].set_title('Florida')
axes[0,1].set_title('Georgia')
axes[1,0].set_title('Texas')
axes[1,1].set_title('North East');