#!/usr/bin/env python
# coding: utf-8

# In[3]:


##basic library - Pandas and Numpy
import pandas as pd
import numpy as np

## Imports for Data Consistency - String Match
import difflib as dff

## Imports for different type of classfiers
from sklearn import tree # <- Decision- Trees
from sklearn import svm # <- Support Vector Machines
import sklearn.linear_model as linear_model # <- Logisitic Regression - Sigmoid Function on the Linear Regression
from sklearn.ensemble import RandomForestClassifier # <- Random Forest Classifier
from sklearn.neural_network import MLPClassifier # <- Neural Networks
from sklearn.naive_bayes import GaussianNB # <- Gaussian Naive-Bayes Classifier

## Imports for recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

## Imports for splitting the data into training and test data
from sklearn.model_selection import train_test_split

## Imports for evaluating the different classifier models selected
import sklearn.metrics as metrics
from sklearn import preprocessing

## Data Visualisation
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


path_csv = "~/Downloads/h1b_kaggle.csv"


# In[5]:


## Define the column names and read the data file into a pandas dataframe
column_names = ['CASE_STATUS', 'EMPLOYER_NAME','SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'FILING_YEAR',               'WORKSITE', 'LONGITUDE', 'LATITUDE']
table_1 = pd.read_table(path_csv, names = column_names, skiprows = 1, error_bad_lines = False, sep = ',')


# In[6]:


pd.set_option('display.max_colwidth', -1)
pd.options.mode.chained_assignment = None


# In[7]:


table_1.head()


# In[11]:


plot_status_numberinit = table_1['CASE_STATUS'].value_counts().plot(title = 'CASE STATUS vs NUMBER OF PETITIONS',                                                                 kind = 'barh', color = 'green')
plot_status_numberinit.set_xlabel("CASE STATUS")
plot_status_numberinit.set_ylabel("NUMBER OF PETITIONS")
plt.show()
print(table_1['CASE_STATUS'].value_counts())


# In[12]:


# Data Analysis1- Row Counts v/s Case Status of the visa petition 

plot_status_number = table_1['CASE_STATUS'].value_counts().plot(title = 'CASE STATUS vs NUMBER OF PETITIONS',                                                                 kind = 'bar', color = 'green')
plot_status_number.set_xlabel("CASE STATUS")
plot_status_number.set_ylabel("NUMBER OF PETITIONS")
for p in plot_status_number.patches:
    plot_status_number.annotate(str(p.get_height()), (p.get_x() * 1.0050, p.get_height() * 1.005))
plot_status_number


# In[13]:


# Data Analysis2- The top 15 employers filing the H1-B visa petitions
plot_status_topemp= table_1['EMPLOYER_NAME'].value_counts().head(15).plot.barh(title = "Top 15 employers filing the petitions",                                                                  color = 'green', figsize = (7, 5))
plot_status_topemp.set_ylabel("name of the employer")
plot_status_topemp.set_xlabel("NUMBER OF PETITIONS")
plot_status_topemp
print(table_1['EMPLOYER_NAME'].value_counts().head(15))


# In[20]:


# Data Analysis3 - The top 15 SOC names for which H1-B visas are raised
plot_status_topsoc=table_1['SOC_NAME'].value_counts().head(15).plot.barh(title="Top 15 SOC",                                                                         color='green',figsize=(7,5))
plot_status_topsoc.set_ylabel("SOC NAME")
plot_status_topsoc.set_xlabel("NUMBER OF PETITIONS")
table_1['SOC_NAME'].value_counts().head(15)
                                                                         
                                                          


# In[ ]:




