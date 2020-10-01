#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns                  
import matplotlib.pyplot as plt   
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                        
warnings.filterwarnings("ignore")
import scipy as sp
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas.plotting import scatter_matrix


# In[2]:


# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']  # names of columns
dataset = pd.read_csv(url, names=names)    # define the dataset


# In[3]:


print(dataset.shape)


# In[4]:


print(dataset.head(30))     # first 30 instances


# In[5]:


print(dataset.describe())     # summary of attribute(to count mean,min,max values)


# In[6]:


print(dataset.groupby('class').size())


# In[7]:


# create univariate plot
dataset.plot(kind='box', subplots = True, layout=(2,2), sharex = False, sharey = False)
plt.show()


# In[8]:


# create histogram
dataset.hist()
plt.show()


# In[9]:


# create scatter plot
scatter_matrix(dataset)
plt.show()


# In[10]:


# create validation dataset(training data set) of unseen data
array=dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


# In[11]:


seed = 6
scoring = 'accuracy'


# In[14]:


# spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:




