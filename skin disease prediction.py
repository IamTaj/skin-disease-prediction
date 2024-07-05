#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv('E:\dermatology.csv')


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


#machine learning
from statsmodels.tools import add_constant as add_constant
df_constant = add_constant(df)
df_constant.head()


# In[8]:


df['outcome'].unique ()


# In[9]:


### split dataset into independent and deprndent features
# for X
X=df.iloc[:,:-1] # include(const dx, dx_type, age, sex) 

# for Y
Y=df.iloc[:,-1] # only localozation


# In[10]:


X


# In[11]:


Y


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1)


# In[46]:


from sklearn.linear_model import LogisticRegression
logistic_regression=LogisticRegression()
logistic_regression.fit(X_train,Y_train)


# In[47]:


from sklearn.model_selection import GridSearchCV
parameter={'penalty':['l1','l2','elasticnet'],'C':[1,2,3,4,5,6,10,20,30,40,50],'max_iter':[100,200,300]}


# In[48]:


logistic_regression=GridSearchCV(logistic_regression,param_grid=parameter,scoring='accuracy',cv=5)


# In[49]:


logistic_regression.fit(X_train,Y_train)


# In[50]:


print(logistic_regression.best_params_)


# In[51]:


print(logistic_regression.best_score_)


# In[52]:


Y_pred=logistic_regression.predict(X_test)


# In[53]:


confusion_matrix = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)


# In[54]:


from sklearn.metrics import accuracy_score


# In[55]:


score=accuracy_score(Y_pred,Y_test)
print(score)


# In[ ]:


#EDA
sns.pairplot(df,hue='outcome')


# In[ ]:




