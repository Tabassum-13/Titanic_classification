#!/usr/bin/env python
# coding: utf-8

# # #BHARAT INTERNSHIP
# 
# ## NAME-SHAIK TABASSUM SHABANA
#  
#  ## TASK2-TITANIC CLASSIFICATION
#   - In this task we predicts if a passenger will survive on the titanic or not
#   
#   # Import Libraries

# In[5]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report


# # Loading the Dataset 

# In[7]:


titanic=pd.read_csv(r"C:\Users\smily\Downloads\archive\tested.csv")


# In[8]:


titanic


# # Cleaning the data

# In[9]:


titanic.shape


# In[10]:


titanic.info()   #statistical info


# In[11]:


titanic.describe()     


# In[12]:


titanic.head(10)


# In[13]:


titanic.columns


# #  Exploratory Data Analysis

# In[14]:


titanic["Sex"].value_counts


# In[15]:


titanic['Survived'].value_counts()       #count of no.of survivors


# # Countplot of survived vs not survived

# In[16]:


sns.countplot(x='Survived',data=titanic)


# # Male vs Female Survival

# In[17]:


sns.countplot(x='Survived',data=titanic,hue='Sex')


# In[18]:


sns.countplot(x='Survived',hue='Pclass',data=titanic)


# # Missing Data

# In[19]:


titanic.isna()     #check for null


# In[20]:


titanic.isnull().sum()


# # Visualize null values

# In[21]:


sns.heatmap(titanic.isna())


# In[22]:


(titanic['Age'].isna().sum()/len(titanic['Age']))*100


# In[23]:


(titanic['Cabin'].isna().sum()/len(titanic['Cabin']))*100


# In[24]:


sns.distplot(titanic['Age'].dropna(),kde=False,color='blue',bins=40)


# In[25]:


titanic['Age'].hist(bins=40,color='blue',alpha=0.4)


# In[26]:


sns.countplot(x='SibSp',data=titanic)


# In[27]:


titanic['Fare'].hist(color='red',bins=40,figsize=(8,4))


# In[28]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=titanic,palette='winter')


# In[29]:


titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)


# In[30]:


titanic['Age'].isna().sum()


# In[31]:


sns.heatmap(titanic.isna())


# - We can see cabin  column has a no.of of null values ,as such we can not use it for prediction so we will drop it 

# In[32]:


titanic.drop('Cabin',axis=1,inplace=True)


# In[33]:


titanic.head()


# # Preparing Data for Model

# In[34]:


titanic.info()


# # convert sex column to numerical values

# In[35]:


gender=pd.get_dummies(titanic['Sex'],drop_first=True)


# In[36]:


titanic['Gender']=gender
titanic.head()


# # Drop the columns which are not required 

# In[37]:


titanic.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)


# In[38]:


titanic.head()


# # Separate Dependent and Independent variables

# In[39]:


x=titanic[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=titanic['Survived']


# In[40]:


y


# # Data Modeling

# >Building model using logestic regression

# In[41]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.22,random_state=41)


# In[42]:


yr=LogisticRegression()


# In[43]:


yr.fit(x_train,y_train)


# In[44]:


predict=yr.predict(x_train)           #predict


# # Testing

# >See how our model works

# In[45]:


pd.DataFrame(confusion_matrix(y_train,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])


# In[46]:


print(classification_report(y_train,predict))


# In[ ]:




